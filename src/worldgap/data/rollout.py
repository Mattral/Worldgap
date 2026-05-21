"""Rollout schema, per TECHNICAL_SPEC.md Section 5.1-5.3.

A Rollout is the single unit of data in WorldGap: one timestamped sequence of
states for one task instance, regardless of modality. Perception rollouts hold
landmark sequences; actuation rollouts hold commanded-pressure/response
sequences. Everything downstream of this schema is modality-agnostic.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Modality = Literal["perception", "actuation"]
Source = Literal["real", "sim", "synthetic"]

# Fixed feature-index layout for perception rollouts (spec 5.1.1).
# This constant is the single source of truth for "which column is what" —
# never rely on positional convention elsewhere in the codebase.
PERCEPTION_FEATURE_LAYOUT = {
    "pose": {"start": 0, "n_landmarks": 33, "dims": ("x", "y", "z", "visibility")},
    "left_hand": {"start": 132, "n_landmarks": 21, "dims": ("x", "y", "z")},
    "right_hand": {"start": 195, "n_landmarks": 21, "dims": ("x", "y", "z")},
}
PERCEPTION_STATE_DIM = 33 * 4 + 21 * 3 + 21 * 3  # 258


def perception_position_channel_mask() -> np.ndarray:
    """Boolean mask, shape (PERCEPTION_STATE_DIM,), True for spatial (x/y/z)
    channels and False for the pose block's `visibility` channel.

    Derived purely from PERCEPTION_FEATURE_LAYOUT so it stays correct if the
    layout ever changes. Purely-spatial synthetic perturbations (tremor,
    reduced range of motion) MUST use this to avoid injecting noise into
    MediaPipe's own confidence/visibility signal — a landmark's visibility
    score isn't a "position" and jittering it silently changes what
    validation-harness ground truth (spec 8.1) would later measure.
    """
    mask = np.zeros(PERCEPTION_STATE_DIM, dtype=bool)
    for group in PERCEPTION_FEATURE_LAYOUT.values():
        start = group["start"]
        dims = group["dims"]
        n_dims = len(dims)
        for landmark_idx in range(group["n_landmarks"]):
            for dim_idx, dim_name in enumerate(dims):
                if dim_name != "visibility":
                    mask[start + landmark_idx * n_dims + dim_idx] = True
    return mask


PERCEPTION_POSITION_CHANNEL_MASK = perception_position_channel_mask()


@dataclass
class Rollout:
    modality: Modality
    source: Source
    condition: dict[str, Any]
    frame_rate_hz: float
    states: np.ndarray  # shape (T, D)
    presence_mask: np.ndarray  # shape (T, K) — see note below
    timestamps_ms: np.ndarray  # shape (T,)
    metadata: dict[str, Any] = field(default_factory=dict)
    rollout_id: str | None = None

    def __post_init__(self) -> None:
        if self.states.ndim != 2:
            raise ValueError(f"states must be 2D (T, D), got shape {self.states.shape}")
        if self.presence_mask.ndim != 2:
            raise ValueError(
                f"presence_mask must be 2D (T, K), got shape {self.presence_mask.shape}"
            )
        if self.states.shape[0] != self.presence_mask.shape[0]:
            raise ValueError(
                "states and presence_mask must have the same number of timesteps: "
                f"{self.states.shape[0]} vs {self.presence_mask.shape[0]}"
            )
        if self.timestamps_ms.shape[0] != self.states.shape[0]:
            raise ValueError("timestamps_ms length must match states' timestep count")
        if self.rollout_id is None:
            self.rollout_id = self._compute_id()

    def _compute_id(self) -> str:
        """Content hash so rollout_id is stable across reruns (spec 5.1: 'content-hashed,
        stable across reruns'), not a random uuid.
        """
        h = hashlib.sha256()
        h.update(self.states.tobytes())
        h.update(json.dumps(self.condition, sort_keys=True).encode())
        h.update(self.modality.encode())
        return h.hexdigest()[:16]

    # -- I/O -----------------------------------------------------------------

    def save(self, processed_dir: Path) -> Path:
        """Writes one .npz per rollout, per spec 5.3 storage layout."""
        out_dir = processed_dir / self.modality
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.rollout_id}.npz"
        np.savez_compressed(
            path,
            states=self.states,
            presence_mask=self.presence_mask,
            timestamps_ms=self.timestamps_ms,
        )
        return path

    @classmethod
    def load(
        cls,
        path: Path,
        modality: Modality,
        source: Source,
        condition: dict[str, Any],
        frame_rate_hz: float,
        metadata: dict[str, Any] | None = None,
    ) -> "Rollout":
        npz = np.load(path)
        return cls(
            modality=modality,
            source=source,
            condition=condition,
            frame_rate_hz=frame_rate_hz,
            states=npz["states"],
            presence_mask=npz["presence_mask"],
            timestamps_ms=npz["timestamps_ms"],
            metadata=metadata or {},
            rollout_id=path.stem,
        )


def validate_perception_rollout(rollout: Rollout) -> None:
    """Sanity-checks a perception rollout against PERCEPTION_STATE_DIM.

    Raises rather than silently accepting a malformed feature vector — a wrong
    state_dim here is exactly the kind of silent bug spec 5.1.1 warns about.
    """
    if rollout.modality != "perception":
        raise ValueError(f"expected modality='perception', got {rollout.modality!r}")
    if rollout.states.shape[1] != PERCEPTION_STATE_DIM:
        raise ValueError(
            f"perception rollout states must have D={PERCEPTION_STATE_DIM}, "
            f"got D={rollout.states.shape[1]}"
        )
