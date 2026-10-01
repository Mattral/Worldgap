"""Spec 5.2 landmark normalization (MUST, V1).

Perception states arrive from MediaPipe in image coordinates, so a person who
sits further back, or moves toward the frame edge, produces different numbers
for the same gesture. Spec 5.2 removes that:

- **pose**: translate by the hip midpoint (landmarks 23, 24), scale by
  shoulder width (landmarks 11, 12);
- **each hand**: translate by its wrist (landmark 0), scale by the diagonal of
  its bounding box.

Width and diagonal are measured in the image plane (x, y), where MediaPipe's
coordinates are reliable; the same scale is applied to z. The pose
`visibility` channel and the presence mask are never touched: they are spec
8.1's ground-truth signals, and normalization must not move them.

Spec 5.2 also requires the parameters to be kept, so raw values stay
recoverable. They live in `metadata["normalization"]` as plain JSON lists
(they have to survive the SQLite index), one entry per frame, and
`denormalize_states` inverts the transform exactly.

This was not implemented through 0.2.0; the first real V1 run found it
(`docs/v1_first_run_results.md`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .rollout import PERCEPTION_FEATURE_LAYOUT, PERCEPTION_STATE_DIM, Rollout

NORMALIZATION_KEY = "normalization"
SCHEME = "spec_5_2"

# MediaPipe pose topology
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24
_WRIST = 0
# Below this, a reference length is treated as degenerate (all landmarks at
# one point, which real detections never produce): the block is translated
# but not scaled, and the frame is counted in `n_degenerate_frames`.
_MIN_SCALE = 1e-6


def _block_xyz(states: np.ndarray, group: str) -> tuple[np.ndarray, list[int]]:
    """(T, n_landmarks, 3) view of a block's x/y/z, plus the column indices."""
    g = PERCEPTION_FEATURE_LAYOUT[group]
    n_dims = len(g["dims"])
    cols = [
        g["start"] + i * n_dims + d
        for i in range(g["n_landmarks"])
        for d in range(3)  # x, y, z are always the first three dims
    ]
    return states[:, cols].reshape(states.shape[0], g["n_landmarks"], 3), cols


def _block_present(presence: np.ndarray, group: str) -> np.ndarray:
    g = PERCEPTION_FEATURE_LAYOUT[group]
    return presence[:, g["start"]] > 0


def _reference(xyz: np.ndarray, group: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame origin (T, 3) and scale (T,) for one block."""
    if group == "pose":
        origin = (xyz[:, _L_HIP] + xyz[:, _R_HIP]) / 2.0
        scale = np.linalg.norm(xyz[:, _L_SHOULDER, :2] - xyz[:, _R_SHOULDER, :2], axis=1)
    else:
        origin = xyz[:, _WRIST]
        extent = xyz[:, :, :2].max(axis=1) - xyz[:, :, :2].min(axis=1)
        scale = np.linalg.norm(extent, axis=1)
    return origin, scale


def normalize_perception_states(
    states: np.ndarray, presence_mask: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (normalized states, JSON-serializable per-frame parameters).

    Absent blocks (presence False) keep their zeros, with origin [0, 0, 0] and
    scale 1.0 recorded so `denormalize_states` leaves them as they were.
    """
    if states.ndim != 2 or states.shape[1] != PERCEPTION_STATE_DIM:
        raise ValueError(f"expected states of shape (T, {PERCEPTION_STATE_DIM}), got {states.shape}")
    out = np.array(states, dtype=np.float64, copy=True)
    per_frame: dict[str, Any] = {}
    n_degenerate = 0
    for group in PERCEPTION_FEATURE_LAYOUT:
        xyz, cols = _block_xyz(out, group)
        present = _block_present(presence_mask, group)
        origin, scale = _reference(xyz, group)
        degenerate = present & (scale < _MIN_SCALE)
        n_degenerate += int(degenerate.sum())
        origin = np.where(present[:, None], origin, 0.0)
        scale = np.where(present & ~degenerate, scale, 1.0)
        normed = (xyz - origin[:, None, :]) / scale[:, None, None]
        normed = np.where(present[:, None, None], normed, xyz)
        out[:, cols] = normed.reshape(out.shape[0], -1)
        per_frame[f"{group}_origin"] = origin.tolist()
        per_frame[f"{group}_scale"] = scale.tolist()
    return out, {"scheme": SCHEME, "n_degenerate_frames": n_degenerate, "per_frame": per_frame}


def denormalize_states(states: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Inverts `normalize_perception_states` using the stored parameters."""
    if params.get("scheme") != SCHEME:
        raise ValueError(f"unknown normalization scheme {params.get('scheme')!r}")
    out = np.array(states, dtype=np.float64, copy=True)
    pf = params["per_frame"]
    for group in PERCEPTION_FEATURE_LAYOUT:
        xyz, cols = _block_xyz(out, group)
        origin = np.asarray(pf[f"{group}_origin"], dtype=np.float64)
        scale = np.asarray(pf[f"{group}_scale"], dtype=np.float64)
        raw = xyz * scale[:, None, None] + origin[:, None, :]
        out[:, cols] = raw.reshape(out.shape[0], -1)
    return out


def is_normalized(rollout: Rollout) -> bool:
    return rollout.metadata.get(NORMALIZATION_KEY, {}).get("scheme") == SCHEME


def normalize_rollout(rollout: Rollout) -> Rollout:
    """Returns a new, normalized perception rollout. Refuses to normalize twice,
    which would silently compound the transform."""
    if rollout.modality != "perception":
        raise ValueError("spec 5.2 normalization applies to perception rollouts only")
    if is_normalized(rollout):
        raise ValueError("rollout is already normalized (metadata['normalization'] is set)")
    states, params = normalize_perception_states(rollout.states, rollout.presence_mask)
    return Rollout(
        modality=rollout.modality,
        source=rollout.source,
        condition=rollout.condition,
        frame_rate_hz=rollout.frame_rate_hz,
        states=states,
        presence_mask=rollout.presence_mask,
        timestamps_ms=rollout.timestamps_ms,
        metadata={**rollout.metadata, NORMALIZATION_KEY: params},
    )


def slice_normalization_params(params: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    """Per-frame parameters for frames [start, end), for windowed rollouts.

    The degenerate-frame count describes the whole source recording, so it is
    renamed rather than copied as if it were this window's.
    """
    sliced = {k: v for k, v in params.items() if k not in ("per_frame", "n_degenerate_frames")}
    sliced["n_degenerate_frames_in_source"] = params.get(
        "n_degenerate_frames", params.get("n_degenerate_frames_in_source")
    )
    sliced["per_frame"] = {k: v[start:end] for k, v in params["per_frame"].items()}
    return sliced
