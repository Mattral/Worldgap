"""Landmark normalization (spec 5.2), with a named, per-study anchor scheme.

Perception states arrive from MediaPipe in image coordinates, so a person who
sits further back, or moves toward the frame edge, produces different numbers
for the same gesture. Normalization removes that:

- **pose**: translate by an anchor point, scale by shoulder width
  (landmarks 11, 12);
- **each hand**: translate by its wrist (landmark 0), scale by the diagonal of
  its bounding box.

Two named schemes differ only in the pose anchor:

- ``"shoulder_midpoint"`` (default): midpoint of landmarks 11 and 12. A
  documented deviation from spec 5.2. In seated, upper-body webcam framing the
  hips are usually out of frame; in the first real V1 run's clean recordings
  MediaPipe's mean visibility was 0.005 for the hips and 0.999 for the
  shoulders, so a hip anchor is MediaPipe's extrapolation, not an observed
  point.
- ``"hip_midpoint"``: midpoint of landmarks 23 and 24, exactly as spec 5.2
  specifies. Kept for framings where the hips are visible.

**The scheme is fixed per study.** It is chosen once, recorded in every
rollout's metadata, and never switched per frame or on visibility: a
switching anchor would put the anchor change inside the measured gap.
`require_single_scheme` enforces this wherever rollouts are combined, and
`GapAnalyzer` refuses to train on, or compare, rollouts with different
schemes.

Width and diagonal are measured in the image plane (x, y), where MediaPipe's
coordinates are reliable; the same scale is applied to z. The pose
`visibility` channel and the presence mask are never touched: they are spec
8.1's ground-truth signals.

Per-frame parameters live in ``metadata["normalization"]`` as plain JSON
lists (they have to survive the SQLite index), and `denormalize_states`
inverts the transform exactly, as spec 5.2 requires.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .rollout import PERCEPTION_FEATURE_LAYOUT, PERCEPTION_STATE_DIM, Rollout

NORMALIZATION_KEY = "normalization"

# MediaPipe pose topology
_L_SHOULDER, _R_SHOULDER, _L_HIP, _R_HIP = 11, 12, 23, 24
_WRIST = 0

#: pose anchor landmarks per scheme
SCHEMES: dict[str, tuple[int, int]] = {
    "shoulder_midpoint": (_L_SHOULDER, _R_SHOULDER),
    "hip_midpoint": (_L_HIP, _R_HIP),
}
DEFAULT_SCHEME = "shoulder_midpoint"

#: What to do when the guard refuses. Normalization is per frame, so applying
#: it to rollouts saved unnormalized (e.g. the first V1 run's stores) gives
#: exactly what re-extraction would; verified on run 1's data, max difference 0.
REMEDY = (
    "Fix: re-extract every rollout from the original recordings with one scheme "
    "(the loaders' normalization_scheme=..., or run_v1_real_data.py --normalization), "
    "then re-fit. For rollouts saved UNnormalized, normalize_rollout(r, scheme) gives "
    "the same states without re-extracting. Do not compare across schemes."
)

# Below this, a reference length is treated as degenerate (all landmarks at
# one point, which real detections never produce): the block is translated
# but not scaled, and the frame is counted in `n_degenerate_frames`.
_MIN_SCALE = 1e-6


def _check_scheme(scheme: str) -> None:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown normalization scheme {scheme!r}; choose one of {sorted(SCHEMES)}")


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


def _reference(xyz: np.ndarray, group: str, scheme: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame origin (T, 3) and scale (T,) for one block."""
    if group == "pose":
        a, b = SCHEMES[scheme]
        origin = (xyz[:, a] + xyz[:, b]) / 2.0
        scale = np.linalg.norm(xyz[:, _L_SHOULDER, :2] - xyz[:, _R_SHOULDER, :2], axis=1)
    else:
        origin = xyz[:, _WRIST]
        extent = xyz[:, :, :2].max(axis=1) - xyz[:, :, :2].min(axis=1)
        scale = np.linalg.norm(extent, axis=1)
    return origin, scale


def normalize_perception_states(
    states: np.ndarray, presence_mask: np.ndarray, scheme: str = DEFAULT_SCHEME
) -> tuple[np.ndarray, dict[str, Any]]:
    """Returns (normalized states, JSON-serializable per-frame parameters).

    One `scheme` for every frame. Absent blocks (presence False) keep their
    zeros, with origin [0, 0, 0] and scale 1.0 recorded so
    `denormalize_states` leaves them as they were.
    """
    _check_scheme(scheme)
    if states.ndim != 2 or states.shape[1] != PERCEPTION_STATE_DIM:
        raise ValueError(f"expected states of shape (T, {PERCEPTION_STATE_DIM}), got {states.shape}")
    out = np.array(states, dtype=np.float64, copy=True)
    per_frame: dict[str, Any] = {}
    n_degenerate = 0
    for group in PERCEPTION_FEATURE_LAYOUT:
        xyz, cols = _block_xyz(out, group)
        present = _block_present(presence_mask, group)
        origin, scale = _reference(xyz, group, scheme)
        degenerate = present & (scale < _MIN_SCALE)
        n_degenerate += int(degenerate.sum())
        origin = np.where(present[:, None], origin, 0.0)
        scale = np.where(present & ~degenerate, scale, 1.0)
        normed = (xyz - origin[:, None, :]) / scale[:, None, None]
        normed = np.where(present[:, None, None], normed, xyz)
        out[:, cols] = normed.reshape(out.shape[0], -1)
        per_frame[f"{group}_origin"] = origin.tolist()
        per_frame[f"{group}_scale"] = scale.tolist()
    return out, {"scheme": scheme, "n_degenerate_frames": n_degenerate, "per_frame": per_frame}


def denormalize_states(states: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """Inverts `normalize_perception_states` using the stored parameters."""
    _check_scheme(params.get("scheme"))
    out = np.array(states, dtype=np.float64, copy=True)
    pf = params["per_frame"]
    for group in PERCEPTION_FEATURE_LAYOUT:
        xyz, cols = _block_xyz(out, group)
        origin = np.asarray(pf[f"{group}_origin"], dtype=np.float64)
        scale = np.asarray(pf[f"{group}_scale"], dtype=np.float64)
        raw = xyz * scale[:, None, None] + origin[:, None, :]
        out[:, cols] = raw.reshape(out.shape[0], -1)
    return out


def normalization_scheme(rollout: Rollout) -> str | None:
    """The rollout's normalization scheme, or None if it is not normalized."""
    return rollout.metadata.get(NORMALIZATION_KEY, {}).get("scheme")


def is_normalized(rollout: Rollout) -> bool:
    return normalization_scheme(rollout) in SCHEMES


def require_single_scheme(rollouts: Iterable[Rollout], purpose: str) -> str | None:
    """Returns the one normalization scheme shared by `rollouts` (None if none
    of them are normalized), or raises if they disagree.

    Raw and normalized rollouts count as different schemes: putting them in
    one comparison is the same mistake as switching anchors.
    """
    schemes = {normalization_scheme(r) for r in rollouts}
    if len(schemes) > 1:
        shown = sorted("unnormalized" if s is None else s for s in schemes)
        raise ValueError(
            f"refusing to {purpose} rollouts with different landmark normalization "
            f"schemes {shown}. The scheme is fixed per study: a gap measured across "
            f"schemes includes the change of anchor itself, not just the change of "
            f"conditions. {REMEDY}"
        )
    return next(iter(schemes)) if schemes else None


def normalize_rollout(rollout: Rollout, scheme: str = DEFAULT_SCHEME) -> Rollout:
    """Returns a new, normalized perception rollout. Refuses to normalize twice,
    which would silently compound the transform."""
    if rollout.modality != "perception":
        raise ValueError("landmark normalization applies to perception rollouts only")
    if NORMALIZATION_KEY in rollout.metadata:
        raise ValueError("rollout is already normalized (metadata['normalization'] is set)")
    states, params = normalize_perception_states(rollout.states, rollout.presence_mask, scheme)
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
