"""EgoHands loader, per TECHNICAL_SPEC.md Section 5.4 (secondary source for
occlusion-heavy conditions).

Same status as hagrid.py: NOT executed or tested in this scaffolding session
(no network access to the dataset host from this sandbox). Structure mirrors
hagrid.py deliberately — both loaders MUST produce Rollout objects with
identical feature-vector layout (PERCEPTION_FEATURE_LAYOUT), since the whole
point is that source and target domains are comparable in the same latent
space.
"""

from __future__ import annotations

from pathlib import Path

from ..rollout import Rollout


def list_egohands_sequences(egohands_root: Path) -> list[Path]:
    if not egohands_root.exists():
        raise FileNotFoundError(
            f"{egohands_root} does not exist — run scripts/download_datasets.sh first"
        )
    return sorted(p for p in egohands_root.iterdir() if p.is_dir())


def extract_rollout_from_frames(
    frame_paths: list[Path],
    landmarker,
    frame_rate_hz: float = 30.0,
    condition: dict | None = None,
) -> Rollout:
    """See hagrid.py's `extract_rollout_from_frames` docstring — same
    MediaPipe-dependent seam, same reason it's not implemented here.
    """
    raise NotImplementedError(
        "Implement against a real MediaPipe HolisticLandmarker instance and "
        "downloaded EgoHands frames in an environment with both available."
    )
