"""EgoHands loader, per TECHNICAL_SPEC.md Section 5.4 (secondary source for
occlusion-heavy conditions).

Same status as hagrid.py: the extraction logic itself is real and shared via
`mediapipe_extract.py` (tested in tests/test_mediapipe_extract.py), but NOT
run against real EgoHands frames in this scaffolding session (no network
access to the dataset host, or to `storage.googleapis.com` for MediaPipe's
model bundle, from this sandbox). Structure mirrors hagrid.py deliberately —
both loaders MUST produce Rollout objects with identical feature-vector
layout (PERCEPTION_FEATURE_LAYOUT), since the whole point is that source and
target domains are comparable in the same latent space.
"""

from __future__ import annotations

from pathlib import Path

from ..rollout import Rollout
from .mediapipe_extract import extract_rollout_from_frames as _extract_rollout_from_frames


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
    """See hagrid.py's `extract_rollout_from_frames` docstring — same shared
    implementation, same real-data/model-file caveat.
    """
    return _extract_rollout_from_frames(
        frame_paths, landmarker, frame_rate_hz=frame_rate_hz, condition=condition, source="real"
    )
