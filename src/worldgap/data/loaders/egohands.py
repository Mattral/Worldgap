"""EgoHands loader, per TECHNICAL_SPEC.md Section 5.4 (secondary source for
occlusion-heavy conditions).

Unlike HaGRID, EgoHands is **video-derived**, so consecutive frames from one
clip really are consecutive moments and `extract_rollout_from_frames()` is the
right operation here — which is why this module keeps it while `hagrid.py`
now raises (see docs/temporal_provenance.md). Callers MUST pass frames from a
single clip, in order; frames pooled across clips are the same
pseudo-trajectory mistake under a different name.

The extraction logic itself is real and shared via `mediapipe_extract.py`
(tested in tests/test_mediapipe_extract.py), but has NOT been run against real
EgoHands frames — that needs the dataset and a MediaPipe `.task` bundle, both
outside this sandbox's network allow-list. Note that
`worldgap.data.loaders.video` is usually the better path even for EgoHands: it
decodes the source video directly and records the file's real frame rate
rather than relying on a caller-supplied default.

Feature-vector layout (PERCEPTION_FEATURE_LAYOUT) is identical to every other
perception loader's, since source and target domains only mean anything if
they are comparable in the same latent space.
"""

from __future__ import annotations

from pathlib import Path

from ..normalization import DEFAULT_SCHEME
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
    normalization_scheme: str = DEFAULT_SCHEME,
) -> Rollout:
    """See hagrid.py's `extract_rollout_from_frames` docstring — same shared
    implementation, same real-data/model-file caveat.
    """
    return _extract_rollout_from_frames(
        frame_paths,
        landmarker,
        frame_rate_hz=frame_rate_hz,
        condition=condition,
        source="real",
        normalization_scheme=normalization_scheme,
    )
