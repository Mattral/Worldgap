"""HaGRID loader, per TECHNICAL_SPEC.md Section 5.4.

Directory scanning (`list_hagrid_sequences`) and the MediaPipe-result-to-
feature-vector conversion (`extract_rollout_from_frames`, implemented in
`mediapipe_extract.py`) are both real and unit tested with zero network
access -- see tests/test_hagrid_loader.py and tests/test_mediapipe_extract.py.

NOT executed against real data in this scaffolding session: this sandbox's
network is allow-listed to GitHub/PyPI/OS package registries only and cannot
reach Kaggle (where HaGRID is hosted) or `storage.googleapis.com` (where
MediaPipe's `.task` model bundles are hosted — confirmed via a direct
request, not assumed). Run this against a real downloaded HaGRID directory
and a real constructed HolisticLandmarker in an environment with both —
your own machine, or Claude Code — see ROADMAP.md Phase 0/1 and
scripts/download_datasets.sh.

Requires the `perception` extra installed for MediaPipe.
"""

from __future__ import annotations

from pathlib import Path

from ..rollout import Rollout
from .mediapipe_extract import extract_rollout_from_frames as _extract_rollout_from_frames

# Canonical gesture subset relevant to the ForceHand glove's controllable DOFs
# (grasp open/close, wrist flexion/extension) — spec 5.4 requirement: MUST
# filter to gestures matched to the target device's DOFs rather than using all
# 18 HaGRID classes indiscriminately (comparing unrelated motion vocabularies
# is not a domain gap measurement).
#
# Phase 0 data audit (ROADMAP.md), partially resolved without Kaggle access:
# all four names below are confirmed real HaGRID v1 class names (the full set
# of 18 is: call, dislike, fist, four, like, mute, ok, one, palm, peace,
# peace_inverted, rock, stop, stop_inverted, three, three2, two_up,
# two_up_inverted), and the dataset's own paper states each class contains
# 30,000+ images (Kapitanov et al. 2022, arXiv:2206.08219) — comfortably
# enough for any reasonable train/test split, so sample-count sufficiency is
# no longer an open question. What's still NOT confirmed, and does need a
# real download to check: whether these four specific gestures are actually
# the best semantic match to the ForceHand glove's real controllable DOFs —
# that's a domain-expertise judgment call, not a data-availability one.
CANONICAL_GESTURES = {"fist", "palm", "stop", "like"}


def list_hagrid_sequences(hagrid_root: Path, gestures: set[str] = CANONICAL_GESTURES) -> list[Path]:
    """Scans a locally-downloaded HaGRID directory (see download_datasets.sh)
    and returns paths for sequences in the canonical gesture subset only.

    This part needs no MediaPipe and no network access — safe to run and unit
    test here once a real (even tiny, fixture-sized) HaGRID-shaped directory
    exists locally.
    """
    if not hagrid_root.exists():
        raise FileNotFoundError(
            f"{hagrid_root} does not exist — run scripts/download_datasets.sh first "
            "(requires network access this sandbox does not have)"
        )
    sequences = []
    for gesture_dir in hagrid_root.iterdir():
        if gesture_dir.is_dir() and gesture_dir.name in gestures:
            sequences.extend(sorted(gesture_dir.glob("*")))
    return sequences


def extract_rollout_from_frames(
    frame_paths: list[Path],
    landmarker,
    frame_rate_hz: float = 30.0,
    condition: dict | None = None,
) -> Rollout:
    """Runs a MediaPipe HolisticLandmarker over a sequence of frames and
    packages the landmark trajectory as a Rollout.

    `landmarker` is expected to be an already-configured
    `mediapipe.tasks.python.vision.HolisticLandmarker` instance — confirm the
    exact construction call against current MediaPipe Tasks API docs at
    implementation time (spec Section 11 flags this explicitly: the API
    surface has moved from legacy `mp.solutions` to the Tasks API since much
    of the reference material on this was written).

    Implemented in `mediapipe_extract.py` (shared with egohands.py) and unit
    tested there with duck-typed fakes — see tests/test_mediapipe_extract.py.
    What's genuinely still untested, because it needs network access this
    sandbox's allowlist blocks (`storage.googleapis.com`, confirmed via a
    direct request) and real downloaded frames: constructing an actual
    HolisticLandmarker (which needs a downloaded `.task` model bundle) and
    running this against real HaGRID images end-to-end. Do that first in an
    environment with both available before trusting real V1 results.
    """
    return _extract_rollout_from_frames(
        frame_paths, landmarker, frame_rate_hz=frame_rate_hz, condition=condition, source="real"
    )
