"""HaGRID loader, per TECHNICAL_SPEC.md Section 5.4.

NOT executed or tested in this scaffolding session: this sandbox's network is
allow-listed to GitHub/PyPI/OS package registries only and cannot reach Kaggle,
where HaGRID is hosted. Run and verify this against real downloaded data in an
environment with broader network access (your own machine, or Claude Code) —
see ROADMAP.md Phase 0/1 and scripts/download_datasets.sh.

Requires the `perception` extra installed for MediaPipe.
"""

from __future__ import annotations

from pathlib import Path

from ..rollout import Rollout

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

    NOT IMPLEMENTED here — this is the one seam in the loader that genuinely
    requires MediaPipe + real image files, neither of which this sandbox has.
    """
    raise NotImplementedError(
        "Implement against a real MediaPipe HolisticLandmarker instance and "
        "downloaded HaGRID frames in an environment with both available. "
        "See docs/data_spec.md for the exact feature-vector layout "
        "(PERCEPTION_FEATURE_LAYOUT in data/rollout.py) this must produce."
    )
