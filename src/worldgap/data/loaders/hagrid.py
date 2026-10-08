"""HaGRID loader, per TECHNICAL_SPEC.md Section 5.4 and the temporal-provenance
extension (docs/temporal_provenance.md).

**Read this before using HaGRID for a V1 run.** HaGRID is, by its own expanded
name, the "HAnd Gesture Recognition **Image** Dataset": ~554,800 unrelated
still photographs across 18 gesture classes, from different subjects, scenes
and sessions. It contains no video and no trajectories.

`Rollout`, the unit everything downstream of this module consumes, is a
*timestamped sequence*. Releases up to 0.1.0 shipped a `list_hagrid_sequences()`
that globbed a gesture folder and returned the individual image files, with an
`extract_rollout_from_frames()` beside it that would happily consume that list
as if it were consecutive frames. That combination produces an object shaped
exactly like a trajectory whose time axis is `sorted(glob())` -- filename
order. Every temporal claim computed on it (the world model predicting
dynamics, tremor as an 8 Hz oscillation, dropout as a run of missed frames)
would then be measuring an artifact of the filesystem. Nothing would error;
the numbers would just mean nothing.

So this module now does two things, and refuses the third:

1. `list_hagrid_images()` -- honest name for what it always returned.
2. `extract_static_pose_rollouts()` -- one T=1 `Rollout` per image, tagged
   `temporal_provenance="static_pose"`. Legitimate for comparing the
   *frame-level landmark distribution* between two image sets.
3. `extract_rollout_from_frames()` -- **raises**. Turning a folder of HaGRID
   stills into one trajectory is not a thing this library will do quietly.

For a real V1 run with genuine temporal structure, use `video.py` against
actual recordings -- including your own webcam captures, which need no dataset
download at all. See `docs/v1_real_data_runbook.md`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

from ..normalization import normalize_rollout
from ..rollout import TEMPORAL_PROVENANCE_KEY, Rollout
from .mediapipe_extract import holistic_result_to_feature_vector

# Canonical gesture subset relevant to the ForceHand glove's controllable DOFs
# (grasp open/close, wrist flexion/extension) -- spec 5.4 requirement: MUST
# filter to gestures matched to the target device's DOFs rather than using all
# 18 HaGRID classes indiscriminately (comparing unrelated motion vocabularies
# is not a domain gap measurement).
#
# Phase 0 data audit (ROADMAP.md), partially resolved without Kaggle access:
# all four names below are confirmed real HaGRID v1 class names (the full set
# of 18 is: call, dislike, fist, four, like, mute, ok, one, palm, peace,
# peace_inverted, rock, stop, stop_inverted, three, three2, two_up,
# two_up_inverted), and the dataset's own paper states each class contains
# 30,000+ images (Kapitanov et al. 2022, arXiv:2206.08219) -- comfortably
# enough for any reasonable train/test split, so sample-count sufficiency is
# no longer an open question. What's still NOT confirmed, and does need a
# domain-expert call rather than a download: whether these four specific
# gestures are the best semantic match to the target glove's real controllable
# DOFs (ROADMAP Phase 0, open question Q11).
CANONICAL_GESTURES = {"fist", "palm", "stop", "like"}

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_hagrid_images(
    hagrid_root: Path, gestures: set[str] = CANONICAL_GESTURES
) -> list[Path]:
    """Scans a locally-downloaded HaGRID directory and returns the individual
    image files belonging to the canonical gesture subset.

    These are **independent stills**, not frames of anything. The return value
    is deliberately named for that. Ordering is sorted-by-filename purely for
    reproducibility; it carries no temporal meaning.

    Needs no MediaPipe and no network access.
    """
    if not hagrid_root.exists():
        raise FileNotFoundError(
            f"{hagrid_root} does not exist -- see docs/v1_real_data_runbook.md for "
            "how to obtain HaGRID, and read that runbook's section on why HaGRID "
            "cannot supply V1's trajectories."
        )
    images: list[Path] = []
    for gesture_dir in sorted(hagrid_root.iterdir()):
        if gesture_dir.is_dir() and gesture_dir.name in gestures:
            images.extend(
                sorted(p for p in gesture_dir.glob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
            )
    return images


def list_hagrid_sequences(
    hagrid_root: Path, gestures: set[str] = CANONICAL_GESTURES
) -> list[Path]:
    """Deprecated alias for `list_hagrid_images`, kept so 0.1.0 callers don't
    break silently -- but the old name asserted something false.

    It never returned sequences. It returned individual image files.
    """
    warnings.warn(
        "list_hagrid_sequences() is misnamed: HaGRID is an image dataset and this "
        "function returns individual stills, not sequences. Use "
        "list_hagrid_images(). Feeding its output to extract_rollout_from_frames() "
        "as if it were a trajectory is the bug this rename exists to surface -- see "
        "docs/temporal_provenance.md.",
        DeprecationWarning,
        stacklevel=2,
    )
    return list_hagrid_images(hagrid_root, gestures)


def extract_static_pose_rollouts(
    image_paths: list[Path],
    landmarker,
    condition: dict | None = None,
    metadata: dict | None = None,
) -> list[Rollout]:
    """Extracts one T=1 `Rollout` per still image.

    This is the honest way to use HaGRID here. Each rollout is a single
    observed pose with no time axis, tagged `temporal_provenance="static_pose"`
    so nothing downstream can mistake it for motion.

    What you can legitimately do with the result: compare the *frame-level
    landmark distribution* of two image sets (e.g. well-lit vs. dim captures of
    the same gestures). That is a real and useful measurement -- it answers
    "does the pose estimator land in a different part of landmark space under
    these conditions" -- and it is a strictly weaker claim than V1's headline
    one, which is about how tracking degrades *over time*.

    What you cannot do: train the world model on these. `GapAnalyzer.fit()`
    will refuse them, because a JEPA objective needs a future window to predict
    and T=1 has none. Use a fitted model from video data, or use the
    non-temporal comparison path in `docs/v1_real_data_runbook.md`.

    `frame_rate_hz` is set to 0.0 rather than an invented 30.0: there is no
    frame rate for a photograph, and writing one down would be the same species
    of quiet fiction this module exists to prevent.
    """
    if not image_paths:
        raise ValueError("image_paths is empty -- nothing to extract")

    try:
        import mediapipe as mp
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "mediapipe is required to load images and run the landmarker "
            "(worldgap[perception]). Install with: pip install 'worldgap[perception]'"
        ) from e

    import numpy as np

    rollouts: list[Rollout] = []
    for path in image_paths:
        image = mp.Image.create_from_file(str(path))
        result = landmarker.detect(image)
        features, presence = holistic_result_to_feature_vector(result)
        rollouts.append(
            normalize_rollout(
                Rollout(
                    modality="perception",
                    source="real",
                    condition={**(condition or {}), "image": path.name},
                    frame_rate_hz=0.0,
                    states=features[None, :],
                    presence_mask=presence[None, :].astype(np.float64),
                    timestamps_ms=np.zeros(1),
                    metadata={
                        **(metadata or {}),
                        TEMPORAL_PROVENANCE_KEY: "static_pose",
                        "gesture": path.parent.name,
                    },
                )
            )
        )
    return rollouts


def extract_rollout_from_frames(*_args, **_kwargs):
    """Removed on purpose. HaGRID stills are not frames of a trajectory.

    If you want a per-image comparison, use `extract_static_pose_rollouts()`.
    If you want real trajectories, use `worldgap.data.loaders.video`.
    """
    raise NotImplementedError(
        "hagrid.extract_rollout_from_frames() has been removed because it was "
        "wrong, not because it was unfinished. HaGRID is an image dataset: "
        "concatenating unrelated stills into one Rollout invents a time axis out "
        "of filename order, and every temporal number computed from it would be "
        "measuring that artifact. Use extract_static_pose_rollouts() for a "
        "frame-level comparison, or worldgap.data.loaders.video for real "
        "trajectories. See docs/temporal_provenance.md."
    )
