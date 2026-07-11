"""Tests for data/loaders/mediapipe_extract.py's conversion logic.

Deliberately uses small duck-typed fake objects instead of real `mediapipe`
types. `holistic_result_to_feature_vector` only needs something with
`.pose_landmarks`/`.left_hand_landmarks`/`.right_hand_landmarks` attributes,
and `extract_rollout_from_frames` takes an already-constructed `landmarker`
as a parameter rather than building one -- that's exactly what makes this
testable without the `perception` extra (mediapipe) installed at all, and
without any real model file or network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from worldgap.data.loaders.mediapipe_extract import (
    holistic_result_to_feature_vector,
    extract_rollout_from_frames,
)
from worldgap.data.rollout import PERCEPTION_FEATURE_LAYOUT, PERCEPTION_STATE_DIM


@dataclass
class FakeLandmark:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    visibility: float = 1.0


@dataclass
class FakeHolisticResult:
    pose_landmarks: list = field(default_factory=list)
    left_hand_landmarks: list = field(default_factory=list)
    right_hand_landmarks: list = field(default_factory=list)


def _full_detection(pose_value=0.5, hand_value=0.25) -> FakeHolisticResult:
    return FakeHolisticResult(
        pose_landmarks=[FakeLandmark(pose_value, pose_value, pose_value, 0.9) for _ in range(33)],
        left_hand_landmarks=[FakeLandmark(hand_value, hand_value, hand_value) for _ in range(21)],
        right_hand_landmarks=[FakeLandmark(hand_value, hand_value, hand_value) for _ in range(21)],
    )


def test_full_detection_produces_correct_shape_and_all_present():
    result = _full_detection()
    features, presence = holistic_result_to_feature_vector(result)
    assert features.shape == (PERCEPTION_STATE_DIM,)
    assert presence.shape == (PERCEPTION_STATE_DIM,)
    assert presence.all()


def test_full_detection_places_values_in_correct_layout_blocks():
    result = _full_detection(pose_value=0.5, hand_value=0.25)
    features, _ = holistic_result_to_feature_vector(result)

    pose = PERCEPTION_FEATURE_LAYOUT["pose"]
    left = PERCEPTION_FEATURE_LAYOUT["left_hand"]
    right = PERCEPTION_FEATURE_LAYOUT["right_hand"]

    # pose x/y/z should be 0.5, visibility 0.9
    assert features[pose["start"]] == pytest.approx(0.5)
    assert features[pose["start"] + 3] == pytest.approx(0.9)  # visibility channel
    # hand blocks are x/y/z only, no visibility channel
    assert features[left["start"]] == pytest.approx(0.25)
    assert features[right["start"]] == pytest.approx(0.25)


def test_missing_hand_detection_zero_fills_and_marks_absent():
    """A hand briefly out of frame is a normal, expected outcome, not an
    error -- must be represented as presence_mask=False, not dropped or
    crash the pipeline.
    """
    result = _full_detection()
    result.left_hand_landmarks = []  # not detected this frame

    features, presence = holistic_result_to_feature_vector(result)
    left = PERCEPTION_FEATURE_LAYOUT["left_hand"]
    left_slice = slice(left["start"], left["start"] + left["n_landmarks"] * len(left["dims"]))

    assert not presence[left_slice].any()
    assert (features[left_slice] == 0.0).all()
    # everything else should still be present
    assert presence[: left["start"]].all()


def test_no_detections_at_all_zero_fills_everything():
    result = FakeHolisticResult()  # nothing detected
    features, presence = holistic_result_to_feature_vector(result)
    assert not presence.any()
    assert (features == 0.0).all()


def test_landmark_count_mismatch_raises():
    result = _full_detection()
    result.pose_landmarks = result.pose_landmarks[:10]  # wrong count
    with pytest.raises(ValueError, match="expected 33 landmarks"):
        holistic_result_to_feature_vector(result)


class _FakeLandmarker:
    """Duck-typed stand-in for mediapipe.tasks.python.vision.HolisticLandmarker
    -- returns a fixed detection regardless of the image, since the point of
    this test is the frame-loop/Rollout-packaging logic, not real detection.
    """

    def __init__(self, results_by_index=None, default=None):
        self._results_by_index = results_by_index or {}
        self._default = default or _full_detection()
        self.calls = []

    def detect(self, image):
        self.calls.append(image)
        return self._results_by_index.get(len(self.calls) - 1, self._default)


def test_extract_rollout_from_frames_builds_correct_shape(tmp_path):
    pytest.importorskip("mediapipe")
    from PIL import Image as PILImage

    frame_paths = []
    for i in range(5):
        p = tmp_path / f"frame_{i}.png"
        PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(p)
        frame_paths.append(p)

    landmarker = _FakeLandmarker()
    rollout = extract_rollout_from_frames(
        frame_paths, landmarker, frame_rate_hz=30.0, condition={"gesture": "fist"}
    )

    assert rollout.states.shape == (5, PERCEPTION_STATE_DIM)
    assert rollout.presence_mask.shape == (5, PERCEPTION_STATE_DIM)
    assert rollout.modality == "perception"
    assert rollout.condition == {"gesture": "fist"}
    assert rollout.frame_rate_hz == 30.0
    assert len(landmarker.calls) == 5


def test_extract_rollout_from_frames_empty_list_raises():
    with pytest.raises(ValueError, match="empty"):
        extract_rollout_from_frames([], _FakeLandmarker())


def test_extract_rollout_preserves_per_frame_dropout(tmp_path):
    """A frame with no detection at all must show up as presence_mask=False
    for that frame, not silently disappear or get interpolated over --
    dropout rate is exactly what spec 8.1's validation ground truth needs.
    """
    pytest.importorskip("mediapipe")
    from PIL import Image as PILImage

    frame_paths = []
    for i in range(3):
        p = tmp_path / f"frame_{i}.png"
        PILImage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(p)
        frame_paths.append(p)

    # frame index 1 has no detection at all
    landmarker = _FakeLandmarker(results_by_index={1: FakeHolisticResult()})
    rollout = extract_rollout_from_frames(frame_paths, landmarker, frame_rate_hz=30.0)

    assert rollout.presence_mask[0].all()
    assert not rollout.presence_mask[1].any()
    assert rollout.presence_mask[2].all()
