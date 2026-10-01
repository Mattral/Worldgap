"""HaGRID loader tests, including the guards added after the static-image vs.
trajectory design gap was resolved (see docs/temporal_provenance.md).
"""

import numpy as np
import pytest

from worldgap.data.loaders.hagrid import (
    extract_rollout_from_frames,
    extract_static_pose_rollouts,
    list_hagrid_images,
    list_hagrid_sequences,
)
from worldgap.data.rollout import PERCEPTION_STATE_DIM


def _make_fixture_hagrid_dir(tmp_path):
    for gesture in ["fist", "palm", "rock", "peace"]:  # only fist/palm are canonical
        gdir = tmp_path / gesture
        gdir.mkdir()
        (gdir / "img_0001.jpg").write_bytes(b"")
        (gdir / "img_0002.jpg").write_bytes(b"")
        (gdir / "labels.json").write_text("{}")  # non-image, must be ignored
    return tmp_path


def test_list_images_filters_to_canonical_gestures_only(tmp_path):
    root = _make_fixture_hagrid_dir(tmp_path)
    images = list_hagrid_images(root, gestures={"fist", "palm"})
    assert {p.parent.name for p in images} == {"fist", "palm"}


def test_list_images_ignores_non_image_files(tmp_path):
    root = _make_fixture_hagrid_dir(tmp_path)
    images = list_hagrid_images(root, gestures={"fist"})
    assert [p.name for p in images] == ["img_0001.jpg", "img_0002.jpg"]


def test_missing_directory_raises_with_actionable_message(tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="runbook"):
        list_hagrid_images(missing)


def test_old_sequences_name_still_works_but_warns_that_it_is_misnamed(tmp_path):
    root = _make_fixture_hagrid_dir(tmp_path)
    with pytest.warns(DeprecationWarning, match="image dataset"):
        out = list_hagrid_sequences(root, gestures={"fist"})
    assert [p.name for p in out] == ["img_0001.jpg", "img_0002.jpg"]


def test_building_one_trajectory_from_hagrid_stills_is_refused(tmp_path):
    """The whole point of the design decision: this must not be quietly possible."""
    root = _make_fixture_hagrid_dir(tmp_path)
    frames = list_hagrid_images(root, gestures={"fist"})
    with pytest.raises(NotImplementedError, match="invents a time axis"):
        extract_rollout_from_frames(frames, landmarker=object())


# --- static-pose path ---------------------------------------------------------


class _FakeLandmark:
    def __init__(self, v=0.5):
        self.x = self.y = self.z = v
        self.visibility = 0.9


class _FakeResult:
    def __init__(self, hands=True):
        self.pose_landmarks = [_FakeLandmark() for _ in range(33)]
        self.left_hand_landmarks = [_FakeLandmark() for _ in range(21)] if hands else []
        self.right_hand_landmarks = [_FakeLandmark() for _ in range(21)] if hands else []


class _FakeLandmarker:
    def detect(self, _image):
        return _FakeResult()


def test_static_pose_rollouts_are_single_frame_and_tagged(tmp_path, monkeypatch):
    root = _make_fixture_hagrid_dir(tmp_path)
    images = list_hagrid_images(root, gestures={"fist"})

    class _FakeImage:
        @staticmethod
        def create_from_file(_path):
            return object()

    import sys
    import types

    fake_mp = types.ModuleType("mediapipe")
    fake_mp.Image = _FakeImage
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)

    rollouts = extract_static_pose_rollouts(images, _FakeLandmarker())

    assert len(rollouts) == 2
    for r in rollouts:
        assert r.states.shape == (1, PERCEPTION_STATE_DIM)
        assert r.temporal_provenance == "static_pose"
        assert r.has_ordered_time_axis is False
        # no invented frame rate for a photograph
        assert r.frame_rate_hz == 0.0
        assert r.metadata["gesture"] == "fist"


def test_static_pose_provenance_rejects_multiframe_states():
    """Stacking several stills into one 'static_pose' rollout is exactly the
    pseudo-trajectory the field exists to block, so the schema refuses it.
    """
    from worldgap.data.rollout import Rollout

    with pytest.raises(ValueError, match="pseudo-trajectory"):
        Rollout(
            modality="perception",
            source="real",
            condition={},
            frame_rate_hz=0.0,
            states=np.zeros((5, PERCEPTION_STATE_DIM)),
            presence_mask=np.ones((5, PERCEPTION_STATE_DIM)),
            timestamps_ms=np.zeros(5),
            metadata={"temporal_provenance": "static_pose"},
        )
