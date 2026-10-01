"""Tests for the video loader and the spec-8.1 ground-truth signals.

Real video decoding needs OpenCV and an actual file, so `extract_rollout_from_video`
is exercised against a genuinely written video only when OpenCV is importable;
everything else here runs with no optional dependencies at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from worldgap.data.loaders.video import (
    landmark_quality_ground_truth,
    list_videos,
    split_into_windows,
)
from worldgap.data.rollout import PERCEPTION_FEATURE_LAYOUT, PERCEPTION_STATE_DIM, Rollout


def _perception_rollout(
    n_frames: int = 30,
    fps: float = 30.0,
    hands_missing_frames: set[int] | None = None,
    provenance: str | None = "video",
    pose_visibility: float = 0.9,
) -> Rollout:
    states = np.zeros((n_frames, PERCEPTION_STATE_DIM))
    mask = np.ones((n_frames, PERCEPTION_STATE_DIM))

    pose = PERCEPTION_FEATURE_LAYOUT["pose"]
    vis_cols = np.arange(
        pose["start"] + pose["dims"].index("visibility"),
        pose["start"] + pose["n_landmarks"] * len(pose["dims"]),
        len(pose["dims"]),
    )
    states[:, vis_cols] = pose_visibility

    for name in ("left_hand", "right_hand"):
        g = PERCEPTION_FEATURE_LAYOUT[name]
        sl = slice(g["start"], g["start"] + g["n_landmarks"] * len(g["dims"]))
        for f in hands_missing_frames or set():
            mask[f, sl] = 0.0

    meta = {} if provenance is None else {"temporal_provenance": provenance}
    return Rollout(
        modality="perception",
        source="real",
        condition={"lighting": "test"},
        frame_rate_hz=fps,
        states=states,
        presence_mask=mask,
        timestamps_ms=np.arange(n_frames) * (1000.0 / fps),
        metadata=meta,
    )


# --- ground truth -------------------------------------------------------------


def test_ground_truth_reports_hand_dropout_rate():
    r = _perception_rollout(n_frames=20, hands_missing_frames={0, 1, 2, 3})
    gt = landmark_quality_ground_truth(r)
    assert gt["hand_dropout_rate"] == pytest.approx(0.2)


def test_ground_truth_separates_run_length_from_rate():
    """Same dropout rate, very different to control through -- so the longest
    run is reported alongside the rate rather than averaged into it.
    """
    scattered = _perception_rollout(n_frames=30, hands_missing_frames={0, 6, 12, 18, 24, 29})
    burst = _perception_rollout(n_frames=30, hands_missing_frames={10, 11, 12, 13, 14, 15})

    a = landmark_quality_ground_truth(scattered)
    b = landmark_quality_ground_truth(burst)

    assert a["hand_dropout_rate"] == pytest.approx(b["hand_dropout_rate"])
    assert a["longest_dropout_run_s"] == pytest.approx(1 / 30.0)
    assert b["longest_dropout_run_s"] == pytest.approx(6 / 30.0)


def test_ground_truth_reads_mediapipes_own_visibility_score():
    r = _perception_rollout(pose_visibility=0.42)
    gt = landmark_quality_ground_truth(r)
    assert gt["mean_pose_visibility"] == pytest.approx(0.42)


def test_ground_truth_refuses_a_rollout_with_no_real_time_axis():
    """A dropout run measured in seconds is meaningless without ordered time."""
    r = _perception_rollout(n_frames=1, provenance="static_pose")
    with pytest.raises(ValueError, match="real time axis"):
        landmark_quality_ground_truth(r)


def test_ground_truth_refuses_undeclared_provenance_too():
    r = _perception_rollout(provenance=None)
    with pytest.raises(ValueError, match="real time axis"):
        landmark_quality_ground_truth(r)


# --- windowing ----------------------------------------------------------------


def test_split_into_windows_produces_non_overlapping_windows_by_default():
    r = _perception_rollout(n_frames=100)
    windows = split_into_windows(r, window_frames=30)
    assert len(windows) == 3
    assert [w.states.shape[0] for w in windows] == [30, 30, 30]
    assert [w.condition["window_start_frame"] for w in windows] == [0, 30, 60]


def test_windows_carry_the_non_independence_warning_in_metadata():
    """They share a subject, session and camera -- the confidence flag computed
    over them is optimistic, and the data says so rather than a docstring only.
    """
    r = _perception_rollout(n_frames=60)
    windows = split_into_windows(r, window_frames=20)
    for w in windows:
        assert w.metadata["windows_are_not_independent_samples"] is True
        assert w.metadata["window_of"] == r.rollout_id
        assert w.temporal_provenance == "video"


def test_split_rejects_nonsense_window_sizes():
    r = _perception_rollout(n_frames=10)
    with pytest.raises(ValueError, match="window_frames"):
        split_into_windows(r, window_frames=0)
    with pytest.raises(ValueError, match="hop_frames"):
        split_into_windows(r, window_frames=5, hop_frames=0)


# --- listing ------------------------------------------------------------------


def test_list_videos_finds_recordings_recursively(tmp_path):
    (tmp_path / "clean").mkdir()
    (tmp_path / "clean" / "a.mp4").write_bytes(b"")
    (tmp_path / "clean" / "b.MOV").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    found = list_videos(tmp_path)
    assert [p.name for p in found] == ["a.mp4", "b.MOV"]


def test_list_videos_raises_on_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_videos(tmp_path / "nope")


# --- real decode path (only when OpenCV is present) ---------------------------


def test_extract_rollout_from_video_uses_the_files_real_frame_rate(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from worldgap.data.loaders.video import extract_rollout_from_video

    path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (64, 64))
    if not writer.isOpened():  # pragma: no cover - codec unavailable in this env
        pytest.skip("no mp4v encoder available")
    for i in range(10):
        writer.write(np.full((64, 64, 3), i * 20 % 255, dtype=np.uint8))
    writer.release()

    class _FakeLandmark:
        x = y = z = 0.5
        visibility = 0.8

    class _FakeResult:
        def __init__(self):
            self.pose_landmarks = [_FakeLandmark() for _ in range(33)]
            self.left_hand_landmarks = [_FakeLandmark() for _ in range(21)]
            self.right_hand_landmarks = [_FakeLandmark() for _ in range(21)]

    class _FakeLandmarker:
        def detect(self, _image):
            return _FakeResult()

    rollout = extract_rollout_from_video(path, _FakeLandmarker(), use_video_mode=False)

    assert rollout.frame_rate_hz == pytest.approx(25.0)
    assert rollout.temporal_provenance == "video"
    assert rollout.has_ordered_time_axis is True
    assert rollout.states.shape[1] == PERCEPTION_STATE_DIM
