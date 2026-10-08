"""scripts/record_v1_session.py: the capture step that run 1 depended on.

The script's job that matters downstream: stamp each file with the frame rate
the camera actually delivered. `extract_rollout_from_video` turns frames into
seconds with that number, so `longest_dropout_run_s` (spec 8.1 ground truth)
is wrong by exactly the rate error if the script gets it wrong. A fake camera
driven by a fake clock makes the timing deterministic.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_v1_session.py"


def _load():
    spec = importlib.util.spec_from_file_location("record_v1_session", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["record_v1_session"] = module
    spec.loader.exec_module(module)
    return module


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Cam:
    """Reports `nominal` fps but delivers frames `1/true_fps` apart."""

    def __init__(self, clock: _Clock, true_fps: float, nominal: float = 30.0, size=(64, 48)):
        self.clock, self.true_fps, self.nominal, self.size = clock, true_fps, nominal, size
        self.i = 0

    def get(self, _prop):
        return self.nominal

    def read(self):
        self.clock.t += 1.0 / self.true_fps
        self.i += 1
        w, h = self.size
        return True, np.full((h, w, 3), self.i % 255, np.uint8)


@pytest.fixture
def rec(monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "show", lambda *a, **k: 255)  # no window, no key
    return module


def _file_fps_and_frames(path: Path) -> tuple[float, int, int]:
    cap = cv2.VideoCapture(str(path))
    out = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return out


def test_writes_the_rate_the_camera_actually_delivered(rec, tmp_path):
    clock = _Clock()
    out = tmp_path / "take0.mp4"
    stats = rec.record_take(_Cam(clock, true_fps=20.0), out, 3.0, 2.0, None, [], clock=clock)
    assert stats["camera_fps_actual"] == pytest.approx(20.0, abs=1e-6)
    assert stats["fps_written"] == pytest.approx(20.0, abs=1e-6)
    fps, frames, _ = _file_fps_and_frames(out)
    assert fps == pytest.approx(20.0, abs=0.01)
    assert frames == stats["frames"] in (60, 61)  # 3 s at 20 fps; the fake clock's float sum may land one frame either side
    assert not list(tmp_path.glob("*.tmp.mp4"))


def test_keeps_the_nominal_rate_when_within_tolerance(rec, tmp_path):
    clock = _Clock()
    out = tmp_path / "take0.mp4"
    stats = rec.record_take(_Cam(clock, true_fps=30.06), out, 2.0, 2.0, None, [], clock=clock)
    assert stats["fps_written"] == 30.0  # 0.2% off: under the 0.5% tolerance, no rewrite


def test_low_resolution_is_downscaled_on_save(rec, tmp_path):
    clock = _Clock()
    out = tmp_path / "take0.mp4"
    stats = rec.record_take(_Cam(clock, 30.0, size=(640, 480)), out, 1.0, 2.0, 320, [], clock=clock)
    assert stats["size"] == [320, 240]
    assert _file_fps_and_frames(out)[2] == 320


def test_quitting_mid_take_leaves_no_file(rec, tmp_path, monkeypatch):
    calls = {"n": 0}

    def _show(*_a, **_k):
        calls["n"] += 1
        return ord("q") if calls["n"] > 200 else 255  # past the countdown, mid-take

    monkeypatch.setattr(rec, "show", _show)
    clock = _Clock()
    out = tmp_path / "take0.mp4"
    assert rec.record_take(_Cam(clock, 30.0), out, 10.0, 2.0, None, [], clock=clock) is None
    assert not list(tmp_path.iterdir())


def test_session_plan_records_clean_first_and_last(rec):
    plan = rec.session_plan()
    names = [c[0] for c, _ in plan]
    assert len(plan) == 33
    assert names[:2] == ["clean", "clean"] and names[-1] == "clean"
    assert sorted({t for c, t in plan if c[0] == "clean"}) == [0, 1, 2]
    by_name = {c[0]: c for c, _ in plan}
    assert by_name["fast_motion"][2] == 1.0
    assert by_name["low_resolution"][3] == 320
    assert [t for _, t in rec.session_plan("dim_light")] == [0, 1, 2]
    with pytest.raises(SystemExit):
        rec.session_plan("not_a_condition")


def test_loader_durations_follow_the_recorded_rate(rec, tmp_path):
    """End to end: a 3 s take from a camera delivering 20 fps (while claiming
    30) must come out of the loader as 3 s, so a dropout over the whole take
    is measured as ~3 s, not 2 s."""
    pytest.importorskip("mediapipe")
    from worldgap.data.loaders.video import (
        extract_rollout_from_video,
        landmark_quality_ground_truth,
    )

    clock = _Clock()
    out = tmp_path / "take0.mp4"
    rec.record_take(_Cam(clock, true_fps=20.0), out, 3.0, 2.0, None, [], clock=clock)

    class _NothingDetected:
        def detect(self, _image):
            class R:
                pose_landmarks = left_hand_landmarks = right_hand_landmarks = None

            return R()

    rollout = extract_rollout_from_video(out, _NothingDetected(), use_video_mode=False)
    assert rollout.frame_rate_hz == pytest.approx(20.0, abs=0.01)
    gt = landmark_quality_ground_truth(rollout)
    assert gt["longest_dropout_run_s"] == pytest.approx(3.0, abs=0.06)
