"""End-to-end smoke test of scripts/run_v1_real_data.py.

The script is the thing the user actually runs on their own machine, where a
crash costs a capture session. So it is tested here the whole way through --
real video files written to disk, real decoding, real rollout stores, real
world-model fit, real gap computation, real report -- with only the
`HolisticLandmarker` faked, since that is the single piece that needs a model
bundle from a host CI cannot reach.

The fake produces landmarks that genuinely differ between conditions (the
"degraded" condition drops hands on a fraction of frames), so the assertions
below are about behaviour, not just absence of exceptions.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v1_real_data.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_v1_real_data", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_v1_real_data"] = module
    spec.loader.exec_module(module)
    return module


class _Landmark:
    def __init__(self, v: float, visibility: float):
        self.x = self.y = self.z = v
        self.visibility = visibility


class _Result:
    def __init__(self, hands: bool, visibility: float, jitter: float):
        self.pose_landmarks = [_Landmark(0.5 + jitter, visibility) for _ in range(33)]
        self.left_hand_landmarks = (
            [_Landmark(0.4 + jitter, visibility) for _ in range(21)] if hands else []
        )
        self.right_hand_landmarks = (
            [_Landmark(0.6 + jitter, visibility) for _ in range(21)] if hands else []
        )


class _FakeLandmarker:
    """Drops hands on `dropout_every`-th frame and jitters landmark positions."""

    def __init__(self, dropout_every: int | None, visibility: float, noise: float):
        self.dropout_every = dropout_every
        self.visibility = visibility
        self.noise = noise
        self._n = 0
        self._rng = np.random.default_rng(0)

    def _next(self):
        hands = True
        if self.dropout_every and self._n % self.dropout_every == 0:
            hands = False
        jitter = float(self._rng.normal(0, self.noise))
        self._n += 1
        return _Result(hands, self.visibility, jitter)

    def detect(self, _image):
        return self._next()

    def detect_for_video(self, _image, _ts):
        return self._next()

    def close(self):
        pass


def _write_video(path: Path, n_frames: int = 80, fps: float = 25.0) -> None:
    cv2 = pytest.importorskip("cv2")
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 64))
    if not writer.isOpened():  # pragma: no cover - codec unavailable
        pytest.skip("no mp4v encoder available")
    for i in range(n_frames):
        writer.write(np.full((64, 64, 3), (i * 3) % 255, dtype=np.uint8))
    writer.release()


@pytest.fixture
def recordings(tmp_path: Path) -> Path:
    root = tmp_path / "recordings"
    for name in ("clean", "dim", "occluded"):
        (root / name).mkdir(parents=True)
        for i in range(2):
            _write_video(root / name / f"take{i}.mp4")
    return root


def test_script_runs_end_to_end_and_writes_every_artifact(recordings, tmp_path, monkeypatch):
    module = _load_script()
    out = tmp_path / "v1_run"

    fakes = {
        "clean": _FakeLandmarker(dropout_every=None, visibility=0.95, noise=0.01),
        "dim": _FakeLandmarker(dropout_every=5, visibility=0.6, noise=0.05),
        "occluded": _FakeLandmarker(dropout_every=2, visibility=0.4, noise=0.09),
    }

    # one landmarker per condition, handed out in the order the script asks
    order = iter(["clean", "dim", "occluded"])

    class _Dispatcher:
        def __init__(self):
            self.current = fakes["clean"]

        def detect(self, image):
            return self.current.detect(image)

        def detect_for_video(self, image, ts):
            return self.current.detect_for_video(image, ts)

        def close(self):
            pass

    dispatcher = _Dispatcher()
    monkeypatch.setattr(module, "build_landmarker", lambda *_a, **_k: dispatcher)

    real_extract = module.extract_condition

    def _extract(name, videos, landmarker, *a, **k):
        dispatcher.current = fakes[name]
        return real_extract(name, videos, landmarker, *a, **k)

    monkeypatch.setattr(module, "extract_condition", _extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_v1_real_data.py",
            "--recordings", str(recordings),
            "--out", str(out),
            "--window-frames", "32",
            "--epochs", "3",
        ],
    )
    del order

    assert module.main() == 0

    # every artifact the runbook promises actually exists
    assert (out / "preregistered_conditions.json").exists()
    assert (out / "ground_truth.json").exists()
    assert (out / "checkpoint.pt").exists()
    assert (out / "v1_report.html").exists()
    assert (out / "stores" / "clean" / "index.db").exists()
    assert list((out / "stores" / "clean" / "perception").glob("*.npz"))


def test_ground_truth_separates_the_conditions_it_should(recordings, monkeypatch):
    """The real check: MediaPipe-side ground truth must actually rank the
    conditions, otherwise there is nothing for a gap score to correlate with.
    """
    module = _load_script()

    values = {}
    for name, fake in {
        "clean": _FakeLandmarker(None, 0.95, 0.01),
        "dim": _FakeLandmarker(5, 0.6, 0.05),
        "occluded": _FakeLandmarker(2, 0.4, 0.09),
    }.items():
        _rollouts, gts = module.extract_condition(
            name, module.list_videos(recordings / name), fake, 32, None, 1
        )
        values[name] = module.aggregate_ground_truth(gts)

    assert values["clean"] < values["dim"] < values["occluded"]


def test_missing_clean_directory_is_a_clear_error(tmp_path, monkeypatch, capsys):
    module = _load_script()
    root = tmp_path / "recordings"
    (root / "dim").mkdir(parents=True)
    _write_video(root / "dim" / "a.mp4", n_frames=10)

    monkeypatch.setattr(
        sys, "argv", ["run_v1_real_data.py", "--recordings", str(root), "--dry-run"]
    )
    assert module.main() == 1
    assert "no 'clean/' directory" in capsys.readouterr().out
