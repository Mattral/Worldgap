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
    def __init__(self, x: float, y: float, visibility: float):
        self.x, self.y, self.z = x, y, 0.0
        self.visibility = visibility


def _points(n: int, cx: float, cy: float, spread: float, noise: np.ndarray, visibility: float):
    """n landmarks with real extent (spec 5.2 normalization divides by
    shoulder width / hand size, so points piled on one spot are degenerate),
    with independent per-landmark noise: a shift shared by every point is a
    translation, which normalization correctly removes."""
    return [
        _Landmark(cx + spread * np.cos(k) + noise[k, 0], cy + spread * np.sin(1.7 * k) + noise[k, 1], visibility)
        for k in range(n)
    ]


class _Result:
    def __init__(self, hands: bool, visibility: float, rng: np.random.Generator, noise: float):
        def jitter(n):
            return rng.normal(0, noise, size=(n, 2))

        self.pose_landmarks = _points(33, 0.5, 0.5, 0.2, jitter(33), visibility)
        self.left_hand_landmarks = _points(21, 0.4, 0.5, 0.05, jitter(21), visibility) if hands else []
        self.right_hand_landmarks = _points(21, 0.6, 0.5, 0.05, jitter(21), visibility) if hands else []


class _FakeLandmarker:
    """Drops hands on `dropout_every`-th frame and jitters landmark positions.

    Enforces the one MediaPipe VIDEO-mode rule that bit the first real run:
    timestamps must increase monotonically over the landmarker's lifetime.
    Without this the fake accepted a single landmarker shared across videos,
    which the real one rejects on the second file.
    """

    def __init__(self, dropout_every: int | None, visibility: float, noise: float, seed: int = 0):
        self.dropout_every = dropout_every
        self.visibility = visibility
        self.noise = noise
        self._n = 0
        self._rng = np.random.default_rng(seed)
        self._last_ts: int | None = None

    def _next(self):
        hands = True
        if self.dropout_every and self._n % self.dropout_every == 0:
            hands = False
        self._n += 1
        return _Result(hands, self.visibility, self._rng, self.noise)

    def detect(self, _image):
        return self._next()

    def detect_for_video(self, _image, ts):
        if self._last_ts is not None and ts <= self._last_ts:
            raise ValueError("Input timestamp must be monotonically increasing.")
        self._last_ts = ts
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

    params = {
        "clean": {"dropout_every": None, "visibility": 0.95, "noise": 0.01},
        "dim": {"dropout_every": 5, "visibility": 0.6, "noise": 0.05},
        "occluded": {"dropout_every": 2, "visibility": 0.4, "noise": 0.09},
    }
    current = {"name": "clean"}
    built = []

    def _build(*_a, **_k):
        # distinct seed per video: real takes never repeat exactly, and
        # identical takes make the Frechet covariance near-singular
        fake = _FakeLandmarker(**params[current["name"]], seed=len(built))
        built.append(fake)
        return fake

    monkeypatch.setattr(module, "build_landmarker", _build)

    real_extract = module.extract_condition

    def _extract(name, videos, make_landmarker, *a, **k):
        current["name"] = name
        return real_extract(name, videos, make_landmarker, *a, **k)

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
    assert module.main() == 0
    assert len(built) == 6  # one landmarker per video (3 conditions x 2 takes)

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
    for name, args in {
        "clean": (None, 0.95, 0.01),
        "dim": (5, 0.6, 0.05),
        "occluded": (2, 0.4, 0.09),
    }.items():
        _rollouts, gts = module.extract_condition(
            name,
            module.list_videos(recordings / name),
            lambda args=args: _FakeLandmarker(*args),
            32,
            None,
            1,
        )
        values[name] = module.aggregate_ground_truth(gts)

    assert values["clean"] < values["dim"] < values["occluded"]


def test_each_video_gets_a_fresh_landmarker(recordings):
    """Regression: the first real run crashed on its second video because one
    VIDEO-mode landmarker was shared across files whose timestamps each start
    at 0. A shared landmarker must fail (as MediaPipe's does), and
    extract_condition must build and close one per video.
    """
    module = _load_script()
    videos = module.list_videos(recordings / "clean")
    assert len(videos) == 2

    shared = _FakeLandmarker(None, 0.95, 0.01)
    with pytest.raises(ValueError, match="monotonically increasing"):
        module.extract_condition("clean", videos, lambda: shared, 32, None, 1)

    built, closed = [], []

    class _Tracked(_FakeLandmarker):
        def close(self):
            closed.append(self)

    def _make():
        built.append(_Tracked(None, 0.95, 0.01))
        return built[-1]

    _rollouts, gts = module.extract_condition("clean", videos, _make, 32, None, 1)
    assert len(gts) == 2
    assert len(built) == 2 and built[0] is not built[1]
    assert closed == built


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
