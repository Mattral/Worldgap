"""scripts/run_v1_run2.py end to end, on synthetic videos with a fake
landmarker, so pipeline bugs surface in CI rather than during the overnight
run. The fake reacts to the degradations (darkness removes the hand, blur and
downscaling move the landmarks), so every stage carries real signal."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("mediapipe")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_v1_run2.py"


def _load():
    spec = importlib.util.spec_from_file_location("run_v1_run2", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_v1_run2"] = module
    spec.loader.exec_module(module)
    return module


def _write_video(path: Path, seed: int, n_frames: int = 150) -> None:
    rng = np.random.default_rng(seed)
    base = rng.integers(60, 200, size=(120, 160, 3)).astype(np.uint8)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120))
    if not w.isOpened():  # pragma: no cover
        pytest.skip("no mp4v encoder")
    for i in range(n_frames):
        w.write(np.roll(base, i, axis=1))
    w.release()


class _L:
    def __init__(self, x, y, v=1.0):
        self.x, self.y, self.z, self.visibility = x, y, 0.0, v


class _Fake:
    """Hand visible only if the frame is bright enough; landmark error grows as
    the frame loses sharpness or resolution."""

    def __init__(self):
        self.t = 0

    def detect_for_video(self, image, _ts):
        img = image.numpy_view().astype(float)
        bright = img.mean()
        sharp = np.abs(np.diff(img, axis=1)).mean()
        width = img.shape[1]
        rng = np.random.default_rng(self.t)
        self.t += 1
        err = 0.02 / (1 + sharp / 5) + 0.02 * (1 - min(1.0, width / 160))
        k = np.arange(21)
        hx = 0.3 + 0.01 * (k % 5) + err + rng.normal(0, 0.002, 21)
        hy = 0.5 + 0.01 * (k // 5) + rng.normal(0, 0.002, 21)

        class R:
            pass

        r = R()
        r.pose_landmarks = [_L(0.5 + 0.01 * np.cos(j) + rng.normal(0, 0.002), 0.5 + 0.01 * np.sin(j)) for j in range(33)]
        r.left_hand_landmarks = [_L(x, y) for x, y in zip(hx, hy)] if bright > 15 else None
        r.right_hand_landmarks = None
        return r

    def close(self):
        pass


@pytest.fixture
def recordings(tmp_path: Path) -> Path:
    root = tmp_path / "rec"
    (root / "clean").mkdir(parents=True)
    for k in range(3):
        _write_video(root / "clean" / f"take{k}.mp4", seed=k)
    (root / "distance_2m").mkdir()
    _write_video(root / "distance_2m" / "take0.mp4", seed=9)
    return root


def test_full_pipeline_then_resume_from_cache(recordings, tmp_path, monkeypatch):
    module = _load()
    built = []

    def _build(_model):
        built.append(1)
        return _Fake()

    monkeypatch.setattr(module, "build_landmarker", _build)
    monkeypatch.setattr(module, "SUMMARY_DIM", 4)  # tiny synthetic data
    out = tmp_path / "out"
    argv = ["run_v1_run2.py", "--recordings", str(recordings), "--out", str(out),
            "--epochs", "1", "--bootstrap", "500"]
    monkeypatch.setattr(sys, "argv", argv)
    assert module.main() == 0

    assert len(built) == 3 + 24 * 3 + 1  # one fresh landmarker per video
    settings = json.loads((out / "study_settings.json").read_text())
    assert settings["smoke_mode"] is True and settings["normalization_scheme"] == "shoulder_midpoint"
    assert len(settings["primary_conditions"]) == 24

    res = json.loads((out / "results.json").read_text())
    assert res["status"] == "smoke"
    assert len(res["conditions"]) == 24
    # Amendment 2: both analyses, side by side
    assert {"A", "B", "interpretation"} <= set(res["rules_all_24"])
    assert res["rules_all_24"]["n_conditions"] == 24
    sens = res["rules_measured_only"]  # the fake leaves some conditions undefined
    assert sens is not None and sens["n_conditions"] == 24 - len(res["secondary"]["undefined_landmark_error_ranked_worst"])
    assert isinstance(res["rules_disagree"], bool)
    # per-take breakdown for every condition
    assert all(len(r["takes"]) == 3 for r in res["conditions"])
    # the fake loses the hand in the darkest conditions: reported, ranked worst, never dropped
    assert "darken_0.06" in res["secondary"]["undefined_landmark_error_ranked_worst"]
    assert {r["condition"] for r in res["conditions"]} == set(module.PRIMARY)
    # physical conditions are descriptive; missing ones are reported as such
    phys = {r["condition"]: r for r in res["physical"]}
    assert "frechet" in phys["distance_2m"] and phys["occlusion_25"]["status"] == "not recorded"
    assert (out / "run2_report.html").exists()

    # second run: everything comes from the cache, nothing is re-extracted
    built.clear()
    assert module.main() == 0
    assert built == []


def test_blind_mode_runs_everything_but_keeps_no_numbers(recordings, tmp_path, monkeypatch):
    module = _load()
    monkeypatch.setattr(module, "build_landmarker", lambda _m: _Fake())
    monkeypatch.setattr(module, "SUMMARY_DIM", 4)
    out = tmp_path / "blind"
    monkeypatch.setattr(sys, "argv", ["run_v1_run2.py", "--recordings", str(recordings), "--out", str(out),
                                      "--epochs", "1", "--bootstrap", "200", "--blind"])
    assert module.main() == 0
    assert not (out / "results.json").exists() and not (out / "run2_report.html").exists()
    health = json.loads((out / "results_blind.json").read_text())
    assert health["blind"] and health["n_primary_conditions"] == 24 and health["report_generated"]
    text = (out / "results_blind.json").read_text()
    for leaked in ("rho", "frechet", "delta", "interpretation"):
        assert leaked not in text


def test_interpretation_follows_the_amended_rules():
    module = _load()
    assert module.interpret(False, False, 0.9).startswith("Rule A failed")
    assert "better than counting" in module.interpret(True, True, 0.9)
    assert "underpowered" in module.interpret(True, False, 0.85)
    assert "underpowered" not in module.interpret(True, False, 0.75)


def test_refuses_a_clean_take_count_other_than_three(tmp_path, monkeypatch, capsys):
    module = _load()
    root = tmp_path / "rec"
    (root / "clean").mkdir(parents=True)
    _write_video(root / "clean" / "take0.mp4", seed=0, n_frames=10)
    monkeypatch.setattr(sys, "argv", ["run_v1_run2.py", "--recordings", str(root), "--dry-run"])
    assert module.main() == 1
    assert "fixes 3 clean takes" in capsys.readouterr().out
