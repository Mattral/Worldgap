"""Run-2 tooling: graded image degradations, the loader's frame hook, and the
paired landmark-error ground truth."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from worldgap import Rollout
from worldgap.data.degradations import ImageDegradation
from worldgap.data.loaders.video import paired_landmark_error
from worldgap.data.normalization import normalize_rollout
from worldgap.data.rollout import PERCEPTION_STATE_DIM, TEMPORAL_PROVENANCE_KEY

cv2 = pytest.importorskip("cv2")

FRAME = np.random.default_rng(0).integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


# -- degradations -------------------------------------------------------------


def test_darken_scales_intensity():
    out = ImageDegradation("darken", 0.25)(FRAME, 0)
    np.testing.assert_allclose(out, np.rint(FRAME * 0.25), atol=0.5)


def test_downscale_sets_the_width_and_keeps_aspect():
    out = ImageDegradation("downscale", 32)(FRAME, 0)
    assert out.shape == (24, 32, 3)
    assert ImageDegradation("downscale", 640)(FRAME, 0) is FRAME  # never upsamples


def test_blur_smooths():
    assert ImageDegradation("blur", 3)(FRAME, 0).astype(float).std() < FRAME.astype(float).std()


def test_noise_is_reproducible_per_frame_and_differs_between_frames():
    d = ImageDegradation("noise", 30, seed=7)
    np.testing.assert_array_equal(d(FRAME, 5), d(FRAME, 5))
    assert not np.array_equal(d(FRAME, 5), d(FRAME, 6))
    assert not np.array_equal(d(FRAME, 5), ImageDegradation("noise", 30, seed=8)(FRAME, 5))


def test_invalid_degradations_are_refused():
    with pytest.raises(ValueError, match="unknown degradation"):
        ImageDegradation("fog", 1)
    with pytest.raises(ValueError):
        ImageDegradation("darken", 1.5)
    with pytest.raises(ValueError):
        ImageDegradation("blur", 0)


def test_loader_applies_and_records_the_transform(tmp_path: Path):
    pytest.importorskip("mediapipe")
    from worldgap.data.loaders.video import extract_rollout_from_video

    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    if not w.isOpened():  # pragma: no cover
        pytest.skip("no mp4v encoder")
    for i in range(6):
        w.write(np.full((48, 64, 3), 40 * i % 255, np.uint8))
    w.release()

    widths = []

    class _Fake:
        def detect(self, image):
            widths.append(image.width)

            class R:
                pose_landmarks = left_hand_landmarks = right_hand_landmarks = None

            return R()

    r = extract_rollout_from_video(video, _Fake(), use_video_mode=False,
                                   frame_transform=ImageDegradation("downscale", 32))
    assert widths and set(widths) == {32}
    assert r.metadata["frame_transform"] == {"kind": "downscale", "severity": 32, "seed": 0}


# -- paired landmark error ------------------------------------------------------


def _hand(cx: float, cy: float, size: float) -> np.ndarray:
    """21 landmarks spread over a square of side `size` (diagonal size*sqrt2)."""
    k = np.arange(21)
    return np.stack([cx + size * (k % 5) / 4, cy + size * (k // 5) / 4, np.zeros(21)], axis=1)


def _rollout(hands_per_frame: list[dict[str, np.ndarray]], normalize: bool = True) -> Rollout:
    t = len(hands_per_frame)
    states = np.zeros((t, PERCEPTION_STATE_DIM))
    presence = np.zeros_like(states)
    for i, hands in enumerate(hands_per_frame):
        states[i, 0:132] = np.tile([0.5, 0.5, 0.0, 1.0], 33)  # some pose
        states[i, 44:48] = [0.4, 0.4, 0.0, 1.0]  # shoulder 11
        states[i, 48:52] = [0.6, 0.4, 0.0, 1.0]  # shoulder 12
        presence[i, 0:132] = 1
        for name, start in (("left", 132), ("right", 195)):
            if name in hands:
                states[i, start:start + 63] = hands[name].ravel()
                presence[i, start:start + 63] = 1
    r = Rollout(modality="perception", source="real", condition={}, frame_rate_hz=30.0,
                states=states, presence_mask=presence, timestamps_ms=np.arange(t) * 33.3,
                metadata={TEMPORAL_PROVENANCE_KEY: "video"})
    return normalize_rollout(r) if normalize else r


def test_identical_rollouts_have_zero_error():
    ref = _rollout([{"left": _hand(0.3, 0.5, 0.1)}] * 4)
    out = paired_landmark_error(ref, ref)
    assert out["hand_landmark_error"] == pytest.approx(0.0, abs=1e-12)
    assert out["frames_compared_fraction"] == 1.0


def test_error_is_in_units_of_the_reference_hand_size():
    size = 0.1
    ref = _rollout([{"left": _hand(0.3, 0.5, size)}] * 3)
    shift = 0.02  # in x only
    deg = _rollout([{"left": _hand(0.3 + shift, 0.5, size)}] * 3)
    expected = shift / (size * np.sqrt(2))
    assert paired_landmark_error(ref, deg)["hand_landmark_error"] == pytest.approx(expected, rel=1e-9)


def test_left_right_label_swap_is_not_scored_as_error():
    ref = _rollout([{"left": _hand(0.3, 0.5, 0.1)}] * 3)
    deg = _rollout([{"right": _hand(0.3, 0.5, 0.1)}] * 3)
    assert paired_landmark_error(ref, deg)["hand_landmark_error"] == pytest.approx(0.0, abs=1e-12)


def test_frames_without_a_hand_on_either_side_are_skipped_not_scored():
    ref = _rollout([{"left": _hand(0.3, 0.5, 0.1)}, {"left": _hand(0.3, 0.5, 0.1)}, {}, {}])
    deg = _rollout([{"left": _hand(0.3, 0.5, 0.1)}, {}, {"left": _hand(0.3, 0.5, 0.1)}, {}])
    out = paired_landmark_error(ref, deg)
    assert out["n_hand_comparisons"] == 1.0
    assert out["frames_compared_fraction"] == 0.25


def test_works_on_unnormalized_rollouts_and_refuses_unpaired_shapes():
    ref = _rollout([{"left": _hand(0.3, 0.5, 0.1)}] * 3, normalize=False)
    assert paired_landmark_error(ref, ref)["hand_landmark_error"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="same frames"):
        paired_landmark_error(ref, _rollout([{"left": _hand(0.3, 0.5, 0.1)}] * 4))


# -- pilot script -----------------------------------------------------------------


def test_pilot_script_reports_ground_truth_and_baselines_without_gap_scores(tmp_path, monkeypatch):
    pytest.importorskip("mediapipe")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_v1_pilot.py"
    spec = importlib.util.spec_from_file_location("run_v1_pilot", script)
    pilot = importlib.util.module_from_spec(spec)
    sys.modules["run_v1_pilot"] = pilot
    spec.loader.exec_module(pilot)

    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(12):
        w.write(np.full((48, 64, 3), 20 * i % 255, np.uint8))
    w.release()

    class _L:
        def __init__(self, x, y):
            self.x, self.y, self.z, self.visibility = x, y, 0.0, 1.0

    class _Fake:
        """Finds a hand only in bright frames, so darkening raises dropout."""

        def detect_for_video(self, image, _ts):
            bright = image.numpy_view().mean() > 60
            hand = [_L(0.3 + 0.01 * (k % 5), 0.5 + 0.01 * (k // 5)) for k in range(21)]

            class R:
                pass

            r = R()
            r.pose_landmarks = [_L(0.5 + 0.005 * k, 0.5) for k in range(33)]
            r.left_hand_landmarks = hand if bright else None
            r.right_hand_landmarks = None
            return r

        def close(self):
            pass

    monkeypatch.setattr(pilot, "build_landmarker", lambda _m: _Fake())
    out = tmp_path / "pilot.json"
    monkeypatch.setattr(sys, "argv", ["run_v1_pilot.py", "--video", str(video),
                                      "--grid", json.dumps({"darken": [0.5, 0.1]}),
                                      "--max-frames", "12", "--out", str(out)])
    assert pilot.main() == 0
    report = json.loads(out.read_text())
    assert report["pilot"] is True
    rows = {(r["kind"], r["severity"]): r for r in report["rows"]}
    assert rows[("darken", 0.1)]["hand_dropout_rate"] > rows[("none", None)]["hand_dropout_rate"]
    assert not any("gap" in k or "frechet" in k for r in report["rows"] for k in r)
