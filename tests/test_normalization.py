"""Spec 5.2 landmark normalization.

The property the missing normalization broke, and that must not regress: a
rollout translated or scaled as a whole (the same person further from the
camera, or shifted in the frame) produces the same normalized states and the
same encoding. Found by the first real V1 run (docs/v1_first_run_results.md),
where the largest gap scores tracked where the person sat in the frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from worldgap import GapAnalyzer, GapConfig, Rollout, split_into_windows
from worldgap.config import EncoderConfig, TrainingConfig, WorldModelConfig
from worldgap.data.index import RolloutIndex
from worldgap.data.normalization import (
    NORMALIZATION_KEY,
    denormalize_states,
    is_normalized,
    normalize_perception_states,
    normalize_rollout,
)
from worldgap.data.rollout import (
    PERCEPTION_FEATURE_LAYOUT,
    PERCEPTION_STATE_DIM,
    TEMPORAL_PROVENANCE_KEY,
)

SCALE, SHIFT = 0.55, np.array([0.21, -0.13, 0.07])


def _scene(t: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Plausible raw landmarks over time: pose (with visibility) and two hands.

    Returns pose xyz (T,33,3), pose visibility (T,33), hands xyz (T,2,21,3).
    """
    rng = np.random.default_rng(seed)
    pose = rng.uniform([0.3, 0.2, -0.2], [0.7, 0.9, 0.2], size=(33, 3))
    pose[11], pose[12] = [0.40, 0.40, 0.0], [0.60, 0.41, 0.0]  # shoulders
    pose[23], pose[24] = [0.43, 0.80, 0.0], [0.57, 0.79, 0.0]  # hips
    hands = rng.normal(0, 0.03, size=(2, 21, 3)) + np.array([[[0.35, 0.5, 0.0]], [[0.65, 0.5, 0.0]]])
    drift = np.linspace(0, 1, t)[:, None]
    pose_t = pose[None] + 0.02 * np.sin(6 * drift)[:, :, None] + rng.normal(0, 0.002, (t, 33, 3))
    hands_t = hands[None] + 0.03 * np.cos(4 * drift)[:, :, None, None] + rng.normal(0, 0.002, (t, 2, 21, 3))
    vis = rng.uniform(0, 1, size=(t, 33))
    return pose_t, vis, hands_t


def _pack(pose, vis, hands) -> np.ndarray:
    t = pose.shape[0]
    states = np.zeros((t, PERCEPTION_STATE_DIM))
    states[:, 0:132] = np.concatenate([pose, vis[:, :, None]], axis=2).reshape(t, -1)
    states[:, 132:195] = hands[:, 0].reshape(t, -1)
    states[:, 195:258] = hands[:, 1].reshape(t, -1)
    return states


def _transformed(pose, vis, hands):
    return pose * SCALE + SHIFT, vis, hands * SCALE + SHIFT


def _rollout(states: np.ndarray, presence: np.ndarray | None = None, seed: int = 0) -> Rollout:
    t = states.shape[0]
    return Rollout(
        modality="perception",
        source="real",
        condition={"seed": seed},
        frame_rate_hz=30.0,
        states=states,
        presence_mask=np.ones_like(states) if presence is None else presence,
        timestamps_ms=np.arange(t) * (1000 / 30),
        metadata={TEMPORAL_PROVENANCE_KEY: "video"},
    )


def _tiny_analyzer(train: list[Rollout]) -> GapAnalyzer:
    cfg = GapConfig(
        modality="perception",
        state_dim=PERCEPTION_STATE_DIM,
        encoder=EncoderConfig(d_model=16, n_layers=1, n_heads=2, dim_feedforward=32),
        world_model=WorldModelConfig(context_frames=4, predict_frames=2, summary_dim=8),
        training=TrainingConfig(max_epochs=1, batch_size=4, seed=0),
    )
    analyzer = GapAnalyzer(cfg)
    analyzer.fit(train)
    return analyzer


# -- the property ------------------------------------------------------------


def test_translated_and_scaled_rollout_normalizes_to_the_same_states():
    raw = _pack(*_scene())
    moved = _pack(*_transformed(*_scene()))
    assert not np.allclose(raw, moved)  # the inputs really differ
    ones = np.ones_like(raw)
    a, _ = normalize_perception_states(raw, ones)
    b, _ = normalize_perception_states(moved, ones)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_translated_and_scaled_rollout_produces_the_same_encoding():
    raw = _rollout(_pack(*_scene()))
    moved = _rollout(_pack(*_transformed(*_scene())))
    train = [normalize_rollout(_rollout(_pack(*_scene(seed=s)), seed=s)) for s in range(1, 5)]
    analyzer = _tiny_analyzer(train)

    enc = analyzer._rollouts_to_summary_latents
    np.testing.assert_allclose(
        enc([normalize_rollout(raw)]), enc([normalize_rollout(moved)]), atol=1e-5
    )
    # not vacuous: without normalization the same model tells them apart
    assert not np.allclose(enc([raw]), enc([moved]), atol=1e-3)


def test_real_video_loader_path_is_invariant(tmp_path: Path):
    """Through extract_rollout_from_video with a fake landmarker: if anyone
    removes normalization from the loader, this fails."""
    cv2 = pytest.importorskip("cv2")
    pytest.importorskip("mediapipe")
    from worldgap.data.loaders.video import extract_rollout_from_video

    t = 24
    video = tmp_path / "v.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (32, 32))
    if not writer.isOpened():  # pragma: no cover
        pytest.skip("no mp4v encoder")
    for i in range(t):
        writer.write(np.full((32, 32, 3), i * 5, np.uint8))
    writer.release()

    class _L:
        def __init__(self, x, y, z, v=1.0):
            self.x, self.y, self.z, self.visibility = x, y, z, v

    class _Fake:
        def __init__(self, pose, vis, hands):
            self.pose, self.vis, self.hands, self.i = pose, vis, hands, 0

        def detect(self, _img):
            class R:
                pass

            r, i = R(), self.i
            r.pose_landmarks = [_L(*self.pose[i, k], self.vis[i, k]) for k in range(33)]
            r.left_hand_landmarks = [_L(*self.hands[i, 0, k]) for k in range(21)]
            r.right_hand_landmarks = [_L(*self.hands[i, 1, k]) for k in range(21)]
            self.i += 1
            return r

    a = extract_rollout_from_video(video, _Fake(*_scene(t)), use_video_mode=False)
    b = extract_rollout_from_video(video, _Fake(*_transformed(*_scene(t))), use_video_mode=False)
    assert is_normalized(a) and is_normalized(b)
    np.testing.assert_allclose(a.states, b.states, atol=1e-6)


# -- correctness of the transform ------------------------------------------


def test_reference_points_land_where_spec_5_2_says():
    states, _ = normalize_perception_states(_pack(*_scene()), np.ones((40, PERCEPTION_STATE_DIM)))
    pose = states[:, 0:132].reshape(40, 33, 4)[:, :, :3]
    hip_mid = (pose[:, 23] + pose[:, 24]) / 2
    np.testing.assert_allclose(hip_mid, 0.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1), 1.0, atol=1e-9)
    for start in (132, 195):
        hand = states[:, start:start + 63].reshape(40, 21, 3)
        np.testing.assert_allclose(hand[:, 0], 0.0, atol=1e-9)  # wrist at origin
        extent = hand[:, :, :2].max(axis=1) - hand[:, :, :2].min(axis=1)
        np.testing.assert_allclose(np.linalg.norm(extent, axis=1), 1.0, atol=1e-9)


def test_visibility_and_presence_are_untouched():
    raw = _rollout(_pack(*_scene()))
    out = normalize_rollout(raw)
    vis_cols = [PERCEPTION_FEATURE_LAYOUT["pose"]["start"] + i * 4 + 3 for i in range(33)]
    np.testing.assert_array_equal(out.states[:, vis_cols], raw.states[:, vis_cols])
    np.testing.assert_array_equal(out.presence_mask, raw.presence_mask)


def test_raw_values_are_recoverable_including_absent_blocks():
    states = _pack(*_scene())
    presence = np.ones_like(states)
    states[5:9, 132:195] = 0.0  # left hand missing for 4 frames
    presence[5:9, 132:195] = 0.0
    normed, params = normalize_perception_states(states, presence)
    np.testing.assert_allclose(normed[5:9, 132:195], 0.0)  # absent stays zero
    np.testing.assert_allclose(denormalize_states(normed, params), states, atol=1e-12)


def test_degenerate_reference_length_is_counted_not_divided_by():
    states = np.full((3, PERCEPTION_STATE_DIM), 0.5)  # every landmark at one point
    normed, params = normalize_perception_states(states, np.ones_like(states))
    assert np.isfinite(normed).all()
    assert params["n_degenerate_frames"] == 9  # 3 frames x 3 blocks
    np.testing.assert_allclose(denormalize_states(normed, params), states)


def test_refuses_to_normalize_twice_or_non_perception():
    once = normalize_rollout(_rollout(_pack(*_scene())))
    with pytest.raises(ValueError, match="already normalized"):
        normalize_rollout(once)
    act = Rollout(
        modality="actuation", source="sim", condition={}, frame_rate_hz=10.0,
        states=np.zeros((5, 3)), presence_mask=np.ones((5, 3)), timestamps_ms=np.arange(5) * 100.0,
    )
    with pytest.raises(ValueError, match="perception"):
        normalize_rollout(act)


# -- parameters survive windowing and storage --------------------------------


def test_windows_carry_their_own_slice_of_parameters(tmp_path: Path):
    full = normalize_rollout(_rollout(_pack(*_scene())))
    windows = split_into_windows(full, window_frames=10)
    assert len(windows) == 4
    for k, w in enumerate(windows):
        p = w.metadata[NORMALIZATION_KEY]
        assert len(p["per_frame"]["pose_scale"]) == 10
        assert "n_degenerate_frames" not in p and "n_degenerate_frames_in_source" in p
        np.testing.assert_allclose(
            denormalize_states(w.states, p), denormalize_states(full.states, full.metadata[NORMALIZATION_KEY])[k * 10:(k + 1) * 10],
        )

    # and through the SQLite index (metadata is JSON there)
    w = windows[1]
    w.save(tmp_path)
    with RolloutIndex(tmp_path / "index.db") as index:
        index.add(w)
        back = index.load_rollout(w.rollout_id, tmp_path)
    np.testing.assert_allclose(
        denormalize_states(back.states, back.metadata[NORMALIZATION_KEY]),
        denormalize_states(w.states, w.metadata[NORMALIZATION_KEY]),
    )
