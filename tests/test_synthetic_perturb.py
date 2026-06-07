import numpy as np

from worldgap.data.loaders.synthetic_perturb import (
    inject_tremor,
    reduce_range_of_motion,
    simulate_occlusion,
)
from worldgap.data.rollout import PERCEPTION_STATE_DIM, Rollout, perception_position_channel_mask


def _clean_rollout(t=60, d=12):
    rng = np.random.default_rng(0)
    return Rollout(
        modality="perception",
        source="real",
        condition={"lighting": "clean"},
        frame_rate_hz=30.0,
        states=rng.normal(size=(t, d)),
        presence_mask=np.ones((t, d)),
        timestamps_ms=np.arange(t) * (1000.0 / 30.0),
    )


def _real_layout_rollout(t=60):
    """A rollout at the actual PERCEPTION_STATE_DIM (258), unlike
    `_clean_rollout`'s generic toy shape — needed to exercise the real
    position/visibility channel split.
    """
    rng = np.random.default_rng(0)
    d = PERCEPTION_STATE_DIM
    return Rollout(
        modality="perception",
        source="real",
        condition={"lighting": "clean"},
        frame_rate_hz=30.0,
        states=rng.normal(size=(t, d)),
        presence_mask=np.ones((t, d)),
        timestamps_ms=np.arange(t) * (1000.0 / 30.0),
    )


def test_tremor_injection_changes_states_but_not_shape():
    r = _clean_rollout()
    perturbed = inject_tremor(r, frequency_hz=5.0, amplitude=0.05, seed=1)
    assert perturbed.states.shape == r.states.shape
    assert not np.allclose(perturbed.states, r.states)
    assert perturbed.source == "synthetic"
    assert "motion_profile" in perturbed.condition


def test_tremor_is_reproducible_given_same_seed():
    r = _clean_rollout()
    p1 = inject_tremor(r, seed=42)
    p2 = inject_tremor(r, seed=42)
    np.testing.assert_allclose(p1.states, p2.states)


def test_reduce_rom_shrinks_displacement_around_mean():
    r = _clean_rollout()
    reduced = reduce_range_of_motion(r, retain_fraction=0.5)
    original_spread = np.abs(r.states - r.states.mean(axis=0)).mean()
    reduced_spread = np.abs(reduced.states - reduced.states.mean(axis=0)).mean()
    assert reduced_spread < original_spread


def test_occlusion_zeroes_states_and_mask_together():
    r = _clean_rollout(t=60, d=12)
    occluded = simulate_occlusion(r, landmark_group_slice=slice(0, 6), occlusion_frac=0.2, seed=0)
    # wherever mask is 0, states must also be 0 -- the "MUST NOT zero-fill and
    # mark present" requirement, checked from the other direction.
    zero_mask_positions = occluded.presence_mask == 0
    assert np.all(occluded.states[zero_mask_positions] == 0.0)
    assert zero_mask_positions.sum() > 0  # something was actually occluded


def test_perturbations_change_rollout_id():
    r = _clean_rollout()
    perturbed = inject_tremor(r, seed=1)
    assert perturbed.rollout_id != r.rollout_id


def test_tremor_does_not_perturb_visibility_channels():
    """Regression test: inject_tremor previously perturbed every one of the
    258 state channels, including the pose block's visibility column, which
    contradicted its own docstring and would corrupt MediaPipe-confidence-
    derived validation ground truth (spec 8.1). It must leave visibility
    exactly unchanged and still perturb at least one spatial channel.
    """
    r = _real_layout_rollout()
    perturbed = inject_tremor(r, frequency_hz=5.0, amplitude=0.05, seed=1)

    position_mask = perception_position_channel_mask()
    visibility_mask = ~position_mask

    np.testing.assert_array_equal(
        perturbed.states[:, visibility_mask], r.states[:, visibility_mask]
    )
    assert not np.allclose(perturbed.states[:, position_mask], r.states[:, position_mask])


def test_reduce_rom_does_not_shrink_visibility_channels():
    """Regression test: reduce_range_of_motion previously compressed every
    channel's displacement toward its mean, including visibility, which isn't
    a spatial-displacement quantity. Visibility must be left exactly
    unchanged; spatial channels must still shrink as before.
    """
    r = _real_layout_rollout()
    reduced = reduce_range_of_motion(r, retain_fraction=0.5)

    position_mask = perception_position_channel_mask()
    visibility_mask = ~position_mask

    np.testing.assert_array_equal(reduced.states[:, visibility_mask], r.states[:, visibility_mask])

    original_spread = np.abs(
        r.states[:, position_mask] - r.states[:, position_mask].mean(axis=0)
    ).mean()
    reduced_spread = np.abs(
        reduced.states[:, position_mask] - reduced.states[:, position_mask].mean(axis=0)
    ).mean()
    assert reduced_spread < original_spread
