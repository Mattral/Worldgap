"""Tests for the V2 simulated PGM baseline and the sim-vs-real rollout pair.

The physics assertions here are deliberately about *properties* the ideal
McKibben model must have, not about numbers copied out of a run. A test that
pins today's output would pass forever and catch nothing.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from worldgap.data.loaders.pgm_actuator import OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA
from worldgap.data.loaders.pgm_sim import (
    ACTUATION_STATE_DIM,
    ACTUATION_STATE_LAYOUT,
    OGAWA_2017_IDEAL_MCKIBBEN,
    IdealMcKibbenParams,
    digitized_pgm_rollout,
    ideal_mckibben_force_n,
    length_residuals_mm,
    simulate_length_at_force,
    simulated_curve_is_saturated,
    simulated_pgm_rollout,
)

# --- model physics ------------------------------------------------------------


def test_ideal_model_produces_no_force_without_pressure():
    """Not an oversight -- it is failure mode #2 the V2 gap is meant to expose,
    so it is pinned here to stop someone "fixing" it and silently erasing the
    finding.
    """
    lengths = np.linspace(300, 500, 20)
    assert np.all(ideal_mckibben_force_n(lengths, pressure_mpa=0.0) == 0.0)


def test_force_increases_with_pressure_at_fixed_length():
    length = 400.0
    forces = [ideal_mckibben_force_n(length, p) for p in (0.05, 0.1, 0.2, 0.3)]
    assert all(b > a for a, b in itertools.pairwise(forces))


def test_force_increases_with_length_at_fixed_pressure():
    """Longer means less contracted, so the muscle balances a heavier load."""
    forces = ideal_mckibben_force_n(np.array([350.0, 400.0, 450.0]), 0.2)
    assert forces[0] < forces[1] < forces[2]


def test_derived_braid_angle_is_physically_plausible():
    """Derived from reported geometry, not reported directly -- so at minimum it
    must lie in the range a braid can actually occupy, below the 54.74 deg
    maximum-contraction angle.
    """
    theta = OGAWA_2017_IDEAL_MCKIBBEN.braid_angle_deg
    assert 20.0 < theta < 54.7356
    # and it must reproduce the 28 mm max diameter it was derived from
    d_max = (
        OGAWA_2017_IDEAL_MCKIBBEN.braid_diameter_mm
        * math.sin(math.radians(54.7356))
        / math.sin(math.radians(theta))
    )
    assert d_max == pytest.approx(28.0, abs=0.01)


def test_inversion_round_trips_within_the_models_invertible_domain():
    """Round-trips only where the model actually carries load.

    Below its free-contraction length the ideal model predicts negative force
    (the muscle would have to push), which is clipped to zero -- so several
    lengths map to F = 0 and the inverse cannot be unique there. That is a
    property of the model, not a bug in the inversion, and the test is written
    to the domain rather than around it.
    """
    pressure = 0.2
    lengths = np.linspace(330.0, 460.0, 40)
    forces = ideal_mckibben_force_n(lengths, pressure)
    loaded = forces > 1e-9
    recovered = simulate_length_at_force(forces[loaded], pressure)
    assert np.abs(recovered - lengths[loaded]).max() < 0.5  # under any noise floor


def test_free_contraction_length_is_far_longer_than_the_real_muscles():
    """The mechanism behind most of the V2 gap, stated as a test.

    At 0.2 MPa the ideal model contracts freely to about 390 mm and can hold no
    load below that. Ogawa's real measured curve sits at roughly 322 mm at
    ~0 N. The real gel-foam PGM contracts substantially further than braid
    geometry alone predicts -- which is why the residuals are largest at low
    load, and why `tube_stiffness_n_per_mm` exists as a hypothesis to test.
    """
    from worldgap.data.loaders.pgm_actuator import load_ogawa2017_fig4a_curve

    forces = np.linspace(0.0, 50.0, 2001)
    lengths = simulate_length_at_force(forces, 0.2)
    free_contraction_mm = float(lengths[forces > 1e-9].min())

    real = load_ogawa2017_fig4a_curve(0.2)
    real_at_lowest_force = float(real.length_mm[0])

    assert free_contraction_mm > real_at_lowest_force + 40.0


def test_optional_tube_stiffness_lets_the_model_hold_load_at_zero_pressure():
    """The variant that tests whether tube elasticity explains the low-pressure
    gap. Off by default, on purpose.
    """
    params = IdealMcKibbenParams(
        resting_length_mm=500.0,
        braid_diameter_mm=23.0,
        braid_angle_deg=42.0,
        tube_stiffness_n_per_mm=0.5,
    )
    assert ideal_mckibben_force_n(520.0, pressure_mpa=0.0, params=params) > 0.0


# --- saturation reporting -----------------------------------------------------


def test_zero_pressure_curve_is_reported_as_fully_saturated():
    """A flat curve because the search grid ran out is not a physical plateau,
    and the difference has to be visible in the output.
    """
    forces = np.arange(0.5, 49.0, 0.5)
    lengths = simulate_length_at_force(forces, 0.0)
    sat = simulated_curve_is_saturated(forces, lengths)
    assert sat["fraction_saturated"] > 0.9


def test_mid_pressure_curve_is_not_saturated():
    forces = np.arange(0.5, 49.0, 0.5)
    lengths = simulate_length_at_force(forces, 0.2)
    assert simulated_curve_is_saturated(forces, lengths)["fraction_saturated"] == 0.0


# --- rollouts -----------------------------------------------------------------


def test_sim_and_real_rollouts_share_a_state_layout():
    """The reusability claim depends on it: two domains are only comparable in
    one latent space if their feature vectors mean the same thing.
    """
    sim = simulated_pgm_rollout(0.2)
    real = digitized_pgm_rollout(0.2)
    assert sim.states.shape[1] == real.states.shape[1] == ACTUATION_STATE_DIM
    assert sim.metadata["state_layout"] == real.metadata["state_layout"]
    assert tuple(sim.metadata["state_layout"]) == ACTUATION_STATE_LAYOUT


def test_rollouts_declare_honest_temporal_provenance():
    assert simulated_pgm_rollout(0.2).temporal_provenance == "simulated"
    real = digitized_pgm_rollout(0.2)
    assert real.temporal_provenance == "quasi_static_sweep"
    # ordered enough to train on, but not clocked
    assert real.has_ordered_time_axis is True
    assert real.metadata["axis_is_applied_load_not_time"] is True


def test_real_rollout_carries_its_digitization_noise_floor():
    """So a gap number can be reported next to the uncertainty of the data it
    came from, rather than in a different document.
    """
    real = digitized_pgm_rollout(0.05)
    assert real.metadata["digitization_noise_floor_mm"] > 0
    assert real.metadata["data_is_digitized_from_a_published_figure"] is True


def test_normalization_puts_channels_on_a_comparable_scale():
    """Raw length is ~400 and raw pressure is ~0.2; unnormalized, the encoder
    would see length and essentially nothing else.
    """
    real = digitized_pgm_rollout(0.2)
    spread = real.states.max(axis=0) - real.states.min(axis=0)
    assert real.states.min() >= -0.5
    assert real.states.max() <= 1.5
    # force and length both actually vary along the sweep
    assert spread[1] > 0.5
    assert spread[2] > 0.1


def test_every_published_pressure_level_produces_both_sides():
    for p in OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA:
        assert simulated_pgm_rollout(p).states.shape[0] > 10
        assert digitized_pgm_rollout(p).states.shape[0] > 10


# --- residuals ----------------------------------------------------------------


def test_model_error_exceeds_digitization_noise_at_every_usable_pressure():
    """The headline V2 result, asserted rather than just printed: the naive
    model's disagreement with the real curves is far larger than the
    uncertainty of the digitization, so it is a real modelling error and not a
    tracing artifact.
    """
    for p in OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA:
        r = length_residuals_mm(p)
        assert r["residual_exceeds_noise_floor"], f"{p} MPa"


def test_residual_is_smallest_in_the_mid_pressure_range():
    """Matches Ogawa Section 2.3: behaviour is most nonlinear at 0.05-0.15 MPa
    and the ideal model has no elastic term, so error should grow towards both
    ends rather than sit flat.
    """
    usable = [p for p in OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA if p > 0]
    errs = {p: length_residuals_mm(p)["mean_abs_residual_mm"] for p in usable}
    best = min(errs, key=errs.get)
    assert 0.05 < best < 0.3
    assert errs[0.3] > errs[best]
    assert errs[0.05] > errs[best]
