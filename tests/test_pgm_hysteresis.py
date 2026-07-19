import numpy as np
import pytest

from worldgap.data.loaders.pgm_actuator import (
    HysteresisCurveFit,
    OGAWA_2017_PGM_VS_PM10RF_AT_0_2MPA,
    OGAWA_2017_PROTOTYPE,
    THAKUR_2018_PROTOTYPE,
    check_residual_structure,
    fit_hysteresis_curve,
    predict,
    thakur2018_force_from_pressure,
)


def _synthetic_hysteresis_loop(n=200, noise=0.01, seed=0):
    """A pressure trace that ramps up then down, with a genuinely different
    response curve on each branch — a minimal synthetic stand-in for a real
    PGM hysteresis loop, used only to check the fitting code is sound.
    """
    rng = np.random.default_rng(seed)
    half = n // 2
    pressure_up = np.linspace(0, 1, half)
    pressure_down = np.linspace(1, 0, n - half)
    pressure = np.concatenate([pressure_up, pressure_down])

    # loading branch: response = p^2 ; unloading branch: response = p^2 - 0.15
    # (unloading lags below loading at the same pressure -- a real hysteresis signature)
    response_up = pressure_up**2
    response_down = pressure_down**2 - 0.15
    response = np.concatenate([response_up, response_down])
    response += rng.normal(scale=noise, size=n)
    return pressure, response


def test_two_branch_fit_separates_loading_and_unloading():
    pressure, response = _synthetic_hysteresis_loop()
    fit = fit_hysteresis_curve(pressure, response, degree=2, digitization_uncertainty=0.02)
    predicted = predict(fit, pressure)
    residuals = np.abs(response - predicted)
    # a correct two-branch fit should track the synthetic loop tightly, well
    # within a few noise-standard-deviations
    assert residuals.mean() < 0.05


def test_residual_check_flags_low_structure_for_correctly_modeled_loop():
    pressure, response = _synthetic_hysteresis_loop(noise=0.005)
    fit = fit_hysteresis_curve(pressure, response, degree=2)
    diagnostics = check_residual_structure(fit, pressure, response)
    assert diagnostics["flag_unmodeled_hysteresis"] is False


def test_residual_check_flags_structure_when_hysteresis_is_ignored():
    """Simulates the exact mistake spec 12.13 warns about: fitting a single
    monotonic curve across both directions instead of a hysteresis-aware
    two-branch fit, and checking that the residual-structure diagnostic
    actually catches it rather than silently absorbing the error.
    """
    pressure, response = _synthetic_hysteresis_loop(noise=0.005)
    naive_coeffs = np.polyfit(pressure, response, deg=2)  # one curve, both directions
    naive_fit = HysteresisCurveFit(
        loading_coeffs=naive_coeffs,
        unloading_coeffs=naive_coeffs,
        digitization_uncertainty=0.02,
        degree=2,
    )
    diagnostics = check_residual_structure(naive_fit, pressure, response)
    assert abs(diagnostics["residual_direction_correlation"]) > 0.3
    assert diagnostics["flag_unmodeled_hysteresis"] is True


# --- Real reference data tests (Ogawa et al. 2017 / Thakur et al. 2018) ------


def test_thakur_stretched_equation_matches_papers_own_reported_check_values():
    """The paper itself reports ~30N at 60kPa and ~44N at 100kPa for the
    stretched (in-suit) configuration -- the fitted equation should reproduce
    those within the paper's own 'approximately' rounding, not just be
    plausible-looking.
    """
    assert thakur2018_force_from_pressure(60.0, stretched=True) == pytest.approx(30.0, abs=1.0)
    assert thakur2018_force_from_pressure(100.0, stretched=True) == pytest.approx(44.0, abs=1.0)


def test_thakur_unstretched_and_stretched_equations_differ():
    # Same pressure, different configuration -- must not collapse to one curve.
    unstretched = thakur2018_force_from_pressure(150.0, stretched=False)
    stretched = thakur2018_force_from_pressure(150.0, stretched=True)
    assert unstretched != stretched


def test_thakur_equation_raises_outside_validated_pressure_range():
    with pytest.raises(ValueError, match="50-300 kPa"):
        thakur2018_force_from_pressure(30.0, stretched=True)  # below 50 kPa
    with pytest.raises(ValueError, match="50-300 kPa"):
        thakur2018_force_from_pressure(350.0, stretched=True)  # above 300 kPa


def test_prototype_specs_reflect_the_two_different_physical_units():
    """Regression guard against ever accidentally merging these -- they are
    genuinely different hardware (see docs/pgm_reference_data.md), and a
    future edit collapsing them into one shared spec would be a real bug.
    """
    assert OGAWA_2017_PROTOTYPE.natural_length_mm == 250.0
    assert THAKUR_2018_PROTOTYPE.natural_length_mm == 300.0
    assert OGAWA_2017_PROTOTYPE.natural_length_mm != THAKUR_2018_PROTOTYPE.natural_length_mm
    # both share the same reported operating pressure range
    assert OGAWA_2017_PROTOTYPE.min_pressure_mpa == THAKUR_2018_PROTOTYPE.min_pressure_mpa == 0.05
    assert OGAWA_2017_PROTOTYPE.max_pressure_mpa == THAKUR_2018_PROTOTYPE.max_pressure_mpa == 0.3


def test_ogawa_comparison_table_pgm_beats_commercial_pam_on_contraction():
    """Sanity check against the paper's own stated conclusion (Section 2.4):
    the PGM has higher contraction ratio than the commercial PM-10RF at every
    force level reported.
    """
    pgm = OGAWA_2017_PGM_VS_PM10RF_AT_0_2MPA["pgm"]
    pm10rf = OGAWA_2017_PGM_VS_PM10RF_AT_0_2MPA["pm10rf"]
    for force_n in (0, 10, 20):
        pgm_contraction, _ = pgm[force_n]
        pm10rf_contraction, _ = pm10rf[force_n]
        assert pgm_contraction > pm10rf_contraction
