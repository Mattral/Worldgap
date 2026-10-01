import numpy as np
import pytest

from worldgap.data.loaders.pgm_actuator import (
    OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA,
    OGAWA_2017_MAX_ELONGATION_MM,
    OGAWA_2017_PGM_VS_PM10RF_AT_0_2MPA,
    OGAWA_2017_PROTOTYPE,
    THAKUR_2018_PROTOTYPE,
    HysteresisCurveFit,
    check_residual_structure,
    fit_hysteresis_curve,
    load_ogawa2017_fig4a_curve,
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


def test_prototype_specs_preserve_each_papers_own_reported_dimensions():
    """Regression guard against "tidying up" the disagreement away.

    These are NOT two different actuators -- Thakur 2018 Section II.A
    describes the PGM as the one "we previously developed" citing Ogawa 2017
    (see docs/pgm_reference_data.md). But the two papers report its
    dimensions inconsistently (Ogawa Section 2.3 says 250 mm natural, Ogawa
    Section 4 says 300 mm normal, Thakur says 300 mm resting), and worldgap
    stores what each paper actually said rather than picking a winner. A
    future edit that collapses these into one "reconciled" spec would be
    fabricating agreement the literature doesn't have.
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


# --- Real digitized Length(Force) curves (Ogawa et al. 2017 Figure 4a) ------


def test_all_seven_pressure_levels_load_and_are_monotonic():
    """Physical sanity check: elongation must be non-decreasing with applied
    load in this quasi-static test. Isotonic smoothing inside the loader
    enforces this, so a violation here means the loader itself is broken.
    """
    assert len(OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA) == 7
    for pressure in OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA:
        curve = load_ogawa2017_fig4a_curve(pressure)
        assert curve.pressure_mpa == pressure
        assert len(curve.force_n) > 50  # digitized at ~0.5N resolution over ~0-49N
        assert np.all(np.diff(curve.length_mm) >= -1e-9)
        assert curve.digitization_noise_floor_mm >= 0.0


def test_higher_pressure_gives_shorter_length_at_shared_force():
    """Cross-curve sanity check: at a force level every curve actually
    covers, higher supply pressure should mean a shorter (more contracted)
    muscle -- this is the whole point of a pneumatic actuator. Checked at
    Force=30N, comfortably inside every curve's digitized range.
    """
    lengths = [
        load_ogawa2017_fig4a_curve(p).length_at(30.0) for p in OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA
    ]
    assert lengths == sorted(lengths, reverse=True)


def test_digitized_0_2mpa_curve_matches_papers_independently_stated_contraction_ratios():
    """The strongest available cross-check: Ogawa's own Figure 6 / Section
    2.4 text states contraction ratios (relative to the 500mm max elongation,
    spec 2.2) at 0.2 MPa independently of Figure 4(a)'s graph. If the digitized
    curve is accurate, converting it to a contraction ratio should reproduce
    those numbers within a few percentage points -- not exactly, since one
    came off a graph and the other is the paper's own rounded text, but close.
    """
    curve = load_ogawa2017_fig4a_curve(0.2)
    paper_stated_pct = {0: 36, 10: 29, 20: 23}
    for force_n, expected_pct in paper_stated_pct.items():
        f = max(force_n, curve.force_n.min())  # curve may not start exactly at 0
        length = curve.length_at(f)
        implied_pct = (OGAWA_2017_MAX_ELONGATION_MM - length) / OGAWA_2017_MAX_ELONGATION_MM * 100
        assert implied_pct == pytest.approx(expected_pct, abs=3.0)


def test_length_at_raises_outside_digitized_force_range():
    curve = load_ogawa2017_fig4a_curve(0.05)
    with pytest.raises(ValueError, match="outside the digitized range"):
        curve.length_at(curve.force_n.max() + 10.0)


def test_load_ogawa2017_fig4a_curve_rejects_untested_pressure():
    with pytest.raises(ValueError, match="not one of the levels"):
        load_ogawa2017_fig4a_curve(0.4)
