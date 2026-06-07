import numpy as np

from worldgap.data.loaders.pgm_actuator import (
    HysteresisCurveFit,
    check_residual_structure,
    fit_hysteresis_curve,
    predict,
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
