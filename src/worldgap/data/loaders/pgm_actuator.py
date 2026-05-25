"""PGM actuator reference-curve fitting, per TECHNICAL_SPEC.md Section 5.5 and
edge case 12.13.

Real requirement, not a formality: PGMs (McKibben-type pneumatic muscles)
exhibit hysteresis — the response at a given pressure depends on whether
pressure is currently increasing or decreasing. A single monotonic curve fit
across both directions will systematically misrepresent the real actuator,
which in turn silently corrupts every downstream V2 gap-score number. This
module fits the loading and unloading branches separately and checks the
residuals for leftover directional structure (spec 12.13: 'MUST check fit
residuals aren't systematically structured').

Input is expected to be a digitized (pressure, response) curve — typically
extracted from a published figure via WebPlotDigitizer, since raw data from
Ogawa et al. (2017) is not expected to be available (spec 5.5, 14). Digitization
uncertainty MUST be recorded and carried through as a noise floor (spec 12.14).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pointbiserialr


@dataclass
class HysteresisCurveFit:
    loading_coeffs: np.ndarray
    unloading_coeffs: np.ndarray
    digitization_uncertainty: float
    degree: int


def fit_hysteresis_curve(
    pressure: np.ndarray,
    response: np.ndarray,
    degree: int = 3,
    digitization_uncertainty: float = 0.02,
) -> HysteresisCurveFit:
    """Splits the digitized curve into loading (pressure increasing) and
    unloading (pressure decreasing) branches by the local sign of dPressure/dt,
    and fits each with its own polynomial — a deliberately simple hysteresis-aware
    alternative to a single monotonic fit.
    """
    if len(pressure) != len(response):
        raise ValueError("pressure and response arrays must be the same length")
    if len(pressure) < 2 * (degree + 1):
        raise ValueError(
            f"need at least {2 * (degree + 1)} points to fit a degree-{degree} "
            "polynomial on each of two branches"
        )

    d_pressure = np.gradient(pressure)
    loading_mask = d_pressure >= 0
    unloading_mask = ~loading_mask

    if loading_mask.sum() < degree + 1 or unloading_mask.sum() < degree + 1:
        raise ValueError(
            "not enough points on the loading or unloading branch to fit the "
            f"requested degree ({degree}) — got {loading_mask.sum()} loading, "
            f"{unloading_mask.sum()} unloading points. This can happen if the "
            "digitized curve is monotonic (no hysteresis loop captured) — check "
            "the source figure before proceeding, per spec 12.13/12.15."
        )

    loading_coeffs = np.polyfit(pressure[loading_mask], response[loading_mask], degree)
    unloading_coeffs = np.polyfit(pressure[unloading_mask], response[unloading_mask], degree)

    return HysteresisCurveFit(
        loading_coeffs=loading_coeffs,
        unloading_coeffs=unloading_coeffs,
        digitization_uncertainty=digitization_uncertainty,
        degree=degree,
    )


def predict(fit: HysteresisCurveFit, pressure_trace: np.ndarray) -> np.ndarray:
    """Predicts response for a commanded pressure trace, selecting the loading
    or unloading branch at each timestep based on the local pressure direction.
    """
    d_pressure = np.gradient(pressure_trace)
    loading_pred = np.polyval(fit.loading_coeffs, pressure_trace)
    unloading_pred = np.polyval(fit.unloading_coeffs, pressure_trace)
    return np.where(d_pressure >= 0, loading_pred, unloading_pred)


def check_residual_structure(
    fit: HysteresisCurveFit, pressure: np.ndarray, response: np.ndarray
) -> dict:
    """Per spec 12.13: MUST check whether the branch-split fit actually absorbed
    the hysteresis, or whether residuals still correlate with pressure
    direction (a sign the two-branch model is too simple for this actuator and
    a fuller model — e.g. Bouc-Wen — is needed).
    """
    predicted = predict(fit, pressure)
    residuals = response - predicted
    d_pressure = np.gradient(pressure)
    loading_indicator = (d_pressure >= 0).astype(float)

    if loading_indicator.std() == 0:
        direction_correlation = 0.0
    else:
        direction_correlation, _ = pointbiserialr(loading_indicator, residuals)

    return {
        "residual_std": float(residuals.std()),
        "residual_direction_correlation": float(direction_correlation),
        "flag_unmodeled_hysteresis": bool(abs(direction_correlation) > 0.3),
    }
