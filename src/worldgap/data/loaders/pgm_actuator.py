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

`fit_hysteresis_curve`/`predict`/`check_residual_structure` below take a
digitized (pressure, response) curve — typically extracted from a published
figure via WebPlotDigitizer. That full-curve digitization is still pending
(see `docs/pgm_reference_data.md`); what IS available now, and is real —
obtained directly from Ogawa et al. (2017) and Thakur et al. (2018), not
estimated — is below: real prototype dimensions, the real operating pressure
range, and Thakur's directly reusable fitted force-pressure equations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pointbiserialr

# --- Real reference data (spec Section 14 / ROADMAP Phase 0 & 6, resolved) --
#
# See docs/pgm_reference_data.md for the full transcription, citations, and
# an explicit explanation of why these two papers describe different physical
# prototypes that are kept separate below rather than merged into one dataset.

OGAWA_2017_CITATION = (
    "Ogawa, K., Thakur, C., Ikeda, T., Tsuji, T., & Kurita, Y. (2017). "
    "Development of a pneumatic artificial muscle driven by low pressure and "
    "its application to the unplugged powered suit. Advanced Robotics, 31(21), "
    "1135-1143. https://doi.org/10.1080/01691864.2017.1392345"
)

THAKUR_2018_CITATION = (
    "Thakur, C., Ogawa, K., Tsuji, T., & Kurita, Y. (2018). Soft Wearable "
    "Augmented Walking Suit With Pneumatic Gel Muscles and Stance Phase "
    "Detection System to Assist Gait. IEEE Robotics and Automation Letters, "
    "3(4), 4257-4264. https://doi.org/10.1109/LRA.2018.2864355"
)


@dataclass(frozen=True)
class PGMPrototypeSpec:
    """Physical dimensions of one specific PGM prototype, as reported in one
    specific paper. Two different prototypes exist across the two papers
    worldgap currently has (250mm vs 300mm natural length) — see
    docs/pgm_reference_data.md for why they're kept as two separate specs
    rather than averaged or merged.
    """

    natural_length_mm: float
    max_elongation_length_mm: float
    min_pressure_mpa: float
    max_pressure_mpa: float
    source_citation: str
    max_contraction_length_mm: float | None = None
    inner_tube_diameter_mm: float | None = None


OGAWA_2017_PROTOTYPE = PGMPrototypeSpec(
    natural_length_mm=250.0,
    max_elongation_length_mm=500.0,
    min_pressure_mpa=0.05,
    max_pressure_mpa=0.3,
    inner_tube_diameter_mm=4.0,
    source_citation=OGAWA_2017_CITATION,
)

THAKUR_2018_PROTOTYPE = PGMPrototypeSpec(
    natural_length_mm=300.0,
    max_contraction_length_mm=250.0,
    max_elongation_length_mm=450.0,
    min_pressure_mpa=0.05,
    max_pressure_mpa=0.3,
    source_citation=THAKUR_2018_CITATION,
)

# Ogawa et al. (2017) Figure 6 / Section 2.4: contraction & elongation ratio
# comparison between the PGM and a commercial low-powered PAM (Squse PM-10RF),
# both at 0.2 MPa supply pressure. Transcribed directly from the paper's own
# text (which explicitly states these percentages), not read off the figure.
# Values are (contraction_pct, elongation_pct); elongation is not reported at
# 0 N for either muscle (elongation is a beyond-natural-length phenomenon that
# only shows up once there's load pulling against the muscle's own contraction).
OGAWA_2017_PGM_VS_PM10RF_AT_0_2MPA = {
    "pgm": {0: (36, None), 10: (29, 11), 20: (23, 20)},
    "pm10rf": {0: (32, None), 10: (14, 29), 20: (3, 41)},
}


def thakur2018_force_from_pressure(pressure_kpa: float, stretched: bool) -> float:
    """Real, directly-usable fitted force-vs-pressure relationship from
    Thakur et al. (2018) Section II.A (Eq. 1 unstretched, Eq. 2 stretched),
    fit via scipy.optimize.curve_fit against measured force at a FIXED PGM
    length (either the natural/unstretched length, or stretched to 45cm as
    used in the actual AWS suit), across a range of supplied air pressures.

    This is NOT the same physical measurement `fit_hysteresis_curve` below
    models — it's force output at one constant length across a pressure
    sweep, not elongation response to a hung load at one pressure, and it's
    from a different (300mm, not 250mm) prototype. Use this as an independent
    force-generation cross-check / simple reference model, not as hysteresis-
    loop ground truth for the same actuator.

    Raises outside the reported valid range (50-300 kPa) rather than silently
    extrapolating a linear fit the source paper never validated there.
    """
    if not (50.0 <= pressure_kpa <= 300.0):
        raise ValueError(
            f"pressure_kpa={pressure_kpa} is outside 50-300 kPa, the range Thakur "
            "et al. (2018) Section II.A actually measured and fit -- extrapolating "
            "this linear equation beyond that range isn't something the source "
            "paper validated, so this function refuses to guess. See "
            "docs/pgm_reference_data.md."
        )
    if stretched:
        return 0.3883 * pressure_kpa + 5.8899  # R^2 = 0.998, Eq. 2
    return 0.1799 * pressure_kpa - 5.1983  # R^2 = 0.993, Eq. 1



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
