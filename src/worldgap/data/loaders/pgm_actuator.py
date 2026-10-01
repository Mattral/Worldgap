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
figure via WebPlotDigitizer, or an equivalent color-tracing method. Also
below: real prototype dimensions, the real operating pressure range,
Thakur's directly reusable fitted force-pressure equations, and a real
digitized Length(Force) curve family from Ogawa et al. (2017) Figure 4(a) —
all obtained directly from the papers, not estimated. See
`docs/pgm_reference_data.md` for what each real-data addition actually
represents and its documented limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import isotonic_regression
from scipy.stats import pointbiserialr

# --- Real reference data (spec Section 14 / ROADMAP Phase 0 & 6, resolved) --
#
# See docs/pgm_reference_data.md for the full transcription, citations, and
# an explicit explanation of why the two papers' measurements are kept
# separate below rather than merged into one dataset.
#
# CORRECTION (supersedes an earlier version of this module and of
# docs/pgm_reference_data.md): these are NOT two different physical
# prototypes. Thakur et al. (2018) Section II.A describes the actuator as
# the one "we previously developed" citing Ogawa et al. (2017), reuses
# Ogawa's elongation figure ("Fig. 2 shows the elongation ratio of the PGM
# as measured by [14]"), cites Ogawa for the 50-300 kPa range, and motivates
# its own experiment precisely because the stretched-length force behaviour
# "is not measured in [14]". The two papers therefore report the same PGM
# design measured in two different ways. The reason the numbers below are
# still kept apart is the MEASUREMENT TYPE, not the hardware:
#
#   Ogawa 2017 Fig. 4(a): Length(Force) at each of 7 FIXED supply pressures
#                         (pressure held, hung load cycled 0->5 kg->0)
#   Thakur 2018 Eq. 1/2:  Force(pressure) at a FIXED PGM length
#                         (length held, pressure swept 50-300 kPa)
#
# Those are different dependent variables under different held-constant
# conditions; averaging or interpolating between them would silently invent
# a measurement neither paper made.
#
# The two papers also report the prototype's dimensions inconsistently, and
# worldgap does not paper over that -- see `PGMPrototypeSpec` below.

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
    """PGM dimensions **as reported by one specific paper** — deliberately one
    spec object per source, because the two papers report the same actuator's
    dimensions inconsistently and worldgap records that disagreement rather
    than resolving it by fiat.

    What each paper says, verbatim in substance:

    - Ogawa 2017 §2.2: inner tube 250 mm (inner dia. 4 mm), outer braided
      mesh 500 mm (outer dia. 9 mm).
    - Ogawa 2017 §2.3: "a natural length of 250 mm and maximum possible
      elongation of 500 mm".
    - Ogawa 2017 §4 (Discussion): "had a fixed length of 300 mm of the normal
      length".
    - Thakur 2018 §II.A: "resting length of 30 cm, maximum contraction length
      of 25 cm, and maximum elongation length of 45 cm".

    Ogawa's own §2.3 and §4 therefore disagree with each other (250 mm vs
    300 mm "normal"/"natural"), and §4 agrees with Thakur. A plausible
    reconciliation — flagged here as **our inference, not either paper's
    claim** — is that Thakur's triple is the careful one (resting 300,
    fully contracted 250, fully elongated 450) and Ogawa §2.3 reported the
    inner-tube length (250 mm) and the braided-mesh length (500 mm) from
    §2.2 as if they were the assembled muscle's natural and maximum lengths.
    Nothing in worldgap depends on that reconciliation being correct; both
    specs are stored as reported so downstream code can pick its source
    explicitly.
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
    sweep, not elongation response to a hung load at one pressure. Thakur
    describes it as the same PGM Ogawa developed (§II.A explicitly: the
    stretched-length behaviour "is not measured in [14]"), so the reason not
    to merge it with the Ogawa curves is the measurement type, not different
    hardware. Use this as an independent force-generation cross-check /
    simple reference model, not as hysteresis-loop ground truth.

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


# --- Real digitized data: Ogawa et al. (2017) Figure 4(a) ---------------------
#
# Resolves the last item in docs/pgm_reference_data.md's "not yet extracted"
# list. Digitized from the published figure (method: pixel-color tracing with
# connected-component analysis per curve color, see DIGITIZATION_README.txt
# bundled alongside the CSVs) -- these are estimates reconstructed from the
# chart, not the original experimental measurements.
#
# IMPORTANT — this is NOT the pressure-ramp hysteresis
# `fit_hysteresis_curve`/`predict` above model. Ogawa's actual experiment held
# pressure FIXED and cycled the applied LOAD (0->5kg in 1kg steps): what's
# digitized here is elongation response to load, at each of 7 fixed supply
# pressures -- a family of Length(Force) curves parameterized by pressure, not
# a single curve showing hysteresis with respect to pressure direction.
# Neither paper worldgap has actually characterizes pressure-ramp hysteresis
# (see docs/pgm_reference_data.md). Use this data as a real Length(Force,
# Pressure) reference surface for validating a simulated actuator model
# against real measurements -- not as input to `fit_hysteresis_curve`, whose
# "loading"/"unloading" branches mean something different (pressure
# direction, not load direction).

_OGAWA_2017_FIG4A_DIR = Path(__file__).parent.parent / "reference_data" / "ogawa2017_fig4a"

OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3)

# Reference length used to convert a digitized Length back into the paper's
# "contraction ratio", for the independent Figure 6 cross-check.
#
# HONESTY NOTE (this is an inference, not a formula either paper writes):
# Ogawa §2.4 says Figure 6 shows "the contraction ratio relative to the
# natural length", and never states the arithmetic. Taking the reference to
# be the 500 mm of §2.3's "maximum possible elongation of 500 mm" (= §2.2's
# braided-mesh length) and computing (500 - L)/500 reproduces Figure 6's
# separately-stated 36/29/23 % at 0/10/20 N to within 1.4 percentage points,
# so 500 mm is evidently the reference the figure used. The literal reading
# of "natural length" (250 mm per §2.3, or 300 mm per §4) does not reproduce
# those percentages at all. Treat this constant as "the reference length that
# reproduces the paper's own published numbers", not as the paper's
# definition.
OGAWA_2017_MAX_ELONGATION_MM = 500.0

# Ogawa 2017 §4 (Discussion), verbatim in substance and directly relevant to
# any hand/wrist application of this reference data: the characterized PGM
# "had a fixed length of 300 mm of the normal length. This length is suitable
# for applications such as assisting walking but not for assisting the hand
# or wrist, which requires shorter muscles." The paper goes on to say future
# work should "model them with a third parameter of the resting length of the
# PGM along with the air pressure and applied force."
#
# Consequence for worldgap, stated plainly rather than buried: the reference
# data below characterizes a walking-suit-scale actuator. Using it as the
# "real" side of a V2 gap for a hand/wrist-scale PGM is an extrapolation
# across a length scale the source paper explicitly says does not carry over,
# and any such result MUST be labelled that way.
OGAWA_2017_SCALE_CAVEAT = (
    "Ogawa et al. (2017) Section 4 states the characterized PGM had a normal "
    "length of 300 mm, which the authors describe as suitable for walking "
    "assistance but NOT for hand or wrist assistance, which requires shorter "
    "muscles; they recommend resting length be modelled as a third parameter "
    "alongside air pressure and applied force. Reference data loaded from this "
    "module therefore characterizes a walking-suit-scale actuator."
)


@dataclass(frozen=True)
class DigitizedLengthForceCurve:
    """One pressure level's digitized Length(Force) curve, smoothed to enforce
    the known physical constraint that elongation is non-decreasing with
    applied load in this quasi-static test.
    """

    pressure_mpa: float
    force_n: np.ndarray
    length_mm: np.ndarray
    digitization_noise_floor_mm: float
    """Max absolute correction isotonic (monotonic) smoothing applied to the
    raw digitized points, per spec 12.14 ("digitization uncertainty MUST be
    recorded and carried through as a noise floor").

    This is a **heuristic proxy** for digitization uncertainty, not a
    statistical confidence interval and not a calibrated error bar. It says:
    "at least this much disagreement with a physically-required constraint
    was present in the traced points here." It is a lower bound on the trace
    error in a loose sense only — error that happens to preserve monotonicity
    is invisible to it. Use it as a floor below which differences should not
    be interpreted, not as a +/- uncertainty to propagate arithmetically.
    """
    source_citation: str = OGAWA_2017_CITATION

    def length_at(self, force_n: float) -> float:
        """Linearly interpolates within the digitized force range. Raises
        outside it rather than silently extrapolating a curve shape that was
        never observed there (same convention as `thakur2018_force_from_pressure`).
        """
        lo, hi = self.force_n.min(), self.force_n.max()
        if not (lo <= force_n <= hi):
            raise ValueError(
                f"force_n={force_n} is outside the digitized range [{lo:.1f}, {hi:.1f}] N "
                f"for pressure={self.pressure_mpa} MPa -- refusing to extrapolate."
            )
        return float(np.interp(force_n, self.force_n, self.length_mm))


def load_ogawa2017_fig4a_curve(pressure_mpa: float) -> DigitizedLengthForceCurve:
    """Loads one pressure level's digitized Length(Force) curve, applying
    isotonic regression to enforce non-decreasing elongation with load and
    recording the correction magnitude as the digitization noise floor.

    Raises for any pressure not in `OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA`
    -- only the 7 levels Ogawa et al. (2017) actually tested are available;
    this deliberately does not interpolate across pressure levels, since the
    PGM's pressure-response is nonlinear and doing so would imply confidence
    in a pressure this data doesn't cover.
    """
    matches = [p for p in OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA if abs(p - pressure_mpa) < 1e-6]
    if not matches:
        raise ValueError(
            f"pressure_mpa={pressure_mpa} was not one of the levels Ogawa et al. (2017) "
            f"Figure 4(a) actually tested: {OGAWA_2017_FIG4A_AVAILABLE_PRESSURES_MPA}. "
            "This function does not interpolate across pressure levels."
        )
    exact_pressure = matches[0]

    suffix = "0" if exact_pressure == 0.0 else str(exact_pressure).replace(".", "p")
    csv_path = _OGAWA_2017_FIG4A_DIR / f"PGM_length_{suffix}MPa.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"expected digitized data at {csv_path}, not found")

    force_n, length_mm = [], []
    with open(csv_path) as f:
        next(f)  # header
        for line in f:
            f_val, l_val = line.strip().split(",")
            force_n.append(float(f_val))
            length_mm.append(float(l_val))
    force_n = np.array(force_n)
    length_mm = np.array(length_mm)

    order = np.argsort(force_n)
    force_n, length_mm = force_n[order], length_mm[order]

    smoothed = isotonic_regression(length_mm, increasing=True).x
    noise_floor_mm = float(np.abs(length_mm - smoothed).max())

    return DigitizedLengthForceCurve(
        pressure_mpa=exact_pressure,
        force_n=force_n,
        length_mm=smoothed,
        digitization_noise_floor_mm=noise_floor_mm,
    )
