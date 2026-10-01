"""Simulated PGM actuator baseline for V2 -- the "sim" side of the sim-to-real gap.

Per TECHNICAL_SPEC.md Section 5.5, with the simulator choice recorded here as a
documented decision (open question Q9 in the project's own notes).

## What this simulates, and why this model rather than another

The point of V2 is to answer: *if you build a soft-exosuit controller against a
simulated pneumatic muscle, how wrong is the simulator?* That question is only
meaningful if the simulator is the one a roboticist would actually reach for
before they had characterization data in hand. So the baseline here is the
**ideal McKibben model** -- the standard textbook force balance for a braided
pneumatic artificial muscle, derived from the braid's geometry alone:

    F(eps, P) = (pi * D0^2 * P / 4) * ( 3*(1-eps)^2 / tan^2(theta0) - 1 / sin^2(theta0) )

with `eps = (L0 - L) / L0` the contraction ratio, `L0` the resting length, `D0`
the braid diameter at the resting braid angle `theta0`. (Chou & Hannaford,
"Measurement and modeling of McKibben pneumatic artificial muscles",
IEEE T-RA 12(1), 1996; the same relation appears in Schulte 1961 and in every
subsequent PAM survey.)

Deliberately NOT chosen:

- **A fit to the Ogawa curves.** Comparing a curve fitted on the reference data
  against that same reference data measures curve-fitting, not sim-to-real.
  The gap would be small by construction and would mean nothing.
- **MuJoCo (`spec 5.5`'s other option).** MuJoCo has no native McKibben
  actuator; using it means supplying a custom gain/bias function -- i.e.
  writing this same analytic model, with a physics engine wrapped around it and
  a heavy optional dependency added. If V3 ever needs contact or full-body
  dynamics, MuJoCo earns its place; for a single actuator's static response it
  does not.

## What this model is known to get wrong

Stated up front, because these are the findings V2 should surface, not
surprises:

1. **No hysteresis.** Force depends only on the current `(eps, P)`, not on
   whether load is increasing or decreasing. Ogawa Section 2.3 reports genuine
   loading/unloading nonlinearity at 0.05-0.15 MPa.
2. **Zero force at zero pressure.** The ideal model has `F = 0` when `P = 0`.
   The real PGM's gel-foam inner tube is an elastic body that resists stretch
   with no air at all -- Ogawa's 0 MPa curve is a real measured curve, and this
   model cannot produce it. An optional linear `tube_stiffness_n_per_mm` term
   is available for the variant that accounts for it; the default baseline
   leaves it at zero precisely so the omission shows up in the gap.
3. **No threshold/dead-band pressure**, no friction between braid and bladder,
   no end-cap effects. All are documented real contributors to PAM modelling
   error.

That list is the hypothesis V2 tests. A large gap concentrated at low pressure
would corroborate it; a large gap that is uniform across pressure would suggest
something else is wrong and is worth chasing.

## Where the geometry numbers come from

`OGAWA_2017_IDEAL_MCKIBBEN` is parameterized from Ogawa et al. (2017)'s own
reported geometry, with every inference marked:

- `L0 = 500 mm` -- Section 2.2's "outer braided mesh of 500 mm". **Inference**:
  this is the same reference length that reproduces Figure 6's published
  contraction ratios (see `docs/pgm_reference_data.md`), which is the strongest
  available evidence for which length the figure's axis is relative to.
- `D0 = 23 mm` -- Section 2.3's "default diameter of the braided mesh was 23 mm".
- `theta0 = 42.1 deg` -- **derived, not reported.** Braid kinematics give
  `D/D0 = sin(theta)/sin(theta0)`; maximum contraction of a braided muscle
  occurs at `theta = 54.74 deg` (where `3cos^2 - 1 = 0`... in fact where the
  braid can contract no further), and Section 2.3 gives the maximum diameter as
  28 mm. Hence `sin(theta0) = D0 * sin(54.74 deg) / D_max`, giving
  `theta0 = asin(23 * 0.8165 / 28) = 42.1 deg`. This is a chain of three
  inferences and is flagged as such wherever it is used.

Nothing downstream should treat these as measured parameters. They are a
defensible parameterization of a naive model, which is exactly what V2 needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..rollout import TEMPORAL_PROVENANCE_KEY, Rollout
from .pgm_actuator import OGAWA_2017_CITATION, load_ogawa2017_fig4a_curve

#: State layout for V2 actuation rollouts produced by this module.
#: Both the simulated and the digitized-real side MUST use it, or the two
#: domains are not comparable in one latent space (spec Section 3).
ACTUATION_STATE_LAYOUT = ("pressure_mpa", "force_n", "length_mm")
ACTUATION_STATE_DIM = len(ACTUATION_STATE_LAYOUT)

#: Fixed scales used to bring the three channels onto a comparable range before
#: encoding. Constants, not fitted from either domain: a normalization derived
#: from one side would leak that side's distribution into the comparison, which
#: is precisely what a domain-gap measurement must not do. Chosen from the
#: experiment's own stated ranges (0-0.3 MPa, 0-50 N, 300-550 mm).
ACTUATION_NORMALIZATION = {
    "pressure_mpa": (0.0, 0.3),
    "force_n": (0.0, 50.0),
    "length_mm": (300.0, 550.0),
}

_MAX_CONTRACTION_BRAID_ANGLE_DEG = 54.7356  # asin(sqrt(2/3)); standard braid result


@dataclass(frozen=True)
class IdealMcKibbenParams:
    """Geometry of the ideal braided-muscle model. See this module's docstring
    for the provenance of each number and which are inferred.
    """

    resting_length_mm: float
    braid_diameter_mm: float
    braid_angle_deg: float
    tube_stiffness_n_per_mm: float = 0.0
    """Optional linear elastic term for the gel-foam inner tube, in N per mm of
    stretch beyond resting length. Zero in the baseline **on purpose**: the
    ideal model's inability to produce any force at zero pressure is one of the
    specific errors V2 exists to quantify, so hiding it behind a fudge factor
    would defeat the measurement. Set it non-zero only to test the hypothesis
    that tube elasticity explains the low-pressure gap.
    """
    source_note: str = (
        "Parameterized from Ogawa et al. (2017) reported geometry; braid angle "
        "derived, not reported. See pgm_sim module docstring."
    )


OGAWA_2017_IDEAL_MCKIBBEN = IdealMcKibbenParams(
    resting_length_mm=500.0,
    braid_diameter_mm=23.0,
    braid_angle_deg=math.degrees(
        math.asin(23.0 * math.sin(math.radians(_MAX_CONTRACTION_BRAID_ANGLE_DEG)) / 28.0)
    ),
)


def ideal_mckibben_force_n(
    length_mm: np.ndarray | float,
    pressure_mpa: float,
    params: IdealMcKibbenParams = OGAWA_2017_IDEAL_MCKIBBEN,
) -> np.ndarray:
    """Force the ideal McKibben model predicts at a given length and pressure.

    Sign convention matches the Ogawa experiment: positive force is the load
    hung on the muscle, which stretches it. The model's own convention produces
    contractile force, so the returned value is the load the muscle can hold at
    that length -- i.e. the balance point. Negative results are clipped to zero:
    a hung weight cannot push.

    Units: MPa in, N out (1 MPa = 1 N/mm^2, so no conversion factor is needed
    given diameters in mm).
    """
    length = np.asarray(length_mm, dtype=float)
    theta0 = math.radians(params.braid_angle_deg)
    eps = (params.resting_length_mm - length) / params.resting_length_mm

    pneumatic = (math.pi * params.braid_diameter_mm**2 * pressure_mpa / 4.0) * (
        3.0 * (1.0 - eps) ** 2 / math.tan(theta0) ** 2 - 1.0 / math.sin(theta0) ** 2
    )
    elastic = params.tube_stiffness_n_per_mm * np.maximum(
        length - params.resting_length_mm, 0.0
    )
    return np.maximum(pneumatic + elastic, 0.0)


def simulate_length_at_force(
    force_n: np.ndarray,
    pressure_mpa: float,
    params: IdealMcKibbenParams = OGAWA_2017_IDEAL_MCKIBBEN,
    length_bounds_mm: tuple[float, float] = (250.0, 550.0),
    n_grid: int = 4001,
) -> np.ndarray:
    """Inverts `ideal_mckibben_force_n` to get the length the model predicts
    under each applied load -- the same quantity Ogawa's Figure 4(a) measures.

    Inverted numerically on a dense length grid rather than algebraically,
    because the optional elastic term makes the closed form messy and the grid
    is exact to its own resolution (0.075 mm at the default settings, well below
    the digitization noise floor of the data it gets compared against).

    Loads the model cannot support at any length in bounds are clipped to the
    bound rather than extrapolated, and that clipping is reported by
    `simulated_curve_is_saturated()` so a flat region is never mistaken for a
    physical plateau.
    """
    grid = np.linspace(length_bounds_mm[0], length_bounds_mm[1], n_grid)
    predicted = ideal_mckibben_force_n(grid, pressure_mpa, params)

    # force is non-decreasing in length for this model (longer -> less
    # contracted -> holds more), so np.interp on (force, length) is valid.
    order = np.argsort(predicted)
    return np.interp(np.asarray(force_n, dtype=float), predicted[order], grid[order])


def simulated_curve_is_saturated(
    force_n: np.ndarray,
    lengths_mm: np.ndarray,
    length_bounds_mm: tuple[float, float] = (250.0, 550.0),
    tol_mm: float = 1e-6,
) -> dict[str, float]:
    """Reports how much of a simulated curve sits pinned at a search bound.

    A high fraction means the model could not represent that load at all and the
    curve is flat because the grid ran out, not because the actuator plateaus.
    Any gap computed over a heavily saturated curve is measuring the clipping.
    """
    lengths = np.asarray(lengths_mm)
    lo = float(np.mean(np.abs(lengths - length_bounds_mm[0]) < tol_mm))
    hi = float(np.mean(np.abs(lengths - length_bounds_mm[1]) < tol_mm))
    return {"fraction_at_lower_bound": lo, "fraction_at_upper_bound": hi, "fraction_saturated": lo + hi}


def _to_states(pressure_mpa: float, force_n: np.ndarray, length_mm: np.ndarray) -> np.ndarray:
    raw = np.stack(
        [np.full_like(force_n, pressure_mpa, dtype=float), np.asarray(force_n, dtype=float),
         np.asarray(length_mm, dtype=float)],
        axis=1,
    )
    normalized = np.empty_like(raw)
    for i, channel in enumerate(ACTUATION_STATE_LAYOUT):
        lo, hi = ACTUATION_NORMALIZATION[channel]
        normalized[:, i] = (raw[:, i] - lo) / (hi - lo)
    return normalized


def simulated_pgm_rollout(
    pressure_mpa: float,
    force_grid_n: np.ndarray | None = None,
    params: IdealMcKibbenParams = OGAWA_2017_IDEAL_MCKIBBEN,
    condition: dict | None = None,
) -> Rollout:
    """One quasi-static load sweep from the simulated actuator, as a `Rollout`.

    The time axis is the load sweep -- ascending applied force -- mirroring
    Ogawa's own procedure of adding weight in steps and recording the resulting
    length. Tagged `temporal_provenance="simulated"`.
    """
    force = (
        np.arange(0.5, 49.0, 0.5) if force_grid_n is None else np.asarray(force_grid_n, float)
    )
    length = simulate_length_at_force(force, pressure_mpa, params)
    states = _to_states(pressure_mpa, force, length)
    return Rollout(
        modality="actuation",
        source="sim",
        condition={**(condition or {}), "pressure_mpa": pressure_mpa},
        frame_rate_hz=1.0,  # nominal: the axis is applied load, not time
        states=states,
        presence_mask=np.ones_like(states),
        timestamps_ms=np.arange(len(force), dtype=float) * 1000.0,
        metadata={
            TEMPORAL_PROVENANCE_KEY: "simulated",
            "model": "ideal_mckibben",
            "state_layout": list(ACTUATION_STATE_LAYOUT),
            "axis_is_applied_load_not_time": True,
            "saturation": simulated_curve_is_saturated(force, length),
            "braid_angle_deg_is_derived_not_reported": True,
        },
    )


def digitized_pgm_rollout(pressure_mpa: float, condition: dict | None = None) -> Rollout:
    """The same sweep from the real digitized Ogawa Figure 4(a) data.

    Identical state layout and normalization to `simulated_pgm_rollout`, which
    is what makes the two comparable in one latent space. Tagged
    `temporal_provenance="quasi_static_sweep"`: genuinely ordered (the load was
    added in steps, in order) but not clocked, so nothing here may be reported
    in seconds or Hz.

    The curve's digitization noise floor travels along in metadata, so a gap
    result can be reported next to the uncertainty of the data it came from
    rather than separately from it.
    """
    curve = load_ogawa2017_fig4a_curve(pressure_mpa)
    states = _to_states(pressure_mpa, curve.force_n, curve.length_mm)
    return Rollout(
        modality="actuation",
        source="real",
        condition={**(condition or {}), "pressure_mpa": pressure_mpa},
        frame_rate_hz=1.0,  # nominal: the axis is applied load, not time
        states=states,
        presence_mask=np.ones_like(states),
        timestamps_ms=np.arange(len(curve.force_n), dtype=float) * 1000.0,
        metadata={
            TEMPORAL_PROVENANCE_KEY: "quasi_static_sweep",
            "state_layout": list(ACTUATION_STATE_LAYOUT),
            "axis_is_applied_load_not_time": True,
            "digitization_noise_floor_mm": curve.digitization_noise_floor_mm,
            "source_citation": OGAWA_2017_CITATION,
            "data_is_digitized_from_a_published_figure": True,
        },
    )


def length_residuals_mm(pressure_mpa: float, params: IdealMcKibbenParams = OGAWA_2017_IDEAL_MCKIBBEN) -> dict:
    """Direct physical-units comparison at one pressure: simulated length minus
    real digitized length, on the real curve's own force grid.

    Reported alongside the latent-space gap score because the two answer
    different questions, and only this one is in millimetres a person can
    reason about. The digitization noise floor is included so the residual can
    be read against it -- a residual below the floor is not evidence of
    anything.
    """
    curve = load_ogawa2017_fig4a_curve(pressure_mpa)
    simulated = simulate_length_at_force(curve.force_n, pressure_mpa, params)
    residual = simulated - curve.length_mm
    return {
        "pressure_mpa": pressure_mpa,
        "mean_residual_mm": float(residual.mean()),
        "mean_abs_residual_mm": float(np.abs(residual).mean()),
        "max_abs_residual_mm": float(np.abs(residual).max()),
        "digitization_noise_floor_mm": curve.digitization_noise_floor_mm,
        "residual_exceeds_noise_floor": bool(
            np.abs(residual).mean() > curve.digitization_noise_floor_mm
        ),
        "saturation": simulated_curve_is_saturated(curve.force_n, simulated),
    }
