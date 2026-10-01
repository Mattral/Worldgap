# PGM reference data (Ogawa et al. 2017 / Thakur et al. 2018)

Resolves the spec Section 14 / ROADMAP Phase 0 & 6 "Ogawa et al. access" item.
Both papers below were obtained directly and are the source of every number
on this page. Nothing here was read off a figure by eye and presented as
precise — anything that would require that (the full continuous
pressure-elongation curves) is explicitly called out as **not yet done** at
the bottom, rather than approximated and quietly presented as solid.

## Citations

1. Ogawa, K., Thakur, C., Ikeda, T., Tsuji, T., & Kurita, Y. (2017). Development
   of a pneumatic artificial muscle driven by low pressure and its application
   to the unplugged powered suit. *Advanced Robotics*, 31(21), 1135–1143.
   https://doi.org/10.1080/01691864.2017.1392345
2. Thakur, C., Ogawa, K., Tsuji, T., & Kurita, Y. (2018). Soft Wearable
   Augmented Walking Suit With Pneumatic Gel Muscles and Stance Phase
   Detection System to Assist Gait. *IEEE Robotics and Automation Letters*,
   3(4), 4257–4264. https://doi.org/10.1109/LRA.2018.2864355

## Why two papers, and why they aren't merged into one dataset

**Correction (this supersedes an earlier version of this document).** An
earlier version of this page claimed these papers describe "two different
physical PGM prototypes." That claim was wrong, and the papers say so
directly. Thakur et al. (2018) §II.A describes the actuator as the one "we
previously developed" citing Ogawa et al. (2017); reuses Ogawa's elongation
figure ("Fig. 2 shows the elongation ratio of the PGM **as measured by
[14]**"); cites Ogawa for the 50–300 kPa operating range; and motivates its
own experiment precisely because the stretched-length force behaviour "is
not measured in [14]." That is one actuator design measured two ways, not
two actuators.

The reason the numbers are still kept strictly separate is the
**measurement type**, which genuinely does differ:

| | Ogawa 2017 (Fig. 4a) | Thakur 2018 (Eq. 1–2) |
|---|---|---|
| Held constant | supply pressure (7 levels, 0–0.3 MPa in 0.05 MPa steps) | PGM length (unstretched, or stretched to 45 cm as used in the suit) |
| Swept | applied load, 0→5 kg→0 in 1 kg steps | supply pressure, 50–300 kPa |
| Measured | PGM length (elongation) | generated force |
| Result shape | `Length(Force)` family parameterized by pressure | `Force(pressure)` at a fixed length |

Different dependent variables, under different held-constant conditions.
Averaging or interpolating between them would silently invent a measurement
neither paper made — so `pgm_actuator.py` exposes them as two separate
entry points and never combines them.

### The prototype dimensions are reported inconsistently, and we don't paper over it

| Source | Statement |
|---|---|
| Ogawa §2.2 | inner tube **250 mm** (inner dia. 4 mm); outer braided mesh **500 mm** (outer dia. 9 mm); mesh dia. 23 mm default / 28 mm on contraction |
| Ogawa §2.3 | "a **natural length of 250 mm** and maximum possible elongation of **500 mm**" |
| Ogawa §4 (Discussion) | "had a fixed length of **300 mm** of the normal length" |
| Thakur §II.A | "resting length of **30 cm**, maximum contraction length of **25 cm**, maximum elongation length of **45 cm**" |

Ogawa's own §2.3 and §4 disagree with each other, and §4 agrees with Thakur.
A plausible reconciliation — flagged here as **our inference, not either
paper's claim** — is that Thakur's triple is the careful one (resting 300,
fully contracted 250, fully elongated 450 mm) and that §2.3 reported §2.2's
inner-tube length (250 mm) and braided-mesh length (500 mm) as if they were
the assembled muscle's natural and maximum lengths. Nothing in `worldgap`
depends on that reconciliation being right: `OGAWA_2017_PROTOTYPE` and
`THAKUR_2018_PROTOTYPE` each store what their own paper reported, and callers
pick a source explicitly.

### Scale caveat: this is walking-suit hardware, and the authors say so

Ogawa §4, in full relevance:

> The PGM developed in this paper could actuate with low air pressure but had
> a fixed length of 300 mm of the normal length. This length is suitable for
> applications such as assisting walking **but not for assisting the hand or
> wrist, which requires shorter muscles.** Future work should consider
> designing and developing PGMs of the required size and model them with a
> third parameter of the resting length of the PGM along with the air
> pressure and applied force.

So: every number on this page characterizes a walking-assist-scale actuator.
Using it as the "real" side of a V2 gap for a hand- or wrist-scale PGM is an
extrapolation across a length scale the source paper explicitly says does not
carry over. `pgm_actuator.OGAWA_2017_SCALE_CAVEAT` carries this string so it
can be surfaced in reports rather than living only in a doc nobody re-reads,
and `docs/v2_actuation_runbook.md` repeats it at the point of use. The
authors' own suggested fix — treat resting length as a third model parameter
alongside pressure and force — is the right shape for any future hand-scale
characterization, and is not something this reference data can substitute for.

**Second correction from an earlier version of this document**: it previously stated
Figure 4(a) shows "a genuine hysteresis loop." Having now actually digitized
that figure (see below), each pressure level renders as a single smooth
curve, not two visually separate loading/unloading branches — either the two
directions overlap too closely to resolve at the chart's resolution, or the
figure shows one direction only despite §2.3's text describing both being
recorded. What's actually usable from this figure is `Length(Force)` at each
of 7 fixed pressures — real, valuable data, but not literally the
loading/unloading loop `pgm_actuator.py`'s two-branch hysteresis fit expects.
See "Ogawa 2017 Figure 4(a)" below for how this is actually integrated.

Thakur's experiment measures something else entirely (force at constant
length vs. swept pressure) on the *same* actuator design, and is integrated
separately as a cross-check / simpler reference model. Averaging or
interpolating between Ogawa and Thakur's numbers would silently invent a
measurement neither paper made.

## Numbers directly usable without digitization

Everything below is either an explicit fitted equation the source paper
reports, or a number stated in the paper's own text/tables — not read off a
graph.

### Shared / consistent across both papers
- Operating pressure range: **0.05–0.3 MPa (50–300 kPa)**, consistent between
  both papers (Thakur 2018 §II.A explicitly cites this range "as reported by
  [Ogawa 2017]").

### Ogawa 2017 — qualitative hysteresis regime (§2.3)
- In the **0.05–0.15 MPa** range: muscle behavior is nonlinear, with a real
  loading/unloading gap (genuine hysteresis).
- In the **0.2–0.3 MPa** range: stretched length changes ~linearly with
  applied force — i.e. hysteresis is much less pronounced at higher pressure.
- This qualitatively confirms a two-branch (or pressure-range-dependent)
  hysteresis model is the right shape, ahead of getting the exact curve.

### Ogawa 2017 — Figure 6 / §2.4 comparison table (PGM vs. commercial PM-10RF, at 0.2 MPa)

| Force | PGM contraction | PGM elongation | PM-10RF contraction | PM-10RF elongation |
|---|---|---|---|---|
| 0 N | 36% | — | 32% | — |
| 10 N | 29% | 11% | 14% | 29% |
| 20 N | 23% | 20% | 3% | 41% |

(PM-10RF = Squse Co. Ltd.'s commercially available low-pressure PAM, the
comparison baseline Ogawa 2017 benchmarks against.)

### Thakur 2018 — fitted force-vs-pressure equations (§II.A, Eq. 1–2)

Force in Newtons, pressure `x` in kPa, valid over the reported range **50–300
kPa only** — the paper does not validate these fits outside that range, so
`pgm_actuator.py`'s implementation raises rather than extrapolates silently.

- Unstretched (resting length): `F = 0.1799x − 5.1983` (R² = 0.993)
- Stretched to 45 cm (as used in the actual AWS suit): `F = 0.3883x + 5.8899` (R² = 0.998)

Sanity check against the paper's own reported values: at 60 kPa the paper
reports ~30 N of assistive force; the stretched equation gives 29.2 N. At 100
kPa the paper reports ~44 N; the equation gives 44.7 N. Both within the
paper's own "approximately" rounding — consistent, not just plausible.

## Ogawa 2017 Figure 4(a) — digitized Length(Force) curves (real, not text-stated)

Digitized via pixel-color tracing with connected-component analysis (one
color mask per curve), not WebPlotDigitizer manual clicking. Bundled as
package data (ships in the wheel, not just the source checkout) at
`src/worldgap/data/reference_data/ogawa2017_fig4a/*.csv`, one CSV per
pressure level, with the digitization method noted in the accompanying
`DIGITIZATION_README.txt`. Loaded via
`worldgap.data.loaders.pgm_actuator.load_ogawa2017_fig4a_curve(pressure_mpa)`.

**What this actually is**: `Length(Force)` at each of the 7 pressure levels
Ogawa et al. (2017) tested (0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3 MPa) — a real
digitized reference surface, useful for validating a simulated actuator
model against real measurements. **What this is NOT**: a pressure-ramp
hysteresis loop. See the correction note above — use this for comparing a
simulated `Length(Force, Pressure)` response against real data, not as input
to `fit_hysteresis_curve()`.

**Digitization noise, quantified rather than hidden** (spec 12.14: digitization
uncertainty MUST be recorded and carried through as a noise floor): the raw
traced points aren't perfectly monotonic (which they physically must be —
this is a quasi-static loading test, elongation cannot decrease as force
increases). `load_ogawa2017_fig4a_curve()` applies isotonic regression
(`scipy.optimize.isotonic_regression`) to enforce that constraint, and
reports the correction magnitude as `digitization_noise_floor_mm`:

| Pressure | Noise floor (max correction) |
|---|---|
| 0 MPa | 0.64 mm |
| 0.05 MPa | 11.39 mm |
| 0.1 MPa | 6.34 mm |
| 0.15 MPa | 4.93 mm |
| 0.2 MPa | 3.05 mm |
| 0.25 MPa | 0.06 mm |
| 0.3 MPa | 0.44 mm |

The noisier curves (0.05/0.1/0.15 MPa) are exactly the ones that visually
cross and cluster tightly with their neighbors in the middle of the chart —
color-based tracing is more error-prone there. Any conclusion drawn from
this data that hinges on a difference smaller than the relevant pressure
level's noise floor isn't distinguishable from digitization artifact.

**What this number is and isn't.** `digitization_noise_floor_mm` is a
**heuristic proxy**, not a statistical confidence interval and not a
calibrated error bar. It reports one thing: how far the traced points had to
move to satisfy a constraint the physics requires. Trace error that happens
to preserve monotonicity is invisible to it, so it is a floor on
interpretability, not a ± to propagate arithmetically. Read it as "do not
interpret differences smaller than this," and never as "the true value is
within this much."

**Independent cross-validation** (this is the strongest evidence the
digitization is accurate, not just plausible-looking): Ogawa's Figure 6 /
§2.4 independently states contraction ratios at 0.2 MPa — 36%/29%/23% at
0/10/20 N. Converting the digitized 0.2 MPa curve as `(500 mm − Length) /
500 mm`, without having used Figure 6's numbers anywhere in the digitization
itself, gives 35.6%/29.9%/24.4% — within 0.4–1.4 percentage points of the
paper's own independently-stated numbers. Tested directly in
`test_digitized_0_2mpa_curve_matches_papers_independently_stated_contraction_ratios`.

**About that 500 mm — it is our inference, not the paper's formula.** §2.4
says Figure 6 shows "the contraction ratio relative to the natural length"
and never writes the arithmetic down. The literal reading of "natural
length" (250 mm per §2.3, or 300 mm per §4) does not reproduce Figure 6's
percentages at all. Taking the reference to be the 500 mm of §2.3's "maximum
possible elongation" (= §2.2's braided-mesh length) does, to within 1.4
percentage points. So `OGAWA_2017_MAX_ELONGATION_MM = 500.0` should be read
as *"the reference length that reproduces the paper's own published
numbers"*, not as a definition the paper states. The cross-check is still
strong evidence about the digitization — it is a test of the traced curve
against independently published percentages, and the fact that one
consistent reference length reconciles them across three separate force
levels is itself the evidence — but it is not evidence about what Ogawa
meant by "natural length."

## What's still NOT extracted (genuinely pending, not skipped)

- Ogawa 2017 Figure 5 (contraction ratio vs. air pressure at each of 9 fixed
  forces) — exists only as a graph, not tabulated numbers.
- Ogawa 2017 Figure 4(b) — the same underlying data as 4(a) shown as a 3D
  surface; redundant with 4(a) and much harder to digitize cleanly, so
  deliberately skipped.
- Thakur 2018 Figure 2 (elongation vs. air pressure and force, 3D surface)
  and Figure 4 (the raw force-profile measurements the two fitted equations
  above were derived from).
- A real pressure-ramp hysteresis characterization (commanded pressure
  increasing/decreasing, response lagging) — genuinely not present in either
  paper (see the correction note above). `fit_hysteresis_curve()` in
  `pgm_actuator.py` remains validated only against synthetic data
  (`test_pgm_hysteresis.py`) pending either real hardware logging (V3) or a
  first-principles pneumatic dynamics model.
