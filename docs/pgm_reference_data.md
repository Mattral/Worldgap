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

These describe **two different physical PGM prototypes**, not two
measurements of the same unit:

| | Ogawa 2017 | Thakur 2018 |
|---|---|---|
| Natural/resting length | 250 mm | 300 mm |
| Inner tube diameter | 4 mm | not reported |
| Max contraction length | not reported as a fixed value | 250 mm |
| Max elongation length | 500 mm (200% of natural length) | 450 mm (150% of resting length) |
| Braided mesh default / max diameter | 23 mm / 28 mm | not reported |
| What was measured | Elongation (length) response to a **hung load** (0–5 kg in 1 kg steps), at each of several fixed supply pressures (0–0.3 MPa in 0.05 MPa steps), recorded both loading (load added) and unloading (load removed) — i.e. a genuine hysteresis loop | Force output at a **fixed PGM length** (either unstretched/resting, or stretched to 45 cm as used in the actual walking suit), across a swept supply pressure (0.05–0.3 MPa) |

Ogawa's experiment is the one that actually produces a loading/unloading
elongation curve — the same physical quantity `pgm_actuator.py`'s two-branch
hysteresis fit models. Thakur's experiment measures something else (force at
constant length vs. pressure) on different hardware, and is integrated
separately as a cross-check / simpler reference model, not folded into the
same curve. Averaging or interpolating between the two would silently imply
they're the same actuator; they aren't reported as such in either paper.

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

## What's still NOT extracted (genuinely pending, not skipped)

- Ogawa 2017 Figures 4 and 5 (full elongation vs. force curves at each of 7
  pressure levels, and contraction ratio vs. air pressure at each of 9 fixed
  forces) — these exist only as graphs in the PDF, not as tabulated numbers.
  Getting a real, citable digitized version of these needs WebPlotDigitizer
  or equivalent, not an eyeballed reading of the page image.
- Thakur 2018 Figure 2 (elongation vs. air pressure and force, 3D surface)
  and Figure 4 (the raw force-profile measurements the two fitted equations
  above were derived from) — same caveat.
- Once digitized, `fit_hysteresis_curve()` in `pgm_actuator.py` should be run
  against the real Ogawa Fig. 4/5 data (matching physical quantity: elongation
  response to hung load, loading vs. unloading), not the current
  synthetic test fixture, which exists only to validate the fitting code
  itself and is explicitly documented as synthetic in `test_pgm_hysteresis.py`.
