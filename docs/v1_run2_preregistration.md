# V1 run 2 — pre-registration

**Status: fixed.** This document was written and committed **before any run-2
recording exists**. Its git commit is its identity. Nothing below may be
changed after run-2 data is recorded; any departure during the run is
reported in the results as a deviation from this document, not edited in.

### Amendments

**Amendment 1 — 2026-10-08, made before any run-2 data existed.** The
original text is in git history (merge `c153902`). Amended passages are marked
*[A1]*.

- **Why:** the original lacked a power analysis for Rule B, and its §8.4 read
  every Rule B failure as "no claim of added value", including failures where
  the test simply had little chance to pass. Interpreting those afterwards
  would look like an excuse; fixing the reading now does not.
- **§8.4:** a Rule B failure with ρ_gap ≥ 0.80 now reads *"underpowered to
  settle it"*, not *"no added value"*. The original row was: "pass | fail |
  The gap score tracks degradation, but not demonstrably better than counting
  dropouts. No claim of added value."
- **§13:** adds a Rule B power table (with its assumptions, so it can be
  re-run) and a limitation: the 24 conditions are 4 factors × 6 ordered
  severities, not 24 exchangeable draws.
- **§12:** adds the testing protocol for the analysis script before run-2
  data exists (synthetic data, and run 1's footage in blind mode).

Run 1 (`v1_first_run_results.md`) was null and underpowered: the conditions
barely made MediaPipe fail, take-to-take noise was comparable to the
between-condition signal, and spec 5.2 normalization was missing. Run 2 is
designed from those findings and from a pilot (§11).

---

## 1. Question

Does the worldgap gap score, computed between clean webcam recordings and
graded degradations of them, **rank how badly MediaPipe hand tracking
degrades**, and does it do so **better than simply counting the frames where
the hand was lost**?

The second half matters because the world model sees the presence mask
(which landmarks are missing) as input. A gap score that only rediscovers
dropout would be an expensive dropout counter. Run 2 pre-registers a trivial
dropout baseline and requires the gap score to beat it.

## 2. Design overview

A hybrid design.

- **Primary (confirmatory): 24 software-degraded conditions** (§4.1). Fresh
  clean recordings are degraded frame by frame (`darken`, `downscale`,
  `blur`, `noise`, 6 severities each) before MediaPipe sees them. Every
  condition reuses the same frames, which removes take-to-take noise and gives
  every degraded frame a clean reference. **These are simulated degradations
  of real footage, not real deployment conditions**, and every result from
  them is labelled that way.
- **Secondary (descriptive only): 6 physical conditions** (§4.2), recorded
  for real. They have no paired clean frame, so only dropout is measurable
  for them. No decision rule applies to them.

## 3. Data collection

| | |
|---|---|
| Subject | One person (the author), as in run 1. Recording anyone else would be human-subjects data needing consent and ethics approval first; not part of this run |
| Camera | The same built-in webcam as run 1, its default resolution (640×480 in run 1), mounted and fixed except where a physical condition requires otherwise |
| Routine | Run 1's: `fist` → `palm` → `stop` → `like`, one gesturing hand, the other out of frame, prompted on screen, 2 s per gesture, repeated |
| Take length | **75 s** (≈2250 frames at 30 fps → 46 non-overlapping 48-frame windows per take) |
| Clean takes | **3** (`clean/take0..2`) |
| Physical takes | 3 per physical condition (§4.2), 18 in total |
| Tool | `scripts/record_v1_session.py`, which writes each file with the frame rate the camera actually delivered |
| Order | `clean` takes 0 and 1 first, the physical conditions, then `clean` take 2 last (drift check, as in run 1) |
| Run-1 data | **Not used.** Run 1's recordings are spent; run 2 uses only fresh recordings. |

**Recording exclusions.** A take may be discarded and re-recorded **only
during the recording session, before any run-2 processing**, and only for a
recording failure (camera fault, the routine interrupted, the wrong condition
set up). Each such re-take is logged with its reason. No take is excluded
after processing starts.

## 4. Conditions

### 4.1 Primary: 24 software-degraded conditions (fixed)

Applied with `worldgap.data.degradations.ImageDegradation(kind, severity,
seed=0)` to every frame of every clean take, before MediaPipe. The last three
columns are the **pilot's** values on run 1's footage (§11), recorded as
expectations only. They play no role in the analysis.

| # | Condition | Kind | Severity | Pilot dropout | Pilot landmark error | Pilot frames compared |
|---|---|---|---|---:|---:|---:|
| 1 | `darken_0.3` | darken | 0.3 | 0.0% | 0.010 | 1.00 |
| 2 | `darken_0.2` | darken | 0.2 | 0.0% | 0.016 | 1.00 |
| 3 | `darken_0.12` | darken | 0.12 | 0.2% | 0.025 | 1.00 |
| 4 | `darken_0.08` | darken | 0.08 | 3.6% | 0.033 | 0.96 |
| 5 | `darken_0.07` | darken | 0.07 | 14.2% | 0.039 | 0.86 |
| 6 | `darken_0.06` | darken | 0.06 | 35.8% | 0.042 | 0.64 |
| 7 | `downscale_192` | downscale | 192 | 0.0% | 0.009 | 1.00 |
| 8 | `downscale_128` | downscale | 128 | 0.0% | 0.016 | 1.00 |
| 9 | `downscale_96` | downscale | 96 | 0.0% | 0.024 | 1.00 |
| 10 | `downscale_64` | downscale | 64 | 0.0% | 0.035 | 1.00 |
| 11 | `downscale_48` | downscale | 48 | 0.0% | 0.048 | 1.00 |
| 12 | `downscale_32` | downscale | 32 | 3.9% | 0.086 | 0.96 |
| 13 | `blur_2` | blur | 2 | 0.0% | 0.009 | 1.00 |
| 14 | `blur_4` | blur | 4 | 0.0% | 0.020 | 1.00 |
| 15 | `blur_6` | blur | 6 | 0.0% | 0.029 | 1.00 |
| 16 | `blur_9` | blur | 9 | 1.0% | 0.052 | 0.99 |
| 17 | `blur_13` | blur | 13 | 19.0% | 0.064 | 0.81 |
| 18 | `blur_18` | blur | 18 | 42.6% | 0.086 | 0.57 |
| 19 | `noise_10` | noise | 10 | 0.0% | 0.005 | 1.00 |
| 20 | `noise_15` | noise | 15 | 0.0% | 0.007 | 1.00 |
| 21 | `noise_20` | noise | 20 | 0.0% | 0.009 | 1.00 |
| 22 | `noise_24` | noise | 24 | 0.8% | 0.011 | 0.99 |
| 23 | `noise_28` | noise | 28 | 13.7% | 0.013 | 0.86 |
| 24 | `noise_32` | noise | 32 | 44.3% | 0.017 | 0.56 |

Severities are `darken`: brightness multiplier; `downscale`: frame width in
pixels; `blur`: Gaussian σ in pixels; `noise`: Gaussian std in 0–255
intensity levels (seeded per frame).

**Handling rule (fixed).** All 24 conditions are processed, analysed and
reported, whatever happens on fresh footage. A condition that lands past its
expected dropout, even at 100%, is **reported, never dropped** and never
replaced by a different severity. If a condition's landmark error is
undefined because no frame had a hand on both sides, it is assigned the
**worst rank** on landmark error (total loss counts as worse than any partial
error; several such conditions tie at the worst rank).

### 4.2 Secondary: 6 physical conditions (descriptive only)

| Condition | Setup |
|---|---|
| `occlusion_25`, `occlusion_50`, `occlusion_75` | A card covering roughly a quarter / half / three quarters of the gesturing hand (by eye), held in place throughout |
| `distance_2m`, `distance_3m`, `distance_4m` | The subject at about 2, 3 or 4 m from the camera (clean takes are at arm's length) |

Reported per condition: gap score (Fréchet and MMD²), dropout, and the
trivial baseline. Landmark error is not defined (no paired frame), and **no
correlation or decision rule is computed for them**.

## 5. Processing

- Landmarks: `extract_rollout_from_video`, MediaPipe `holistic_landmarker.task`
  (float16; sha256 `e2dab61191e2dcd0…`), **VIDEO running mode, a fresh
  landmarker for every video**, every frame (`stride=1`).
- Degradations: the condition's `ImageDegradation` as `frame_transform`,
  recorded in each rollout's metadata.
- Normalization: **`shoulder_midpoint`**, fixed for the whole study
  (written to `study_settings.json`). Every rollout in the study uses it; the
  analyzer refuses any other.
- Windows: `split_into_windows(window_frames=48)`, non-overlapping, so 46 per
  75 s take and **138 per condition**.
- Per-(take, condition) results are cached, so an overnight run can be
  interrupted and resumed. Caching never changes a result: a cached rollout is
  the same extraction, stored.

## 6. Model

`GapConfig(modality="perception")` with:

| Setting | Value |
|---|---|
| encoder | d_model 128, 2 layers, 4 heads, feedforward 256 |
| world model | context 16 frames, predict 8 frames, **summary_dim 16** |
| training | 30 epochs, batch 16, **seed 0**; other settings at `GapConfig` defaults |
| training data | the **138 clean windows only** (3 clean takes) |

**`summary_dim` = 16, chosen deliberately** (run 1 used 32). The confidence
rule (spec 7.3) is "low" below 5 × summary_dim windows per domain and
"medium" below 10 ×. With 138 windows per condition, 32 guarantees "low"
(138 < 160); 16 gives "medium" (80 ≤ 138 < 160). Windows cut from one
recording are **not independent samples**; the effective sample size per
condition is nearer 3 recordings than 138 windows, and every confidence flag
is reported with that caveat.

**Collapse rule.** If the collapse safeguard fires (spec 12.7), no gap score
from that model is reported. The training failure is fixed and the run is
repeated under this same document, and the failed attempt is reported.

## 7. Measures

### 7.1 Gap score (the thing being tested)

For each condition: **Fréchet distance** (`GapAnalyzer.compute_gap`) between
the 138 clean windows (source) and the condition's 138 windows (target),
using the one model of §6. MMD² is reported alongside as a secondary score.

### 7.2 Primary ground truth: paired landmark error

For each condition and each clean take *k*: `paired_landmark_error(clean_k,
degraded_k)` over the full take (all frames, not windows). For each frame and
each hand detected in the degraded version, it measures the distance to the
nearest hand detected in the clean version of **the same frame**: the mean
over the 21 landmarks of the image-plane (x, y) distance, divided by that
clean hand's bounding-box diagonal. A left/right label swap is not counted.

**Frames where the degraded condition lost the hand are excluded from this
metric** (as are frames where the clean take had no hand). Landmark error
therefore measures **tracking quality given detection**; *availability* is
carried separately by dropout (§7.3). This is fixed: it changes the number,
and it means landmark error at high-dropout severities is computed on the
frames MediaPipe still tracked, which are likely the easier ones.

**Per-condition value:** the pooled mean over all compared (frame, hand) pairs
of the 3 takes, i.e. Σ(meanₖ × nₖ) / Σ nₖ. Also reported: the number of
comparisons and the fraction of frames compared. Undefined values follow the
§4.1 handling rule.

This replaces dropout as the primary ground truth: a **documented deviation
from spec 8.1**. Reason: with dropout as the ground truth, the trivial
dropout baseline *is* the ground truth (ρ = 1), so it could never be beaten.
And in the pilot, dropout was 0% in 13 of these 24 conditions (ties), while
landmark error was graded throughout. Landmark error is computed from
MediaPipe's own output and never from the world model, so spec 8.1's
independence requirement still holds.

### 7.3 Secondary ground truth: dropout

Per condition: the mean over the 3 takes of `hand_dropout_rate` (fraction of
frames with no hand), plus `longest_dropout_run_s`.

### 7.4 Trivial baselines

- **Primary baseline: `no_hand_fraction`**, the condition's mean
  `hand_dropout_rate` over its 3 degraded takes (higher = worse).
- Secondary baseline: `1 − presence_density`, one minus the mean of the
  presence mask (higher = worse).

They are computed from the same degraded rollouts the gap score sees, and
need no model.

## 8. Analysis

All correlations are Spearman across the **24 primary conditions**. All
confidence intervals use a percentile bootstrap over conditions (resampled
with replacement): **10,000 resamples, seed 0**, 95%. A resample is discarded
if either correlation in it is undefined (constant input), and the count of
valid resamples is reported.

### 8.1 Primary estimand

ρ_gap = Spearman ρ(Fréchet gap score, landmark error), with its bootstrap CI.

### 8.2 Decision rules (the only confirmatory tests)

- **Rule A — does the score track degradation?** Passes if the 95% CI of
  ρ_gap has its **lower bound > 0**.
- **Rule B — is the tool worth more than counting dropouts?** Let ρ_base =
  Spearman ρ(`no_hand_fraction`, landmark error) and Δ = ρ_gap − ρ_base.
  The CI of Δ comes from a **paired** bootstrap: the same resampled conditions
  for both correlations in every resample. Rule B passes if the 95% CI of Δ
  has its **lower bound > 0**.

For reference, in the pilot ρ_base was **+0.56** over these 24 conditions. With 24
conditions, passing Rule B is expected to need ρ_gap well above that. This is
stated in advance so a near-miss is not later read as a success. *[A1]* The
estimated power is in §13: about 17–51% at ρ_gap = 0.80 and 54–98% at 0.90.

### 8.3 Secondary analyses (descriptive; no decision attached)

1. ρ(gap, dropout) and ρ(`no_hand_fraction`, dropout), with CIs.
2. **Present-frames-only gap:** the gap recomputed using only windows in
   which every frame has at least one hand (in source and target alike),
   correlated with landmark error. This shows whether the result survives when
   the gap score cannot see missing hands. If a condition has fewer than 80
   such windows (5 × summary_dim), its value is still reported, flagged.
3. The `1 − presence_density` baseline against landmark error.
4. ρ(MMD², landmark error).
5. Per-factor rank agreement (within each of the 4 factors), descriptive.
6. **Practical note, not a decision:** whether ρ_gap ≥ 0.6.
7. Physical conditions (§4.2): table of gap scores, dropout and baseline.
8. Confidence flag per condition, with the non-independence caveat.

### 8.4 Interpretation, fixed in advance

| Rule A | Rule B | What will be said |
|---|---|---|
| pass | pass | On one subject and one camera, across 24 pre-registered software-degraded conditions, the gap score ranked MediaPipe landmark degradation, and did so better than counting dropouts. Not "validated" in general. |
| pass | fail, **ρ_gap ≥ 0.80** *[A1]* | The gap score tracks degradation. Whether it adds value over counting dropouts is **not settled: the test was underpowered** to detect an advantage (§13, power ≈ 17–51% at ρ_gap = 0.80). This is **not** evidence of no added value. |
| pass | fail, **ρ_gap < 0.80** *[A1]* | The gap score tracks degradation, but not demonstrably better than counting dropouts. No claim of added value. |
| fail | — | The gap score did not track landmark degradation across these conditions. Reported as a negative result. |

*[A1]* ρ_gap here is the point estimate of §8.1. The 0.80 threshold is fixed
now, before data, from the §13 power table; it is not moved afterwards.

None of these outcomes is generalised beyond one person, one camera and
simulated degradations of real footage.

## 9. What would void the run

- The collapse safeguard firing (§6), handled as stated there.
- Any change to §3–§8 after recording begins. If the analysis script turns out
  to contain a bug, it is fixed, the fix is documented, and the analysis
  re-run. Every run of the analysis is reported, not just the last.

## 10. Deviations from the technical spec, fixed here

- **Spec 8.1:** primary ground truth is paired landmark error, not dropout
  (§7.2); dropout is secondary.
- **Spec 5.2:** shoulder-midpoint anchor (already a documented deviation;
  hips were visible at 0.005 vs shoulders 0.999 in run 1).
- **Spec 8.3:** conditions are 4 software-degradation factors × 6
  severities, not (lighting, occlusion, motion) combinations; physical
  occlusion and distance are secondary and descriptive.

## 11. Pilot (labelled pilot; not part of run 2's data)

Run with `scripts/run_v1_pilot.py` on the first 900 frames (30 s) of run 1's
`clean/take0.mp4`, footage that is not part of run 2. The pilot computed
**no gap scores and no gap correlations**, so severities were chosen without
seeing the tool's performance. 35 passes: 4 factors × 7 severities, plus 7 to
refine the two cliffs.

What it established:

- Dropout is a cliff, not a ramp, for `darken` (0.2% at ×0.12, 3.6% at
  ×0.08, 70% at ×0.055, 86% at ×0.05) and `noise` (0.8% at std 24, 44% at
  32, 100% from 35). `downscale` barely causes dropout (3.9% at 32 px).
  `blur` ramps (1%, 19%, 43% at σ 9, 13, 18).
- Landmark error rises smoothly with every factor before the cliffs.
- The 24 severities in §4.1 stop just short of each cliff: at least 56% of
  frames compared in every condition.
- Over those 24, the dropout baseline already ranks landmark error at
  ρ = +0.56 (presence density, |ρ| = 0.48). That is the bar for Rule B.

Processing speed varied about fivefold between pilot passes (≈1 min vs ≈5 min
per 30 s clip), possibly from repeated re-detection near the cliffs or other
load on the machine; run 2 is therefore cached and run overnight.

## 12. Reproducibility

- Software at pre-registration: worldgap `main` @ `ff54d98` (unreleased,
  after 0.2.0), mediapipe 1.0.1, OpenCV 5.0.0, torch 2.14 (CPU), Python
  3.14.
- The run-2 analysis script, which implements exactly §5–§8 including caching,
  is committed **before any run-2 recording is processed**. Its commit hash and
  `study_settings.json` are recorded with the results.
- *[A1]* **Testing before run-2 data exists.** The analysis script
  (`scripts/run_v1_run2.py`) is tested end to end in CI on synthetic videos,
  and on run 1's footage in **blind mode** (`--blind`). Blind mode runs every
  stage (extraction, caching, fitting, gap scores, both rules, secondary
  analyses) but writes only a health report and discards every score and
  correlation unread. A non-blind run on real footage, which would preview the
  result for these exact conditions, is not made before run 2.
- Recordings stay on the recording machine (gitignored; spec 12.16). Results
  are published as numbers only.

## 13. Known limitations, stated in advance

- One subject, one camera, one session. Any result says nothing about other
  people, cameras or devices.
- The primary conditions are simulated degradations of real footage.
  Real-world conditions are represented only by the descriptive physical
  conditions.
- Windows within a recording are not independent; confidence flags are
  optimistic.
- Landmark error is quality given detection (§7.2), so it is likely
  understated at high-dropout severities.
- The 4 factors put landmark error on different scales (noise peaks near
  0.017, blur near 0.086). Spearman ranks across factors, so a score that
  orders *factors* differently from MediaPipe will be penalised even if it
  orders severities within each factor correctly. §8.3 item 5 reports this.
- 24 conditions limit the precision of every correlation, and of Δ most of
  all.
- *[A1]* **The 24 conditions are not exchangeable draws.** They are 4 factors
  × 6 ordered severities. The bootstrap resamples them as if they were
  exchangeable, ignoring that severities within a factor are related, so every
  CI is **narrower than the dependence structure warrants**, Δ's most of all.
  A Rule A or B pass should be read with that in mind.

### Rule B power *[A1]* (an estimate)

Probability that Rule B passes (95% paired-bootstrap CI of Δ above 0), for
n = 24 and ρ_base = 0.56, by the true ρ_gap and the unknown correlation
between the gap score and the dropout baseline:

| ρ_gap | r(gap, base) = 0.2 | 0.4 | 0.6 | 0.8 | range |
|---|---:|---:|---:|---:|---:|
| 0.70 | 0.08 | 0.11 | 0.10 | 0.21 | 0.08–0.21 |
| 0.80 | 0.17 | 0.25 | 0.33 | 0.51 | 0.17–0.51 |
| 0.85 | 0.34 | 0.41 | 0.50 | 0.76 | 0.34–0.76 |
| 0.90 | 0.54 | 0.57 | 0.77 | 0.98 | 0.54–0.98 |
| 0.95 | infeasible | 0.88 | 0.97 | infeasible | 0.88–0.97 |

Assumptions, so the numbers can be reproduced with
`python scripts/power_rule_b.py --sims 300` (seed 0):

- The 24 conditions are exchangeable draws from a **Gaussian copula** over
  (landmark error, gap score, baseline). Spearman targets are converted to
  latent Pearson correlations as r = 2·sin(π·ρ/6).
- **300 simulated studies per cell**, each analysed exactly as pre-registered
  (paired percentile bootstrap, 10,000 resamples, 95%). The Monte Carlo
  error is about ±3 percentage points per cell.
- Scores are continuous, with no ties. Real dropout ties heavily at 0%, so the
  baseline will have coarser resolution than modelled.
- Cells whose correlation matrix is not positive definite are infeasible.
- The real conditions violate the exchangeability assumption (see the
  limitation above).

An independent simulation by the author under the same copula model gave
similar ranges (0.70: 9–16%; 0.80: 29–47%; 0.85: 48–70%; 0.90: 75–86%; 0.95:
94–98%). The differences come from the assumed range of r(gap, base) and from
Monte Carlo error.
