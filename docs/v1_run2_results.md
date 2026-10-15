# V1 run 2 — results (2026-10-15)

The second real run of worldgap's V1 pipeline, analysed exactly as
pre-registered in [`v1_run2_preregistration.md`](v1_run2_preregistration.md)
(with Amendments 1 and 2). That document fixed the conditions, measures,
decision rules and interpretation before any run-2 recording existed. This
page reports what came out, with every analysis the pre-registration asks for.
Departures from it are listed under [Deviations](#deviations-from-the-pre-registration).

The 24 primary conditions are **simulated degradations of real footage**, not
real deployment conditions. Every number below is from one subject (the
author), one camera and one recording session.

## Headline

**Rule A passed; Rule B failed. Both analyses of Amendment 2 agree.**

| Analysis | Conditions | ρ_gap (95% CI) | Rule A | ρ_base | Δ = ρ_gap − ρ_base (95% CI) | Rule B |
|---|---:|---|---|---:|---|---|
| (i) All conditions, undefined landmark error ranked worst | 24 | **+0.611** [+0.247, +0.845] | **pass** | +0.654 | **−0.043** [−0.306, +0.220] | **fail** |
| (ii) Measured conditions only | 22 | **+0.495** [+0.090, +0.777] | **pass** | +0.551 | **−0.055** [−0.390, +0.278] | **fail** |

The two analyses carry equal weight, and neither is the headline over the
other. They agree on both rules, so a single verdict applies. ρ_gap = 0.611 is
below 0.80, so the §8.4 row that applies is the one fixed in advance:

> The gap score tracks degradation, but not demonstrably better than counting
> dropouts. **No claim of added value.**

The underpowered reading of Amendment 1 applies only at ρ_gap ≥ 0.80, so it
does not apply here. In both analyses the point estimate of Δ is negative: the
dropout count ranked landmark error slightly better than the gap score did.

Bootstrap: 10,000 percentile resamples over conditions, seed 0, paired for Δ.
All 10,000 were valid for every correlation reported here. As §13 states in
advance, the 24 conditions are 4 factors × 6 ordered severities, not
exchangeable draws, so every CI here is narrower than the data's dependence
structure warrants.

## What was run

| | |
|---|---|
| Subject | One person (the author), one hand gesturing, the other out of frame |
| Recording | One session on 2026-10-15 with `scripts/record_v1_session.py --plan run2`, in the pre-registered order: `clean` takes 0 and 1, the physical conditions, `clean` take 2 last |
| Takes | **15 of the planned 21**: 3 `clean`, and 3 each of `occlusion_25`, `occlusion_50`, `occlusion_75` and `distance_2m`. `distance_3m` and `distance_4m` were not recorded (see Deviations) |
| Frame rate | The camera **delivered 29.99–30.02 fps** in every take, measured while recording. Every file was written at 30.0 fps with 2250–2252 frames per 75 s take. No take needed a rate correction |
| Re-takes | None. No take was discarded or re-recorded |
| Pipeline | `main` @ `69a4df7` (clean working tree), `scripts/run_v1_run2.py` sha256 `0701918b…`, unchanged since the blind test |
| Software | worldgap 0.2.0 + unreleased `main`, mediapipe 1.0.1 (`holistic_landmarker.task`, sha256 `e2dab611…`), Python 3.14 |
| Settings | `shoulder_midpoint` normalization, 48-frame windows, summary_dim 16, 30 epochs, seed 0, 10,000 bootstrap resamples. No smoke flag (`smoke_mode: false`) |
| Extraction passes | 87: 3 clean + 24 conditions × 3 takes + 12 physical takes |

Recordings and run outputs stay on the author's machine (`recordings_run2/`
and `v1_run2/` are gitignored; spec 12.16). Only numbers are published.

## Model fit (§6)

**Collapse safeguard: did not fire.** Final loss 0.0214 after 270 steps on the
138 clean windows (3 takes × 46); no rollout skipped; `collapsed=False`.

## Results per condition

Gap scores are the 138 clean windows (source) against each condition's 138
windows (target). The confirmatory tests use the **Fréchet** column, ranked
against **landmark error**. Landmark error is the paired, image-plane hand
landmark error given detection (§7.2), pooled over the 3 takes. Dropout is
the mean over the takes of the fraction of frames with no hand. The
present-only gap uses only windows in which every frame has a hand (§8.3
item 2); "flag" marks fewer than 80 such windows.

| # | Condition | Fréchet | MMD² | Landmark error | Frames compared | Dropout | Longest dropout (s) | 1 − presence | Present windows | Present-only Fréchet |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `darken_0.3` | 0.001282 | −0.005667 | 0.0107 | 0.98 | 1.8% | 0.48 | 0.249 | 127 | 0.001445 |
| 2 | `darken_0.2` | 0.001410 | −0.004548 | 0.0164 | 0.98 | 1.7% | 0.48 | 0.248 | 122 | 0.000948 |
| 3 | `darken_0.12` | 0.001368 | −0.005248 | 0.0246 | 0.96 | 4.2% | 0.81 | 0.254 | 95 | 0.029382 |
| 4 | `darken_0.08` | 0.024283 | 0.007867 | 0.0334 | 0.84 | 16.0% | 1.59 | 0.283 | 57 | 0.304942 (flag) |
| 5 | `darken_0.07` | 2.866289 | 0.220512 | 0.0356 | 0.29 | 71.3% | 42.24 | 0.695 | 11 | 4.059711 (flag) |
| 6 | `darken_0.06` | 5.311813 | 0.613167 | **undefined** | 0.00 | 100.0% | 75.00 | 1.000 | 0 | not computable |
| 7 | `downscale_192` | 0.007558 | 0.009625 | 0.0120 | 1.00 | 0.2% | 0.09 | 0.245 | 136 | 0.007647 |
| 8 | `downscale_128` | 0.008817 | 0.013014 | 0.0200 | 1.00 | 0.3% | 0.16 | 0.245 | 135 | 0.007937 |
| 9 | `downscale_96` | 0.009139 | 0.013565 | 0.0285 | 1.00 | 0.3% | 0.06 | 0.245 | 134 | 0.007614 |
| 10 | `downscale_64` | 0.006667 | 0.005987 | 0.0418 | 0.99 | 1.1% | 0.27 | 0.247 | 131 | 0.004668 |
| 11 | `downscale_48` | 0.006336 | 0.004808 | 0.0564 | 0.98 | 1.7% | 0.26 | 0.248 | 125 | 0.005539 |
| 12 | `downscale_32` | 0.026646 | 0.021826 | 0.0911 | 0.93 | 7.1% | 0.84 | 0.259 | 72 | 0.029515 (flag) |
| 13 | `blur_2` | 0.009325 | 0.014415 | 0.0101 | 1.00 | 0.2% | 0.12 | 0.245 | 136 | 0.009337 |
| 14 | `blur_4` | 0.010297 | 0.017541 | 0.0222 | 0.99 | 0.6% | 0.14 | 0.246 | 131 | 0.007282 |
| 15 | `blur_6` | 0.007565 | 0.009054 | 0.0312 | 0.98 | 1.6% | 0.40 | 0.248 | 129 | 0.005713 |
| 16 | `blur_9` | 0.014113 | 0.015155 | 0.0493 | 0.97 | 2.8% | 0.66 | 0.250 | 104 | 0.012585 |
| 17 | `blur_13` | 0.033896 | 0.017019 | 0.0743 | 0.85 | 15.4% | 1.99 | 0.286 | 57 | 0.040712 (flag) |
| 18 | `blur_18` | 0.134537 | 0.040245 | 0.0776 | 0.49 | 51.0% | 4.31 | 0.381 | 8 | 0.312119 (flag) |
| 19 | `noise_10` | 0.001454 | −0.003803 | 0.0058 | 0.99 | 0.7% | 0.14 | 0.246 | 131 | 0.005000 |
| 20 | `noise_15` | 0.003345 | −0.001481 | 0.0078 | 0.99 | 1.1% | 0.16 | 0.247 | 131 | 0.006352 |
| 21 | `noise_20` | 0.007235 | 0.001175 | 0.0109 | 0.99 | 1.5% | 0.28 | 0.248 | 128 | 0.011704 |
| 22 | `noise_24` | 0.020220 | 0.004713 | 0.0132 | 0.97 | 2.7% | 0.76 | 0.256 | 115 | 0.032902 |
| 23 | `noise_28` | 0.153248 | 0.006915 | 0.0158 | 0.90 | 9.9% | 6.16 | 0.310 | 105 | 0.069172 |
| 24 | `noise_32` | 5.089919 | 0.604824 | **undefined** | 0.00 | 100.0% | 75.00 | 1.000 | 0 | not computable |

**Undefined landmark error: `darken_0.06` and `noise_32`.** MediaPipe found no
hand in any frame of any of their 3 takes. Per §4.1 they are reported, not
dropped. In analysis (i) they tie at the worst rank, and analysis (ii) leaves
them out. They are also the two conditions with the largest gap scores. That
is the partly definitional agreement Amendment 2's sensitivity analysis was
written for, and analysis (ii) shows both rules come out the same without them.

Negative MMD² values (conditions 1–3 and 19–20) are the unbiased estimator
reporting that no difference is detectable at this sample size, not an error.

**Against the pilot's expectations (§4.1, run 1's footage):** two conditions
landed well past their expected dropout. `darken_0.06` reached 100% against
an expected 35.8%, and `noise_32` reached 100% against 44.3%.
`darken_0.07` (71.3% vs 14.2%) and `darken_0.08` (16.0% vs 3.6%) were also
higher. The noise and darken cliffs sat at milder severities on this footage
than on the pilot's.

## Secondary analyses (§8.3; descriptive, no decision attached)

Each value below is given for (i) all 24 conditions and (ii) the 22 measured
ones, where the analysis applies to both.

| # | Analysis | (i) 24 conditions | (ii) 22 measured |
|---|---|---|---|
| 1 | ρ(gap, dropout) | +0.619 [+0.160, +0.865] | — |
| 1 | ρ(`no_hand_fraction`, dropout) | +1.000 (identical by construction: `no_hand_fraction` *is* the mean dropout) | — |
| 2 | ρ(present-frames-only gap, landmark error) | — | **+0.383 [−0.048, +0.697]**, 22 computable conditions; 5 of them flagged (< 80 present windows) |
| 3 | ρ(1 − presence density, landmark error) | +0.625 [+0.288, +0.828] | +0.513 [+0.155, +0.743] |
| 4 | ρ(MMD², landmark error) | +0.687 [+0.349, +0.880] | +0.593 [+0.203, +0.824] |
| 6 | ρ_gap ≥ 0.6 (practical note, not a decision) | yes (0.611) | no (0.495) |

The present-frames-only gap (item 2) was not computable, because fewer than 3
windows had a hand in every frame, for exactly the two conditions with
undefined landmark error. So its correlation covers the same 22 conditions as
analysis (ii).

**Item 5, per-factor rank agreement** (Spearman ρ of Fréchet against landmark
error, within each factor's 6 severities; undefined ranked worst):

| darken | downscale | blur | noise |
|---:|---:|---:|---:|
| +0.943 | **+0.086** | +0.829 | +1.000 |

The darken and noise values each include one condition ranked worst by
convention.

**Item 7, physical conditions** (descriptive only; no paired frame, so no
landmark error and no correlation):

| Condition | Fréchet | MMD² | Dropout (mean) | Per take | Longest dropout per take (s) |
|---|---:|---:|---:|---|---|
| `occlusion_25` | 0.141929 | 0.078020 | 44.5% | 29.6 / 68.8 / 35.0% | 1.97 / 4.90 / 1.20 |
| `occlusion_50` | 0.535723 | 0.169690 | 63.9% | 63.3 / 31.7 / 96.8% | 5.00 / 3.40 / 18.43 |
| `occlusion_75` | 0.808476 | 0.224693 | 71.9% | 69.9 / 56.6 / 89.4% | 7.37 / 4.17 / 10.23 |
| `distance_2m` | 0.511127 | 0.124047 | 0.0% | 0.0 / 0.0 / 0.0% | 0 / 0 / 0 |
| `distance_3m` | — | — | — | not recorded | — |
| `distance_4m` | — | — | — | not recorded | — |

The trivial baseline (`no_hand_fraction`) for a physical condition is its
dropout column. `distance_2m` lost no hand in any frame, yet it has the
third-largest physical gap. Its take 2 also had presence density 1.00, against
0.76 in takes 0 and 1.

**Item 8, confidence flags:** `medium` for every condition, primary and
physical (138 windows each; 80 ≤ 138 < 160 at summary_dim 16, as §6 planned).
The flag is optimistic: windows cut from one recording are not independent,
and the effective sample size per condition is nearer 3 recordings than 138
windows. In the present-frames-only analysis the 5 flagged conditions were
`low`.

## Per-take breakdown (§8.3 item 9)

Each cell is dropout / landmark error / fraction of frames compared /
presence density, for that take of the degraded condition.

| Condition | Take 0 | Take 1 | Take 2 |
|---|---|---|---|
| `darken_0.3` | 3.9% / 0.0090 / 0.96 / 0.75 | 0.0% / 0.0115 / 1.00 / 0.76 | 1.4% / 0.0115 / 0.99 / 0.75 |
| `darken_0.2` | 4.2% / 0.0149 / 0.96 / 0.75 | 0.4% / 0.0184 / 1.00 / 0.75 | 0.4% / 0.0160 / 1.00 / 0.75 |
| `darken_0.12` | 8.7% / 0.0230 / 0.91 / 0.73 | 1.4% / 0.0280 / 0.99 / 0.75 | 2.4% / 0.0227 / 0.98 / 0.75 |
| `darken_0.08` | 18.9% / 0.0339 / 0.81 / 0.71 | 6.0% / 0.0361 / 0.94 / 0.74 | 23.2% / 0.0297 / 0.77 / 0.70 |
| `darken_0.07` | 73.2% / 0.0424 / 0.27 / 0.26 | **100.0% / undefined / 0.00 / 0.00** | 40.5% / 0.0325 / 0.59 / 0.66 |
| `darken_0.06` | 100.0% / undefined / 0.00 / 0.00 | 100.0% / undefined / 0.00 / 0.00 | 100.0% / undefined / 0.00 / 0.00 |
| `downscale_192` | 0.5% / 0.0093 / 0.99 / 0.75 | 0.0% / 0.0132 / 1.00 / 0.76 | 0.0% / 0.0135 / 1.00 / 0.76 |
| `downscale_128` | 0.8% / 0.0157 / 0.99 / 0.75 | 0.0% / 0.0244 / 1.00 / 0.76 | 0.0% / 0.0199 / 1.00 / 0.76 |
| `downscale_96` | 0.8% / 0.0226 / 0.99 / 0.75 | 0.0% / 0.0344 / 1.00 / 0.76 | 0.0% / 0.0284 / 1.00 / 0.76 |
| `downscale_64` | 3.3% / 0.0365 / 0.97 / 0.75 | 0.0% / 0.0517 / 1.00 / 0.76 | 0.0% / 0.0369 / 1.00 / 0.76 |
| `downscale_48` | 4.8% / 0.0500 / 0.95 / 0.74 | 0.2% / 0.0705 / 1.00 / 0.76 | 0.0% / 0.0484 / 1.00 / 0.76 |
| `downscale_32` | 9.7% / 0.0906 / 0.90 / 0.73 | 7.9% / 0.1100 / 0.92 / 0.74 | 3.6% / 0.0742 / 0.96 / 0.76 |
| `blur_2` | 0.7% / 0.0078 / 0.99 / 0.75 | 0.0% / 0.0103 / 1.00 / 0.76 | 0.0% / 0.0121 / 1.00 / 0.76 |
| `blur_4` | 1.8% / 0.0187 / 0.98 / 0.75 | 0.0% / 0.0251 / 1.00 / 0.76 | 0.0% / 0.0227 / 1.00 / 0.76 |
| `blur_6` | 4.6% / 0.0292 / 0.95 / 0.74 | 0.0% / 0.0363 / 1.00 / 0.76 | 0.1% / 0.0281 / 1.00 / 0.76 |
| `blur_9` | 6.3% / 0.0490 / 0.94 / 0.74 | 1.8% / 0.0591 / 0.98 / 0.75 | 0.4% / 0.0399 / 1.00 / 0.76 |
| `blur_13` | 17.3% / 0.0693 / 0.83 / 0.71 | 18.1% / 0.0903 / 0.82 / 0.71 | 10.9% / 0.0645 / 0.89 / 0.72 |
| `blur_18` | 54.2% / 0.0695 / 0.46 / 0.62 | 68.8% / 0.0798 / 0.31 / 0.55 | 30.0% / 0.0818 / 0.70 / 0.68 |
| `noise_10` | 2.0% / 0.0048 / 0.98 / 0.75 | 0.0% / 0.0052 / 1.00 / 0.76 | 0.1% / 0.0074 / 1.00 / 0.76 |
| `noise_15` | 2.3% / 0.0070 / 0.98 / 0.75 | 0.0% / 0.0080 / 1.00 / 0.76 | 1.2% / 0.0084 / 0.99 / 0.75 |
| `noise_20` | 2.8% / 0.0088 / 0.97 / 0.75 | 0.0% / 0.0135 / 1.00 / 0.76 | 1.6% / 0.0102 / 0.98 / 0.75 |
| `noise_24` | 4.4% / 0.0104 / 0.96 / 0.74 | 0.2% / 0.0164 / 1.00 / 0.76 | 3.4% / 0.0126 / 0.97 / 0.74 |
| `noise_28` | 6.0% / 0.0131 / 0.94 / 0.73 | 4.4% / 0.0218 / 0.96 / 0.72 | 19.3% / 0.0119 / 0.81 / 0.61 |
| `noise_32` | 100.0% / undefined / 0.00 / 0.00 | 100.0% / undefined / 0.00 / 0.00 | 100.0% / undefined / 0.00 / 0.00 |

What the breakdown shows:

- **All-or-nothing behaviour appeared once.** In `darken_0.07`, MediaPipe
  never found the hand in take 1 but tracked 27% and 59% of frames in takes 0
  and 2. The blind test on run 1's footage had found this under moderate
  noise; here `noise_20` and `noise_24` tracked steadily in all three takes,
  and `noise_32` failed completely in all three.
- **Take 0 shows more dropout at mild severities** than takes 1 and 2 in most
  factors, for example `blur_6` (4.6% vs 0.0% and 0.1%) and `downscale_64`
  (3.3% vs 0.0% and 0.0%). Take 2, recorded last as the drift check, mostly
  sits with take 1. The exceptions are `darken_0.08` and `noise_28`, where
  take 2 had the most dropout.

## Deviations from the pre-registration

1. **`distance_3m` and `distance_4m` were not recorded.** They could not be
   set up during the session, and their 6 takes were skipped in the recording
   tool. The choice was then between recording them in a second sitting
   before any processing, or dropping them. They were dropped, and processing
   began with 15 of the 21 planned takes. Under §3 no take can be added after
   processing starts, so they will not be recorded for run 2. **Effect:** the
   secondary physical set has 4 of its 6 conditions (12 of 18 takes). The
   primary analysis and both decision rules are unaffected: they use only the
   3 `clean` takes, all recorded in order.
2. **The analysis ran in two parts.** The first invocation stopped during
   extraction pass 79 of 87, when the process hosting it ended, before any
   model was fitted or any score computed. The same command was rerun.
   Passes 1–78 loaded from the §5 cache, and passes 79–87, the fit and the
   analysis ran fresh. Before the rerun, mediapipe (1.0.1), the model file and
   the repository were checked unchanged, since the cache key does not cover
   the first two. A cached rollout is the stored result of the same
   extraction, so this does not change any number. §9 asks for every run of
   the analysis to be reported. The interrupted one produced no results.

No take was re-recorded, the collapse safeguard did not fire, and the
analysis script was not changed after recording.

## Reading the result (after the fact)

The verdict is the fixed §8.4 row above. What follows was identified **after
seeing the results**. It describes the pattern and suggests why, but it does
not change the verdict or test any new claim.

- **On these conditions, the gap score moved mostly with missing hands, not
  with tracking quality given detection.** Its correlation with dropout
  (+0.62) matches its correlation with landmark error (+0.61). The largest
  gaps are the two total-loss conditions (≈5) and `darken_0.07`
  (2.87, 71% dropout). With only fully tracked windows (item 2), the
  correlation falls to +0.38 and its CI includes zero.
- **`downscale` is where this shows most clearly.** Landmark error rose about
  7.6-fold across its 6 severities (0.012 to 0.091), the widest absolute range
  of any factor, while dropout stayed at or below 7.1%. The gap score stayed between
  0.006 and 0.027, not in severity order (within-factor ρ = +0.09). Within the
  other three factors, which do lose hands as severity rises, it ranked well
  (+0.83 to +1.00).
- **That is the case Rule B was designed to catch.** The world model sees the
  presence mask, so a gap score driven by dropout is close to a dropout counter
  (§1). Here it did no better than one: Δ = −0.04 and −0.06, both CIs spanning
  zero.
- **Rule B had little power at this effect size** (§13: about 8–21% even at
  ρ_gap = 0.70). But the point estimate of Δ favours the baseline, so this is
  not a near-miss that a larger study would be expected to reverse.
- **The cliffs moved.** Two conditions chosen to stop short of total loss
  reached it on this footage. As Amendment 2 anticipated, that gives the
  convention-ranked conditions weight in analysis (i), and analysis (ii)
  checks that the verdict does not depend on them.

## What this does and does not establish

**Does:** on one subject, one camera and one session, across 24 pre-registered
software-degraded conditions of real footage, the worldgap Fréchet gap score
ranked MediaPipe hand-landmark degradation with ρ = +0.61 (+0.50 without the
two conditions whose landmark error was undefined), and **did not rank it
better than counting the frames where the hand was lost**. No claim of added
value over that baseline is made.

**Does not:** say anything about other people, cameras, devices or real
deployment conditions. The primary conditions are simulated degradations, and
the physical conditions are descriptive and incomplete. It is not a general
result about the gap score.
