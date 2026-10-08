# V1 first real run — results (2026-10-01)

The first run of worldgap's V1 pipeline on real recordings. The validation was
pre-registered, and **its result is null**. This page records it as it came
out, before any code change prompted by it. Procedure:
[`v1_real_data_runbook.md`](v1_real_data_runbook.md).

## Headline

On the author's own recordings, across 10 pre-registered capture conditions,
the worldgap Fréchet gap score **did not predict** MediaPipe hand dropout:

> Spearman ρ = **−0.224**, 95% bootstrap CI **[−0.810, 0.539]**, p = 0.533

The interval includes zero. Per the runbook (Step 4, check 4) and spec
Section 14, that is the result. It is not re-run with different conditions.

## What was run

| | |
|---|---|
| Subject | One person (the author), one hand gesturing, the other out of frame |
| Camera | One built-in webcam, 640×480, fixed except in `camera_shake` |
| Frame rate | Camera **delivered 29.99–30.01 fps** in every take (measured by `scripts/record_v1_session.py` while recording); all 33 files written at 30.0 fps, 2250–2251 frames per 75 s take. No take needed a rate correction, so timestamps (and `longest_dropout_run_s`) are off by at most ~0.03% |
| Capture | `scripts/record_v1_session.py` (on-screen prompts; saves raw, unmirrored frames with the measured rate) |
| Routine | `fist` → `palm` → `stop` → `like`, prompted on screen, 2 s each (1 s in `fast_motion`), repeated for 75 s per take |
| Takes | 11 folders × 3 takes = 33 recordings (≈41 min). `clean`: 2 takes first, 1 take last |
| `low_resolution` | Downscaled to 320×240 at capture time |
| Pre-registration | The 10 target conditions written to `preregistered_conditions.json` before any score was computed |
| Pipeline | `main` @ `52b587c` (includes the per-video landmarker fix; the first attempt crashed on the second video before scoring anything) |
| Model | `GapConfig(perception)`: d_model 128, 2 layers, context 16 / predict 8 frames, summary_dim 32, 30 epochs, seed 0; 48-frame windows |
| Software | Python 3.14.7, worldgap 0.2.0, mediapipe 1.0.1 (`holistic_landmarker.task` float16, sha256 `e2dab611…`), OpenCV 5.0.0, torch 2.14.1 (CPU) |
| Ground truth | Mean `hand_dropout_rate` per condition: fraction of frames where neither hand was detected (spec 8.1; MediaPipe's own output, independent of the model) |

Recordings and run outputs stay on the author's machine (`recordings/` and
`v1_run/` are gitignored; spec 12.16).

## Results per condition

Gap scores are `clean` (source) vs. each condition. The validation correlates
the **Fréchet** column with the hand-dropout mean.

| Condition | Fréchet | MMD² | Confidence | Hand dropout (mean) | Per take |
|---|---:|---:|---|---:|---|
| `clean` | — (source) | — | — | 0.1% | 0.0 / 0.0 / 0.4% |
| `backlit` | 0.015836 | 0.116868 | low | 10.7% | 5.6 / 25.7 / 0.9% |
| `camera_shake` | 0.019142 | 0.153276 | low | 3.2% | 2.0 / 4.0 / 3.5% |
| `cluttered_background` | 0.171987 | 0.611298 | low | 1.8% | 0.7 / 0.8 / 3.8% |
| `dim_light` | 0.056559 | 0.100014 | low | 1.9% | 0.9 / 3.0 / 1.8% |
| `far_from_camera` | 0.248979 | 0.874873 | low | 3.6% | 7.5 / 1.2 / 2.2% |
| `fast_motion` | 0.090096 | 0.640635 | low | 1.0% | 0.8 / 0.6 / 1.5% |
| `hand_edge_of_frame` | 0.266878 | 0.543044 | low | 3.3% | 1.1 / 0.2 / 8.5% |
| `hand_partial_occlusion` | 0.051918 | 0.446613 | low | 8.7% | 2.3 / 20.3 / 3.5% |
| `low_resolution` | 0.104846 | 0.563346 | low | 3.1% | 0.1 / 7.0 / 2.3% |
| `side_light` | 0.039318 | 0.101262 | low | 2.1% | 5.7 / 0.0 / 0.7% |

## The runbook's four checks

1. **Collapse safeguard: did not fire.** Final loss 0.024 after 270 steps on
   138 clean windows; `collapsed=False`.
2. **Confidence: `low` for every condition**, and the flag is itself
   optimistic. Each condition has 138 windows (3 recordings × 46), below the
   5 × summary_dim = 160 that spec 7.3 asks for. More importantly, windows cut
   from one recording share a session, lighting and camera, so they are not
   independent samples (`windows_are_not_independent_samples`). The effective
   sample size per condition is closer to **3 recordings than 138 windows**.
   The Fréchet covariance estimates, and therefore the bootstrap CI above, are
   less certain than the numbers suggest.
3. **Fréchet and MMD broadly agree** on the ordering (Spearman 0.685,
   p = 0.029), so the gap ranking is consistent across the two divergences
   rather than estimator noise.
4. **The CI includes zero.** Reported as the result.

## Why it came out null — two findings, given equal weight

Both were identified **after** seeing the result. They explain it; they do not
change it.

### 1. Design finding: the ground truth had almost no dynamic range

MediaPipe held up under most conditions. Condition means ranged from 1.0%
(`fast_motion`) to 10.7% (`backlit`), against 0.1% for `clean`, and most sat
between 1.8% and 3.6%. Within a condition, takes disagreed with each other as
much as conditions disagreed with each other:

| | Standard deviation of hand dropout |
|---|---:|
| Within a condition (take to take, mean over the 10 targets) | **3.44 pp** |
| Between conditions (of the 10 condition means) | **3.02 pp** |

`backlit` takes ran 5.6%, 25.7% and 0.9%; `hand_partial_occlusion` 2.3%,
20.3% and 3.5%. A single take dominates each of the two highest condition
means. Take-to-take noise was comparable to the between-condition signal,
and each condition rests on only ~3 independent recordings, so the study had
**little power to detect a relationship in either direction**. A null from it
is closer to *uninformative* than to evidence against the gap score. That is
a property of the **condition set**, not of the code: these conditions mostly
did not make MediaPipe fail. With n = 10 conditions the CI also spans 1.35
units of ρ.

*Corrected 2026-10-08:* an earlier version of this paragraph said that no gap
score could correlate with this ground truth, however good the model. Two
similar standard deviations (3.44 vs 3.02 pp) do not support "could not";
they support "had little power to".

### 2. Code finding: spec 5.2 normalization was not implemented

Spec 5.2 makes landmark normalization a MUST for V1: translate pose landmarks
relative to the hip midpoint and each hand relative to its wrist, scale by
shoulder width and hand size. Nothing in the pipeline did this, so the model
saw **raw image coordinates**, an undocumented deviation from the spec.

The pattern of results fits that. The two largest gaps were
`hand_edge_of_frame` (0.267) and `far_from_camera` (0.249), the conditions
that by design move the body within the frame. Third was
`cluttered_background` (0.172), where setting up a different background may
also have changed the framing. The two conditions with the most tracking failure, `backlit`
and `hand_partial_occlusion`, had among the *smallest* gaps (0.016, 0.052).
Without normalization, the gap score largely measures **where the person is in
the image**, which is not what spec 8.1 asks it to predict.

This is a hypothesis about the mechanism, consistent with the numbers, not a
tested claim.

## What this does and does not establish

**Does:** on one subject, one camera and these 10 pre-registered conditions,
this version of the pipeline (no spec 5.2 normalization) did not show a gap
score that tracks MediaPipe hand dropout.

**Does not:** say anything about other people, cameras or devices; and it does
not test the *specified* design, since normalization was missing. It is also
not evidence that the approach fails in general. Given finding 1, this
condition set had little power to show a correlation either way.

## Next, in order

1. Implement spec 5.2 normalization fully, including per-frame normalization
   parameters in metadata, with a test that translating or scaling a rollout as
   a whole leaves its encoding unchanged.
2. Design run 2 so that MediaPipe actually fails across a range of severities
   (the ground truth needs dynamic range before the gap score can be tested),
   and pre-register it before any re-scoring.

Re-scoring these same recordings after the normalization fix would be a
follow-up analysis prompted by the result above, and must be labelled as such.
It is not a substitute for a fresh pre-registered run.

**Run 1's saved stores are unnormalized** (raw image coordinates), so they
cannot be fed to the current analyzer as they are; it refuses to mix them with
normalized rollouts. A re-score has to normalize first, in one of two
equivalent ways: re-extract from the recordings with the chosen scheme, or
apply `normalize_rollout(r, scheme)` to each saved rollout. Normalization is
per frame, so these agree exactly. Checked on `clean/take0.mp4` with the
shoulder scheme: all 46 windows matched re-extraction with maximum state
difference 0.0. Either way it is a follow-up analysis: its numbers are not
comparable with the pre-registered ρ = −0.224 above.
