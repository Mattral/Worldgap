# Roadmap

Phases per [`docs/TECHNICAL_SPEC.md`](docs/TECHNICAL_SPEC.md) Section 16. Checkboxes
reflect actual status, not aspiration — update this file every session, not just
at the end of a phase.

## Phase 0 — Data audit
- [x] Confirm HaGRID canonical-gesture-subset sample counts are sufficient —
      resolved without needing Kaggle access: all four names in
      `CANONICAL_GESTURES` are confirmed real HaGRID v1 class names, and the
      dataset's own paper (Kapitanov et al. 2022, arXiv:2206.08219) states
      each of the 18 classes contains 30,000+ images.
- [x] **Resolve: can HaGRID supply V1's trajectories at all?** — **No.** It is
      an image dataset; a `Rollout` is a timestamped sequence. Assembling
      stills into one would give a time axis that is filename order. V1's
      primary real source is now **video**, including own webcam recordings
      (no download needed). Enforced in code via
      `metadata["temporal_provenance"]`; HaGRID is retained for frame-level
      distribution comparison only. See `docs/temporal_provenance.md`.
- [ ] Whether the four canonical gestures are the best semantic match to the
      target glove's actual controllable DOFs — still open, and a
      domain-expertise call rather than a data-availability one.
- [x] Confirm Ogawa et al. (2017) is accessible — **resolved**. Both Ogawa et al.
      (2017) and Thakur et al. (2018, a follow-up paper with a directly reusable
      fitted force-pressure equation) have been obtained. See
      `docs/pgm_reference_data.md` for the full transcription and citations.

## Phase 1 — Rollout schema, storage, loaders, perturbation
- [x] `Rollout` schema + save/load round-trip (`data/rollout.py`) — tested
- [x] Synthetic perturbation pipeline: tremor, reduced ROM, occlusion
      (`data/loaders/synthetic_perturb.py`) — tested
- [x] HaGRID/EgoHands directory scanning + canonical gesture filtering — tested
      (network-free portion only)
- [x] HaGRID/EgoHands MediaPipe landmark extraction — the conversion logic
      itself (`data/loaders/mediapipe_extract.py`: MediaPipe detection result
      → `PERCEPTION_FEATURE_LAYOUT`-shaped array, including graceful
      per-frame dropout handling) is real and unit tested against duck-typed
      fakes, with zero network access needed (`tests/test_mediapipe_extract.py`).
      Constructing a real `HolisticLandmarker` needs the
      `holistic_landmarker.task` model bundle, a one-time download on the
      machine that does the V1 run (`docs/v1_real_data_runbook.md`, Step 0);
      no real landmarker has produced a result yet. `egohands.py` delegates to
      the shared, tested implementation. **Update**: `hagrid.extract_rollout_from_frames()` now
      *raises* rather than delegating — see Phase 0 and
      `docs/temporal_provenance.md`. Also found here: the `perception` extra's
      `mediapipe>=0.10` floor permitted 0.10.x wheels, which do **not**
      contain `vision.HolisticLandmarker` at all (only `PoseLandmarker` and
      `HandLandmarker`, with `mp.solutions` gone). Floor raised to
      `mediapipe>=1.0`; `scripts/check_mediapipe_setup.py` verifies it by
      constructing a real landmarker before a capture session starts.
- [x] **Video loader** (`data/loaders/video.py`) — real trajectories from real
      recordings, with the file's own frame rate, plus
      `landmark_quality_ground_truth()` computing spec 8.1's independent
      signals (hand dropout rate, longest dropout run in seconds, mean pose
      visibility) from MediaPipe's own output. Tested, including a real
      OpenCV decode round-trip.
- [x] SQLite metadata index (spec 5.3) — `data/index.py`, tested
      (documented decision: indexed per rollout-store directory, not one
      shared repo-wide index — see `cli.py` module docstring)

## Phase 2 — World model core
- [x] `LandmarkEncoder` (Transformer, spec 6.1) — tested via forward pass
- [x] `ActuatorEncoder` (TCN, spec 6.2) — tested via forward pass
- [x] Shared `WorldModel` (JEPA-style core, EMA target encoder, spec 6.3) — tested
- [x] `CollapseSafeguard` — tested, both in isolation and wired into `fit()`
- [x] **Spec 6.3 per-frame prediction** — the predictor broadcast one vector
      across the whole future window, an undocumented deviation that made it
      structurally unable to model how a trajectory evolves. Now uses learned
      mask tokens per future offset, as 6.3 always specified. Guarded by
      `tests/test_world_model_predictor.py`. Breaks 0.1.0 checkpoints.
- [x] **Reproducibility fix (spec 12.18)** — seeding now happens before model
      construction, not only in `fit()`. Previously the same config produced
      different gap scores on every run. `tests/test_reproducibility.py`.
- [x] Real convergence check on non-synthetic data — done for **actuation**:
      the V2 run fits on the simulated PGM and encodes the real digitized
      curves without collapse (`scripts/run_v2_actuation.py`).
- [x] Real convergence check on **perception** data — done 2026-10-01: fitted
      on 138 windows from 3 real clean recordings without collapse (final
      loss 0.024). See `docs/v1_first_run_results.md`.
- [x] **Spec 5.2 landmark normalization** — a MUST for V1 that was never
      implemented (the first real run used raw image coordinates). Now in
      `data/normalization.py`, applied by every real-data loader, with
      per-frame parameters in metadata. `tests/test_normalization.py` checks
      that translating or scaling a rollout as a whole leaves its encoding
      unchanged.
- [x] **Pose anchor per study** — default scheme `shoulder_midpoint`
      (hips had visibility 0.005 vs. shoulders 0.999 in run 1), spec-faithful
      `hip_midpoint` kept; a documented deviation from spec 5.2. The scheme is
      fixed per study and recorded; `GapAnalyzer` refuses to train on or
      compare mixed schemes (`tests/test_normalization_scheme_guard.py`).

## Phase 3 — Divergence module
- [x] Fréchet distance with Ledoit-Wolf shrinkage + complex-component handling
      (spec 7.1) — tested, including the sample-size confidence flag (7.3)
- [x] MMD cross-check (spec 7.2) — tested

## Phase 4 — Validation harness
- [x] `ValidationHarness` with enforced pre-registration (anti-cherry-picking,
      spec 8.3) — tested
- [x] Spearman + bootstrap CI (spec 8.2) — tested
- [x] First V1 validation run against real MediaPipe ground truth — done
      2026-10-01, **null**: ρ = −0.224, 95% CI [−0.810, 0.539] across 10
      pre-registered conditions. Ground truth had almost no dynamic range
      (within-condition sd 3.44 pp ≥ between-condition 3.02 pp), and spec 5.2
      normalization was missing. `docs/v1_first_run_results.md`.
- [ ] Run 2: a condition set where MediaPipe actually fails across a range of
      severities, pre-registered before any re-scoring, run after spec 5.2
      normalization lands.

## Phase 5 — Packaging, CLI, demo notebook
- [x] `pyproject.toml`, src-layout, editable install — verified working
- [x] `GapAnalyzer` public API (spec 9.1) — tested across both modalities
      (`tests/test_modality_swap.py` — the concrete reusability check)
- [x] CLI (`worldgap train/analyze/validate`) — wired end-to-end against local
      rollout stores; verified via `tests/test_cli.py` (train->analyze
      round trip, empty/missing-store errors, validate join + anti-cherry-pick
      rejection) and against the real installed console-script entry point.
      The CLI consumes rollout stores rather than building them;
      `scripts/run_v1_real_data.py` builds V1 stores from video in the same
      `{dir}/index.db` + `{dir}/{modality}/*.npz` layout
- [x] Demo notebook (`notebooks/demo.ipynb`) — executes top-to-bottom via
      `jupyter nbconvert --execute` with zero manual intervention (acceptance
      criterion, spec Section 13); covers V1, the V1/V2 reusability claim, and
      the validation harness's anti-cherry-picking rejection on synthetic
      data, plus a Part 4 on the bundled digitized PGM reference data
- [x] Report generation (spec 9.3) — `report.py`, HTML and Markdown output,
      tested; includes the spec 7.2 Fréchet/MMD rank-disagreement diagnostic
- [x] CI (GitHub Actions) on Python 3.10/3.11/3.12 with `.[dev,perception]`
      installed, so the 6 MediaPipe/OpenCV tests — including all of
      `tests/test_v1_script_smoke.py` — run instead of skipping. Until 0.2.0
      they were skipped on every run. On Linux the `perception` extra also
      needs `libegl1 libgles2`, which CI installs.
- [x] `tests/test_version.py` keeps `pyproject.toml` and
      `worldgap.__version__` in sync.
- [x] **0.2.0 released** to PyPI (tag `v0.2.0`). Breaks 0.1.0 checkpoints;
      see `CHANGELOG.md`.

## Phase 6 — V2 actuation gap
- [x] Two-branch hysteresis-aware curve fit (spec 5.5, edge case 12.13) — tested
      on synthetic hysteresis data, including a test that the naive
      single-branch mistake is actually caught by the residual-structure check
- [x] Real Ogawa et al. (2017) / Thakur et al. (2018) reference data obtained
      and integrated — real prototype dimensions, pressure ranges, and
      Thakur's fitted force-pressure equations are in `pgm_actuator.py` (see
      `docs/pgm_reference_data.md`).
- [x] Ogawa 2017 Figure 4(a) digitized (`load_ogawa2017_fig4a_curve`) —
      real `Length(Force)` at each of the 7 tested pressure levels, isotonic
      smoothing applied with the correction magnitude recorded as a
      documented noise floor (spec 12.14), independently cross-validated
      against Figure 6's separately-stated contraction ratios (within
      0.4–1.4 percentage points — see `docs/pgm_reference_data.md`).
      **Correction**: an earlier version of this roadmap and
      `docs/pgm_reference_data.md` assumed this figure would show a
      pressure-ramp hysteresis loop; it doesn't (see that doc's correction
      note) — what's real and integrated is a `Length(Force, Pressure)`
      reference surface, not hysteresis-loop ground truth for
      `fit_hysteresis_curve()`.
- [ ] Ogawa Figure 5 and Thakur Figure 2/4 remain undigitized (lower
      priority than Figure 4a was — see `docs/pgm_reference_data.md`).
- [ ] A genuine pressure-ramp hysteresis characterization isn't present in
      either paper. `fit_hysteresis_curve()` remains validated only against
      synthetic data — needs either real hardware logging (V3) or a
      first-principles pneumatic dynamics model.
- [x] **End-to-end run on real PGM data — done.** `scripts/run_v2_actuation.py`
      compares an ideal-McKibben simulated actuator (parameterized from
      Ogawa's reported geometry, deliberately *not* fitted to the reference
      curves) against the digitized Figure 4(a) surface. Headline results,
      reproducible with `seed: 0`: the model's error is 13–43 mm across the
      usable pressure range, one to two orders of magnitude above each
      curve's digitization noise floor, minimised near 0.10 MPa and worst at
      both ends; the 0 MPa case is 100% saturated because the ideal model
      produces no force without pressure. The latent Fréchet gap
      rank-correlates with the physical residual at rho = +0.943 (p = 0.005) —
      an internal consistency check, explicitly **not** spec 8.1 validation,
      since both quantities derive from the same curves. Sample-size
      confidence is `low` and cannot be improved: only 7 pressure levels were
      ever published. See `docs/v2_actuation_runbook.md`.
- [x] Scale caveat carried into code (`OGAWA_2017_SCALE_CAVEAT`): Ogawa §4
      states the characterized 300 mm muscle is unsuitable for hand or wrist
      assistance, so applying this data to a glove extrapolates across a scale
      the source paper says does not carry over.

## Phase 7 — V3 (deferred)
- [ ] Not started. Contingent on real hardware access (spec Section 15). What
      changes and what doesn't is already documented there — no new design work
      needed until access exists, only a new data loader.

---

## What's actually done vs. what's scaffolded

**Genuinely implemented and tested** (134 passing tests as of 0.2.0, with the
`perception` extra installed; without it, 6 skip):
Rollout schema with temporal-provenance enforcement, SQLite metadata index,
synthetic perturbation, video loading and MediaPipe-side ground-truth
extraction, Fréchet + MMD metrics, EMA, collapse safeguard, both encoders, the
shared World Model Core with spec-6.3 per-frame mask-token prediction,
`GapAnalyzer` end-to-end across both modalities (including checkpoint
save/load and bit-identical reproducibility from a seed), the validation
harness's anti-cherry-picking enforcement, the PGM hysteresis fit (on
synthetic data), the ideal-McKibben simulated actuator, HTML/Markdown report
generation, the full CLI, and a demo notebook that runs top-to-bottom —
including a Part 4 that reaches the real digitized reference data.

**Run on real data**: V2. `scripts/run_v2_actuation.py` produces reproducible
sim-vs-real numbers with no download required; see
`docs/v2_actuation_runbook.md`.

**Run on real data, result null**: V1. The first run (2026-10-01, 33 webcam
recordings, 10 pre-registered conditions) did not show the gap score
predicting MediaPipe hand dropout. The condition set gave the ground truth too
little range, and spec 5.2 normalization was missing. See
`docs/v1_first_run_results.md`.

**Not done, and not claimed**: any result validated against independently
measured transfer degradation (spec 8.1). Nothing in this repository is a
validated result. Also outstanding: Ogawa Fig. 5 and Thakur Fig. 2/4
digitization, a real pressure-ramp hysteresis characterization (absent from
both papers, so `fit_hysteresis_curve()` stays synthetic-validated), any
hand- or wrist-scale PGM data (Ogawa §4 states the characterized 300 mm muscle
is unsuitable for that scale), GPU-scale training, and V3.
