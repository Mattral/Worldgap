# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

> **Upgrading from 0.2.0: the same video now yields different states.**
> 0.2.0's loaders (`extract_rollout_from_video`, `extract_rollout_from_frames`,
> `extract_static_pose_rollouts`) return **unnormalized** image coordinates.
> From the next release they **normalize by default**, anchoring the pose at
> the **shoulder midpoint** (`normalization_scheme="shoulder_midpoint"`), and
> record the scheme in `metadata["normalization"]`. Consequences:
> - Rollout stores, checkpoints and gap scores produced with 0.2.0 are **not
>   comparable** with ones produced now, even from identical recordings.
> - `GapAnalyzer` refuses to train on or compare rollouts with different
>   schemes, and treats unnormalized vs. normalized as different.
> - To bring 0.2.0 stores forward, either re-extract from the original
>   recordings, or apply `normalize_rollout(r, scheme)` to each saved
>   (unnormalized) rollout: normalization is per frame, so this gives exactly
>   the re-extracted states. Then re-fit. Use `hip_midpoint` only if every
>   rollout in the study uses it.

### Added

- **Run-2 recording plan:** `scripts/record_v1_session.py --plan run2` records
  pre-registration §3's 21 takes in its fixed order (clean 0 and 1, the 6
  physical conditions, clean 2) into `recordings_run2/`. That is a separate
  folder because existing takes are skipped, so run 1's spent clean footage
  can never be picked up. `recordings*/` is gitignored. A test ties the plan's
  order and names to `run_v1_run2.PHYSICAL`.
- **`scripts/run_v1_run2.py`, the run-2 analysis**, implementing
  `docs/v1_run2_preregistration.md` §5–§8 (with Amendment 1). Every
  (condition, take) extraction is cached, keyed on the video, transform,
  scheme and version, and written atomically, so an overnight run resumes
  without redoing work. It writes `study_settings.json` (git commit, script
  and model hashes) and `preregistered_conditions.json` before any score. It
  fits one model on the clean windows (`summary_dim` 16), scores the 24
  software-degraded conditions and the physical ones, and applies Rules A and
  B from one paired bootstrap, with the amended §8.4 reading. **`--blind`**
  runs every stage on real footage but writes only a health report,
  discarding every score and correlation unread. That is how it is tested on
  run 1's footage without previewing run 2's result. `--max-frames`,
  `--epochs` and `--bootstrap` exist for smoke tests and mark the output as
  not run 2. `tests/test_v1_run2_script.py` (4 tests, synthetic videos and a
  fake landmarker) covers the full pipeline, resuming from cache, blind mode
  leaking no numbers, the 3-clean-take rule and the interpretation.
- **`worldgap.validation.stats.paired_spearman_bootstrap`**: Spearman ρ of
  several scores against one ground truth, plus differences between them, from
  a single percentile bootstrap over conditions. The same resample is used for
  every score, so ρ_a − ρ_b gets a properly paired CI. `+inf` ground truth
  ranks worst. With the same seed, a single score's CI equals
  `spearman_with_bootstrap_ci`'s. 5 new tests in `tests/test_stats.py`.
- **`docs/v1_run2_preregistration.md`: run 2, pre-registered** before any
  run-2 recording exists. 24 software-degraded conditions from fresh clean
  recordings (`darken`, `downscale`, `blur`, `noise` × 6 severities, chosen by
  a pilot on run 1's footage that computed no gap scores), plus 6 physical
  conditions reported descriptively. Primary ground truth: paired landmark
  error, a documented spec 8.1 deviation, because with dropout as ground truth
  the trivial dropout baseline would *be* the ground truth. Two confirmatory
  rules: the CI of ρ(gap, landmark error) excludes zero, and the gap score
  beats a dropout-counting baseline (paired-bootstrap CI of Δρ excludes zero;
  the pilot's baseline ρ was +0.56). Also fixed in advance: 75 s takes,
  `summary_dim` 16 (so 138 windows clear "low"), the shoulder anchor, and that
  conditions are never dropped.
- **Run-2 tooling.** `worldgap.data.degradations.ImageDegradation`: graded,
  deterministic image degradations (`darken`, `downscale`, `blur`, `noise`)
  for software-degraded conditions built from real clean recordings, so every
  condition reuses the same frames. `extract_rollout_from_video(...,
  frame_transform=...)` applies one before MediaPipe and records it in
  `metadata["frame_transform"]`. `paired_landmark_error(reference, degraded)`:
  distance from the same frame's clean detection, in hand sizes, with
  left/right label swaps not counted as error. It measures tracking quality,
  which the presence mask (an input to the world model) does not contain.
  `scripts/run_v1_pilot.py` measures dropout, landmark error and the trivial
  presence baselines per severity, and by design computes no gap scores.
  `tests/test_degradations.py` (12 tests).
- **`scripts/record_v1_session.py`**, the guided capture script run 1 was
  recorded with (previously only in temporary storage, so run 1 was not
  reproducible). It prompts the gesture routine on screen, saves raw
  unmirrored frames into the runbook's folder layout (`clean` first and
  last), and writes each file with the frame rate the camera actually
  delivered, rewriting it when that differs from the nominal rate by more than
  0.5%. That number matters: `extract_rollout_from_video` converts frames to
  seconds with it, so `longest_dropout_run_s` inherits any error. Run 1's
  camera delivered 29.99–30.01 fps (no take needed correcting), now recorded
  in `docs/v1_first_run_results.md`. `tests/test_record_v1_session.py` (6
  tests, fake camera and clock) includes an end-to-end check that a 3 s take
  from a camera delivering 20 fps while claiming 30 is measured as a 3 s
  dropout, not 2 s.

- **`docs/v1_first_run_results.md`: the first real V1 run, result null.** 33
  webcam recordings (one subject, 11 folders × 3 takes), 10 pre-registered
  conditions. Spearman ρ = −0.224 between the Fréchet gap score and MediaPipe
  hand dropout, 95% CI [−0.810, 0.539]. Recorded unchanged, before any code
  change it prompted. Two causes found afterwards and given equal weight: the
  condition set gave the ground truth almost no dynamic range (take-to-take
  sd 3.44 pp ≥ between-condition sd 3.02 pp), and spec 5.2 normalization had
  never been implemented (now noted in spec 5.2). The low-confidence flag is
  itself optimistic: windows from one recording are not independent, so the
  effective n per condition is nearer 3 than 138.

### Fixed

- **`LandmarkEncoder` crashed on a window in which nothing was detected in
  any frame.** Such a window is fully masked. PyTorch's inference fast path
  raised `to_padded_tensor: at least one constituent tensor should have
  non-zero numel` (the analyzer encodes windows one at a time, so one empty
  window is enough), and attention is undefined for it in training mode. Fully
  masked windows are now unmasked. Their frame outputs are discarded by the
  presence-weighted pooling anyway, so every empty window encodes to the same
  "nothing detected" summary. Found by the run-2 blind test on run 1's
  footage, where the harshest degradations lost the whole body for 48 frames.
  Two regression tests; one reproduces the exact error against the old code.
- **Spec 5.2 landmark normalization, implemented** (`data/normalization.py`).
  It is a MUST for V1 and was never implemented; the first real run fed raw
  image coordinates to the model. Pose is translated by the hip midpoint and
  scaled by shoulder width, each hand by its wrist and bounding-box diagonal;
  visibility and presence are untouched. Per-frame parameters go to
  `metadata["normalization"]`, survive the SQLite index and windowing, and
  `denormalize_states()` recovers raw values exactly. Applied by
  `extract_rollout_from_video`, `extract_rollout_from_frames` and
  `extract_static_pose_rollouts`. **Behaviour change:** perception rollouts
  from these loaders are now normalized, so stores and checkpoints built
  before this are not comparable with new ones (the first run's numbers stay
  as recorded in `docs/v1_first_run_results.md`). `tests/test_normalization.py`
  (9 tests) includes the property the defect broke: a rollout translated or
  scaled as a whole produces the same states and the same encoding, checked
  through the real video-loader path too. The V1 smoke test's fake landmarks
  now have real extent and per-landmark noise; with every point on one spot
  (and one shared jitter, i.e. a translation) they normalized to constants.
  Spec 5.2 notes an open question: with the hips out of frame (mean
  visibility 0.005 in the first run's clean recordings), the hip-midpoint
  origin is MediaPipe's extrapolation.
- `scripts/run_v1_real_data.py` closes each rollout store's SQLite index
  (`save_store` leaked the connection).
- V2 runbook: results are bit-identical only for the same platform, torch
  build and CPU thread count; with 4 threads the overall MMD² reads −0.014865
  instead of −0.014864.
- **`scripts/run_v1_real_data.py` crashed on the second video of the first
  real run** with `ValueError: Input timestamp must be monotonically
  increasing`. It built one VIDEO-mode `HolisticLandmarker` and reused it for
  every file, but each file's timestamps restart at 0. It now builds a fresh
  landmarker per video and closes it afterwards. That also stops MediaPipe's
  frame-to-frame tracking state from carrying from the end of one recording
  into the next (shifting timestamps instead would have hidden the error but
  kept that contamination). The test fake now enforces the same timestamp
  rule, which is why the smoke tests had passed; the new
  `test_each_video_gets_a_fresh_landmarker` fails against the old script.

### Removed

- **The `actuation` extra** (`mujoco>=3.1`). Nothing imported MuJoCo: the V2
  simulator is an ideal McKibben model in numpy/scipy (`pgm_sim.py`), so the
  extra downloaded a large dependency the design had deliberately rejected.
  **V2 needs nothing beyond the core install.** If you had
  `worldgap[actuation]` pinned, change it to `worldgap`. The rejection is now
  recorded as a documented deviation in spec 5.5 and 11.

### Changed

- **Default pose anchor is now the shoulder midpoint** (a documented
  deviation from spec 5.2). Normalization has two named schemes:
  `shoulder_midpoint` (default) and the spec-faithful `hip_midpoint`. In run
  1's clean recordings MediaPipe's mean visibility was 0.005 for the hips and
  0.999 for the shoulders, so a hip anchor was an extrapolation. The scheme is
  a per-study choice: every loader takes `normalization_scheme=`,
  `run_v1_real_data.py` takes `--normalization` and records it in a new
  `study_settings.json` before any score, and nothing ever switches anchors per
  frame or on visibility.
- **`GapAnalyzer` refuses to mix normalization schemes.** `fit()` rejects
  training rollouts with different schemes; `compute_gap()` rejects source and
  target rollouts whose schemes differ from each other or from the training
  scheme (normalized vs. raw counts as different), in the same spirit as the
  temporal-provenance guard. The training scheme is saved in checkpoints
  (older checkpoints load with it unknown, and only the source/target check
  applies). The refusal names the fix: re-extract with one scheme, or `normalize_rollout()` saved unnormalized rollouts. `tests/test_normalization_scheme_guard.py` (9 tests).
- `docs/v1_first_run_results.md`: the claim that no gap score could have
  correlated with run 1's ground truth was too strong for two similar
  standard deviations (3.44 vs 3.02 pp). Now: take-to-take noise comparable
  to the between-condition signal and ~3 recordings per condition, so the
  study had little power in either direction and its null is closer to
  uninformative than to evidence against the gap score. The correction is
  marked in place.
- Project URLs (`pyproject.toml`, README, notebook) use the repository's real
  casing, `Mattral/Worldgap`.
- Docs brought up to date with 0.2.0. README: PyPI install, CI badge, the V1
  source described as video rather than HaGRID, the checkpoint-upgrade note.
  `docs/architecture.md`: module map now covers the video loader, the
  simulator and the reference data; its "incomplete seams" list described
  long-fixed stubs. `docs/data_spec.md`: index and temporal provenance
  documented as implemented, and the PGM section no longer suggests feeding
  the Fig. 4(a) `Length(Force)` curves to `fit_hysteresis_curve` (a different
  hysteresis axis). ROADMAP: CI, version guard and release status, 134 tests.
  `docs/v2_actuation_runbook.md`: run instructions start from a clone, and the
  reproducibility claim now says what differs across platforms (last digits of
  per-level MMD²) and what does not.
- The demo notebook's install cell explains why it installs from GitHub `main`
  rather than PyPI (so the library matches the notebook).

## [0.2.0] - 2026-10-01

> **Upgrading from 0.1.0: saved checkpoints will not load.** The world-model
> predictor now has learned per-offset mask tokens (`future_mask_tokens`, see
> *Fixed* below), so the model's `state_dict` changed shape.
> `GapAnalyzer.load_checkpoint()` on a 0.1.0 checkpoint fails with a
> `RuntimeError` (missing key `future_mask_tokens`; size mismatch for
> `predictor.0.weight`). Re-run `worldgap train` to produce a new
> checkpoint. Gap scores from 0.1.0 are also not comparable with 0.2.0 ones:
> both the predictor and the seeding order changed.

### Added — real-data paths, V1 and V2

- **`docs/temporal_provenance.md` + `Rollout.metadata["temporal_provenance"]`**
  — resolves the static-image-vs-trajectory design gap. Every rollout now
  records where its *time axis* came from (`video`, `simulated`,
  `quasi_static_sweep`, `static_pose`, `synthetic_from_static`), separately
  from where its values came from. `Rollout` rejects a multi-frame
  `static_pose`; `GapAnalyzer.fit()` raises when every rollout declares a
  non-temporal axis and warns when none declare anything.
- **`data/loaders/video.py`** — V1's real data source. Decodes a video file,
  runs the landmarker over its frames in order, and produces a rollout with a
  real time axis and the file's own frame rate (refusing to assume 30 fps).
  Plus `landmark_quality_ground_truth()`, which computes spec 8.1's
  independent degradation signals from MediaPipe's own output: hand dropout
  rate, longest dropout *run* in seconds (reported separately from the rate,
  since they are different control problems), and mean pose visibility.
- **`Rollout.split_into_windows()`** — cuts one long recording into several
  rollouts so the divergence metrics have an adequate `n`. Each window carries
  `windows_are_not_independent_samples=True`, because they share a subject,
  session and camera and the resulting confidence flag is therefore optimistic.
- **`data/loaders/pgm_sim.py`** — the V2 simulated actuator baseline (resolves
  the open "which simulator" question). Ideal McKibben braid model
  parameterized from Ogawa's own reported geometry, with the derived braid
  angle flagged as derived. Deliberately not fitted to the reference curves
  and deliberately not MuJoCo; reasons documented in the module.
- **`scripts/run_v1_real_data.py`** — end-to-end V1 on real recordings:
  extraction, rollout stores, pre-registration written to disk *before* any
  score is computed, gap scores, report, validation. Smoke-tested end to end
  against real video files with a faked landmarker
  (`tests/test_v1_script_smoke.py`).
- **`scripts/run_v2_actuation.py`** — end-to-end V2, needing no downloads:
  physical residuals in mm against each curve's noise floor, latent gap
  scores, saturation reporting, and an internal-consistency rank correlation
  explicitly labelled as *not* spec 8.1 validation.
- **`scripts/check_mediapipe_setup.py`** — preflight that constructs a real
  `HolisticLandmarker` and runs it, so setup problems surface before a capture
  session rather than during one.
- `docs/v1_real_data_runbook.md`, `docs/v2_actuation_runbook.md`.
- `.gitattributes` (`* text=auto eol=lf`) to stop CRLF/LF churn showing
  untouched files as modified on Windows.

### Fixed

- **Reproducibility (spec 12.18).** `GapAnalyzer.__init__` now seeds before
  constructing the model. Previously `torch.manual_seed()` ran only inside
  `fit()`, by which point every encoder weight had already been initialized
  from ambient global RNG state — so the same config produced different gap
  scores on every run. Found by running the V2 script twice.
  `tests/test_reproducibility.py` guards it.
- **Spec 6.3 deviation: per-frame prediction.** The predictor took the pooled
  context, produced one vector, and broadcast it across the whole future
  window — structurally unable to represent how a trajectory evolves over the
  horizon. It now uses learned mask tokens per future offset, as 6.3 specified
  all along, and those tokens are in the optimizer's parameter list.
  `tests/test_world_model_predictor.py` guards against the broadcast
  regression. **This changes the model's `state_dict`: 0.1.0 checkpoints will
  not load.**
- **`mediapipe>=1.0`** in the `perception` extra, up from `>=0.10`.
  `vision.HolisticLandmarker` — the entry point every V1 loader is written
  against — does not exist in the 0.10.x wheels, which export only
  `PoseLandmarker` and `HandLandmarker` and no longer ship `mp.solutions`
  either. The old floor resolved to a version where a real V1 run fails with
  `AttributeError`.
- **`hagrid.extract_rollout_from_frames()` now raises** instead of silently
  assembling unrelated stills into one trajectory whose time axis is filename
  order. `list_hagrid_sequences()` is deprecated in favour of
  `list_hagrid_images()` (it never returned sequences), and
  `extract_static_pose_rollouts()` provides the honest `T=1` path.
- `MMDResult.mmd_squared` documented as the *unbiased* estimator, which can
  legitimately be negative when the true value is near zero; `GapResult` now
  warns when it is, saying what it means and that its magnitude must not be
  used to rank conditions. A test asserting `mmd_squared >= 0` encoded the
  false property and was corrected.
- Documented that `summary_head` sits outside the JEPA loss and is therefore a
  fixed random projection, not a learned one (spec 6.3 implementation note).
- Spec cross-references: `cli.py` cited "Section 16" for a risks item that is
  in Section 14; `report.py`/CHANGELOG cited "spec 210" (a line number) for a
  requirement in Section 7.2.
- `configs/v2_default.yaml`: `state_dim` 2 → 3, matching the real actuation
  state layout `[pressure_mpa, force_n, length_mm]`; `summary_dim` 32 → 8,
  because only 7 pressure levels exist in the literature.
- `scripts/check_mediapipe_setup.py`, `run_v1_real_data.py`,
  `run_v2_actuation.py` and `download_datasets.sh` are now committed as
  executable (`100755`). They carry shebangs but were stored as `100644`
  because git on Windows does not record the executable bit, so `ruff check .`
  reported three EXE001 errors on any Unix checkout (ruff only checks the
  Python files) and `./scripts/download_datasets.sh` was permission-denied.
  CI was unaffected (it lints `src tests`).
- The version string lives in both `pyproject.toml` and
  `src/worldgap/__init__.py`; `tests/test_version.py` now fails if they drift.

### Changed — corrections to earlier claims

- **The two papers do not describe two different physical prototypes.** Thakur
  et al. (2018) §II.A describes the actuator as the one "we previously
  developed" citing Ogawa, reuses Ogawa's elongation figure, and motivates its
  own experiment because the stretched-length behaviour "is not measured in
  [14]". The papers' measurements are still kept separate — because they
  measure different quantities, not different hardware. Corrected in
  `docs/pgm_reference_data.md`, `pgm_actuator.py`, the spec and the tests.
- The contraction-ratio reference length (500 mm) is now labelled an
  **inference** that reproduces Figure 6's published numbers, not a formula
  either paper writes down — the literal reading of "natural length" (250 mm
  or 300 mm) does not reproduce them.
- `digitization_noise_floor_mm` is now labelled a **heuristic proxy**, not a
  confidence interval.
- Ogawa §4's scale caveat is now carried in code
  (`OGAWA_2017_SCALE_CAVEAT`): the characterized muscle is 300 mm and the
  authors state it is unsuitable for hand or wrist assistance.

### Added
- `pgm_actuator.py`: `load_ogawa2017_fig4a_curve()` — real digitized
  `Length(Force)` data at each of the 7 pressure levels Ogawa et al. (2017)
  Figure 4(a) tested, bundled as package data (`data/reference_data/
  ogawa2017_fig4a/*.csv`, ships in the wheel — verified against an actual
  clean-venv install, not just the source checkout). Isotonic regression
  (`scipy.optimize.isotonic_regression`, bumping the scipy floor to >=1.12)
  enforces the known physical constraint that elongation is non-decreasing
  with load, with the correction magnitude recorded per curve as a
  documented digitization noise floor (spec 12.14) — ranges from 0.06mm on
  the cleanest curve to 11.39mm on the noisiest. Independently
  cross-validated against Figure 6's separately-stated contraction ratios at
  0.2 MPa (35.6%/29.9%/24.4% digitized vs. 36%/29%/23% paper-stated — within
  0.4-1.4 percentage points, without having used Figure 6's numbers anywhere
  in the digitization). 6 new tests.
- **Correction**: an earlier version of `docs/pgm_reference_data.md` assumed
  Ogawa Figure 4(a) would show a pressure-ramp hysteresis loop (matching
  `fit_hysteresis_curve()`'s loading/unloading model). Having now actually
  digitized it, each pressure level is a single curve, not two resolvable
  branches — what's real and integrated is a `Length(Force, Pressure)`
  reference surface, not hysteresis-loop ground truth. Documented clearly
  in `docs/pgm_reference_data.md` and `ROADMAP.md` rather than silently
  reusing the old (incorrect) framing.
- `docs/pgm_reference_data.md`: real PGM reference data obtained directly
  from Ogawa et al. (2017) and Thakur et al. (2018) — resolves the spec
  Section 14 / ROADMAP Phase 0 & 6 "Ogawa et al. access" item. Documents why
  the two papers' measurements are not merged into one dataset (they measure
  different quantities under different held-constant conditions) and how the
  two papers report the prototype's dimensions inconsistently.
- `pgm_actuator.py`: added `OGAWA_2017_PROTOTYPE`/`THAKUR_2018_PROTOTYPE`
  (real dimensions + citations), `OGAWA_2017_PGM_VS_PM10RF_AT_0_2MPA` (real
  contraction/elongation comparison table transcribed from the paper's own
  text), and `thakur2018_force_from_pressure()` — Thakur's two directly
  reusable fitted force-pressure equations (R²=0.993/0.998), validated
  against the paper's own reported sanity-check values (60kPa→~30N,
  100kPa→~44N stretched) and raising outside the paper's validated 50-300 kPa
  range rather than silently extrapolating. 5 new tests
  (`tests/test_pgm_hysteresis.py`). Still pending: digitizing the full
  continuous pressure-elongation curves (Ogawa Fig. 4/5) for the hysteresis
  fit itself — see `docs/pgm_reference_data.md` for exactly what's extracted
  vs. still needed.
- `data/loaders/mediapipe_extract.py`: real, unit-tested implementation of
  the MediaPipe-result → `PERCEPTION_FEATURE_LAYOUT` conversion (spec 5.4),
  shared by `hagrid.py` and `egohands.py` (both now delegate here instead of
  raising `NotImplementedError`). Handles per-frame dropout (no detection at
  all) by setting `presence_mask=False` rather than dropping or interpolating
  the frame, matching spec 8.1's requirement that MediaPipe dropout rate stay
  measurable. Tested against duck-typed fake landmark/result objects
  (`tests/test_mediapipe_extract.py`) — no real `mediapipe` install needed for
  8 of 10 tests; the 2 that build real `mp.Image` objects skip cleanly via
  `pytest.importorskip` rather than erroring when the `perception` extra isn't
  installed. Confirmed via a direct request that `storage.googleapis.com`
  (where MediaPipe's `.task` model bundles are hosted) is blocked by this
  sandbox's network allowlist (`x-deny-reason: host_not_allowed`) — so
  constructing a real `HolisticLandmarker` and running this against real
  HaGRID/EgoHands frames remains the one genuinely blocked piece.
- `pyproject.toml`: added `pillow` to the `dev` extra (used by the two
  fixture-image tests above).

### Changed
- `docs/TECHNICAL_SPEC.md`/`ROADMAP.md`: replaced references to an unpublished
  source document with citations to the published Ogawa et al. (2017) and
  Thakur et al. (2018) papers — same technical grounding, fully citable.
- CI installs `.[dev,perception]` instead of `.[dev]`. Before this, the six
  tests that need `mediapipe`/`cv2` — including all of
  `tests/test_v1_script_smoke.py` — were skipped on every CI run, and the
  `perception` extra had never been installed by CI on any Python version.
  Doing so surfaced a real requirement: on Linux, mediapipe's
  `libmediapipe.so` links `libEGL.so.1` and `libGLESv2.so.2`, so CI now
  installs `libegl1 libgles2`, and the README and V1 runbook say so. The
  matrix also runs with `fail-fast: false`, so one Python version failing no
  longer cancels the others.

---

## [0.1.0] - 2026-07-12

First published release (PyPI).

### Added
- Initial repo scaffolding: `pyproject.toml`, src-layout package, CI workflow.
- `Rollout` schema with content-hashed IDs, save/load round-trip (spec 5.1–5.3).
- Synthetic perturbation pipeline: tremor injection, reduced range-of-motion,
  occlusion simulation — all seeded and reproducible (spec 5.4).
- `LandmarkEncoder` (Transformer, spec 6.1) and `ActuatorEncoder` (TCN, spec 6.2).
- Shared `WorldModel` JEPA-style core with EMA target encoder and an automated
  representation-collapse safeguard (spec 6.3, edge case 12.7).
- Fréchet-distance divergence metric with Ledoit-Wolf covariance shrinkage,
  numerically-stable matrix-sqrt handling, and a sample-size confidence flag
  (spec 7.1, 7.3, edge case 12.9).
- MMD cross-check metric (spec 7.2).
- `GapAnalyzer` top-level API, verified identical across the perception and
  actuation modalities via `tests/test_modality_swap.py` — the concrete test
  of the "one core, swappable encoder" reusability claim (spec Section 3).
- `ValidationHarness` with pre-registration enforced in code, not just
  documented — submitting a condition set that doesn't exactly match what was
  pre-registered raises rather than silently proceeding (spec 8.3).
- Two-branch (loading/unloading) hysteresis-aware curve fit for PGM actuator
  reference data, with a residual-structure diagnostic that catches the naive
  single-branch mistake (spec 5.5, edge case 12.13). Tested against synthetic
  data only at this release — no real PGM characterization data yet.
- HaGRID/EgoHands loaders: directory scanning and canonical-gesture filtering
  implemented and tested; MediaPipe landmark extraction left as a documented
  `NotImplementedError` seam at this release (needs real data + network
  access unavailable in the scaffolding sandbox).
- `data/index.py`: `RolloutIndex`, a SQLite metadata index (spec 5.3) — closes
  the gap where `Rollout.save()`/`load()` alone can't round-trip a rollout's
  condition/source/metadata without the caller already knowing them
  out-of-band.
- `GapAnalyzer.save_checkpoint()` / `GapAnalyzer.load_checkpoint()`: model +
  optimizer + config persistence, so `worldgap train` and `worldgap analyze`
  can be separate processes.
- `GapConfig.from_yaml()`: loads `configs/v1_default.yaml`-style files for
  `worldgap train --config`.
- `report.py`: HTML/Markdown report generation (spec 9.3) — condition table,
  Fréchet/MMD trend plot, low-confidence warning surfacing, and a Fréchet/MMD
  rank-disagreement diagnostic (spec 7.2).
- CLI (`worldgap train/analyze/validate`) wired end-to-end against local
  rollout stores (spec 9.2). Documented decision: each of
  `--data-dir`/`--source`/`--target` is a self-contained rollout store
  (`{dir}/index.db` + `{dir}/{modality}/*.npz`) rather than one shared
  repo-wide `data/` root.
- `notebooks/demo.ipynb`: runs top-to-bottom via `jupyter nbconvert --execute`
  with zero manual intervention (spec Section 13 acceptance criterion) —
  covers V1 perception gap with a perturbation-severity sanity sweep, the
  V1/V2 reusability claim, report generation, and the validation harness's
  anti-cherry-picking rejection. Entirely synthetic data.
- `pyproject.toml`: `[project.urls]`, complete classifier list (Python
  3.10/3.11/3.12, OS Independent, `Human Machine Interfaces` in place of a
  nonexistent `Robotics` classifier — verified against the real
  `trove-classifiers` package). Verified `python -m build` + `twine check`
  pass and the built wheel installs and runs correctly in a clean venv.
- `matplotlib` added as a core dependency (needed by `report.py`).
- `docs/TECHNICAL_SPEC.md`, `docs/architecture.md`, `docs/data_spec.md`,
  `ROADMAP.md`.

### Fixed
- `inject_tremor` / `reduce_range_of_motion` (`data/loaders/synthetic_perturb.py`)
  were perturbing every state channel, including the pose block's `visibility`
  column — contradicting `inject_tremor`'s own docstring and silently injecting
  noise into what would later be MediaPipe-confidence-derived validation
  ground truth (spec 8.1). Both now use a new `perception_position_channel_mask`
  (`data/rollout.py`) to restrict spatial perturbations to x/y/z channels only,
  for real 258-dim perception rollouts.
- `cli.py`'s module docstring claimed `train`/`analyze` were "wired to
  GapAnalyzer" before they actually were; corrected once the real wiring
  landed.
- `GapResult` (`analyzer.py`) exposes `n_source`/`n_target`/`confidence` as
  read-only properties proxying `.frechet`, matching spec 9.1's API example
  literally rather than only via `result.frechet.n_source`.
- `spearman_with_bootstrap_ci` (`validation/stats.py`) raises a clear error
  instead of crashing on `np.percentile` if every bootstrap resample produces
  a NaN rho (fully degenerate/constant input).
- `docs/TECHNICAL_SPEC.md` Sections 9, 10, 13 referenced the project's old
  name (`simgap`) in code samples and the repo-tree block; corrected to
  `worldgap` throughout. Section 12.13 gained an explicit note documenting
  that the hysteresis fit is a two-branch polynomial split, not literally
  Bouc-Wen/Hammerstein-Wiener as the MUST names — a conscious, tested
  simplification rather than an unremarked deviation.

### Known gaps at this release
- HaGRID/EgoHands MediaPipe landmark extraction not yet implemented (fixed
  post-release — see Unreleased above).
- No real PGM characterization data; hysteresis fit tested only against
  synthetic data (fixed post-release — see Unreleased above).
- No end-to-end V1 validation run against real ground truth yet (still open;
  see ROADMAP.md).
