# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- `docs/pgm_reference_data.md`: real PGM reference data obtained directly
  from Ogawa et al. (2017) and Thakur et al. (2018) — resolves the spec
  Section 14 / ROADMAP Phase 0 & 6 "Ogawa et al. access" item. Documents that
  the two papers describe two *different* physical PGM prototypes (250mm vs
  300mm natural length, measuring different physical quantities) and should
  not be merged into one dataset.
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
- `docs/TECHNICAL_SPEC.md`/`ROADMAP.md`: removed personal-correspondence
  framing ("the Kurita proposal, internal document, shared in this
  conversation") in favor of citing the now-public Ogawa/Thakur papers
  directly — same technical grounding, no implication of private
  correspondence.

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
  rank-disagreement diagnostic (spec 210).
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
