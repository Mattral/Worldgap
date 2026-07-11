# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-07-12

## [Unreleased]

### Added
- `pyproject.toml`: added `[project.urls]` (Homepage/Repository/Issues/Changelog)
  and completed the classifier list (Python 3.10/3.11/3.12, OS Independent,
  `Human Machine Interfaces` in place of a nonexistent `Robotics` topic
  classifier — verified against the real `trove-classifiers` package, not
  guessed). Verified `python -m build` + `twine check` pass, and that the
  built wheel installs and runs correctly (`import worldgap`, `GapAnalyzer`,
  and the `worldgap` console script) in a completely clean venv.
- `notebooks/demo.ipynb`: added a Colab badge and an auto-install cell so the
  notebook works from a fresh Colab runtime (installs from GitHub until the
  package is on PyPI, then from PyPI). Re-verified zero errors after the change.
- `README.md`: added PyPI and Colab badges.
- `data/index.py`: `RolloutIndex`, a SQLite metadata index per spec 5.3.
  `Rollout.save()`/`Rollout.load()` alone can't round-trip a rollout's
  condition/source/metadata without the caller already knowing them
  out-of-band; this closes that gap and is what makes "load every rollout in
  a directory" possible. Tested in `tests/test_index.py`.
- `GapAnalyzer.save_checkpoint()` / `GapAnalyzer.load_checkpoint()`: model +
  optimizer + config persistence, needed so `worldgap train` and
  `worldgap analyze` can be separate processes. Tested in
  `tests/test_checkpoint.py`.
- `GapConfig.from_yaml()`: loads `configs/v1_default.yaml`-style files for
  `worldgap train --config`. Tested against the real default configs in
  `tests/test_config_yaml.py`.
- `report.py`: HTML/Markdown report generation per spec 9.3 — condition
  table, Frechet/MMD trend plot, low-confidence warning surfacing, and a
  Frechet/MMD rank-disagreement diagnostic per spec Section 210. Tested in
  `tests/test_report.py`.
- CLI wired end-to-end: `train`, `analyze`, and `validate` now actually call
  `GapAnalyzer`/`ValidationHarness` against local rollout stores rather than
  only parsing arguments. Documented decision: each of
  `--data-dir`/`--source`/`--target` is a self-contained rollout store
  (`{dir}/index.db` + `{dir}/{modality}/*.npz`) rather than assuming one
  shared repo-wide `data/` root — see `cli.py`'s module docstring and the
  corresponding note added to spec 9.2. Tested end-to-end in
  `tests/test_cli.py`, including against the real installed console-script.
- `notebooks/demo.ipynb`: runs top-to-bottom via `jupyter nbconvert --execute`
  with zero manual intervention (spec Section 13 acceptance criterion) —
  covers V1 perception gap with a perturbation-severity sanity sweep, the
  V1/V2 reusability claim via the identical `GapAnalyzer` class, report
  generation, and the validation harness's anti-cherry-picking rejection.
  Entirely synthetic data; no Kaggle/MediaPipe/GPU access needed.
- `matplotlib` added as a core dependency (needed by `report.py`).

### Fixed
- `inject_tremor` / `reduce_range_of_motion` (`data/loaders/synthetic_perturb.py`)
  were perturbing every state channel, including the pose block's `visibility`
  column — contradicting `inject_tremor`'s own docstring and silently injecting
  noise into what would later be MediaPipe-confidence-derived validation
  ground truth (spec 8.1). Both now use a new `perception_position_channel_mask`
  (`data/rollout.py`) to restrict spatial perturbations to x/y/z channels only,
  for real 258-dim perception rollouts; behavior on non-standard-shape
  (test-fixture/actuation) rollouts is unchanged.
- `cli.py`'s module docstring claimed `train`/`analyze` were "wired to
  GapAnalyzer" — they aren't yet (matches ROADMAP.md, which was already
  accurate); docstring corrected to match actual behavior.
- `GapResult` (`analyzer.py`) now exposes `n_source`/`n_target`/`confidence`
  as read-only properties proxying `.frechet`, matching spec 9.1's API
  example literally rather than only via `result.frechet.n_source`.
- `spearman_with_bootstrap_ci` (`validation/stats.py`) now raises a clear
  error instead of crashing on `np.percentile` if every bootstrap resample
  produces a NaN rho (fully degenerate/constant input).
- `docs/TECHNICAL_SPEC.md` Sections 9, 10, 13 still referenced the project's
  old name (`simgap`) in code samples and the repo-tree block; corrected to
  `worldgap` throughout. Section 12.13 also gained an explicit note
  documenting that the hysteresis fit is a two-branch polynomial split, not
  literally Bouc-Wen/Hammerstein-Wiener as the MUST names — a conscious,
  tested simplification rather than an unremarked deviation.

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
  single-branch mistake (spec 5.5, edge case 12.13).
- HaGRID/EgoHands loaders: directory scanning and canonical-gesture filtering
  implemented and tested; MediaPipe landmark extraction left as a documented
  `NotImplementedError` seam (needs real data + network access unavailable in
  the scaffolding sandbox).
- CLI skeleton (`worldgap train/analyze/validate`).
- `docs/TECHNICAL_SPEC.md`, `docs/architecture.md`, `docs/data_spec.md`,
  `ROADMAP.md`.

### Known gaps (see ROADMAP.md)
- No dataset actually downloaded or run through MediaPipe yet.
- No real PGM characterization data digitized yet (synthetic data only, for
  testing the fitting code itself).
- No end-to-end V1 validation run against real ground truth yet.
- CLI does not yet call the loaders end-to-end.
- Report generation (spec 9.3) not implemented.
