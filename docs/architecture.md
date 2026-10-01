# Architecture

Ground truth is [`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md) — this file is a shorter,
implementation-facing map of where each spec section lives in the code, kept up
to date as the repo evolves. If this file and the spec ever disagree, the spec
wins; open a PR to reconcile them.

## Module map

| Spec section | Code |
|---|---|
| 5.1 Rollout schema | `src/worldgap/data/rollout.py` |
| 5.1 Temporal provenance | `src/worldgap/data/rollout.py` (`temporal_provenance`, enforced in `Rollout.__post_init__` and `GapAnalyzer.fit()`; see `docs/temporal_provenance.md`) |
| 5.2 Landmark normalization | `src/worldgap/data/normalization.py` (applied by the video, frame and still-image loaders) |
| 5.3 SQLite metadata index | `src/worldgap/data/index.py` (`RolloutIndex`) |
| 5.4 Synthetic perturbation | `src/worldgap/data/loaders/synthetic_perturb.py` |
| 5.4 V1 real source: video | `src/worldgap/data/loaders/video.py` (`extract_rollout_from_video`, `landmark_quality_ground_truth`) |
| 5.4 MediaPipe result → feature vector | `src/worldgap/data/loaders/mediapipe_extract.py` |
| 5.4 HaGRID / EgoHands loaders | `src/worldgap/data/loaders/hagrid.py` (still images: `T=1` rollouts only), `egohands.py` (video-derived frames) |
| 5.5 PGM reference data | `src/worldgap/data/loaders/pgm_actuator.py` (digitized Ogawa 2017 Fig. 4(a), Thakur 2018 equations) + `data/reference_data/` |
| 5.5 Simulated PGM actuator | `src/worldgap/data/loaders/pgm_sim.py` (ideal McKibben model) |
| 5.5 / 12.13 PGM hysteresis fit | `src/worldgap/data/loaders/pgm_actuator.py` (`fit_hysteresis_curve`, synthetic-validated only) |
| 6.1 Landmark encoder | `src/worldgap/models/encoders/landmark_encoder.py` |
| 6.2 Actuator encoder | `src/worldgap/models/encoders/actuator_encoder.py` |
| 6.3 World Model Core (JEPA-style) | `src/worldgap/models/world_model.py` |
| 6.3 EMA update | `src/worldgap/models/ema.py` |
| 6.3 Collapse safeguard | `src/worldgap/models/collapse.py` |
| 7.1 Fréchet distance | `src/worldgap/metrics/frechet.py` |
| 7.2 MMD | `src/worldgap/metrics/mmd.py` |
| 8 Validation harness | `src/worldgap/validation/harness.py`, `stats.py` |
| 9.1 Top-level API | `src/worldgap/analyzer.py` (`GapAnalyzer`, `GapResult`, checkpoint save/load) |
| 9.2 CLI | `src/worldgap/cli.py` — wired end-to-end against local rollout stores |
| 9.3 Report generation | `src/worldgap/report.py` |

## The one thing to protect

Only `_build_encoder()` in `analyzer.py` should ever branch on `modality`.
Everything downstream of the encoder — `WorldModel`, `frechet_distance`,
`mmd_squared`, `ValidationHarness` — must stay modality-agnostic. This is
checked, not just asserted: see `tests/test_modality_swap.py`.

If implementing V3 (spec Section 15) ever requires an `if modality ==` branch
outside `_build_encoder`, treat that as a regression in this property and fix
the abstraction rather than adding the branch.

## End-to-end scripts

| Script | What it does |
|---|---|
| `scripts/run_v2_actuation.py` | V2 on real digitized data; reproducible, no downloads (`docs/v2_actuation_runbook.md`) |
| `scripts/run_v1_real_data.py` | V1 from recorded video; pre-registers conditions before scoring (`docs/v1_real_data_runbook.md`) |
| `scripts/check_mediapipe_setup.py` | Preflight: builds a real `HolisticLandmarker` from the model bundle |

## Known incomplete seams (see ROADMAP.md for status)

- **V1 has not been run on real recordings.** `run_v1_real_data.py` is tested
  end to end against real video files with a faked landmarker
  (`tests/test_v1_script_smoke.py`, which CI runs with the `perception` extra
  installed); no real `HolisticLandmarker` has produced a result yet.
- **HaGRID cannot supply trajectories.** It is a still-image dataset, so
  `hagrid.extract_rollout_from_frames()` raises by design; use
  `extract_static_pose_rollouts()` for frame-level comparison
  (`docs/temporal_provenance.md`).
- **`fit_hysteresis_curve` is validated on synthetic data only.** Neither
  source paper publishes a pressure-ramp hysteresis loop; the digitized
  Fig. 4(a) data is `Length(Force)` at fixed pressure and must not be fed to it.
- **The PGM reference data is walking-assist scale.** Ogawa et al. (2017) §4
  says the characterized 300 mm muscle does not carry over to hand or wrist
  scale (`OGAWA_2017_SCALE_CAVEAT`).
