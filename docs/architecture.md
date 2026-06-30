# Architecture

Ground truth is [`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md) — this file is a shorter,
implementation-facing map of where each spec section lives in the code, kept up
to date as the repo evolves. If this file and the spec ever disagree, the spec
wins; open a PR to reconcile them.

## Module map

| Spec section | Code |
|---|---|
| 5.1 Rollout schema | `src/worldgap/data/rollout.py` |
| 5.3 SQLite metadata index | `src/worldgap/data/index.py` (`RolloutIndex`) |
| 5.4 Synthetic perturbation | `src/worldgap/data/loaders/synthetic_perturb.py` |
| 5.4 HaGRID / EgoHands loaders | `src/worldgap/data/loaders/hagrid.py`, `egohands.py` (**not yet run against real data** — see ROADMAP Phase 0/1) |
| 5.5 PGM hysteresis fit | `src/worldgap/data/loaders/pgm_actuator.py` |
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

## Known incomplete seams (see ROADMAP.md for status)

- `hagrid.py` / `egohands.py`: directory-scanning and gesture-filtering logic
  is real and tested; the actual MediaPipe landmark-extraction call is a
  `NotImplementedError` stub, since it needs real downloaded frames and the
  `perception` extra installed, neither available in the sandbox that
  produced this scaffolding.
- `pgm_actuator.py`: the reference "real" curve is currently synthetic
  (test-only); the actual Ogawa et al. (2017) figure still needs digitizing,
  and that paper is likely paywalled (Taylor & Francis) — see ROADMAP Phase 6.

Everything else that was previously listed here as incomplete — the CLI's
end-to-end data path, and report generation (spec 9.3) — is now implemented
and tested (`tests/test_cli.py`, `tests/test_report.py`).
