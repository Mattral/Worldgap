# Roadmap

Phases per [`docs/TECHNICAL_SPEC.md`](docs/TECHNICAL_SPEC.md) Section 16. Checkboxes
reflect actual status, not aspiration — update this file every session, not just
at the end of a phase.

## Phase 0 — Data audit
- [ ] Confirm HaGRID canonical-gesture-subset sample counts are sufficient
      (`CANONICAL_GESTURES` in `hagrid.py` is currently a placeholder mapping)
- [ ] Confirm Ogawa et al. (2017) is accessible (open access, or via institutional
      library) — flagged in spec Section 14 as the one real external dependency

## Phase 1 — Rollout schema, storage, loaders, perturbation
- [x] `Rollout` schema + save/load round-trip (`data/rollout.py`) — tested
- [x] Synthetic perturbation pipeline: tremor, reduced ROM, occlusion
      (`data/loaders/synthetic_perturb.py`) — tested
- [x] HaGRID/EgoHands directory scanning + canonical gesture filtering — tested
      (network-free portion only)
- [ ] HaGRID/EgoHands MediaPipe landmark extraction — **blocked**: needs real
      downloaded frames + the `perception` extra, neither available in the
      sandbox that produced this scaffolding. Currently a `NotImplementedError`
      stub with the exact contract documented.
- [ ] SQLite metadata index (spec 5.3) — not started; `.npz`-per-rollout works
      without it for now

## Phase 2 — World model core
- [x] `LandmarkEncoder` (Transformer, spec 6.1) — tested via forward pass
- [x] `ActuatorEncoder` (TCN, spec 6.2) — tested via forward pass
- [x] Shared `WorldModel` (JEPA-style core, EMA target encoder, spec 6.3) — tested
- [x] `CollapseSafeguard` — tested, both in isolation and wired into `fit()`
- [ ] Real convergence check on actual (non-synthetic-placeholder) data — blocked
      on Phase 0/1 data access

## Phase 3 — Divergence module
- [x] Fréchet distance with Ledoit-Wolf shrinkage + complex-component handling
      (spec 7.1) — tested, including the sample-size confidence flag (7.3)
- [x] MMD cross-check (spec 7.2) — tested

## Phase 4 — Validation harness
- [x] `ValidationHarness` with enforced pre-registration (anti-cherry-picking,
      spec 8.3) — tested
- [x] Spearman + bootstrap CI (spec 8.2) — tested
- [ ] Actual V1 validation run against real MediaPipe-confidence ground truth —
      blocked on Phase 0/1 data access

## Phase 5 — Packaging, CLI, demo notebook
- [x] `pyproject.toml`, src-layout, editable install — verified working
- [x] `GapAnalyzer` public API (spec 9.1) — tested across both modalities
      (`tests/test_modality_swap.py` — the concrete reusability check)
- [x] CLI skeleton (`worldgap train/analyze/validate`) — argument parsing verified,
      not yet wired to real data loaders end-to-end
- [ ] Demo notebook — not started
- [ ] Report generation (spec 9.3) — not started

## Phase 6 — V2 actuation gap
- [x] Two-branch hysteresis-aware curve fit (spec 5.5, edge case 12.13) — tested
      on synthetic hysteresis data, including a test that the naive
      single-branch mistake is actually caught by the residual-structure check
- [ ] Digitize real Ogawa et al. curve via WebPlotDigitizer — blocked on Phase 0
- [ ] `test_modality_swap.py`-style end-to-end run on real PGM data — blocked

## Phase 7 — V3 (deferred)
- [ ] Not started. Contingent on Kurita Lab access (spec Section 15). What
      changes and what doesn't is already documented there — no new design work
      needed until access exists, only a new data loader.

---

## What's actually done vs. what's scaffolded

**Genuinely implemented and tested** (35+ passing tests as of this writing):
Rollout schema, synthetic perturbation, Fréchet + MMD metrics, EMA, collapse
safeguard, both encoders, the shared World Model Core, `GapAnalyzer` end-to-end
across both modalities, the validation harness's anti-cherry-picking
enforcement, and the PGM hysteresis fit (on synthetic data).

**Scaffolded but not runnable yet**: the HaGRID/EgoHands MediaPipe extraction
step (needs real data + network access this sandbox didn't have) and the CLI's
end-to-end data-loading path. Both have a clearly marked seam in the code
rather than a silent gap.
