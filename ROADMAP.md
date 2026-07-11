# Roadmap

Phases per [`docs/TECHNICAL_SPEC.md`](docs/TECHNICAL_SPEC.md) Section 16. Checkboxes
reflect actual status, not aspiration — update this file every session, not just
at the end of a phase.

## Phase 0 — Data audit
- [x] Confirm HaGRID canonical-gesture-subset sample counts are sufficient —
      resolved without needing Kaggle access: all four names in
      `CANONICAL_GESTURES` are confirmed real HaGRID v1 class names, and the
      dataset's own paper (Kapitanov et al. 2022, arXiv:2206.08219) states
      each of the 18 classes contains 30,000+ images. Still open, and this
      part *does* need real data: whether these four gestures are the best
      semantic match to the ForceHand glove's actual DOFs.
- [ ] Confirm Ogawa et al. (2017) is accessible (open access, or via institutional
      library) — flagged in spec Section 14 as the one real external dependency.
      Checked so far: it's published in *Advanced Robotics* (Taylor & Francis,
      subscription), and the only copy found online is a ResearchGate
      "Request PDF" listing, not an open one — likely needs institutional
      access or emailing the Kurita Lab directly.

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
- [x] SQLite metadata index (spec 5.3) — `data/index.py`, tested
      (documented decision: indexed per rollout-store directory, not one
      shared repo-wide index — see `cli.py` module docstring)

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
- [x] CLI (`worldgap train/analyze/validate`) — wired end-to-end against local
      rollout stores; verified via `tests/test_cli.py` (train->analyze
      round trip, empty/missing-store errors, validate join + anti-cherry-pick
      rejection) and against the real installed console-script entry point.
      Still out of scope: producing rollout stores from raw HaGRID/EgoHands
      frames in the first place (Phase 1's MediaPipe blocker)
- [x] Demo notebook (`notebooks/demo.ipynb`) — executes top-to-bottom via
      `jupyter nbconvert --execute` with zero manual intervention (acceptance
      criterion, spec Section 13); covers V1, the V1/V2 reusability claim, and
      the validation harness's anti-cherry-picking rejection, entirely on
      synthetic data
- [x] Report generation (spec 9.3) — `report.py`, HTML and Markdown output,
      tested; includes the spec-210 Frechet/MMD rank-disagreement diagnostic

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

**Genuinely implemented and tested** (71 passing tests as of this writing):
Rollout schema, SQLite metadata index, synthetic perturbation, Fréchet + MMD
metrics, EMA, collapse safeguard, both encoders, the shared World Model Core,
`GapAnalyzer` end-to-end across both modalities (including checkpoint
save/load), the validation harness's anti-cherry-picking enforcement, the PGM
hysteresis fit (on synthetic data), HTML/Markdown report generation, the full
CLI (`train`/`analyze`/`validate`) against local rollout stores, and a demo
notebook that runs top-to-bottom on synthetic data with zero manual steps.

**Still blocked, not skipped for convenience**: the HaGRID/EgoHands MediaPipe
extraction step (needs real downloaded frames + network access this sandbox
doesn't have) and digitizing the real Ogawa et al. (2017) curve (needs a
likely-paywalled Taylor & Francis figure). Both have a clearly marked seam in
the code rather than a silent gap. Everything else in Phase 5 that doesn't
require those two inputs is done.
