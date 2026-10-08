# worldgap

[![PyPI](https://img.shields.io/pypi/v/worldgap.svg)](https://pypi.org/project/worldgap/)
[![CI](https://github.com/Mattral/Worldgap/actions/workflows/ci.yml/badge.svg)](https://github.com/Mattral/Worldgap/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/Worldgap/blob/main/notebooks/demo.ipynb)

Reusable world-model-based domain-gap quantification — for perception pipelines and
actuator/mechanism models — before any hardware is touched.

Ground truth for this project's design is [`docs/TECHNICAL_SPEC.md`](docs/TECHNICAL_SPEC.md).
Read that before changing architecture, data schemas, or metric definitions.

## What this is

Given two sets of rollouts (a source domain and a target domain — e.g. clean-lighting
hand-tracking data vs. occluded/low-light data, or a simulated actuator vs. a published
real-actuator characterization curve), `worldgap` trains a shared world model, encodes
both domains into a common latent space, and reports a divergence score that is designed
to *predict* real-world transfer degradation — validated against independently measured
ground truth, not asserted.

One core library. Two current use cases, distinguished only by which encoder plugs in:

- **V1 — perception gap**: MediaPipe landmark sequences. No hardware required.
- **V2 — actuation gap**: pneumatic gel muscle (PGM) pressure/response sequences, using
  a published characterization curve as the "real" reference. No hardware required.
- **V3 — closed loop** (not started, contingent on lab access): swaps in live logged
  telemetry from real hardware. Same core code, new data loader only — see spec Section 15.

## Status

Be suspicious of libraries that don't say this plainly, so: **nothing here has been
validated against real measured ground truth.** V2's numbers come from data
digitized off a published figure. V1 has now been run once on real recordings
against spec 8.1's independent ground truth, and that pre-registered test came
out **null** (below). The word "validated" is reserved for a test that passes.

What *is* real:

- Core library, CLI, report generation and demo notebook — implemented and tested
  end-to-end on synthetic and local data. CI runs the full suite on Python
  3.10–3.12 with the `perception` extra installed, so the V1 video/MediaPipe path
  is exercised too (against a faked landmarker, not a real model bundle).
- **Bundled real reference data** — Ogawa et al. (2017) Figure 4(a) digitized into
  `Length(Force)` curves at all 7 tested supply pressures, shipped in the wheel and
  loaded via `load_ogawa2017_fig4a_curve()`, with the digitization's own noise floor
  recorded per curve and an independent cross-check against the paper's separately
  stated Figure 6 numbers. Plus Thakur et al. (2018)'s fitted force–pressure
  equations, which refuse to extrapolate outside the 50–300 kPa the paper measured.
- A **simulated PGM actuator baseline** (`pgm_sim.py`) and a V2 end-to-end path that
  compares it against those digitized curves — see `docs/v2_actuation_runbook.md`.

**First real V1 run (2026-10-01): null result.** On the author's own webcam
recordings across 10 pre-registered conditions, the gap score did not predict
MediaPipe hand dropout (Spearman ρ = −0.22, 95% CI [−0.81, 0.54]). Two reasons,
both found afterwards: the conditions barely made MediaPipe fail (take-to-take
noise was as large as the difference between conditions), and spec 5.2's
landmark normalization had not been implemented. Full write-up:
[`docs/v1_first_run_results.md`](docs/v1_first_run_results.md). (HaGRID is a
still-image dataset and cannot supply V1's trajectories — see
`docs/temporal_provenance.md`.)

A note on what the bundled PGM data covers: Ogawa et al. (2017) Section 4 states the
characterized actuator is a 300 mm walking-assist-scale muscle and explicitly says
that length is *not* suitable for hand or wrist assistance. Applying this reference
data to a hand-scale device is an extrapolation across a scale the source paper says
does not carry over — `worldgap` says so rather than letting the number travel
silently.

See [`ROADMAP.md`](ROADMAP.md) for phase-by-phase status and
[`CHANGELOG.md`](CHANGELOG.md) for what's landed.

## Install

From PyPI:

```bash
pip install worldgap                 # core: torch + the World Model; V1 and V2 analysis, V2 data
pip install "worldgap[perception]"   # + MediaPipe/OpenCV, to turn video into V1 landmark rollouts
```

From a clone (for development, or to run `scripts/`):

```bash
pip install -e ".[dev]"              # + test tooling
pip install -e ".[perception,dev]"   # what CI installs
```

V2 needs nothing beyond the core install: the simulated actuator is an ideal
McKibben model in numpy/scipy, and the digitized reference data ships in the
wheel. (0.1.0 had an `actuation` extra pulling in MuJoCo; nothing used it, and
it was removed.)

On Linux, `perception` also needs the system libraries `libEGL.so.1` and
`libGLESv2.so.2` (Debian/Ubuntu: `sudo apt-get install libegl1 libgles2`).

## Quickstart

The library API works with any `Rollout` objects you construct yourself. Producing
perception rollouts from video needs the `perception` extra and MediaPipe's
`holistic_landmarker.task` model bundle (see `docs/v1_real_data_runbook.md`);
V2 rollouts come straight from `worldgap.data.loaders.pgm_sim`.

```python
from worldgap import GapAnalyzer
from worldgap.config import GapConfig

config = GapConfig(modality="perception")
analyzer = GapAnalyzer(config)
analyzer.fit(train_rollouts)
result = analyzer.compute_gap(source_rollouts, target_rollouts)
print(result.frechet.distance, result.confidence)
```

See [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for a runnable end-to-end example
(no downloads needed) covering both modalities, report generation, and the
validation harness — on synthetic data, plus a final part on the bundled digitized
PGM reference data.

For the two real-data paths, see [`docs/v2_actuation_runbook.md`](docs/v2_actuation_runbook.md)
(runs today, reproducible) and [`docs/v1_real_data_runbook.md`](docs/v1_real_data_runbook.md)
(needs recordings).

**Upgrading from 0.2.0 (unreleased changes on `main`):** the perception loaders
now normalize landmarks by default with a shoulder anchor, so **the same video
yields different states** than in 0.2.0. Stores, checkpoints and gap scores from
0.2.0 are not comparable with new ones, and `GapAnalyzer` refuses to mix them.
Re-extract, or apply `normalize_rollout()` to saved unnormalized rollouts, then
re-fit. See [`CHANGELOG.md`](CHANGELOG.md).

**Upgrading from 0.1.0:** saved checkpoints will not load (the predictor changed) —
retrain.

## CLI

```bash
worldgap train    --modality perception --data-dir ./data/processed --config configs/v1_default.yaml
worldgap analyze  --source ./data/clean --target ./data/perturbed --modality perception --output report.html
worldgap validate --gap-scores results.csv --ground-truth degradation.csv
```

`--data-dir`/`--source`/`--target` are each a self-contained rollout store
(`{dir}/index.db` + `{dir}/{modality}/*.npz`, built with `Rollout.save()` +
`RolloutIndex.add()` — see `tests/test_index.py`). See `src/worldgap/cli.py`'s module
docstring for why this differs slightly from spec 5.3's single-shared-index diagram.

## Why not a webapp

This is infra meant to be dropped into someone else's pipeline, not a hosted service.
See spec Section 9 for the full API/CLI contract.

## License

MIT — see [`LICENSE`](LICENSE).
