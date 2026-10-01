# worldgap

[![PyPI](https://img.shields.io/pypi/v/worldgap.svg)](https://pypi.org/project/worldgap/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Mattral/worldgap/blob/main/notebooks/demo.ipynb)

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
validated against real measured ground truth yet.** Every gap number this repo has
produced so far came from synthetic data or from data digitized off a published
figure. The word "validated" is reserved for when spec 8.1's independent ground
truth actually exists.

What *is* real:

- Core library, CLI, report generation and demo notebook — implemented and tested
  end-to-end (`pytest`: see the badge/CI), on synthetic and local data.
- **Bundled real reference data** — Ogawa et al. (2017) Figure 4(a) digitized into
  `Length(Force)` curves at all 7 tested supply pressures, shipped in the wheel and
  loaded via `load_ogawa2017_fig4a_curve()`, with the digitization's own noise floor
  recorded per curve and an independent cross-check against the paper's separately
  stated Figure 6 numbers. Plus Thakur et al. (2018)'s fitted force–pressure
  equations, which refuse to extrapolate outside the 50–300 kPa the paper measured.
- A **simulated PGM actuator baseline** (`pgm_sim.py`) and a V2 end-to-end path that
  compares it against those digitized curves — see `docs/v2_actuation_runbook.md`.

What is *not* done: a real V1 run (HaGRID/EgoHands frames through MediaPipe on a
machine with dataset + model-bundle access — `docs/v1_real_data_runbook.md` and
`scripts/run_v1_real_data.py` are written and waiting for that machine), and any
validation against real MediaPipe confidence/dropout.

A note on what the bundled PGM data covers: Ogawa et al. (2017) Section 4 states the
characterized actuator is a 300 mm walking-assist-scale muscle and explicitly says
that length is *not* suitable for hand or wrist assistance. Applying this reference
data to a hand-scale device is an extrapolation across a scale the source paper says
does not carry over — `worldgap` says so rather than letting the number travel
silently.

See [`ROADMAP.md`](ROADMAP.md) for phase-by-phase status and
[`CHANGELOG.md`](CHANGELOG.md) for what's landed.

## Install

```bash
pip install -e .                 # core: torch + the World Model, works for both modalities
pip install -e ".[perception]"   # + MediaPipe, for V1 data loading (HaGRID/EgoHands)
pip install -e ".[actuation]"    # + MuJoCo, for V2 data loading/simulation
pip install -e ".[dev]"          # test tooling
```

On Linux, `perception` also needs the system libraries `libEGL.so.1` and
`libGLESv2.so.2` (Debian/Ubuntu: `sudo apt-get install libegl1 libgles2`).

## Quickstart

The library API works with any `Rollout` objects you construct yourself — the note
below only applies to *producing* rollouts from raw HaGRID/EgoHands data, which still
needs MediaPipe + real downloads (see ROADMAP Phase 0/1).

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
(synthetic data, no external dependencies) covering both modalities, report
generation, and the validation harness.

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
