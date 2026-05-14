# Data Spec (quick reference)

Full detail: [`TECHNICAL_SPEC.md`](TECHNICAL_SPEC.md) Section 5. This file is the
fast-lookup version for when you're mid-implementation and don't want to
re-read the whole spec.

## Perception feature layout (spec 5.1.1)

Fixed order — see `PERCEPTION_FEATURE_LAYOUT` in `src/worldgap/data/rollout.py`,
the single source of truth. Never rely on positional convention elsewhere.

| Group | Landmarks | Dims | Column range |
|---|---|---|---|
| Pose | 33 | x, y, z, visibility | 0–131 |
| Left hand | 21 | x, y, z | 132–194 |
| Right hand | 21 | x, y, z | 195–257 |

Total: 258. `PERCEPTION_STATE_DIM` in `rollout.py` enforces this.

## Normalization (spec 5.2)

- Pose: translate relative to hip-midpoint, scale by shoulder width.
- Each hand: translate relative to wrist, scale by hand bounding-box diagonal.
- Store the normalization parameters in `metadata`, not just the normalized
  values — raw values must stay recoverable.

## Storage (spec 5.3)

```
data/
  raw/                    # gitignored, populated by scripts/download_datasets.sh
  processed/
    perception/{rollout_id}.npz
    actuation/{rollout_id}.npz
  index.db                # metadata index (not yet implemented -- see ROADMAP)
```

`Rollout.save()` / `Rollout.load()` in `rollout.py` implement the per-file
part of this; the SQLite metadata index is a Phase 1 item.

## Canonical gesture subset (spec 5.4)

`CANONICAL_GESTURES` in `hagrid.py` is a **placeholder** — confirm against
HaGRID's real class list and the ForceHand glove's actual controllable DOFs
during the Phase 0 data audit before trusting it.

## Synthetic perturbations (spec 5.4)

All three in `data/loaders/synthetic_perturb.py`, all seeded and reproducible:

- `inject_tremor` — band-limited noise, 4–6Hz default, position channels only.
- `reduce_range_of_motion` — clips displacement around the per-channel mean.
- `simulate_occlusion` — zeroes a contiguous frame window for a landmark
  subgroup, **and** sets `presence_mask` to 0 for the same window (never
  zero-fill without also masking — edge case 12.1).

## PGM reference curve (spec 5.5)

Digitize published Ogawa et al. (2017) figures via WebPlotDigitizer into
`(pressure, response)` arrays, then fit with
`data/loaders/pgm_actuator.fit_hysteresis_curve` — a two-branch (loading vs.
unloading) polynomial fit, not a single monotonic curve (edge case 12.13).
Always run `check_residual_structure` afterward and don't trust the fit if
`flag_unmodeled_hysteresis` is `True`.
