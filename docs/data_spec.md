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

- Pose: translate relative to the anchor midpoint of the study's scheme
  (default `shoulder_midpoint`, landmarks 11/12; spec-faithful
  `hip_midpoint`, 23/24), scale by shoulder width.
- Each hand: translate relative to wrist, scale by hand bounding-box diagonal.
- Store the normalization parameters in `metadata`, not just the normalized
  values — raw values must stay recoverable.

Implemented in `data/normalization.py` (`normalize_rollout`,
`denormalize_states`) and applied by every loader that creates real
perception rollouts. Parameters are per frame, in
`metadata["normalization"]["per_frame"]` as JSON lists (`pose_origin`,
`pose_scale`, `left_hand_origin`, ...); `split_into_windows` slices them to
each window. Visibility and presence are never changed. Hand-built synthetic
rollouts are not normalized automatically: call `normalize_rollout()` if you
want them comparable with real ones.

**One scheme per study.** The default anchors on the shoulders because, in
seated webcam framing, MediaPipe barely sees the hips (mean visibility 0.005
vs. 0.999 for the shoulders in the first run). The scheme is recorded in
`metadata["normalization"]["scheme"]`, never switched per frame, and
`GapAnalyzer` refuses to train on or compare rollouts whose schemes differ
(spec 5.2 deviation note).

## Storage (spec 5.3)

```
data/
  raw/                    # gitignored, populated by scripts/download_datasets.sh
  processed/
    perception/{rollout_id}.npz
    actuation/{rollout_id}.npz
  index.db                # SQLite metadata index (RolloutIndex, data/index.py)
```

`Rollout.save()` / `Rollout.load()` in `rollout.py` implement the per-file
part of this; `RolloutIndex` in `data/index.py` is the metadata index. The CLI
treats each `--data-dir`/`--source`/`--target` as its own self-contained store
(`{dir}/index.db` + `{dir}/{modality}/*.npz`) — see spec 9.2's implementation
note.

## Temporal provenance (spec 5.1, 5.4)

Every rollout records where its *time axis* came from in
`metadata["temporal_provenance"]`: `video`, `simulated`, `quasi_static_sweep`,
`static_pose` or `synthetic_from_static`. `static_pose` rollouts must have one
frame. V1's real trajectories come from video (including your own webcam), not
HaGRID, which is a still-image dataset. Full rationale:
[`temporal_provenance.md`](temporal_provenance.md).

## Canonical gesture subset (spec 5.4)

`CANONICAL_GESTURES` in `hagrid.py` (`fist`, `palm`, `stop`, `like`) are all
confirmed real HaGRID v1 class names, and the V1 runbook's recording routine
uses the same four. Whether they match the ForceHand glove's actual
controllable DOFs is still open — treat the set as a **placeholder** for that
purpose.

## Synthetic perturbations (spec 5.4)

All three in `data/loaders/synthetic_perturb.py`, all seeded and reproducible:

- `inject_tremor` — band-limited noise, 4–6Hz default, position channels only.
- `reduce_range_of_motion` — clips displacement around the per-channel mean.
- `simulate_occlusion` — zeroes a contiguous frame window for a landmark
  subgroup, **and** sets `presence_mask` to 0 for the same window (never
  zero-fill without also masking — edge case 12.1).

## PGM reference data (spec 5.5)

What ships in the wheel, with full provenance in
[`pgm_reference_data.md`](pgm_reference_data.md):

- **Ogawa et al. (2017) Fig. 4(a), digitized** — `Length(Force)` at each of the
  7 tested pressures (0–0.3 MPa), via `load_ogawa2017_fig4a_curve(p)`.
  Isotonic-smoothed, with a per-curve `digitization_noise_floor_mm` (a
  heuristic floor, not a confidence interval). `.length_at(F)` refuses to
  extrapolate.
- **Thakur et al. (2018) force–pressure equations** —
  `thakur2018_force_from_pressure(kPa, stretched)`, valid 50–300 kPa only.

Keep the two papers' numbers separate: they measure different dependent
variables under different held-constant conditions.

V2 state layout is `[pressure_mpa, force_n, length_mm]`, normalized by the
fixed `ACTUATION_NORMALIZATION` constants (never fitted from either domain).
`pgm_sim.simulated_pgm_rollout(p)` and `digitized_pgm_rollout(p)` build the two
sides.

`fit_hysteresis_curve` (two-branch loading/unloading fit, edge case 12.13) is
for **pressure-ramp** hysteresis data, which neither paper publishes, so it is
validated on synthetic data only. Do **not** feed it the Fig. 4(a)
`Length(Force)` curves — they are a different hysteresis axis. If you do use it
on real ramp data, run `check_residual_structure` afterward and don't trust the
fit if `flag_unmodeled_hysteresis` is `True`.
