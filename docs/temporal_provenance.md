# Temporal provenance: where a rollout's time axis came from

Status: **resolved design decision**, extending `TECHNICAL_SPEC.md` §5.1 and
§5.4. Recorded here rather than left implicit because it changes what V1's real
data source can be.

## The problem this fixes

`Rollout` is a timestamped sequence. Everything downstream assumes that:

- the world model's JEPA objective predicts a **future window** from a
  **context window** — it is a model of how a trajectory continues;
- `inject_tremor()` takes a frequency in Hz, which only means something if
  consecutive rows are consecutive moments;
- `simulate_occlusion()` and real MediaPipe dropout are interesting because
  they come in **runs** — a hand lost for half a second is a different control
  problem from the same total dropout scattered one frame at a time.

HaGRID — the dataset the spec named as V1's primary source — is, by its own
expanded name, the "HAnd Gesture Recognition **Image** Dataset". Roughly
554,800 unrelated still photographs, ~30,000 per class, across 18 classes,
from different subjects, scenes and sessions. There is no video in it and no
trajectories.

Releases up to 0.1.0 shipped `list_hagrid_sequences()`, which globbed a gesture
folder and returned the individual image files, next to an
`extract_rollout_from_frames()` that would consume that list as consecutive
frames. Nothing connected the two — but nothing stopped them being connected
either, and the obvious next line of code does exactly that.

The result would have been an object shaped exactly like a trajectory, whose
time axis is `sorted(glob())`. Filename order. The model would have trained
without error. The tremor sweep would have produced monotonic numbers. Every
one of them would have been measuring an artifact of the filesystem, and
nothing in the output would have said so.

That is the failure mode this project's whole ethos is against: not a crash, a
confidently wrong number.

## The decision

**Every `Rollout` records where its time axis came from**, in
`metadata["temporal_provenance"]`. `source` already says where the *values*
came from (`real` / `sim` / `synthetic`); this is the orthogonal question about
the *ordering*, and it was the one silently unanswered.

| Provenance | Meaning | May train the world model? |
|---|---|---|
| `video` | Consecutive samples of one continuous recording at the stated `frame_rate_hz`. | **Yes** — this is the real case. |
| `simulated` | Produced by a simulator or analytic model (e.g. the V2 PGM baseline). Time is real within the model's terms. | Yes. |
| `static_pose` | One still image, `T == 1`. No time axis exists. | No — there is no future window to predict. |
| `synthetic_from_static` | A trajectory built from a real static pose by applying a known motion model. Real pose, invented dynamics. | Technically, but any result is a statement about the motion model, not the world. |
| *(undeclared)* | Legacy or hand-constructed. Treated as **unknown, therefore unsafe**. | Warns. |

Enforced in three places, so it is a property of the code rather than of this
document:

1. `Rollout.__post_init__` rejects `static_pose` with `T != 1` — you cannot
   stack stills and label the result a static pose.
2. `hagrid.extract_rollout_from_frames()` **raises**. Building one trajectory
   out of HaGRID stills is not something this library will do quietly. The
   honest operation, `extract_static_pose_rollouts()`, sits next to it.
3. `GapAnalyzer.fit()` raises when every rollout *declares* a non-temporal
   provenance, and warns when none declare anything. The asymmetry is
   deliberate: a declared `static_pose` is a caller stating in writing that the
   ordering is not observation order, while an undeclared rollout is just code
   written before this field existed.

`list_hagrid_sequences()` is kept as a deprecated alias that warns, because the
old name asserted something false about its own return value.

## What V1's real data source actually is

The important practical consequence: **a real V1 run needs no dataset download
at all.**

A webcam plus `worldgap.data.loaders.video` produces genuinely measured
trajectories. Record the gesture set twice — once in good light, once in the
deployment-like condition you care about (dim room, partial occlusion, hand at
the edge of frame) — and you have a source/target pair of real recordings, from
the same person and camera, differing in exactly the variable under study.

For this specific claim that is a *stronger* artifact than HaGRID could
provide, not a weaker fallback: it contains real temporal degradation, the
conditions are controlled rather than incidental, and the ground truth (§8.1)
comes out of MediaPipe's own reporting on the very same frames.

Recommended sources, in order:

1. **Own recordings** (`video.py`) — available immediately, real temporal
   structure, controlled conditions. The primary source.
2. **EgoHands** — video-derived frames with real occlusion, if a second
   independent source is wanted.
3. **HaGRID** — retained for the frame-level distribution comparison below, and
   for its breadth of subjects. Not a trajectory source.

Because the divergence metrics need `n ≳ 5 × latent_dim` rollouts (§7.3),
`video.split_into_windows()` cuts a long recording into several. Its honest
caveat travels with the data: windows from one recording share a subject,
session and camera, so they are **not independent samples** and the resulting
confidence flag is optimistic. Each window carries
`metadata["windows_are_not_independent_samples"] = True`. Use several separate
recordings too.

## What HaGRID is still good for

`extract_static_pose_rollouts()` gives one `T = 1` rollout per image. With
those you can compare the **frame-level landmark distribution** between two
image sets: does the pose estimator land in a different part of landmark space
under these conditions?

That is a real, useful measurement. It is also a strictly weaker claim than
V1's headline one, and should never be reported as if it were the same thing.
It says nothing about how tracking degrades over time, because a photograph
cannot.

## Ground truth for validation (§8.1)

`video.landmark_quality_ground_truth()` computes the independent degradation
signals §8.1 requires, directly from MediaPipe's own output — never touching
the encoder under test:

- `hand_dropout_rate` — fraction of frames with neither hand detected;
- `longest_dropout_run_s` — reported separately from the rate, because two
  brief blips and one long blackout can share a rate and be very different to
  control through;
- `mean_pose_visibility` — MediaPipe's own per-landmark confidence;
- `any_dropout_rate` — any missing block.

These are what a deployed confidence-threshold safety layer thresholds on, which
is what makes them the right ground truth rather than merely an available one.

## Open question this does *not* settle

Whether `CANONICAL_GESTURES = {fist, palm, stop, like}` is the right match for a
given glove's real controllable DOFs is a domain-expertise call, not a data one,
and remains open (ROADMAP Phase 0 / Q11).
