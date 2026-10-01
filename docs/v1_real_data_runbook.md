# V1 real-data runbook

How to produce worldgap's first result on real measured data, on your own
machine, in about an hour. No dataset download required.

Read `temporal_provenance.md` first if you have not — it explains why the
source here is video rather than HaGRID, and that choice is the reason this
runbook exists in its current form.

---

## What this produces

- Real landmark trajectories from real recordings, in worldgap's rollout stores.
- A gap score per capture condition, with Fréchet and MMD both reported.
- Ground truth from MediaPipe's own output (dropout rate, dropout run length,
  pose visibility) — independent of the model under test, as §8.1 requires.
- If you record ≥ 10 conditions: a pre-registered Spearman correlation with a
  bootstrap CI. Under 10, the script reports the gap scores and **skips** the
  correlation rather than running it underpowered.

Everything below has been exercised end-to-end in the test suite
(`tests/test_v1_script_smoke.py`) with real video files and a faked
landmarker. The one step never executed anywhere is a real
`HolisticLandmarker` over real frames, because that needs the model bundle.
Watch the first run; don't trust it blindly.

---

## Step 0 — install and preflight

```bash
pip install -e ".[perception,dev]"
mkdir -p models
curl -L -o models/holistic_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task

python scripts/check_mediapipe_setup.py --model models/holistic_landmarker.task
```

The preflight must print all `ok` before you record anything.

**The one failure worth knowing about in advance.** `mediapipe` 0.10.x does
**not** ship `vision.HolisticLandmarker` — those wheels export only
`PoseLandmarker` and `HandLandmarker`, and the legacy `mp.solutions` API is
gone as well. Every V1 loader here is written against the holistic task, so
0.10.x fails with an `AttributeError` partway in. The `perception` extra pins
`mediapipe>=1.0` for this reason. If you already have 0.10.x installed:

```bash
pip install -U "mediapipe>=1.0"
```

**On Linux**, mediapipe's native library links `libEGL.so.1` and
`libGLESv2.so.2`, which headless machines and slim Docker images often lack.
The symptom is `OSError: libEGL.so.1: cannot open shared object file` on the
first image call. On Debian/Ubuntu: `sudo apt-get install libegl1 libgles2`.
Windows and macOS need nothing extra.

If the downloaded `.task` file is a few kilobytes, it is an HTML error page
with a `.task` name. The preflight catches that too.

---

## Step 1 — record

Directory layout. One subfolder per capture condition; `clean/` is the source
domain, everything else is a target condition:

```
recordings/
  clean/         take0.mp4  take1.mp4  take2.mp4
  dim_light/     take0.mp4  ...
  backlit/
  hand_partially_out_of_frame/
  ...
```

**What to record.** The same short gesture routine every time — the four
canonical gestures (`fist`, `palm`, `stop`, `like`), held a second or two each,
then repeat. 60–90 seconds per take. Three takes per condition.

The routine must be identical across conditions. That is the whole
experimental design: the only thing differing between `clean/` and a target
folder should be the deployment variable under study. If you also change the
gesture set, you are measuring "different gestures" and calling it "worse
conditions".

**Conditions worth capturing**, roughly in order of how much they resemble
real deployment failure:

| Condition | How |
|---|---|
| `dim_light` | Lights off, screen glow only |
| `backlit` | Window or lamp behind you |
| `side_light` | Single lamp at 90° — hard shadows across the hand |
| `hand_partial_occlusion` | Other hand or an object crossing the tracked one |
| `hand_edge_of_frame` | Hand at the frame border, entering and leaving |
| `far_from_camera` | 2–3 m instead of arm's length |
| `fast_motion` | Same gestures, roughly twice the speed |
| `cluttered_background` | Busy background rather than a wall |
| `low_resolution` | Downscale the recording afterwards (`ffmpeg -vf scale=320:-1`) |
| `camera_shake` | Handheld rather than mounted |

Ten conditions is not an arbitrary target — it is §8.3's minimum for the
correlation to be computed at all. Nine conditions gets you gap scores and no
validation. It is worth the extra twenty minutes of recording.

**Recording hygiene.** Fix the camera for everything except `camera_shake`.
Same clothing, same position, same time of day if you can. Record `clean/`
first and last, so if something drifts across the session you can see it.

**A note on participants.** If you ever record anyone other than yourself,
that is human-subjects data and needs whatever consent and ethics process your
institution requires, before recording, not after. Recordings stay off the
repository regardless (§12.16 and `.gitignore`).

---

## Step 2 — dry run

```bash
python scripts/run_v1_real_data.py --recordings ./recordings --dry-run
```

Prints the conditions it found and the plan, without loading MediaPipe. Check
the list matches what you recorded — a typo'd folder name silently becoming its
own "condition" is the easy mistake here.

---

## Step 3 — run

```bash
python scripts/run_v1_real_data.py \
  --recordings ./recordings \
  --model ./models/holistic_landmarker.task \
  --out ./v1_run
```

On CPU, expect a few minutes per minute of footage for extraction, plus a
couple of minutes to fit. A GPU is not needed at this scale.

Outputs under `./v1_run/`:

| File | What it is |
|---|---|
| `preregistered_conditions.json` | Written **before** any score is computed (§8.3) |
| `stores/<condition>/` | Self-contained rollout stores — reusable by the CLI |
| `ground_truth.json` | MediaPipe's own quality signals, per recording |
| `checkpoint.pt` | The fitted world model |
| `v1_report.html` | Gap scores, Fréchet/MMD trend, confidence flags |
| `validation.json` | Spearman ρ + bootstrap CI (only if ≥ 10 conditions) |

The stores are ordinary worldgap stores, so the CLI works on them directly:

```bash
worldgap analyze --source ./v1_run/stores/clean \
                 --target ./v1_run/stores/dim_light \
                 --modality perception \
                 --checkpoint ./v1_run/checkpoint.pt \
                 --output dim_light.html
```

---

## Step 4 — read the output honestly

*The first real run's results, read against these four checks, are in
[`v1_first_run_results.md`](v1_first_run_results.md).*

Four things to check before believing any number.

**1. Did the collapse safeguard fire?** The script stops if it did, and it is
right to. A collapsed world model maps everything to nearly the same latent,
which makes every gap score small and meaningless. That is a training failure,
not a finding of "no gap".

**2. What is the confidence flag?** `low` means fewer than 5 × latent_dim
rollouts (§7.3) and the Fréchet covariance estimate is not trustworthy.
`split_into_windows()` raises the count, but windows from one recording share a
subject, session and camera — they are **not independent samples**, each one
carries `windows_are_not_independent_samples` in its metadata, and the
confidence flag computed over them is optimistic. More separate recordings is
the real fix; more windows is not.

**3. Do Fréchet and MMD agree on the ordering?** When they disagree, the report
says so. That disagreement is a signal about non-Gaussian latent structure
(§7.2), not noise to average away.

**4. If the CI includes zero, that is the result.** It means the gap score did
not predict measured degradation across these conditions. Report it. Re-running
with different conditions until the interval excludes zero is exactly the
cherry-picking the pre-registration exists to prevent — and the harness will
refuse a trimmed condition set.

---

## What this run does and does not establish

**Does**: that the gap score, computed on real measured trajectories, does or
does not track a real independently-measured degradation signal across
conditions you fixed in advance.

**Does not**: anything about a different camera, a different person, or a
different device. One subject on one camera is one experiment. Say "on my own
recordings, across N pre-registered conditions" and the result is solid; drop
that clause and it is overclaiming.

It also establishes nothing about actuation — that is V2, see
`v2_actuation_runbook.md`.

---

## Optional: HaGRID, for what it is actually good for

HaGRID is an image dataset and cannot supply trajectories (see
`temporal_provenance.md`). It can supply a **frame-level landmark distribution**
comparison, which is a real but weaker measurement:

```python
from worldgap.data.loaders.hagrid import list_hagrid_images, extract_static_pose_rollouts

images = list_hagrid_images(Path("./hagrid"))
rollouts = extract_static_pose_rollouts(images, landmarker)   # T = 1 each
```

These cannot train the world model — `GapAnalyzer.fit()` refuses them, because
a JEPA objective needs a future window and a photograph has none. Encode them
with a model fitted on video. And report the result as what it is: a statement
about where the pose estimator lands in landmark space, not about how tracking
degrades over time.

Check HaGRID's own licence terms before redistributing anything derived from
it (open question Q13).
