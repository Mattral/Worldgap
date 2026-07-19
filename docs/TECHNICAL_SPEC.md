# Technical Specification: World-Model-Based Domain-Gap Quantification Tool

**Status**: Draft v0.1 — ground truth for implementation
**Owner**: Min (Mattral)
**Scope**: V1 (perception gap) and V2 (actuation gap), with explicit forward-compatibility notes for V3 (closed-loop, real hardware)
**Convention**: Requirements use MUST / SHOULD / MAY per RFC 2119 — MUST is non-negotiable for a version to be considered complete, SHOULD is a strong default that can be consciously overridden and documented, MAY is optional.

---

## Table of Contents

1. Project Summary & Goals
2. Scope & Non-Goals
3. System Overview
4. Glossary
5. Data Specification
6. Model Specification
7. Divergence Metric Specification
8. Validation Harness Specification
9. Library & CLI API Specification
10. Repository Structure
11. Tech Stack
12. Edge Cases & Failure Modes
13. Acceptance Criteria
14. Risks & Open Dependencies
15. V3 Forward-Compatibility Notes
16. Milestone Phases
17. References

---

## 1. Project Summary & Goals

This project builds a **reusable, embodiment-agnostic tool that quantifies the gap between two data domains before any policy or control system is deployed on real hardware**. The original framing was sim-vs-real for robot control policies. This spec targets a more specific and more immediately buildable version: quantifying the gap between **controlled-condition hand/pose tracking data** and **target-deployment-condition data** (home lighting, occlusion, motor-impairment-like motion), as a preparatory risk-quantification instrument for vision-based rehabilitation exoskeleton control — directly relevant to vision-based non-contact gesture control for pneumatic soft exoskeletons (ForceHand-style glove, PGM actuation per Ogawa et al. 2017 and Thakur et al. 2018, MediaPipe Pose+Hands front end, confidence-threshold safety layer).

**Primary goal**: build a tool that, given two sets of rollouts (source domain, target domain), outputs a single defensible number (plus supporting diagnostics) predicting how much a downstream system's behavior will degrade moving from source to target — validated against measurable ground truth, without touching robot or exoskeleton hardware.

**Secondary goal**: the same core code MUST work unmodified for a second modality (PGM actuator pressure→response) by swapping only the encoder — proving the "reusable infra" thesis rather than asserting it.

**Non-goal for this spec**: building any part of the actual ForceHand/exosuit control system, gesture-mapping engine, or safety layer for this application domain. This tool produces evidence that could inform that system later; it does not implement it.

---

## 2. Scope & Non-Goals

### In scope (V1)
- Landmark-sequence world model trained on public hand/pose data plus synthetically perturbed variants.
- Fréchet-style + MMD divergence metrics between domains.
- Validation harness correlating divergence against MediaPipe's own confidence/dropout degradation.
- Packaged library + CLI + demo notebook.

### In scope (V2)
- Actuator-response world model using MuJoCo simulation vs. digitized published PGM characterization data (Ogawa et al., 2017) as the "real" reference.
- Same divergence + validation code reused unmodified from V1.

### Explicitly out of scope for V1/V2
- Any physical robot, gripper, or exoskeleton hardware.
- Real patient data collection (no IRB/ethics approval exists or is sought at this stage).
- The gesture-mapping engine, PGM pressure command mapping, or safety layer as functional deliverables (they are referenced only as the downstream consumer this tool's output would eventually inform).
- Hosting, dashboards, or any multi-user service. This is a local library.

### Deferred to V3 (not started, contingent on lab access — see Section 15)
- Live ForceHand glove telemetry as the actuation "real" reference.
- Real deployment-condition perception data (with appropriate ethics approval, owned by the lab).

---

## 3. System Overview

Four modules, in a fixed pipeline, with exactly one module swapped between V1 and V2:

```
[ Encoder (swappable) ] -> [ Shared World Model Core ] -> [ Divergence Module ] -> [ Validation Harness ] -> Gap Score + Report
```

- **Encoder**: V1 = landmark-sequence encoder. V2 = actuator-state-sequence encoder. This is the *only* module that changes between versions.
- **World Model Core**: JEPA-style predictive model (context encoder, EMA target encoder, predictor, latent-prediction loss). Identical code path for both versions; only input/output dimensions differ via config.
- **Divergence Module**: Fréchet-style distance + MMD between latent trajectory distributions of two rollout sets. Identical code path for both versions.
- **Validation Harness**: correlates divergence scores against an independently measured ground-truth degradation signal. Identical code path for both versions; only the ground-truth signal source differs (MediaPipe confidence for V1, actuator residual error for V2).

Any change required outside the encoder to support V2 or (later) V3 MUST be treated as a design defect in V1 and fixed at the architecture level, not patched per-version.

---

## 4. Glossary

- **Rollout**: a single timestamped sequence of states for one task instance (one gesture performed once, or one commanded-pressure trace and its response).
- **Domain A / source**: the better-characterized, more controlled condition (e.g., HaGRID clean subset; simulated actuator).
- **Domain B / target**: the condition being evaluated for transfer risk (e.g., perturbed/occluded data; real PGM characterization curve).
- **Latent trajectory**: the sequence of embeddings produced by the world model's context encoder for a rollout.
- **Gap score**: the scalar (or small vector) output of the Divergence Module for a given (source-set, target-set) pair.
- **Ground truth degradation**: an independently measured quantity (not derived from the world model) that reflects how much a real downstream system actually degrades under the target condition.

---

## 5. Data Specification

### 5.1 Rollout schema (MUST)

```python
@dataclass
class Rollout:
    rollout_id: str                 # content-hashed, stable across reruns
    modality: Literal["perception", "actuation"]
    source: Literal["real", "sim", "synthetic"]
    condition: dict                 # e.g. {"lighting": "low", "occlusion": "partial", "motion_profile": "tremor_5hz"}
    frame_rate_hz: float
    states: np.ndarray               # shape (T, D) — see 5.1.1 for D per modality
    presence_mask: np.ndarray        # shape (T, K) — per-landmark-group or per-channel validity, see 5.4
    timestamps_ms: np.ndarray        # shape (T,)
    metadata: dict                   # free-form: subject_id (if applicable), gesture_label, notes
```

#### 5.1.1 `states` layout per modality

- **Perception (V1)**: per-frame feature vector, D = 258:
  - Pose: 33 landmarks × (x, y, z, visibility) = 132
  - Left hand: 21 landmarks × (x, y, z) = 63
  - Right hand: 21 landmarks × (x, y, z) = 63
  - Feature order MUST be fixed and documented in `data/rollout.py` as a named-index constant, not implied by position alone — this is the single most common source of silent bugs in landmark pipelines.
- **Actuation (V2)**: per-timestep vector, D = 2 to 6 depending on DOF count modeled (commanded pressure + resulting joint angle per DOF, minimum viable = 1 DOF = wrist flexion).

### 5.2 Normalization (MUST, V1)

- Translate landmarks relative to a reference point: hip-midpoint for pose, wrist for each hand (a standard normalization convention for pose-relative landmark features, so results transfer conceptually to other gesture-tracking pipelines using the same convention).
- Scale by a reference length: shoulder width for pose, hand bounding-box diagonal for each hand.
- Normalization parameters (reference point, scale factor) MUST be stored per-frame in metadata, not discarded, so raw values are always recoverable.

### 5.3 Storage layout (MUST)

- One file per rollout: `.npz` containing `states`, `presence_mask`, `timestamps_ms`.
- A single metadata index (SQLite or Parquet) with one row per rollout, columns matching the non-array fields of the schema above, indexed by `rollout_id`.
- Raw source datasets (HaGRID, EgoHands) MUST NOT be redistributed in this repo. A download+preprocess script MUST fetch them from their original hosts at setup time (license compliance — see 12.16).
- Directory convention:
```
data/
  raw/              # gitignored, downloaded by script
  processed/
    perception/
      {rollout_id}.npz
    actuation/
      {rollout_id}.npz
  index.db          # SQLite metadata index
```

### 5.4 V1 data sources & preprocessing

- **HaGRID**: 554,800 crowd-sourced images, 18 gesture classes, natural lighting/scene variation, subject-to-camera distance 0.5–4m. MUST be filtered to a canonical gesture subset relevant to the ForceHand glove's controllable DOFs (grasp open/close, wrist flexion/extension) rather than all 18 classes indiscriminately — using unrelated gesture classes (e.g. "rock", "peace") would compare unrelated motion vocabularies between domains, not a true domain gap. SHOULD stratify by subject-to-camera distance metadata (if available) to avoid conflating "far away" with "target domain," since distance affects landmark noise independent of the gap being measured.
- **EgoHands**: secondary source for occlusion-heavy conditions.
- **Synthetic perturbation pipeline** (MUST implement, since no public dataset captures motor-impairment-like hand motion):
  - *Tremor injection*: band-limited noise (4–6 Hz, matching literature-typical pathological tremor frequency) added to landmark position trajectories, amplitude configurable per condition.
  - *Reduced range of motion*: clip the normalized landmark displacement range to a configurable fraction (e.g. 50%, 70%) of the original.
  - *Occlusion*: zero out a contiguous window of frames for a randomly chosen landmark subgroup (fingers, whole hand) and set the corresponding `presence_mask` to 0 — MUST NOT set the mask to 1 with zeroed values (see 12.1).
  - Every synthetic condition MUST be reproducible from a stored seed + parameter set, not applied ad hoc.

### 5.5 V2 data sources & preprocessing

- **Reference curve**: Ogawa et al. (2017) PGM pressure-response characterization, extracted via WebPlotDigitizer from published figures (raw data is not expected to be available). Digitization MUST record an estimated reading-error bound (e.g., ± pixel-to-unit conversion uncertainty) — this becomes a documented noise floor, not an assumed-exact ground truth.
- **Simulated actuator**: MuJoCo custom actuator (general actuator with custom gain/bias) OR a standalone hysteresis model (Bouc-Wen or Hammerstein-Wiener) fit via `scipy.optimize.curve_fit`, chosen based on which better matches the hysteresis loop shape in the digitized data (MUST check: PGMs are McKibben-type pneumatic muscles and exhibit hysteresis — a naive monotonic fit is a known failure mode, see 12.13).

---

## 6. Model Specification

### 6.1 Landmark encoder (V1)

- Input projection: `Linear(258 -> d_model)`, `d_model = 256` (default, tunable).
- Positional encoding: sinusoidal (fixed, no learned params needed at this scale).
- Backbone: Transformer encoder, 4 layers, 4 heads, feedforward dim 1024, dropout 0.1 (defaults — these are starting points to validate empirically, not fixed requirements).
- Presence mask MUST be passed through as an attention mask / additive bias, not silently dropped, so the model can distinguish "missing" from "value near zero."

### 6.2 Actuator encoder (V2)

- Input projection: `Linear(D_actuation -> d_model)`, `d_model = 64` (smaller — much lower-dimensional signal).
- Backbone: 2-layer TCN or 2-layer Transformer encoder — SHOULD start with the simpler TCN given the low dimensionality, only move to a Transformer if the TCN underfits the hysteresis dynamics in validation.

### 6.3 Shared world model core (JEPA-style, MUST be identical code path for V1/V2)

- **Context encoder**: the modality-specific encoder above, applied to the context window (`T_ctx` frames, default 16 — 0.5s at 30fps for V1; V2 window length is configurable since actuator sampling rates differ).
- **Target encoder**: exponential-moving-average (EMA) copy of the context encoder, applied to the future window (`T_pred` frames, default 8). EMA decay default 0.996 (standard BYOL/JEPA-family value — MUST be tuned if collapse is observed, see 12.7).
- **Predictor**: a small MLP or lightweight transformer taking the context encoder's pooled representation (+ learned mask tokens for future positions) and predicting the target encoder's latent output for each future frame.
- **Loss**: smooth-L1 (Huber) between predicted and target-encoder latents, computed with a stop-gradient on the target branch. Gradient MUST NOT flow into the target encoder directly — only via the EMA update.
- **Latent summary head**: a final linear projection from `d_model` down to a **small summary dimension, default 32–64**, used specifically for the Divergence Module. This dimension is deliberately kept small — see 7.3 for why (sample-size-vs-dimensionality tradeoff in Fréchet distance estimation).
- **Collapse safeguard (MUST)**: track per-batch latent variance across the summary dimensions during training; if mean variance drops below a configurable threshold for more than N consecutive validation checks, flag training as collapsed. A variance-regularization term (VICReg-style) MAY be added if this triggers in practice — don't add it preemptively; add it if the safeguard actually fires.

### 6.4 Training procedure & hyperparameter defaults

- Optimizer: AdamW, lr 3e-4, weight decay 0.01 (defaults, not requirements).
- Batch size: as large as fits on a free-tier T4 — this workload is small enough that batch size is a non-issue, not a bottleneck to plan around.
- Training MUST log latent variance, loss, and EMA decay-effective values to Weights & Biases every epoch, not just final metrics — the collapse safeguard depends on this history existing.

---

## 7. Divergence Metric Specification

### 7.1 Fréchet-style distance (MUST)

For two sets of latent summary vectors (domain A, domain B), estimate mean and covariance and compute:

```
FD(A, B) = ||μ_A - μ_B||^2 + Tr(Σ_A + Σ_B - 2 * sqrtm(Σ_A @ Σ_B))
```

- `sqrtm` (matrix square root) MUST use a numerically stable implementation (`scipy.linalg.sqrtm` with real-part extraction and small-eigenvalue clipping) — floating-point error commonly produces small complex components that MUST be discarded, not treated as an error.
- Covariance estimation MUST use a shrinkage estimator (Ledoit-Wolf, `sklearn.covariance.LedoitWolf`) rather than the naive empirical covariance — with rollout counts in the hundreds rather than the tens of thousands used in image-domain FID, naive covariance estimates will be poorly conditioned or singular (see 12.9).

### 7.2 MMD cross-check (MUST)

- Maximum Mean Discrepancy with an RBF kernel, bandwidth chosen via the median heuristic.
- MMD and Fréchet distance MUST both be reported together, not one in isolation. Fréchet distance assumes approximately Gaussian latents; disagreement between the two metrics is itself a diagnostic signal (non-Gaussian latent structure) that MUST be surfaced in the output, not hidden.

### 7.3 Numerical stability & sample-size requirements (MUST)

- The summary latent dimension is deliberately kept at 32–64 (see 6.3) specifically because Fréchet-distance covariance estimation requires sample count to substantially exceed dimensionality to be well-conditioned. A rule of thumb requirement: **n ≥ 5 × latent_dim per domain**, else the result MUST be flagged `confidence: "low"` in the output rather than reported as a bare number.
- Any `GapResult` MUST include `n_source`, `n_target`, `latent_dim`, and a `confidence` field derived from the above rule — this is not optional metadata, it is required output.

---

## 8. Validation Harness Specification

### 8.1 Ground-truth degradation signals (MUST be independent of the world model)

- **V1**: MediaPipe's own per-frame confidence score and landmark-dropout rate, computed directly from the MediaPipe task output — MUST NOT be derived from, or otherwise depend on, the trained world model's outputs. This independence is what prevents the validation from being circular (see 12.10).
- **V2**: residual error between the simulated actuator's predicted response and the digitized real curve, at matched commanded-pressure setpoints — also independent of the world model.

### 8.2 Correlation methodology (MUST)

- Spearman rank correlation (not Pearson) between gap scores and ground-truth degradation across a set of distinct conditions — rank correlation is appropriate since we care about monotonic risk-ranking, not a linear relationship.
- Report a bootstrap confidence interval on the Spearman ρ (MUST, minimum 1000 resamples), not a bare point estimate.
- **Minimum condition count**: at least 10–15 distinct conditions MUST be evaluated before any correlation claim is made. Fewer than that produces a correlation estimate with too much sampling variance to be defensible.

### 8.3 Anti-circularity requirement (MUST)

- The set of conditions used to validate the metric MUST be pre-specified (a fixed list of (lighting, occlusion, motion-profile) combinations) before running the correlation analysis. Selecting or trimming conditions after seeing results, or reporting only the best-correlating subset, is a multiple-comparisons violation and MUST NOT be done — if this spec is later revisited to relax this, that must be a deliberate, documented decision, not a default.

### 8.4 Reporting (MUST)

- The validation harness's output MUST include: the full table of (condition, gap score, ground-truth degradation) pairs, the Spearman ρ with CI, and an explicit statement of `n_conditions`. A correlation number without this supporting table is not acceptable output.

---

## 9. Library & CLI API Specification

### 9.1 Python API (MUST)

```python
from worldgap import Rollout, GapAnalyzer, GapConfig

config = GapConfig(modality="perception", latent_dim=64, ...)
analyzer = GapAnalyzer(config)
analyzer.fit(train_rollouts)                       # trains or loads the world model
result = analyzer.compute_gap(source_rollouts, target_rollouts)
# result: GapResult(frechet=..., mmd=..., n_source=..., n_target=..., confidence=..., warnings=[...])
```

- `GapAnalyzer` MUST accept `modality: Literal["perception", "actuation"]` and internally select the correct encoder — this is the concrete test of the reusability claim (Section 3). A unit test asserting both modalities run through the identical `GapAnalyzer` class and produce a `GapResult` of the same schema is a MUST, not optional.
- **Implementation note (documented decision, not a deviation)**: `n_source`, `n_target`, and `confidence` above are exposed as read-only properties on `GapResult` that proxy `GapResult.frechet`, rather than being duplicated as separately-stored fields. Callers still write `result.confidence` exactly as shown; the class just doesn't store that value twice. See `src/worldgap/analyzer.py`.

### 9.2 CLI (MUST)

```
worldgap train    --modality perception --data-dir ./data/processed --config configs/v1_default.yaml
worldgap analyze   --source ./data/clean --target ./data/perturbed --modality perception --output report.html
worldgap validate  --gap-scores results.csv --ground-truth degradation.csv
```

**Implementation note (documented decision, not a deviation)**: `--data-dir`/`--source`/`--target` are each treated as a self-contained rollout store (`{dir}/index.db` + `{dir}/{modality}/*.npz`) rather than assuming a single shared repo-wide `data/` root as the 5.3 diagram depicts — this is what lets `analyze` compare two independently-produced stores (e.g. a "clean" directory and a "perturbed" directory) without them needing to share one index. `--data-dir` therefore points at a store root (e.g. `./data/processed`), not a modality-specific subdirectory; `--modality` still selects which rollouts within that store get loaded. The 5.3 diagram remains the right layout for the primary download+preprocess script's own output. See `src/worldgap/cli.py` module docstring.

### 9.3 Report output (SHOULD)

- A generated HTML or Markdown report per `analyze` run, bundling the `GapResult`, the condition table, and Fréchet/MMD trend plots (matplotlib or plotly). This is a generated artifact, not a hosted service — no server component.

---

## 10. Repository Structure

```
worldgap/
  pyproject.toml
  README.md
  src/worldgap/
    __init__.py
    data/
      rollout.py
      loaders/
        hagrid.py
        egohands.py
        synthetic_perturb.py
        pgm_actuator.py
    models/
      encoders/
        landmark_encoder.py
        actuator_encoder.py
      world_model.py
      ema.py
    metrics/
      frechet.py
      mmd.py
    validation/
      harness.py
      stats.py
    cli.py
    config.py
  configs/
    v1_default.yaml
    v2_default.yaml
  scripts/
    digitize_ogawa_curve.py
    download_datasets.sh
  tests/
    test_rollout_schema.py
    test_frechet_numerics.py
    test_modality_swap.py     # asserts V1/V2 share GapAnalyzer unmodified
    test_collapse_safeguard.py
  notebooks/
    v1_demo.ipynb
    v2_demo.ipynb
  docs/
    architecture.md
    data_spec.md
  paper/
    draft.md
```

---

## 11. Tech Stack

| Layer | Choice | Note |
|---|---|---|
| Language | Python 3.11 | |
| ML core | PyTorch | pin exact version in `pyproject.toml`; verify current stable at install time |
| Perception front-end | MediaPipe Tasks API (`HolisticLandmarker`) | confirm exact config parameter names against current docs at implementation time — Google's API surface has moved from legacy `mp.solutions` to the Tasks API; this spec assumes Tasks API |
| Simulation (V2) | MuJoCo (`pip install mujoco`) | |
| Curve fitting | SciPy (`curve_fit`), optionally a small MLP | |
| Covariance estimation | scikit-learn `LedoitWolf` | |
| Digitization | WebPlotDigitizer (manual, browser-based) | |
| Experiment tracking | Weights & Biases | |
| Config | Pydantic v2 | matches moe-engine convention |
| Testing | pytest, hypothesis | |
| Packaging | `pyproject.toml`, src-layout, hatchling | matches KANX convention |
| CI | GitHub Actions | |

All versions MUST be pinned in a lockfile before any result is considered reproducible (see 12.18).

---

## 12. Edge Cases & Failure Modes

This section is intentionally exhaustive. Each item MUST have its stated mitigation implemented before the corresponding version is considered complete — not treated as future work.

**Data / perception**

1. **Missing detection**: MediaPipe fails to detect hand/pose in a frame. MUST mark via `presence_mask = 0`, MUST NOT zero-fill the state vector and mark it present (a zeroed-but-"present" landmark looks like a valid detection at the origin and silently corrupts normalization).
2. **Handedness flip**: occlusion or crossed hands cause left/right misclassification frame-to-frame. MUST apply a temporal-consistency check (handedness shouldn't flip within a short window under normal motion) and flag/exclude rollouts with frequent flips from training.
3. **Multiple people in frame**: relevant for home deployment (caregiver present). MUST explicitly configure and document a selection policy (largest bounding box / closest to camera) rather than relying on undocumented default behavior.
4. **Variable frame rate / dropped frames**: MUST resample to a fixed canonical rate (30fps default) via interpolation; windows with more than 20% interpolated frames MUST be flagged invalid for training/eval, not silently included.
5. **Subject-to-camera distance confound**: HaGRID spans 0.5–4m; distance affects landmark noise independent of the actual domain gap being measured. SHOULD stratify or control for this rather than pooling all distances into one comparison.
6. **Gesture vocabulary mismatch**: comparing HaGRID's 18 general gesture classes against rehab-relevant motions is comparing different tasks, not a true domain gap. MUST restrict to a canonical gesture subset matched to the ForceHand glove's DOFs.

**Modeling**

7. **Representation collapse**: known JEPA/self-supervised failure mode where the predictor learns a trivial constant output. MUST implement the variance-based collapse safeguard (6.3) as an automated check, not a manual inspection.
8. **Domain-specific shortcut features**: the model might key on background-clutter artifacts correlated with a particular dataset rather than task-relevant motion, even at the landmark level (MediaPipe's own detections can carry small systematic biases per background type). SHOULD run an ablation removing/perturbing background-correlated conditions to check sensitivity.

**Statistical / validation**

9. **Small-sample covariance instability**: naive covariance estimation is singular or poorly conditioned when sample count is close to or below latent dimensionality. MUST use Ledoit-Wolf shrinkage and MUST enforce the `n ≥ 5 × latent_dim` confidence rule (7.3).
10. **Circular validation**: if the "ground truth" degradation signal is derived from the same network being evaluated, the correlation is tautological. MUST keep ground-truth signals fully independent (8.1).
11. **Multiple-comparisons / cherry-picking**: reporting only the best-correlating condition subset after the fact overstates significance. MUST pre-register the condition list before running correlation analysis (8.3).
12. **Non-stationarity across subjects**: gesture speed/style varies by subject; pooling everything into one number can hide subject-level effects. SHOULD report stratified results by subject group where subject metadata exists, not only a pooled aggregate.

**V2-specific**

13. **Hysteresis non-monotonicity**: PGMs exhibit different response depending on whether pressure is increasing or decreasing at a given setpoint. A naive monotonic curve fit will systematically misrepresent the real gap. MUST use a hysteresis-aware model and MUST check fit residuals for structured (non-random) error indicating unmodeled hysteresis. **Implementation note (documented decision, not a deviation)**: `pgm_actuator.py` implements this as a two-branch (loading/unloading) polynomial split-fit rather than a full Bouc-Wen or Hammerstein-Wiener model — simpler, and backstopped by `check_residual_structure`'s residual-direction-correlation check, which is specifically designed to catch the case where the simpler model is insufficient. Accepted as-is unless/until real Ogawa et al. data causes `flag_unmodeled_hysteresis=True`, at which point upgrading to Bouc-Wen/Hammerstein-Wiener is required before trusting V2 results.
14. **Digitization error**: reading data points off a published figure via WebPlotDigitizer introduces quantization/reading noise. MUST record and report an estimated digitization uncertainty as a noise floor; gap-score differences smaller than this floor MUST NOT be over-interpreted.
15. **Single-paper generalization risk**: Ogawa et al.'s characterization reflects one specific PGM batch/configuration; manufacturing variance, wear, and temperature will shift real device behavior. MUST document this explicitly as a known limitation of V2's reference data, not a settled ground truth.

**Compliance**

16. **Dataset licensing**: HaGRID/EgoHands usage terms MUST be checked before any derived data or model trained on them is publicly released (PyPI package, GitHub repo). Raw dataset files MUST NOT be redistributed — only a download+preprocess script.
17. **Privacy boundary**: V1/V2 use only public crowd-sourced or synthetic data — no real patient data, no IRB requirement at this stage. This boundary MUST be preserved; any future use of real deployment/patient data (V3) is explicitly the lab's ethics-approval responsibility, not this tool's.

**Reproducibility**

18. **Non-determinism**: MediaPipe internals, CUDA non-deterministic ops, and dataset shuffling can all introduce run-to-run variance. MUST seed all RNGs, MUST pin exact package versions in a lockfile, and MUST report variance across at least 3 seeds for any headline correlation number rather than a single run.

---

## 13. Acceptance Criteria

### V1 complete when:
- [ ] World model trains to convergence without triggering the collapse safeguard on the HaGRID + synthetic-perturbation split.
- [ ] Fréchet + MMD scores are stable across ≥5 seeds (coefficient of variation reported, not assumed small).
- [ ] Validation harness runs on ≥10 pre-registered conditions and reports Spearman ρ with bootstrap CI, plus the full supporting table (8.4).
- [ ] `worldgap` installs cleanly in a fresh environment; demo notebook runs top-to-bottom without manual intervention.
- [ ] CI green: unit tests including `test_frechet_numerics.py` and `test_collapse_safeguard.py`.

### V2 complete when:
- [ ] Digitized Ogawa curve + documented digitization-uncertainty estimate exists.
- [ ] Hysteresis-aware actuator model fit, with residual diagnostics showing no strong structured error.
- [ ] `test_modality_swap.py` passes — i.e., `GapAnalyzer(modality="actuation")` runs through the identical core code as V1 with zero core-module changes.
- [ ] Validation harness produces a V2-specific correlation table analogous to V1's.

---

## 14. Risks & Open Dependencies

- **Ogawa et al. (2017) access**: **resolved** — obtained, along with Thakur et al. (2018), a follow-up paper with a directly reusable fitted force-pressure equation for the same class of actuator. See `docs/pgm_reference_data.md`. What's still open: the two papers describe two different physical PGM prototypes (250mm vs. 300mm natural length) with different measured quantities (elongation-under-load hysteresis loop vs. force-at-fixed-length), so they are integrated as two separate, clearly-labeled reference points rather than merged into one dataset — and the full continuous pressure-elongation curves (Ogawa Fig. 4/5) still need WebPlotDigitizer-grade extraction for the hysteresis fit itself; only the text-quoted numbers (dimensions, pressure range, Thakur's equations, Ogawa's Fig. 6 comparison table) are integrated so far.
- **Validation correlation may simply be weak**: it's possible the pre-registered condition set produces a Spearman ρ with a CI that includes zero. This is a legitimate, reportable outcome, not a failure to hide — the spec's acceptance criteria (13) require the analysis to be run and reported honestly, not that a specific ρ threshold be hit.
- **JEPA collapse risk is real, not hypothetical**: budget implementation time for the collapse safeguard and possible VICReg-style regularization, rather than assuming default hyperparameters will simply work.
- **HaGRID gesture vocabulary may not map cleanly** onto ForceHand-relevant DOFs; the canonical-subset filtering (5.4) may leave fewer usable samples than expected — check this early with a quick data audit before committing to the full training pipeline.

---

## 15. V3 Forward-Compatibility Notes

V3 is **not started** and is contingent on real ForceHand-style hardware access. What changes, and what does not:

**Does NOT change**: World Model Core, Divergence Module, Validation Harness, `GapAnalyzer` API, `Rollout` schema.

**DOES change**:
- A new data loader replacing synthetic perturbation (V1) / digitized-curve proxy (V2) with live logged telemetry from the real ForceHand glove and real deployment-condition camera footage.
- The ground-truth degradation signal for validation becomes real measured task performance (a human-participant study comparing vision-based control vs. sensor-glove interface, as is standard for evaluating this class of assistive device — see Thakur et al. 2018 §III for the sEMG-based evaluation methodology this would extend), replacing MediaPipe-confidence-only and simulated-actuator-residual proxies.
- Ethics/IRB approval becomes a hard prerequisite, owned by whatever institution runs the human-participant study, not this project.

If implementing V3 ever requires touching the World Model Core, Divergence Module, or Validation Harness, that is a signal the V1/V2 architecture had a hidden assumption that should be fixed retroactively, not worked around.

---

## 16. Milestone Phases

Relative phases, not calendar dates (adjust to actual available time):

- **Phase 0**: Data audit — confirm HaGRID canonical-subset sample counts are sufficient (done, see ROADMAP.md); confirm Ogawa et al. accessibility (done — see Section 17 references and `docs/pgm_reference_data.md`).
- **Phase 1**: Rollout schema, storage layer, HaGRID/EgoHands loaders, synthetic perturbation pipeline.
- **Phase 2**: World model core + landmark encoder + collapse safeguard, trained and sanity-checked.
- **Phase 3**: Divergence module (Fréchet + MMD) with numerical-stability tests.
- **Phase 4**: Validation harness on pre-registered condition set; V1 acceptance criteria met.
- **Phase 5**: Packaging, CLI, demo notebook, V1 release.
- **Phase 6**: V2 — digitization, actuator encoder, modality-swap test, V2 acceptance criteria met.
- **Phase 7 (deferred)**: V3, contingent on real hardware access.

---

## 17. References

- Ogawa, K., Thakur, C., Ikeda, T., Tsuji, T., & Kurita, Y. (2017). Development of a pneumatic artificial muscle driven by low pressure and its application to the unplugged powered suit. *Advanced Robotics*, 31(21), 1135–1143. https://doi.org/10.1080/01691864.2017.1392345
- Thakur, C., Ogawa, K., Tsuji, T., & Kurita, Y. (2018). Soft Wearable Augmented Walking Suit With Pneumatic Gel Muscles and Stance Phase Detection System to Assist Gait. *IEEE Robotics and Automation Letters*, 3(4), 4257–4264. https://doi.org/10.1109/LRA.2018.2864355
- See `docs/pgm_reference_data.md` for the full transcription of PGM prototype specs, pressure ranges, and fitted equations extracted from the above two papers, and an explicit note on why they describe two different physical prototypes that should not be merged into one dataset.
- MediaPipe Tasks API documentation (Pose Landmarker, Hand Landmarker, Holistic Landmarker) — `ai.google.dev/edge/mediapipe`.
- HaGRID dataset — Kapitanov et al., "HaGRID — HAnd Gesture Recognition Image Dataset," arXiv:2206.08219.
- Heusel et al. (2017), Fréchet Inception Distance formulation — basis for Section 7.1's metric.
