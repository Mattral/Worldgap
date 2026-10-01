# V2 runbook — simulated PGM vs. real characterization

```bash
git clone https://github.com/Mattral/Worldgap.git && cd Worldgap
pip install -e .
python scripts/run_v2_actuation.py --out ./v2_run
```

That is the whole setup: the core install, no extras. No downloads, no model
bundle, no recordings — the real reference data ships inside the package. (The
script lives in the repo, not the wheel, hence the clone.) This is the
reproducible half of worldgap: anyone can rerun it and get the numbers below
(see the reproducibility note under the results for what "same" means across
platforms).

---

## The question

If you build a soft-exosuit controller against a *simulated* pneumatic gel
muscle, how wrong is the simulator, and where?

Answering that needs a simulator someone would plausibly use, not one fitted to
the answer.

## What is being compared

**Sim side — the ideal McKibben model.** The standard textbook force balance
for a braided pneumatic artificial muscle, from braid geometry alone (Chou &
Hannaford, IEEE T-RA 12(1), 1996):

```
F(ε, P) = (π·D₀²·P/4) · ( 3(1−ε)²/tan²θ₀ − 1/sin²θ₀ )      ε = (L₀−L)/L₀
```

Parameterized from Ogawa et al. (2017)'s own reported geometry: `L₀ = 500 mm`
(§2.2 braided mesh), `D₀ = 23 mm` (§2.3 default mesh diameter), and
`θ₀ = 42.1°` — **derived, not reported**, from the 28 mm maximum diameter via
braid kinematics. That derivation is a chain of three inferences and is flagged
as such everywhere it appears.

**Real side — Ogawa 2017 Figure 4(a), digitized.** `Length(Force)` at each of
the 7 supply pressures actually tested, isotonic-smoothed with the correction
magnitude carried along as a per-curve noise floor, cross-checked against
Figure 6's independently published contraction ratios. See
`pgm_reference_data.md`.

Both sides use the same state layout — `[pressure_mpa, force_n, length_mm]`,
normalized by fixed constants chosen from the experiment's stated ranges, never
fitted from either domain (a normalization derived from one side would leak
that side's distribution into the comparison).

### Why not MuJoCo, and why not a fit

Spec 5.5 offered MuJoCo or a fitted hysteresis model. Neither was used:

- **A fit to the Ogawa curves** would be compared against the same data it was
  fitted on. The gap would be small by construction and would measure
  curve-fitting, not sim-to-real.
- **MuJoCo** has no native McKibben actuator. Using it means writing this same
  analytic model as a custom gain/bias function with a physics engine and a
  heavy optional dependency wrapped around it. If V3 needs contact or full-body
  dynamics, MuJoCo earns its place; for one actuator's static response it does
  not.

### What the model is known to get wrong

Stated in advance, so the results are a test of a hypothesis rather than a
surprise:

1. **No hysteresis** — force depends only on current `(ε, P)`. Ogawa §2.3
   reports genuine loading/unloading nonlinearity at 0.05–0.15 MPa.
2. **Zero force at zero pressure** — the real gel-foam inner tube is an elastic
   body that resists stretch with no air at all.
3. **No threshold pressure, no braid/bladder friction, no end-cap effects.**

---

## Results

Reproducible with the command at the top (`seed: 0`): bit-identical across
runs on the same platform and torch build. Across platforms, expect the last
digits of the per-level MMD² values to move: on Windows with Python 3.14 and
torch 2.14 (CPU), they differ from the table below by up to 3×10⁻⁶ (e.g.
−0.128881 vs −0.128882), while the physical residuals, every Fréchet value, the
overall MMD² and ρ = +0.943 match to all printed digits. Floating-point
reduction order differs between torch builds; spec 12.18's lockfile is what
pins it exactly.

### 1. Physical residuals — the interpretable number

Simulated length minus real digitized length, against each curve's own
digitization noise floor:

| P (MPa) | mean abs. residual | max abs. residual | noise floor | above floor? | saturated |
|---:|---:|---:|---:|:--:|---:|
| 0.00 | 63.37 mm | 146.33 mm | 0.64 mm | yes | **100 %** |
| 0.05 | 26.72 mm | 68.89 mm | 11.39 mm | yes | 5 % |
| 0.10 | **13.55 mm** | 57.31 mm | 6.34 mm | yes | 0 % |
| 0.15 | 20.17 mm | 63.14 mm | 4.93 mm | yes | 0 % |
| 0.20 | 27.31 mm | 67.91 mm | 3.05 mm | yes | 0 % |
| 0.25 | 30.90 mm | 64.05 mm | 0.06 mm | yes | 0 % |
| 0.30 | 43.03 mm | 70.46 mm | 0.44 mm | yes | 0 % |

**The model's error exceeds the data's own uncertainty at every pressure**, by
one to two orders of magnitude. The gap is a real modelling error, not a
digitization artifact — which is the first thing that had to be established
before anything else here means anything.

**0 MPa is 100 % saturated.** The ideal model predicts zero force at zero
pressure, so it holds no load at any length and the inversion pins to its search
bound. That is failure mode #2 above, confirmed — and the flat curve is the
grid running out, not a physical plateau. The script reports it and excludes
that level from the latent gap rather than letting a clipped curve contribute a
number.

**Error is minimised at 0.10 MPa and grows towards both ends.** The mechanism
is visible directly: at 0.2 MPa the ideal model contracts freely to ~390 mm and
can hold nothing below that, while the real muscle sits at ~322 mm under
near-zero load. The real gel-foam PGM contracts substantially further than
braid geometry alone predicts — which is why residuals are largest at low load,
and why `--tube-stiffness` exists as a hypothesis to test rather than a fudge
factor applied by default.

### 2. Latent-space gap — the reusability demonstration

The same `GapAnalyzer` class V1 uses, only `config.modality` changed, on real
data instead of a toy signal:

| P (MPa) | Fréchet | MMD² | confidence |
|---:|---:|---:|:--|
| 0.05 | 0.000282 | −0.128882 | low |
| 0.10 | 0.000054 | −0.153983 | low |
| 0.15 | 0.000200 | −0.149938 | low |
| 0.20 | 0.000354 | −0.143778 | low |
| 0.25 | 0.000637 | −0.129680 | low |
| 0.30 | 0.000613 | −0.119835 | low |
| all | 0.000263 | −0.014864 | low |

**Every MMD² is negative, and that is not a bug.** `mmd_squared` is the
*unbiased* estimator, and an unbiased estimator of a non-negative quantity must
go negative when the true value is near zero — otherwise it would be biased
upward (Gretton et al. 2012). A negative value means "no difference detectable
at this sample size". Its magnitude is noise and must not be used to rank
conditions; `GapResult.warnings` says so, and the script disqualifies the MMD
correlation below for exactly this reason.

**"low" confidence is a fact about the literature, not a fixable setting.**
Only 7 pressure levels were ever published. After windowing that is ~35
rollouts per side against `5 × summary_dim = 40`. Raising `summary_dim` would
lower the flag further; lowering it further would not add information that does
not exist.

### 3. Internal consistency — not validation

Across the 6 usable levels, the latent Fréchet gap rank-correlates with the
physical residual in millimetres at **Spearman ρ = +0.943 (p = 0.005)**.

That is worth something: it shows the gap score is tracking a physically
meaningful error rather than encoder noise. A tool whose latent number moved
independently of the actual millimetre disagreement would be measuring itself.

**It is not spec 8.1 validation and must never be reported as such.** §8.1
requires ground truth computed *independently of the model under test*, and
both quantities here come from the same two curve families. Passing this check
is necessary, not sufficient.

The MMD² correlation (ρ = +0.829) is reported by the script and then explicitly
disqualified, because all six values it ranks are negative — ranking them ranks
noise.

---

## What this establishes, and what it does not

**Does**: a standard braid-geometry model of this actuator is wrong by 13–43 mm
across its usable pressure range, far above the reference data's uncertainty;
the error is smallest near 0.1 MPa and worst at the extremes; the model cannot
represent the unpressurized case at all; and the same `GapAnalyzer` that
handles landmark trajectories handles this, with the latent score tracking the
physical error.

**Does not**: predict real transfer degradation. That needs an independent
measurement of how a controller built on the simulator actually performs on
hardware — V3, and it needs the hardware.

**Scale caveat, carried from Ogawa §4 and repeated because it matters:** the
characterized muscle has a 300 mm normal length, and the authors state plainly
that this length is *"suitable for applications such as assisting walking but
not for assisting the hand or wrist, which requires shorter muscles."* They
recommend resting length be modelled as a third parameter alongside pressure
and force. Applying these numbers to a hand- or wrist-scale device is an
extrapolation across a scale the source paper says does not carry over. That is
not a caveat to bury in a footnote — for a glove application it is the main
limitation, and no amount of careful analysis of this data set can remove it.

---

## Things worth trying next

- `--tube-stiffness 0.3` (etc.) — does a linear elastic term for the gel-foam
  inner tube account for the low-pressure error? If residuals at 0.05–0.15 MPa
  drop sharply while the high-pressure ones do not, that is a real mechanistic
  finding about what the naive model is missing.
- Digitize Ogawa Fig. 5 (contraction ratio vs. pressure at fixed forces) as an
  independent second view of the same actuator — same rigour: cross-check and
  recorded noise floor.
- A hand-scale PGM characterization. It does not exist in either paper, and per
  §4 it is the thing a glove application actually needs.
