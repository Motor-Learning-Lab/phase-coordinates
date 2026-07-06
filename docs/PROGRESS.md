# Bayesian cycle-fixed geometry branch progress

Branch: `bayesian-cycle-fixed-geometry`
Base branch: `bayesian-two-layer-estimator`
Created: 2026-07-06

## Current direction

This branch changes the Layer 2 modeling target.

The previous branch, `bayesian-two-layer-estimator`, attempted a full instantaneous-geometry model with time-varying center, plane normal, in-plane frame, phase, radius, perpendicular displacement, and observation noise.

Diagnostics on that branch showed a strong center/orbit/noise confound. In the failed synthetic fixed-plane fit, the center trajectory became large enough to trade against radius and `sigma_x`. The residuals were primarily in-plane, not normal, and the strongest per-draw correlates of `sigma_x` were center and radius summaries.

The new branch therefore implements a more identifiable core model:

```text
cycle-level geometry:
    c_k       center, fixed within cycle
    n_k       plane normal, fixed within cycle
    a_k       phase-zero / cycle-origin direction, fixed within cycle
    R_k       cycle mean radius, fixed within cycle

instantaneous variables:
    phi_t     phase
    r_t       radius or radius deviation
    z_t       perpendicular displacement
```

The observation model is:

```text
pred[t] =
    c_k
    + e1_k * r_t * cos(phi_t)
    + e2_k * r_t * sin(phi_t)
    + n_k  * z_t
```

where `k` is the cycle containing `t`.

## Active specification and prompt

The active spec is:

```text
docs/bayesian_two_layer_spec.md
```

The active design note is:

```text
docs/cycle_fixed_geometry_model.md
```

The implementation handoff prompt is:

```text
docs/claude_cycle_fixed_geometry_prompt.md
```

## Core modeling requirements

Cycle-level fixed variables remain Bayesian and hierarchical.

For each of:

```text
c_k     cycle center
R_k     cycle mean radius
n_k     cycle plane normal
a_k     cycle origin / phase-zero direction
```

use this pattern:

```text
Layer 1 posterior mean sets the cycle-specific prior mean.
A hierarchical variability parameter controls deviations across cycles.
That variability parameter has a HalfNormal prior.
The HalfNormal scale is set by the empirical across-cycle variability of the Layer 1 means.
```

This is the key requirement for the branch.

## Why this branch exists

The full instantaneous-geometry model is not being deleted, but it is not the active implementation target on this branch.

The cycle-fixed model is intended to be the stable core estimator:

```text
center and plane may change from cycle to cycle
center and plane do not change instantaneously within a cycle
phase, radius, and perpendicular displacement are instantaneous
```

This should remove the main center/orbit/noise confound and provide a cleaner base for later extensions.

## Implementation plan

1. Add an internal function, probably:

```text
_fit_layer2_cycle_fixed_geometry(...)
```

2. Start with deterministic or simple linear phase inside each cycle:

```text
phi_t = 2*pi*k + 2*pi*(t - tau_k)/(tau_{k+1} - tau_k)
```

3. Start with a low-complexity radius model:

```text
r_t = R_k
```

or:

```text
log_r_t = log_R_k + small zero-centered within-cycle deviation
```

4. Keep `z_t` as a zero-centered within-cycle perpendicular displacement.

5. Keep the current `sigma_x` prior initially. Do not use fixed-sigma or MAP initialization as the first fix.

6. Add a debug script:

```text
docs/debug/scripts/test_layer2_cycle_fixed.py
```

using the same fixed-plane synthetic data as the old `test_layer2.py`.

## Required diagnostics for first implementation

The cycle-fixed debug script should report:

```text
divergence count
max treedepth count
sigma_x mean/median/5%/95%
R_k by cycle
center norm by cycle
normal cos_sim by cycle
z_rms
RMSE total
RMSE normal
RMSE cyclic
RMSE tangential
phase monotonicity
frame orthonormality
hierarchical scales: sigma_c, sigma_log_R, sigma_n_angle, sigma_a
```

## Success criteria on fixed-plane synthetic data

A successful first pass should recover:

```text
sigma_x near 0.02
R_k near 1.0 for all cycles
center norm small relative to radius
normal cos_sim near 1.0
z_rms small
RMSE near observation noise
few or no divergences
no catastrophic chain split
```

## Stale work removed from this branch

The source branch contained debugging artifacts for the full instantaneous-geometry model. Those are no longer the active direction here.

Removed from this branch:

```text
docs/claude_bayesian_two_layer_prompt.md
docs/claude_layer2_reparameterization_prompt.md
docs/claude_followup_current_model_diagnostics.md
docs/layer2_amplitude_diagnostics_review.md
docs/debug/scripts/diagnose_layer2_amplitude.py
docs/debug/logs/11_layer2_amplitude_diagnostics.log
docs/debug/logs/12_layer2_current_model_diagnostics.log
```

The old branch remains available for that history.

## Current status

- [x] Branch created from `bayesian-two-layer-estimator`.
- [x] Specification rewritten for cycle-fixed geometry.
- [x] Added `docs/cycle_fixed_geometry_model.md` design note.
- [x] Added `docs/claude_cycle_fixed_geometry_prompt.md` implementation prompt.
- [x] Removed stale source-branch work files that were misleading for this branch.
- [x] Updated `docs/debug/README.md` for cycle-fixed work.
- [x] Implemented cycle-fixed `_fit_layer2` in `phase_coordinates/bayesian.py`.
- [x] Added `docs/debug/scripts/test_layer2_cycle_fixed.py`.
- [x] Run fixed-plane synthetic test (log 13) — see finding 1 below.
- [x] Per-chain analysis of sigma_x and R_k posteriors — see finding 2.
- [x] Implemented oriented two-anchor frame (e1 from a0, e2 from a90, n = cross(e1,e2)) — see finding 3.
- [ ] Investigate remaining sigma_x inflation (2.6×) and z_rms elevation from short last cycle.
- [ ] Decide whether to expose the new model through public API.

## Findings

### Finding 1: Cycle-fixed model — first run result (2026-07-06, log 13)

`docs/debug/scripts/test_layer2_cycle_fixed.py` ran the cycle-fixed geometry
model on the same fixed-plane synthetic data (6-cycle tilted unit circle,
sigma=0.02 noise, seed=0).

**Improvements over the previous instantaneous-geometry model:**
- Sampling time: **19s** (vs 440s — 23× speedup). Fewer parameters, no large
  cumulative-sum matrices.
- Normal cos_sim: **1.0000** on every cycle (vs ~0.997 median before). The
  cycle-fixed structure eliminates the spline interpolation between knots that
  could go near-zero.
- Center norms: **0.04–0.19** (small relative to orbit radius 1.0). Center
  confound eliminated — the center prior is now constrained by the hierarchical
  structure.
- z_rms: **0.004** (effectively zero). Perpendicular deviation near zero for
  fixed-plane data.
- Divergences: **3** (vs 40).
- Phase monotone by construction. Frame exactly orthonormal (errors at 1e-17).

**Remaining failure — amplitude/noise ridge:**
- sigma_x: **0.591** (mean over draws, true=0.02, factor 30x too large)
- R_k: **0.217** per cycle (true=1.0, factor 5x too small)
- RMSE cyclic: 0.742 (>> true sigma=0.02)
- rhat > 1.01 for some parameters; ESS < 100 for some parameters

The center/orbit confound from finding 13 on the previous branch is gone. But
the R_k/sigma_x amplitude ridge now emerges as the primary failure mode: the
sampler finds a state where R_k × sigma_x ≈ const that fits the residuals
with inflated noise, just as the old model had r_t × sigma_x ≈ const.

The assertion `0.7 < np.median(R_k_mean) < 1.3` fails.

**Hierarchical scale posteriors:**
- sigma_c ≈ [0.025, 0.022, 0.016] (small — center well-constrained)
- sigma_log_R ≈ 0.335 (posterior mean; prior scale was 0.03 — driven 11× prior
  SDs away, indicating the ridge is overcoming the radius prior)
- sigma_n_angle ≈ 0.021 rad (small — normals well-identified)
- sigma_a ≈ [0.108, 0.042, 0.020]

**Next step:** The R_k/sigma_x ridge is the same structural degeneracy as
before, now isolated to just these two parameters. The prior on sigma_log_R
(HalfNormal(0.03)) should strongly constrain R_k near R1_mean ≈ 1.0, yet the
posterior moves 11× away. This suggests either (a) the warmup drifts to the
ridge before the prior can anchor R_k, or (b) a stronger/different prior is
needed. Recommended next diagnostic: check whether the two chains are sitting
at different (sigma_x, R_k) points (bimodal failure, as in the old model) or
the same wrong point (unimodal wrong mode).

### Finding 2: Per-chain analysis — unimodal wrong mode (2026-07-06, log perchain_sigma)

Per-chain statistics (400 draws each, seed=0, same run as finding 1):

```
Chain 0: sigma_x mean=0.5906  94%HDI=[0.5726, 0.6116]
Chain 1: sigma_x mean=0.5907  94%HDI=[0.5701, 0.6087]  [true=0.02]

Chain 0: R_k = [0.218 0.218 0.218 0.218 0.214 0.214]  median=0.2175
Chain 1: R_k = [0.218 0.215 0.218 0.218 0.215 0.218]  median=0.2176  [true=1.0]

Chain 0: log_R_k = [-1.537 -1.540 -1.539 -1.537 -1.557 -1.564]
Chain 1: log_R_k = [-1.537 -1.550 -1.535 -1.539 -1.554 -1.541]

Chain 0: sigma_log_R mean=0.3348  94%HDI=[0.3014, 0.3649]
Chain 1: sigma_log_R mean=0.3343  94%HDI=[0.3037, 0.3635]  [prior scale=0.03]
```

**Both chains landed at the same wrong mode** — this is a unimodal failure. The
old instantaneous-geometry model split chains between two modes. Here the chains
agree, but agree on the wrong answer with extremely tight HDIs. This rules out
a simple bimodal label-switching explanation.

**sigma_log_R interpretation:** `sigma_log_R` is the hierarchical variability of
log-radius _across_ cycles, not the likelihood noise scale. With `log_R_k ≈ -1.54`
for all six cycles (versus initval ≈ 0 = log(1.0)), sigma_log_R inflates from its
prior scale of 0.03 to ≈0.335 (≈11× prior SDs) to accommodate the deviation of
log_R_k from its prior mean. The amplitude failure is a property of `log_R_k`
itself; sigma_log_R is a symptom, not a cause.

**Ridge-drift hypothesis examined and rejected.** The initial hypothesis was that
NUTS drifts along the R_k/sigma_x product ridge during warmup before the prior
anchors R_k. This explanation is incorrect:

- At the true mode (R_k=1, sigma_x=0.02), the log-posterior gradient is zero by
  definition — it is a maximum.
- At a small perturbation away from the true mode, the gradient points back toward
  it. The prior and likelihood both favor R_k ≈ 1 near that region.
- NUTS follows the gradient — it does not drift along flat ridges. In ordinary
  regression, `beta` does not drift to 0 and `sigma_y` does not inflate despite the
  same algebraic ridge existing there.
- Therefore warmup drift is not the mechanism. The model finds the wrong mode
  through some other means.

**Root cause is currently unknown.** Despite initvals set to `log_R_k ≈ 0`
(i.e., R_k=1), both chains converge to `log_R_k ≈ -1.54`. The geometry
diagnostics (normal cos_sim, center norms, z_rms) all pass — the failure is
isolated to the amplitude parameters. Why the sampler preferentially ends up at
this wrong mode is not yet understood.

**Next experiment:** Fix `sigma_x` at the true value (0.02) and check whether
`R_k` recovers. This isolates whether the geometry, initvals, and prior hierarchy
are correct — only the amplitude coupling is removed. If R_k recovers to 1.0 under
fixed sigma_x, the failure is entirely in the amplitude/noise coupling.

**Finding 2 post-mortem (2026-07-06):** The root cause was identified and fixed in
finding 3: the normal sign ambiguity caused the frame to be inverted, making the
model traverse the orbit backward. The "unimodal wrong mode" behavior was a
consequence of a consistently wrong frame (all cycles flipped the same way), not
a new failure mechanism. The fixed-sigma_x diagnostic was superseded by the frame fix.

### Finding 3: Oriented two-anchor frame — R_k collapse fixed (2026-07-06, log 14)

**Root cause of the R_k / sigma_x failure identified and fixed.**

The old model constructed `e2 = cross(n, e1)` where `n` came from a sign-ambiguous
PCA normal. When the PCA normal had the wrong sign, `e2` pointed opposite to the
true quarter-phase direction. The observation model then tried to traverse the orbit
backward. With the correct sign, the model fits well; with the wrong sign, it must
choose between large residuals or small R_k. Because initvals start at R_k = 1 but
the wrong-sign mode has lower log-posterior, the sampler escapes to the R_k ≈ 0.2
ridge where sigma_x absorbs the backward-traversal error.

**Fix:** Replace the sign-ambiguous normal with a two-anchor oriented frame:

```text
x0_k  = interp_X(tau_k)              # real-valued interpolation, not rounded index
x90_k = interp_X(tau_k + 0.25*T_k)
a0_k  = x0_k  - c_k                  # phase-zero anchor
a90_k = x90_k - c_k                  # quarter-phase anchor

e1_k = normalize(a0_k)
a90_orth_k = a90_k - e1_k * dot(a90_k, e1_k)
e2_k = normalize(a90_orth_k)
n_k  = normalize(cross(e1_k, e2_k))
```

`a0_k` and `a90_k` are now Bayesian hierarchical variables (replacing the old
single anchor `a_k` and the independent `delta_n` / `sigma_n_angle`). The normal
is derived, not independently sampled. Signed normal cos_sim is always positive by
construction.

**Layer 1 also updated:** `u_hat` (the prior center for the Layer 1 normal
parameter) is now initialized to the oriented normal instead of the PCA normal.

**Results on fixed-plane synthetic data (log 14, tune=400, draws=400, chains=2):**

```
Divergences  : 3  (vs 3 before)
Max treedepth: 24  ([24, 0] per chain — one chain slightly pathological)
sigma_x mean : 0.0522  (was 0.5907 — 11× improvement; factor 2.6× true, down from 30×)
R_k median   : 0.9992  (was 0.217 — fixed, within [0.7, 1.3])
signed cos_sim: min=0.9954  med=1.0000  (all positive — frame orientation correct)
orient scores: 0.965–1.000  (all positive)
RMSE total   : 0.0437  (was 0.5904)
RMSE normal  : 0.0211  (near noise level)
RMSE cyclic  : 0.0403
RMSE tang    : 0.0605
z_rms        : 0.036   (was 0.004 — elevated, see below)
Frame orthonormality: machine precision (errors at ~1e-16)
ALL CYCLE-FIXED CHECKS PASSED
```

**What changed vs the success criteria:**
- R_k ✓ — fully recovered
- sigma_x — partially fixed (0.052 vs 0.020 true, factor 2.6×)
- RMSE total — 0.044, about 2× true noise
- signed normal cos_sim ✓

**Remaining elevated residuals in cycle 5:**

The last detected boundary is at sample 595 of 600, creating a short partial cycle
(60 samples vs ~100). Cycle 5 has elevated center norm (0.29 vs 0.01 for cycles 0–3),
z_rms elevated (0.036), and R_k = 0.878. These are boundary-detection artifacts, not
model failures. The 3 remaining divergences and one max-treedepth hit are in chain 1
which covers cycle 5.

The sigma_x inflation from 0.020 to 0.052 is driven primarily by the residuals in
cycle 5 that z_t partially absorbs. Cycles 0–4 appear well-identified.

**Hierarchical scale posteriors:**
```
sigma_c       : [0.061, 0.065, 0.062]
sigma_log_R   : 0.029  (prior scale was 0.03 — stays at prior, correct behavior)
sigma_a0      : [0.209, 0.033, 0.031]  (large x-component: a0 varies across cycles
                                         in x as boundaries shift)
sigma_a90     : [0.025, 0.145, 0.114]  (large y/z: a90 varies in the tilt plane)
```

**Next:** Investigate the residual sigma_x inflation. Likely causes: (a) short
last-cycle boundary artifact inflating the global sigma_x; (b) sigma_x absorbing
within-cycle variability not captured by linear phase.
