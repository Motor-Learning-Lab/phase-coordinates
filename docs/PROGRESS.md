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
- [ ] Fix amplitude/noise ridge in R_k/sigma_x space.
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
