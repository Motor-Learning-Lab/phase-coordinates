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

## Active specification

The active spec is now:

```text
docs/bayesian_two_layer_spec.md
```

It has been rewritten for the cycle-fixed geometry model.

A short design note should also be added during implementation:

```text
docs/cycle_fixed_geometry_model.md
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

## Stale work from source branch

The source branch contained many debugging artifacts for the full instantaneous-geometry model. They are not the active direction here.

On this branch, stale prompt/review/debug files from the instantaneous-geometry effort should be removed or ignored so future agents do not follow the wrong trail.

The old branch remains available for that history.

## Current status

- [x] Branch created from `bayesian-two-layer-estimator`.
- [x] Specification rewritten for cycle-fixed geometry.
- [ ] Add `docs/cycle_fixed_geometry_model.md` design note.
- [ ] Remove stale source-branch work files that are misleading for this branch.
- [ ] Implement `_fit_layer2_cycle_fixed_geometry`.
- [ ] Add `docs/debug/scripts/test_layer2_cycle_fixed.py`.
- [ ] Run fixed-plane synthetic test.
- [ ] Decide whether to expose the new model through public API.
