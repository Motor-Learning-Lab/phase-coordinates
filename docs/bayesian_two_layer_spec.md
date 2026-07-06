# Bayesian Cycle-Fixed Geometry Specification

This branch replaces the previous instantaneous-geometry Layer 2 direction with a more identifiable cycle-fixed geometry model.

The motivating diagnostic result from `bayesian-two-layer-estimator` was that allowing `center(t)` and other geometric quantities to vary instantaneously created a strong center/orbit/noise confound. The stable core model on this branch therefore keeps the movement geometry fixed within each cycle and estimates only phase, radius, and perpendicular displacement at the instantaneous time scale.

This document is the active specification for branch:

```text
bayesian-cycle-fixed-geometry
```

## Model summary

Layer 1 remains a coarse cycle model. It samples cycle boundaries `tau` and cycle centers `c`. After sampling, it derives an oriented frame per cycle from real-valued data interpolation at the phase-zero and quarter-phase anchor times.

Layer 2 is a cycle-fixed geometry model. PCA normals are not part of the active model. Normals are derived as `n = cross(e1, e2)` from the oriented two-anchor frame.

For each cycle `k`, the Bayesian cycle-level variables are:

```text
c_k      cycle center
R_k      cycle mean radius
```

The in-plane frame and normal are derived deterministically from the sampled center and fixed anchor points interpolated at the Layer 1 posterior mean boundaries:

```text
x0_k   = interp_X(tau_k)              # real-valued interpolation
x90_k  = interp_X(tau_k + 0.25*T_k)  # quarter-period anchor

a0_k   = x0_const  - c_k             # moves with sampled center
a90_k  = x90_const - c_k             # moves with sampled center

e1_k   = normalize(a0_k)
a90_orth_k = a90_k - e1_k * dot(a90_k, e1_k)
e2_k   = normalize(a90_orth_k)
n_k    = normalize(cross(e1_k, e2_k))
```

For each time point `t` inside cycle `k`, the instantaneous variables are:

```text
phi_t    phase
r_t      radius
z_t      perpendicular displacement from the cycle plane
```

The observation model is:

```text
pred[t] =
    c_k
    + e1_k * r_t * cos(phi_t)
    + e2_k * r_t * sin(phi_t)
    + n_k  * z_t

X[t] ~ Normal(pred[t], sigma_x)
```

The defining restriction of this branch is:

```text
center and plane are fixed within each cycle
```

They may vary across cycles, but they do not vary instantaneously.

## Why this model exists

The previous instantaneous-geometry model estimated:

```text
center(t)
normal(t)
in-plane frame(t)
phase(t)
radius(t)
z(t)
sigma_x
```

Diagnostics showed that the center trajectory could absorb large parts of the orbit. In the failed fixed-plane synthetic run, center summaries were strongly coupled to `sigma_x` and radius, while residuals were overwhelmingly in-plane rather than normal. This indicated a three-way confound:

```text
center(t) <-> orbit/radius <-> sigma_x
```

The cycle-fixed model removes that confound by making `c_k`, `n_k`, `R_k`, and `a_k` cycle-level variables.

## Layer 1 handoff

Layer 1 samples only `tau` (cycle boundaries) and `c` (cycle centers). It does not sample normal vectors or orientation vectors.

After sampling, Layer 1 computes the oriented frame by real-valued interpolation:

```text
x0_arr  = interp_X(tau_mean[:-1])                     # phase-zero data points
x90_arr = interp_X(tau_mean[:-1] + 0.25*T_k)          # quarter-phase data points
a0_mean = x0_arr - c_mean                              # phase-zero anchor vectors
a90_mean = x90_arr - c_mean                            # quarter-phase anchor vectors
e1_mean, e2_mean, normal_mean = gram_schmidt(a0_mean, a90_mean)
```

Layer 1 provides to Layer 2:

```text
tau_mean[k]                 posterior mean cycle boundary
c1_mean[k, 3]               cycle center posterior mean
c1_sd[k, 3]                 cycle center posterior SD
x0_arr[k, 3]                data interpolated at tau_mean[k]  (fixed constant)
x90_arr[k, 3]               data interpolated at tau_mean[k] + 0.25*T_k  (fixed constant)
```

Layer 1 normal vectors are diagnostic outputs, not Bayesian parameters. They are derived after sampling and passed as orientation seeds for reporting, but they do not enter the Layer 2 likelihood.

Layer 2 hierarchical scale parameters control how far `c_k` may move from `c1_mean[k]`. The frame adjusts to match because `a0_k = x0_const - c_k` and `a90_k = x90_const - c_k` both depend on `c_k`.

## Hierarchical priors for cycle-level fixed variables

The cycle-level variables are Bayesian, not deterministic. Each has:

```text
Layer 1 mean as cycle-specific prior mean
hierarchical variability across cycles
HalfNormal prior on that variability
HalfNormal scale set by empirical across-cycle variability
```

Use robust floors so that the prior does not become degenerate when synthetic cycles are nearly identical.

### Center hierarchy

For each coordinate `d`:

```text
empirical_sd_c[d] = sd across k of c1_mean[k, d]
sigma_c[d] ~ HalfNormal(scale = max(empirical_sd_c[d], center_floor))
c_k[d] ~ Normal(c1_mean[k, d], sigma_c[d])
```

Suggested floor:

```text
center_floor = 0.02 * R_X
```

The center is fixed within cycle:

```text
c(t) = c_k for tau_k <= t < tau_{k+1}
```

Do not spline `c_k` across time in this branch.

### Cycle mean radius hierarchy

Compute a Layer 1 / deterministic cycle radius prior mean:

```text
R1_mean[k] = median in-plane distance from c1_mean[k] within cycle k
```

Prefer a log-scale hierarchy:

```text
log_R1_mean[k] = log(max(R1_mean[k], radius_floor))
empirical_sd_log_R = sd across k of log_R1_mean[k]
sigma_log_R ~ HalfNormal(scale = max(empirical_sd_log_R, log_radius_floor))
log_R_k ~ Normal(log_R1_mean[k], sigma_log_R)
R_k = exp(log_R_k)
```

Suggested floor:

```text
log_radius_floor = 0.03
radius_floor = 1e-3 * R_X
```

The instantaneous radius should be anchored to `R_k`:

```text
log_r_t = log_R_k + h_r_t
```

Start with either:

```text
r_t = R_k
```

or a low-flexibility, zero-centered within-cycle deviation `h_r_t`.

### Normal and frame (derived, not sampled)

Normals are derived from the two-anchor frame construction. They are not independently sampled Bayesian variables.

After sampling `c_k`, compute:

```text
a0_k   = x0_const  - c_k
a90_k  = x90_const - c_k

e1_k          = normalize(a0_k)
dot_a90_e1    = dot(a90_k, e1_k)
a90_orth_k    = a90_k - e1_k * dot_a90_e1
e2_k          = normalize(a90_orth_k)
n_k           = normalize(cross(e1_k, e2_k))
```

The signed normal is always consistent with the data ordering because `x0_const` and `x90_const` are both anchored to the observed trajectory. There is no sign ambiguity.

The normal is fixed within cycle:

```text
n(t) = n_k for tau_k <= t < tau_{k+1}
```

There are no `sigma_n_angle`, `delta_n`, or independent anchor variables in the Bayesian model. Add a diagnostic warning when `norm(a90_orth_k) / norm(a90_k)` is too small (indicates near-degenerate frame).

## Instantaneous variables

### Phase

Start simple with linear phase inside each cycle:

```text
phi_t = 2*pi*k + 2*pi*(t - tau_k)/(tau_{k+1} - tau_k)
```

This phase is deterministic conditional on cycle boundaries. It should be monotone by construction.

After the cycle-fixed geometry model works, optional phase flexibility can be added as a cycle-local positive-speed model. Do not add that complexity in the first implementation.

### Radius

Initial implementation may use:

```text
r_t = R_k
```

A next step can add low-flexibility within-cycle radius deviation:

```text
log_r_t = log_R_k + h_r_t
h_r_t ~ smooth zero-centered within-cycle process
```

For the fixed-plane synthetic test, radius should recover approximately 1.0.

### Perpendicular displacement

Use a zero-centered within-cycle process:

```text
z_t ~ smooth zero-centered process
```

For the fixed-plane synthetic test, `z_t` should remain near zero.

## Observation noise

Keep the existing `sigma_x` prior initially:

```text
rho_x = sigma_x / R_X
rho_x ~ LogNormal(log(0.03), 0.5)
sigma_x = R_X * rho_x
```

If the cycle-fixed model still inflates `sigma_x`, inspect center, radius, z, phase, and residual decomposition before tightening the noise prior.

## Implementation targets

The cycle-fixed model is implemented in `phase_coordinates/bayesian.py` as the internal function `_fit_layer2`.

The sampled parameters are:

```text
Layer 1: tau, c, mu_tau, rho_tau
Layer 2: sigma_c, c_k, sigma_log_R, log_R_k, h_z_knots, rho_x
```

All frame variables (`e1_k`, `e2_k`, `n_k`) are `pm.Deterministic` quantities derived from `c_k` and fixed constants.

The public API wrapper is `fit_bayesian_phase_coordinates`.

## Debug scripts

```text
docs/debug/scripts/test_layer2_cycle_fixed.py
docs/debug/scripts/test_cycle_fixed_synthetic_suite.py
```

The first script uses the same fixed-plane synthetic data as `test_layer2.py`. The suite script adds clean complete cycles and mild phase warp scenarios.

Report:

```text
divergence count
max treedepth count
sigma_x mean/median/5%/95%
R_k mean/median by cycle
center norm by cycle
normal cos_sim by cycle (signed and absolute)
orientation scores (e2 alignment with quarter-phase anchor)
z_rms
RMSE total
RMSE normal
RMSE cyclic
RMSE tangential
phase monotonicity
frame orthonormality
hierarchical scales: sigma_c, sigma_log_R
```

## Success criteria on fixed-plane synthetic data

The first working version should aim for:

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

If these do not hold, report the failing component rather than loosening thresholds.

## Non-goals for this branch

Do not implement within-cycle changing centers.

Do not implement within-cycle changing normals.

Do not treat fixed-sigma or MAP initialization as the primary solution unless the cycle-fixed model still exhibits a sampler-only problem after the center/orbit confound has been removed.

Do not delete the old instantaneous model from code in the first pass. Keep it available as a separate experimental path.
