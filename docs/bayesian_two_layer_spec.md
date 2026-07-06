# Bayesian Cycle-Fixed Geometry Specification

This branch replaces the previous instantaneous-geometry Layer 2 direction with a more identifiable cycle-fixed geometry model.

The motivating diagnostic result from `bayesian-two-layer-estimator` was that allowing `center(t)` and other geometric quantities to vary instantaneously created a strong center/orbit/noise confound. The stable core model on this branch therefore keeps the movement geometry fixed within each cycle and estimates only phase, radius, and perpendicular displacement at the instantaneous time scale.

This document is the active specification for branch:

```text
bayesian-cycle-fixed-geometry
```

## Model summary

Layer 1 remains a coarse cycle model. It provides posterior summaries for cycle boundaries and cycle-level geometry.

Layer 2 is now a cycle-fixed geometry model.

For each cycle `k`, estimate cycle-level geometric variables:

```text
c_k      cycle center
n_k      cycle plane normal
R_k      cycle mean radius
a_k      cycle origin / phase-zero direction
```

For each time point `t` inside cycle `k`, estimate instantaneous movement variables:

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

where:

```text
e1_k = normalized projection of a_k into the plane orthogonal to n_k
e2_k = cross(n_k, e1_k)
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

Layer 1 should provide, at minimum:

```text
tau_mean[k]                 posterior mean cycle boundary
c1_mean[k, 3]               cycle center posterior mean
n1_mean[k, 3]               cycle normal posterior mean, sign-aligned
boundary direction estimate  a1_mean[k, 3]
optional posterior SDs       c1_sd, n angular sd, a1_sd
```

Layer 1 means set Layer 2 prior means. Layer 2 hierarchical scale parameters control how far cycle-level variables may move away from those Layer 1 means.

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

### Normal hierarchy

Normals must remain unit vectors.

Use tangent-plane deviations around the Layer 1 normal:

```text
Q_k = orthonormal tangent basis at n1_mean[k]
delta_n[k] in R^2
n_k = normalize(n1_mean[k] + Q_k @ delta_n[k])
```

Compute empirical angular variability from Layer 1 normal means. For example:

```text
angle_sd = sd of adjacent-cycle angles between n1_mean[k] and n1_mean[k+1]
```

Then:

```text
sigma_n_angle ~ HalfNormal(scale = max(angle_sd, normal_angle_floor))
delta_n[k, d] ~ Normal(0, sigma_n_angle)
```

Suggested floor:

```text
normal_angle_floor = 0.03 rad
```

The normal is fixed within cycle:

```text
n(t) = n_k for tau_k <= t < tau_{k+1}
```

### Cycle origin / phase-zero direction hierarchy

The cycle origin `a_k` defines the in-plane phase-zero direction. It should represent the vector from the cycle center to the boundary / phase-zero point.

Recommended prior mean:

```text
a1_mean[k] = X(tau_k) - c1_mean[k]
```

or an equivalent Layer 1 boundary-origin estimate that is explicitly tied to phase zero.

Start with a vector hierarchy:

```text
empirical_sd_a[d] = sd across k of a1_mean[k, d]
sigma_a[d] ~ HalfNormal(scale = max(empirical_sd_a[d], a_floor))
a_k[d] ~ Normal(a1_mean[k, d], sigma_a[d])
```

Suggested floor:

```text
a_floor = 0.02 * R_X
```

Then construct the in-plane basis:

```text
proj_a_k = a_k - n_k * dot(n_k, a_k)
e1_k = proj_a_k / norm(proj_a_k)
e2_k = cross(n_k, e1_k)
```

Add a diagnostic warning or failure when `norm(proj_a_k) / norm(a_k)` is too small.

A later alternative is an in-plane angular hierarchy around the Layer 1 origin direction, but do not start there unless the vector hierarchy fails.

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

Add a new model path rather than overwriting the old one.

Suggested internal function:

```text
_fit_layer2_cycle_fixed_geometry(...)
```

Suggested public option later:

```text
fit_bayesian_phase_coordinates(..., layer2_geometry="cycle_fixed")
```

It is acceptable to begin with an internal function and a debug script before wiring the public API.

## Required debug script

Create:

```text
docs/debug/scripts/test_layer2_cycle_fixed.py
```

Use the same fixed-plane synthetic data as `test_layer2.py`.

Report:

```text
divergence count
max treedepth count
sigma_x mean/median/5%/95%
R_k mean/median by cycle
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
