# Claude Prompt: Implement Cycle-Fixed Bayesian Geometry Model

You are working in `Motor-Learning-Lab/phase-coordinates` on branch `bayesian-cycle-fixed-geometry`.

Read first:

```text
docs/bayesian_two_layer_spec.md
docs/cycle_fixed_geometry_model.md
docs/PROGRESS.md
docs/debug/README.md
phase_coordinates/bayesian.py
```

## Goal

Implement a new Layer 2 Bayesian model with cycle-fixed geometry.

Do not continue the old instantaneous-geometry debugging path. Do not reintroduce instantaneous `center(t)` or instantaneous `normal(t)` in this branch.

## Model structure

For each cycle `k`, define Bayesian cycle-level variables:

```text
c_k      cycle center, fixed within cycle
n_k      cycle plane normal, fixed within cycle
R_k      cycle mean radius
a_k      phase-zero / cycle-origin vector, fixed within cycle
```

For each time point `t` inside cycle `k`, estimate or compute:

```text
phi_t    phase
r_t      radius
z_t      perpendicular displacement
```

Prediction:

```text
pred[t] =
    c_k
    + e1_k * r_t * cos(phi_t)
    + e2_k * r_t * sin(phi_t)
    + n_k  * z_t
```

where:

```text
e1_k = normalized projection of a_k into the plane orthogonal to n_k
e2_k = cross(n_k, e1_k)
```

## Required hierarchical priors

The cycle-level variables must be hierarchical.

For each fixed cycle-level variable:

```text
c_k     center
R_k     cycle mean radius
n_k     normal
a_k     cycle origin direction
```

use this pattern:

```text
Layer 1 posterior mean sets the cycle-specific prior mean.
A hierarchical variability parameter controls across-cycle deviations.
That variability parameter has a HalfNormal prior.
The HalfNormal scale is set by empirical across-cycle variability of the Layer 1 means.
```

Use robust floors where the empirical across-cycle variability is near zero.

### Center

```text
sigma_c[d] ~ HalfNormal(scale=max(sd_k(c1_mean[k,d]), 0.02*R_X))
c_k[d] ~ Normal(c1_mean[k,d], sigma_c[d])
```

### Radius

Compute `R1_mean[k]` from Layer 1 / deterministic geometry, preferably median in-plane distance from `c1_mean[k]` within cycle.

Use log scale:

```text
sigma_log_R ~ HalfNormal(scale=max(sd_k(log(R1_mean[k])), 0.03))
log_R_k ~ Normal(log(R1_mean[k]), sigma_log_R)
R_k = exp(log_R_k)
```

Start with `r_t = R_k`. Add within-cycle radius deviations only after this version works.

### Normal

Use tangent-plane deviations around Layer 1 normals:

```text
Q_k = tangent basis at n1_mean[k]
delta_n[k] ~ Normal(0, sigma_n_angle)
n_k = normalize(n1_mean[k] + Q_k @ delta_n[k])
sigma_n_angle ~ HalfNormal(scale=max(empirical angular sd, 0.03 rad))
```

### Cycle origin

Use:

```text
a1_mean[k] = X(tau_k) - c1_mean[k]
sigma_a[d] ~ HalfNormal(scale=max(sd_k(a1_mean[k,d]), 0.02*R_X))
a_k[d] ~ Normal(a1_mean[k,d], sigma_a[d])
```

Then project `a_k` into the plane to get `e1_k`.

## Phase

Use linear phase within each cycle for the first implementation:

```text
phi_t = 2*pi*k + 2*pi*(t - tau_k)/(tau_{k+1} - tau_k)
```

Do not add a positive-speed phase model in this first pass.

## z / perpendicular displacement

Keep `z_t` as a zero-centered instantaneous variable. Use a simple smooth or weakly regularized prior. For fixed-plane synthetic data, z should remain near zero.

## Observation noise

Keep the existing `sigma_x` prior initially. Do not fix `sigma_x` and do not tune it before testing the cycle-fixed geometry model.

## Implementation steps

1. Add internal function:

```text
_fit_layer2_cycle_fixed_geometry(...)
```

2. Add debug script:

```text
docs/debug/scripts/test_layer2_cycle_fixed.py
```

3. Use the same fixed-plane synthetic data as the existing Layer 2 test.

4. Report:

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

5. Update `docs/PROGRESS.md` with the implementation status and test result.

## Constraints

Do not delete the existing instantaneous-geometry model code.

Do not use fixed-sigma, MAP, Laplace, adapt_full, or ADVI as the first solution.

Do not loosen tests to make the branch pass. If the cycle-fixed model fails, report which component fails.

## Final response expected

Report:

```text
1. Files changed
2. Model structure implemented
3. Hierarchical priors used for c_k, R_k, n_k, and a_k
4. Test command run
5. Fixed-plane synthetic results
6. Whether center/radius/sigma_x confounding improved
7. Remaining problems
8. Recommended next step
```
