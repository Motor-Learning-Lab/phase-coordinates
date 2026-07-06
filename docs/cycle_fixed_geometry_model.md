# Cycle-Fixed Geometry Model

This note describes the active Layer 2 modeling direction on branch `bayesian-cycle-fixed-geometry`.

## Motivation

The previous full instantaneous-geometry model allowed center, plane normal, in-plane frame, radius, phase, perpendicular displacement, and observation noise to vary at the instantaneous time scale.

Diagnostics showed that this model was weakly identified on the fixed-plane synthetic data. The most important failure was a center/orbit/noise confound: the inferred center trajectory could become large and trade against radius and `sigma_x`.

The cycle-fixed model is a more identifiable core estimator.

## Central assumption

```text
The center and plane are properties of a cycle.
They may vary across cycles.
They do not vary instantaneously within a cycle.
```

Within a cycle, only phase, radius, and perpendicular displacement vary over time.

## Variables

For cycle `k`:

```text
c_k      cycle center
n_k      unit normal of the cycle plane
R_k      cycle mean radius
a_k      phase-zero / cycle-origin vector
```

For time point `t` in cycle `k`:

```text
phi_t    phase
r_t      instantaneous radius
z_t      signed perpendicular displacement
```

The in-plane basis is defined from `n_k` and `a_k`:

```text
proj_a_k = a_k - n_k * dot(n_k, a_k)
e1_k = proj_a_k / norm(proj_a_k)
e2_k = cross(n_k, e1_k)
```

The prediction is:

```text
pred[t] =
    c_k
    + e1_k * r_t * cos(phi_t)
    + e2_k * r_t * sin(phi_t)
    + n_k  * z_t
```

## Hierarchical cycle-level geometry

The cycle-level variables are Bayesian parameters, not fixed deterministic estimates.

For each cycle-level variable, Layer 1 provides the cycle-specific prior mean. Across-cycle variation in the Layer 1 estimates sets the scale of the hierarchical variability prior.

The generic pattern is:

```text
empirical_across_cycle_sd = sd of Layer 1 means across cycles
sigma_variable ~ HalfNormal(scale = max(empirical_across_cycle_sd, floor))
variable_k ~ Normal(Layer1_mean_k, sigma_variable)
```

For normals, use angular/tangent-plane deviations rather than raw unnormalized components.

## Center

```text
sigma_c[d] ~ HalfNormal(scale = max(sd_k(c1_mean[k,d]), 0.02*R_X))
c_k[d] ~ Normal(c1_mean[k,d], sigma_c[d])
```

`c_k` is fixed within cycle.

## Cycle mean radius

Prefer log scale:

```text
sigma_log_R ~ HalfNormal(scale = max(sd_k(log(R1_mean[k])), 0.03))
log_R_k ~ Normal(log(R1_mean[k]), sigma_log_R)
R_k = exp(log_R_k)
```

`R1_mean[k]` should be a data-derived radius estimate for cycle `k`, such as median in-plane distance from the Layer 1 center.

## Normal

Use tangent-plane deviations:

```text
Q_k = tangent basis at n1_mean[k]
delta_n[k] ~ Normal(0, sigma_n_angle)
n_k = normalize(n1_mean[k] + Q_k @ delta_n[k])
```

with:

```text
sigma_n_angle ~ HalfNormal(scale = max(empirical angular sd across cycles, 0.03 rad))
```

## Cycle origin

Start with a vector hierarchy:

```text
a1_mean[k] = X(tau_k) - c1_mean[k]
sigma_a[d] ~ HalfNormal(scale = max(sd_k(a1_mean[k,d]), 0.02*R_X))
a_k[d] ~ Normal(a1_mean[k,d], sigma_a[d])
```

Then project `a_k` into the plane to define `e1_k`.

## Phase

First implementation should use linear phase inside each cycle:

```text
phi_t = 2*pi*k + 2*pi*(t - tau_k)/(tau_{k+1} - tau_k)
```

This avoids reintroducing a phase/speed confound while the geometry model is being validated.

## Radius

The first implementation may use:

```text
r_t = R_k
```

A later extension may add a zero-centered within-cycle radius deviation.

## Perpendicular displacement

`z_t` is instantaneous and zero-centered. For fixed-plane synthetic data it should remain near zero.

## Diagnostics to pass before extending the model

On fixed-plane synthetic data, the cycle-fixed model should recover:

```text
sigma_x near 0.02
R_k near 1.0
small center norm
normal cos_sim near 1.0
small z_rms
RMSE near noise level
few or no divergences
no catastrophic chain split
```

Only after these diagnostics pass should the branch revisit richer within-cycle phase or radius flexibility.
