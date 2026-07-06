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

Layer 1 samples only `tau` and `c`. Normals are derived, not sampled.

Layer 2 Bayesian variables per cycle `k`:

```text
c_k      cycle center
R_k      cycle mean radius
```

Fixed anchor constants per cycle (computed from Layer 1 posterior mean boundaries by real-valued data interpolation — not rounded to integer indices):

```text
x0_k   = interp_X(tau_mean[k])              # phase-zero data point
x90_k  = interp_X(tau_mean[k] + 0.25*T_k)  # quarter-phase data point
```

Deterministic frame per cycle (derived inside the PyMC model from the sampled `c_k` and the fixed constants):

```text
a0_k       = x0_const  - c_k            # moves with c_k
a90_k      = x90_const - c_k            # moves with c_k

e1_k       = normalize(a0_k)
a90_orth_k = a90_k - e1_k * dot(a90_k, e1_k)
e2_k       = normalize(a90_orth_k)
n_k        = normalize(cross(e1_k, e2_k))
```

For time point `t` in cycle `k`:

```text
phi_t    phase (deterministic linear-in-cycle)
r_t      instantaneous radius
z_t      signed perpendicular displacement
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

Only `c_k` and `R_k` are independently sampled Bayesian parameters. `n_k`, `e1_k`, `e2_k` are derived from `c_k` and fixed data constants.

## Center

```text
sigma_c[d] ~ HalfNormal(scale = max(sd_k(c1_mean[k,d]), 0.02*R_X))
c_k[d] ~ Normal(c1_mean[k,d], sigma_c[d])
```

`c_k` is fixed within cycle. Moving `c_k` automatically rotates the frame because both anchors subtract `c_k`.

## Cycle mean radius

```text
sigma_log_R ~ HalfNormal(scale = max(sd_k(log(R1_mean[k])), 0.03))
log_R_k ~ Normal(log(R1_mean[k]), sigma_log_R)
R_k = exp(log_R_k)
```

`R1_mean[k]` is the median in-plane distance from the Layer 1 center within each cycle.

## Normal (derived)

The normal is not an independently sampled variable. It is derived inside the model as `n_k = normalize(cross(e1_k, e2_k))` from the two-anchor frame.

No `sigma_n_angle`, `delta_n`, or sign-alignment operations are part of the active model.

## Cycle anchors (fixed constants, not sampled)

The phase-zero and quarter-phase anchor points are computed once from the Layer 1 posterior mean boundaries and fixed as PyTensor constants before sampling begins:

```text
x0_arr  = interp_X(tau_mean[:-1])
x90_arr = interp_X(tau_mean[:-1] + 0.25 * T_k)
x0_const  = pt.constant(x0_arr)
x90_const = pt.constant(x90_arr)
```

These are not sampled. Inside the model, the anchor vectors adjust with `c_k`:

```text
a0_k  = x0_const  - c_k
a90_k = x90_const - c_k
```

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
signed normal cos_sim near +1.0 (all positive)
orientation score (e2 . normalize(a90)) near 1.0
small z_rms
RMSE near noise level
few or no divergences
no catastrophic chain split
```

The signed normal check and orientation score replace the old absolute cos_sim check. The sign is guaranteed correct by construction because `n = cross(e1, e2)` uses the oriented anchors.

Only after these diagnostics pass should the branch revisit richer within-cycle phase or radius flexibility.
