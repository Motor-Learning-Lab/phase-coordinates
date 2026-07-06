# Review: Layer 2 amplitude diagnostics

Date: 2026-07-06
Branch: `bayesian-two-layer-estimator`

This note records a more cautious interpretation of the current Layer 2 amplitude diagnostics. It is meant to sit alongside `docs/PROGRESS.md`, not replace it.

## Summary

The latest diagnostics are useful, but the conclusion in `docs/PROGRESS.md` is too strong.

The evidence supports this statement:

```text
The current free-sigma Layer 2 NUTS run is invalid. The two chains are stuck in different amplitude/noise regions, sigma_x and radius are strongly anti-correlated, and the posterior-mean trajectory is a poor fit to the synthetic data.
```

The evidence does not yet fully support this stronger statement:

```text
The root cause is proven to be adapt_diag warmup drift, and the next fix should be fixed-sigma or MAP initialization.
```

That may turn out to be correct, but the current diagnostic run did not rule out several other failure points, especially phase/in-plane-frame error, center/radius confounding, and z/radius compensation.

## What is solidly established

### 1. The old normal and phase pathologies are mostly fixed

The tangent-plane normal parameterization and boundary-normalized phase model removed the original localized normal artifact and the earlier phase-boundary NUTS geometry problem. Normal recovery in the current free-sigma run remains good:

```text
normal cos_sim min about 0.989
normal cos_sim median about 0.997
```

So the current failure is probably not primarily the normal direction.

### 2. The current free-sigma posterior samples are not trustworthy

The latest diagnostic run reports:

```text
free-sigma fit:
    40 divergences
    sigma_x mean about 0.362
    radius median about 1.123
    corr(sigma_x, radius_median) about -0.993
```

Per-chain behavior is especially important:

```text
Chain 0:
    sigma_x about 0.52
    radius about 0.57

Chain 1:
    sigma_x about 0.21
    radius about 1.59
```

The chains are stuck at different amplitude/noise points from the first saved draw. This means posterior summaries such as `sigma_x_mean` and radius median are not reliable estimates of the intended posterior.

### 3. The posterior-mean trajectory is geometrically poor

The diagnostic reports:

```text
RMS total residual about 0.556
RMS normal residual about 0.038
RMS cyclic residual about 0.419
RMS tangential residual about 0.364
MLE sigma_x for posterior-mean trajectory about 0.321
true synthetic sigma_x = 0.02
```

This is a major clue. The residual is mostly in-plane, not normal. That points toward some combination of:

```text
phase convention or phase indexing error
in-plane frame orientation problem
radius/phase interaction
center absorbing or distorting cyclic structure
posterior averaging across incompatible chains
```

It is not enough to say that sigma_x and radius trade off. The trajectory direction itself is wrong in the current posterior mean.

### 4. The known synthetic solution is much better than the sampled solution

The diagnostic compares the known data-generating trajectory to the posterior-mean trajectory and finds a very large log-likelihood gap. This proves the current sampled region is very poor relative to the known synthetic truth.

However, this comparison does not by itself prove that warmup visited the true mode and then drifted away. To prove warmup drift directly, warmup states or initial logp/trajectory diagnostics would be needed. The safer statement is:

```text
The known good solution is vastly better than the sampled posterior-mean solution. The current NUTS run failed to reach or remain near that solution.
```

## What is not yet established

### 1. Fixed-sigma initialization is not yet a robust fix

The latest diagnostic script ran a fixed-sigma variant despite the intended diagnostics-only constraint. That run was not a stable success:

```text
fixed sigma_x = 0.02, seed 42:
    458 divergences
    normal cos_sim median about 0.58
    normal cos_sim min about 0.21
```

An earlier fixed-sigma run with seed 0 reportedly recovered radius better. These two results together suggest:

```text
Fixed sigma_x can help amplitude, but the fixed-sigma model is also initialization-sensitive.
```

Therefore it is premature to call fixed-sigma initialization a reliable fix. It may still be useful later, but it should not be the next step before the missing current-model diagnostics are completed.

### 2. The current diagnostics did not complete the planned current-model audit

The current diagnostic script did not fully check:

```text
phase error to true synthetic phase
center cyclic amplitude
z/radius and z/residual compensation
sigma_x correlations with center, z, phase error, RMSE, and projected-radius error
projected radius after subtracting normal*z
projected radius by cycle
Rhat/ESS summaries for the key variables
```

Those checks are needed before selecting a model fix.

## Concern about the current `diagnose_layer2_amplitude.py`

The script is useful but mixes two different tasks:

```text
current-model posterior diagnostics
new fixed-sigma model-variant fit
```

For the next diagnostic pass, avoid running model variants. The goal should be to interrogate the current free-sigma model only.

Also, the current OLS radius diagnostic computes the projection after subtracting center but not after subtracting normal*z. The intended projection diagnostic was:

```text
y[t] = X_fit[t] - center[t] - normal[t] * z[t]
cyclic_unit[t] = e1[t] * cos(phase[t]) + e2[t] * sin(phase[t])
r_projected[t] = dot(y[t], cyclic_unit[t])
```

The current script instead uses:

```text
y[t] = X_fit[t] - center[t]
```

Because z appears small, this may not change the result much, but the diagnostic should be corrected before interpreting it as decisive.

## Recommended next step

Do not change the model yet. Do not run fixed-sigma, MAP, Laplace, adapt_full, ADVI, or other model variants yet.

The next pass should update or replace `docs/debug/scripts/diagnose_layer2_amplitude.py` so that it performs a current-model-only diagnostic audit:

```text
1. phase error to true synthetic phase after best constant circular offset
2. center cyclic amplitude and center/sigma_x correlations
3. z/radius and z/residual compensation diagnostics
4. projected radius after subtracting normal*z
5. projected radius by cycle
6. per-draw correlations of sigma_x with center, z, phase error, RMSE, and projected-radius error
7. Rhat/ESS for sigma_x/rho_x, h_r_knots, q_knots, center, and z
```

Only after those diagnostics should the next model change be chosen.

## Decision criteria for the next pass

Use the following interpretation rules:

```text
If phase error is large or correlates with residuals:
    focus on phase convention or boundary indexing.

If center cyclic amplitude is large or correlates with sigma_x:
    focus on center/radius confounding.

If z_rms or z/radius is large or correlates with sigma_x:
    focus on z/radius/normal compensation.

If projected radius after subtracting normal*z is near 1.0 but posterior radius is wrong:
    focus on sampling or radius/sigma_x parameterization.

If projected radius is far from 1.0 by cycle or phase:
    focus on geometry, phase, center, or frame rather than sigma_x alone.

If all component diagnostics look clean but chains remain stuck with sigma_x/radius anti-correlation:
    then a dedicated initialization or blocked-inference strategy becomes more plausible.
```
