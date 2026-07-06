# Claude Prompt: Complete Current-Model Layer 2 Diagnostics

You are working in `Motor-Learning-Lab/phase-coordinates` on branch `bayesian-two-layer-estimator`.

Read first:

```text
docs/layer2_amplitude_diagnostics_review.md
docs/PROGRESS.md
docs/debug/README.md
docs/debug/scripts/diagnose_layer2_amplitude.py
phase_coordinates/bayesian.py
```

The review note records the current interpretation: the latest diagnostics show invalid current posterior samples and large in-plane residuals, but they do not yet prove that fixed-sigma or MAP initialization is the correct next fix.

## Hard boundary for this pass

This pass is diagnostics only.

Use the current free-sigma Layer 2 model. Do not change the model and do not run alternative model variants. In particular, do not run fixed-sigma fits, MAP, Laplace, adapt_full, ADVI, four-condition comparisons, frozen-center fits, z-fixed fits, or altered-prior fits.

The previous diagnostic script accidentally included a fixed-sigma variant. Remove or disable that section before running the follow-up diagnostics.

## Goal

Complete the missing current-model diagnostics. The existing evidence says:

```text
sigma_x and radius are strongly anti-correlated
chains are stuck in different amplitude/noise regions
posterior-mean trajectory has large in-plane residuals
normal recovery is good
```

Now determine whether the amplitude/noise failure is coupled to phase error, center cyclic motion, z compensation, or projected-radius mismatch.

## Sampling rule

Use existing current-model posterior samples if available. If none are available, run exactly one current-model fit using the same free-sigma configuration as `docs/debug/scripts/test_layer2.py`. Do not pass any diagnostic or override arguments to the Layer 2 fitting function.

## File to update

Update:

```text
docs/debug/scripts/diagnose_layer2_amplitude.py
```

The updated script should perform only current-model diagnostics. It should not run fixed-sigma or other model variants.

Archive output to a new log such as:

```text
docs/debug/logs/12_layer2_current_model_diagnostics.log
```

## Diagnostic A: chain quality

Print:

```text
divergences per chain
max treedepth hits per chain
Rhat and ESS for sigma_x or rho_x
Rhat and ESS for h_r_knots
Rhat and ESS for q_knots
Rhat and ESS for c2
Rhat and ESS for h_z_knots
per-chain sigma_x mean, median, 5%, 95%
per-chain radius median-over-time mean, median, 5%, 95%
```

Purpose: distinguish a stable amplitude/noise structure from chain-specific pathology and bad summaries.

## Diagnostic B: phase error to true phase

The synthetic data uses:

```text
phase_true = 2*pi*t
```

Use the fitted time window. Estimate the best constant circular offset between estimated phase and true phase, then compute wrapped phase errors after removing that offset.

Print:

```text
best phase offset
median absolute phase error
95% absolute phase error
phase error by cycle
phase error near boundaries
correlation between absolute phase error and residual norm
correlation between absolute phase error and absolute projected-radius error
```

Purpose: check whether the boundary-normalized phase is misindexed, shifted, or distorted enough to drive in-plane residuals.

## Diagnostic C: center cyclic amplitude

Using the current posterior only, compute:

```text
cyclic_unit[t] = e1[t] * cos(phase[t]) + e2[t] * sin(phase[t])
center_cyclic_projection[t] = dot(center[t], cyclic_unit[t])
```

Also regress each center coordinate on `cos(phase[t])` and `sin(phase[t])`.

Print:

```text
center_rms
center_drift
median abs center_cyclic_projection
center cyclic amplitude by coordinate
total center cyclic amplitude
center cyclic amplitude / median radius
correlation of sigma_x with center_rms
correlation of sigma_x with center cyclic amplitude
correlation of radius median with center_rms
correlation of radius median with center cyclic amplitude
```

Purpose: check whether center(t) is absorbing cyclic motion that should belong to radius.

## Diagnostic D: z and perpendicular compensation

Using the current posterior only, print:

```text
median abs z
95% abs z
z_rms
median abs(z / radius)
percent time abs(z / radius) > 0.25
percent time abs(z / radius) > 0.50
```

Project residuals onto the current frame:

```text
residual_normal[t] = dot(residual[t], normal[t])
residual_e1[t] = dot(residual[t], e1[t])
residual_e2[t] = dot(residual[t], e2[t])
residual_cyclic[t] = dot(residual[t], cyclic_unit[t])
residual_tangential[t] = dot(residual[t], tangential_unit[t])
```

Print RMS values for each residual component and correlations between z summaries and sigma_x/radius.

Purpose: check whether perpendicular deviation is compensating for amplitude, normal, or frame errors.

## Diagnostic E: corrected projected radius

Correct the previous projected-radius diagnostic by subtracting the normal*z component:

```text
y[t] = X_fit[t] - center[t] - normal[t] * z[t]
cyclic_unit[t] = e1[t] * cos(phase[t]) + e2[t] * sin(phase[t])
r_projected[t] = dot(y[t], cyclic_unit[t])
```

Print:

```text
median r_projected
mean r_projected
5%, 95% r_projected
r_projected by cycle
r_projected by phase bin
median fitted radius
correlation between r_projected[t] and fitted radius[t]
best scalar alpha for fitted radius
RMSE at alpha = 1
RMSE at best alpha
```

Purpose: check whether the current phase/frame/center/z geometry implies radius near 1.0.

## Diagnostic F: sigma_x correlation table

Build a posterior-draw table with one row per draw. Include, where feasible:

```text
sigma_x
log_sigma_x
radius_median
radius_mean
radius_sd_over_time
RMSE
RMSE_normal
RMSE_cyclic
RMSE_tangential
center_rms
center_cyclic_amplitude
z_rms
median_abs_z_over_radius
phase_error_median
phase_error_95
projected_radius_median
projected_radius_error
normal_resultant_length_min
```

Print the top 15 strongest absolute correlations with sigma_x and log_sigma_x.

Purpose: identify which latent component is actually coupled to high sigma_x.

## Documentation updates

Update `docs/PROGRESS.md` with a new finding titled:

```text
13. Current-model component diagnostics
```

Update `docs/debug/README.md` to list the revised diagnostic script and new log.

Do not claim a model fix unless the diagnostics directly support it. If the diagnostics are ambiguous, say so.

## Final response expected

Report:

```text
1. Whether a new current-model fit was run or existing samples were reused
2. Commands run
3. Chain quality results
4. Phase error results
5. Center cyclic-amplitude results
6. z/perpendicular results
7. Corrected projected-radius results
8. sigma_x correlation findings
9. Most likely failure point now
10. What remains unresolved
11. Files changed
```
