"""
Layer 2 amplitude diagnostics — no model changes, read-only analysis.

Runs one full Layer 1 + Layer 2 fit on the canonical synthetic dataset
(same setup as test_layer2.py) and then applies 7 diagnostics to characterise
the amplitude-noise ridge that prevents sigma_x and radius from converging.

Usage:
    pixi run python docs/debug/scripts/diagnose_layer2_amplitude.py

Output: stdout (tee to a log file for archiving).
"""

import sys, os, time
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _repo_root)

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic data (identical to test_layer2.py)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
fs = 100.0
n_cycles = 6
samples_per_cycle = 100
n_time = n_cycles * samples_per_cycle
t = np.arange(n_time) / fs
phase_true = 2 * np.pi * t          # 6 full cycles
tilt = np.pi / 6
u = np.cos(phase_true)
v = np.sin(phase_true)
X = np.column_stack([u, v * np.cos(tilt), v * np.sin(tilt)])
X += rng.normal(scale=0.02, size=X.shape)

true_normal = np.array([0.0, -np.sin(tilt), np.cos(tilt)])
true_radius = 1.0
true_sigma_x = 0.02

print("=" * 70)
print("LAYER 2 AMPLITUDE DIAGNOSTICS")
print("=" * 70)
print(f"True normal  : {true_normal}")
print(f"True radius  : {true_radius}")
print(f"True sigma_x : {true_sigma_x}")
print()

# ---------------------------------------------------------------------------
# Run Layer 1 + Layer 2
# ---------------------------------------------------------------------------
from phase_coordinates.bayesian import (
    robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
    seed_boundary_indices, _fit_layer1, _fit_layer2, _numba_available,
    construct_frame,
)

R_X, xbar = robust_movement_scale(X)
ref = dominant_reference_signal(X)
T0 = estimate_dominant_period(ref, fs)
tau_idx = seed_boundary_indices(ref, fs, T0)
use_numba = _numba_available()

print(f"R_X = {R_X:.4f}   tau_idx = {tau_idx}")
print()

t0 = time.time()
layer1 = _fit_layer1(
    X, fs, tau_idx, T0, R_X, xbar,
    draws=400, tune=400, chains=2, target_accept=0.9,
    random_seed=0, use_numba=use_numba,
)
print(f"Layer 1 took {time.time()-t0:.1f}s")

t0 = time.time()
layer2 = _fit_layer2(
    X, fs, layer1, T0, R_X, n_velocity_knots=None,
    draws=400, tune=400, chains=2, target_accept=0.9,
    random_seed=0, use_numba=use_numba,
)
print(f"Layer 2 took {time.time()-t0:.1f}s")

idata = layer2.idata
post = idata.posterior

def pmean(name):
    return post[name].mean(("chain", "draw")).values

def pstd(name):
    return post[name].std(("chain", "draw")).values

def psamples(name):
    v = post[name].values            # (chain, draw, ...)
    s = v.shape
    return v.reshape(-1, *s[2:])     # (chain*draw, ...)

n_chains = int(post.sizes["chain"])
n_draws  = int(post.sizes["draw"])
n_samples = n_chains * n_draws
n_time_fit = pmean("radius").shape[0]
t_fit = layer2.time

print()
print(f"Posterior shape: {n_chains} chains × {n_draws} draws = {n_samples} samples")
print(f"n_time_fit = {n_time_fit}")

# Divergence count
n_div = int(idata["sample_stats"]["diverging"].sum().item())
print(f"Divergences: {n_div}")

# Quick summary of key scalars
sigma_x_samples  = psamples("sigma_x")           # (n_samples,)
radius_samples   = psamples("radius")             # (n_samples, n_time_fit)
r_med_samples    = np.median(radius_samples, axis=-1)   # per-draw median radius

print(f"\nPosterior sigma_x : mean={sigma_x_samples.mean():.4f}  "
      f"median={np.median(sigma_x_samples):.4f}  "
      f"sd={sigma_x_samples.std():.4f}  "
      f"[true={true_sigma_x}]")
print(f"Posterior radius median: mean={r_med_samples.mean():.4f}  "
      f"median={np.median(r_med_samples):.4f}  "
      f"sd={r_med_samples.std():.4f}  "
      f"[true={true_radius}]")

# ---------------------------------------------------------------------------
# DIAGNOSTIC 1: Ridge geometry — scatter of (sigma_x, median radius) per draw
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC 1: Amplitude-noise ridge — (sigma_x, r_median) scatter")
print("-" * 70)

product = sigma_x_samples * r_med_samples      # should cluster near r*sigma ≈ const
print(f"r_median * sigma_x : mean={product.mean():.4f}  "
      f"sd={product.std():.4f}  "
      f"true value={true_radius * true_sigma_x:.4f}")

# Correlation between sigma_x and r_median
corr = np.corrcoef(sigma_x_samples, r_med_samples)[0, 1]
print(f"Pearson corr(sigma_x, r_median) = {corr:.4f}  "
      f"[ridge if |corr| >> 0.5]")

# Percentiles of sigma_x
pcts = [5, 25, 50, 75, 95]
pct_vals = np.percentile(sigma_x_samples, pcts)
print(f"sigma_x percentiles ({pcts}): {pct_vals.round(4)}")
pct_r = np.percentile(r_med_samples, pcts)
print(f"r_median  percentiles ({pcts}): {pct_r.round(4)}")

# Check how much of the posterior product r*sigma spans
print(f"r*sigma_x range: [{product.min():.4f}, {product.max():.4f}]  "
      f"(width/mean={product.std()/product.mean():.3f})")

# ---------------------------------------------------------------------------
# DIAGNOSTIC 2: Log-likelihood slice along the ridge vs across it
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC 2: Likelihood geometry — OLS radius estimates")
print("-" * 70)

# For each posterior draw, compute the OLS-optimal radius given that draw's
# trajectory (center, e1, e2, phase) and sigma_x.  If the chain has found
# the right trajectory but wrong amplitude scale, the OLS radius should
# cluster near 1.0 even when posterior radius doesn't.

center_s  = psamples("center")       # (n_samples, n_time_fit, 3)
e1_s      = psamples("e1")           # (n_samples, n_time_fit, 3)
e2_s      = psamples("e2")           # (n_samples, n_time_fit, 3)
phase_s   = psamples("phase")        # (n_samples, n_time_fit)

# Residual from center: y[t] = X_fit[t] - center[s,t]
# Cyclic unit: u_c[s,t] = e1[s,t]*cos(phi[s,t]) + e2[s,t]*sin(phi[s,t])
# OLS alpha: sum_t dot(y_t, u_ct) / sum_t ||u_ct||^2
# But ||u_ct|| = 1 always (e1,e2 orthonormal), so denominator = n_time_fit

# Trim to t_fit extent
i0_fit = int(round(t_fit[0] * fs))
i1_fit = int(round(t_fit[-1] * fs))
X_fit = X[i0_fit : i1_fit + 1]       # (n_time_fit, 3)

ols_radius = np.empty(n_samples)
for s in range(n_samples):
    y     = X_fit - center_s[s]       # (n_time_fit, 3)
    cos_p = np.cos(phase_s[s])        # (n_time_fit,)
    sin_p = np.sin(phase_s[s])        # (n_time_fit,)
    u_c   = e1_s[s] * cos_p[:, None] + e2_s[s] * sin_p[:, None]   # (n_time_fit, 3)
    num   = np.sum(y * u_c)
    den   = np.sum(u_c**2)
    ols_radius[s] = num / den if den > 1e-12 else np.nan

ols_med = np.nanmedian(ols_radius)
ols_mean = np.nanmean(ols_radius)
ols_std  = np.nanstd(ols_radius)
print(f"OLS optimal radius (per draw) : mean={ols_mean:.4f}  "
      f"median={ols_med:.4f}  sd={ols_std:.4f}  [true={true_radius}]")
print(f"Posterior radius mean         : {r_med_samples.mean():.4f}")
print(f"Ratio OLS/posterior           : {ols_med/np.median(r_med_samples):.4f}  "
      f"[1.0 = trajectory is correct, only scale wrong]")

# Fraction of draws where OLS radius is within 10% of true
frac_good = np.mean(np.abs(ols_radius - true_radius) < 0.1 * true_radius)
print(f"Fraction draws with OLS radius within 10% of true: {frac_good:.3f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC 3: Fixed-sigma_x comparison  (sigma_x = true_sigma_x = 0.02)
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC 3: Fixed sigma_x fit (sigma_x = 0.02, no amplitude ridge)")
print("-" * 70)

t0 = time.time()
layer2_fixed = _fit_layer2(
    X, fs, layer1, T0, R_X, n_velocity_knots=None,
    draws=400, tune=400, chains=2, target_accept=0.9,
    random_seed=42, use_numba=use_numba,
    _sigma_x_override=true_sigma_x,
)
elapsed_fixed = time.time() - t0
print(f"Fixed-sigma run took {elapsed_fixed:.1f}s")

idata_fx = layer2_fixed.idata
n_div_fx = int(idata_fx["sample_stats"]["diverging"].sum().item())
post_fx = idata_fx.posterior

r_fx = post_fx["radius"].values.reshape(-1, n_time_fit)
r_med_fx = np.median(r_fx, axis=-1)
print(f"Divergences       : {n_div_fx}")
print(f"Radius median     : mean={r_med_fx.mean():.4f}  "
      f"median={np.median(r_med_fx):.4f}  [true={true_radius}]")
print(f"sigma_x_mean      : {layer2_fixed.sigma_x_mean:.4f}  [true={true_sigma_x}]")

# Normal recovery in fixed-sigma run
cos_sim_fx = np.abs(layer2_fixed.normal_mean @ true_normal)
print(f"Normal cos_sim    : min={cos_sim_fx.min():.4f}  median={np.median(cos_sim_fx):.4f}")

# Phase monotonicity
dphi_fx = np.diff(layer2_fixed.phase_mean)
print(f"Phase monotone    : {np.all(dphi_fx >= -1e-6)}  min dphi={dphi_fx.min():.6f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC 4: Per-chain amplitude bias
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC 4: Per-chain radius and sigma_x statistics")
print("-" * 70)

for ch in range(n_chains):
    r_ch     = post["radius"].values[ch]          # (draw, n_time_fit)
    sig_ch   = post["sigma_x"].values[ch]         # (draw,)
    r_med_ch = np.median(r_ch, axis=-1)           # (draw,)
    print(f"  Chain {ch}: r_median  mean={r_med_ch.mean():.4f}  "
          f"sd={r_med_ch.std():.4f}  "
          f"| sigma_x mean={sig_ch.mean():.4f}  sd={sig_ch.std():.4f}"
          f"  | r*sigma mean={( r_med_ch * sig_ch).mean():.4f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC 5: Warmup trajectory — did the chain ever visit the true mode?
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC 5: Trace trajectory for sigma_x and r_median")
print("-" * 70)

# Check the first and last 50 draws of each chain
for ch in range(n_chains):
    sig_ch = post["sigma_x"].values[ch]           # (draw,)
    r_ch   = np.median(post["radius"].values[ch], axis=-1)   # (draw,)
    print(f"  Chain {ch} first 50 draws:  "
          f"sigma_x [{sig_ch[:50].min():.3f}, {sig_ch[:50].max():.3f}]  "
          f"r [{r_ch[:50].min():.3f}, {r_ch[:50].max():.3f}]")
    print(f"  Chain {ch} last  50 draws:  "
          f"sigma_x [{sig_ch[-50:].min():.3f}, {sig_ch[-50:].max():.3f}]  "
          f"r [{r_ch[-50:].min():.3f}, {r_ch[-50:].max():.3f}]")
    # fraction of draws where sigma_x is within 2x of true
    frac = np.mean(sig_ch < 4 * true_sigma_x)
    print(f"  Chain {ch} fraction draws sigma_x < 4*true (={4*true_sigma_x}): {frac:.3f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC 6: Residual geometry — does the trajectory direction matter?
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC 6: Residuals breakdown — trajectory vs amplitude error")
print("-" * 70)

# For the mean posterior trajectory:
#   total_resid[t] = X_fit[t] - pred_mean[t]
# Decompose into:
#   normal component   : dot(resid, n_mean)
#   cyclic component   : dot(resid, u_c_mean)   [along instantaneous orbit direction]
#   tangential component : dot(resid, u_t_mean) [perpendicular to orbit in plane]

pred_mean = pmean("predicted_trajectory")    # (n_time_fit, 3)
n_mean    = layer2.normal_mean               # (n_time_fit, 3)
e1_mean   = layer2.e1_mean                  # (n_time_fit, 3)
e2_mean   = layer2.e2_mean                  # (n_time_fit, 3)
phi_mean  = layer2.phase_mean               # (n_time_fit,)

resid = X_fit - pred_mean                   # (n_time_fit, 3)

u_c_mean  = e1_mean * np.cos(phi_mean)[:, None] + e2_mean * np.sin(phi_mean)[:, None]
u_t_mean  = -e1_mean * np.sin(phi_mean)[:, None] + e2_mean * np.cos(phi_mean)[:, None]

resid_n   = np.sum(resid * n_mean,   axis=-1)   # (n_time_fit,)
resid_c   = np.sum(resid * u_c_mean, axis=-1)
resid_t   = np.sum(resid * u_t_mean, axis=-1)
resid_tot = np.linalg.norm(resid, axis=-1)

print(f"  RMS total residual           : {np.sqrt(np.mean(resid_tot**2)):.4f}  [true sigma_x={true_sigma_x}]")
print(f"  RMS residual (normal dir)    : {np.sqrt(np.mean(resid_n**2)):.4f}")
print(f"  RMS residual (cyclic dir)    : {np.sqrt(np.mean(resid_c**2)):.4f}")
print(f"  RMS residual (tangential)    : {np.sqrt(np.mean(resid_t**2)):.4f}")

# What the cyclic residual should look like if radius is wrong
r_median_est = np.median(layer2.radius_mean)
resid_bias_expected = true_radius - r_median_est
print(f"\n  Expected cyclic residual from radius bias: "
      f"r_true - r_est = {resid_bias_expected:.4f}")
print(f"  Observed mean(resid_c) = {resid_c.mean():.4f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC 7: Log-likelihood at true params vs posterior mean params
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC 7: Log-likelihood comparison — true vs estimated parameters")
print("-" * 70)

def gaussian_loglik(X_obs, pred, sigma):
    diff = X_obs - pred
    return -0.5 * np.sum(diff**2) / sigma**2 - 0.5 * X_obs.size * np.log(2 * np.pi * sigma**2)

# Build true prediction using true params
# True: center=0, normal=[0,-sin(tilt),cos(tilt)], e1=[1,0,0], e2=[0,cos(tilt),sin(tilt)]
# radius=1, phase = phase_true[i0_fit:i1_fit+1]
i0_fit_idx = int(round(t_fit[0] * fs))
i1_fit_idx = int(round(t_fit[-1] * fs))
phase_fit_true = phase_true[i0_fit_idx : i1_fit_idx + 1]

e1_true = np.array([1.0, 0.0, 0.0])
e2_true = np.array([0.0, np.cos(tilt), np.sin(tilt)])
pred_true = (
    np.zeros((n_time_fit, 3))                               # center = 0
    + e1_true[None, :] * np.cos(phase_fit_true)[:, None]
    + e2_true[None, :] * np.sin(phase_fit_true)[:, None]
)

ll_true = gaussian_loglik(X_fit, pred_true, true_sigma_x)

# Posterior-mean prediction
ll_post = gaussian_loglik(X_fit, pred_mean, layer2.sigma_x_mean)

# Fixed-sigma posterior mean
pred_fx_mean = layer2_fixed.predicted_trajectory_mean
ll_fx = gaussian_loglik(X_fit, pred_fx_mean, true_sigma_x)

# Posterior-mean prediction evaluated at true sigma_x
ll_post_at_true_sigma = gaussian_loglik(X_fit, pred_mean, true_sigma_x)

print(f"  log-lik (true params, true sigma_x)         : {ll_true:.1f}")
print(f"  log-lik (posterior mean, posterior sigma_x)  : {ll_post:.1f}")
print(f"  log-lik (fixed-sigma posterior mean, 0.02)   : {ll_fx:.1f}")
print(f"  log-lik (posterior mean, true sigma_x=0.02)  : {ll_post_at_true_sigma:.1f}")
print(f"  gap (true - posterior)                       : {ll_true - ll_post:.1f} nats")
print(f"  gap (true - fixed-sigma posterior)           : {ll_true - ll_fx:.1f} nats")

# Also check: what sigma_x maximises lik for the posterior mean trajectory?
# MLE sigma = sqrt(MSE per observation)
mse_post = np.mean((X_fit - pred_mean)**2)
sigma_mle_post = np.sqrt(mse_post)
print(f"\n  MLE sigma_x for posterior-mean trajectory : {sigma_mle_post:.4f}  [true={true_sigma_x}]")
mse_true = np.mean((X_fit - pred_true)**2)
sigma_mle_true = np.sqrt(mse_true)
print(f"  MLE sigma_x for true trajectory           : {sigma_mle_true:.4f}")

# Ridge characterisation: evaluate log-lik at several (r, sigma) pairs along the
# ridge r * sigma ≈ r_est * sigma_est, using the posterior-mean trajectory scaled
# by r / r_est.
r_est = np.median(layer2.radius_mean)
sig_est = layer2.sigma_x_mean

print(f"\n  Ridge scan (r * sigma ≈ {r_est*sig_est:.4f}):")
print(f"  {'r_scale':>8}  {'r_val':>8}  {'sigma_x':>10}  {'log_lik':>12}")

# posterior-mean trajectory without the radius factor (just directions)
# pred_mean = center + r * (e1*cos(phi) + e2*sin(phi)) + n*z
# cyclic part: (e1*cos + e2*sin) scaled by r
center_mean_arr = pmean("center")
perp_mean_arr   = pmean("perp_deviation")
n_mean_arr      = layer2.normal_mean

for r_scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
    r_new   = r_est * r_scale
    sig_new = (r_est * sig_est) / r_new          # keep r * sigma constant
    # Rebuild pred at new r scale
    r_mean_arr = layer2.radius_mean * r_scale
    pred_scaled = (
        center_mean_arr
        + e1_mean * (r_mean_arr * np.cos(phi_mean))[:, None]
        + e2_mean * (r_mean_arr * np.sin(phi_mean))[:, None]
        + n_mean_arr * perp_mean_arr[:, None]
    )
    ll_scaled = gaussian_loglik(X_fit, pred_scaled, sig_new)
    print(f"  {r_scale:>8.2f}  {r_new:>8.4f}  {sig_new:>10.4f}  {ll_scaled:>12.1f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nFull-model fit (free sigma_x):")
print(f"  Divergences : {n_div}")
print(f"  sigma_x     : {layer2.sigma_x_mean:.4f}  [true={true_sigma_x}]  "
      f"factor={layer2.sigma_x_mean/true_sigma_x:.1f}x too large")
print(f"  radius med  : {np.median(layer2.radius_mean):.4f}  [true={true_radius}]")
print(f"  r*sigma     : {np.median(layer2.radius_mean)*layer2.sigma_x_mean:.4f}  "
      f"[true={true_radius*true_sigma_x:.4f}]")
print(f"  Normal cos_sim min : {np.abs(layer2.normal_mean @ true_normal).min():.4f}")

print(f"\nFixed-sigma_x fit (sigma_x = {true_sigma_x}):")
print(f"  Divergences : {n_div_fx}")
print(f"  radius med  : {np.median(r_fx):.4f}  [true={true_radius}]")
print(f"  Normal cos_sim min : {cos_sim_fx.min():.4f}")

print(f"\nRidge correlation (sigma_x vs r_median): {corr:.4f}")
print(f"OLS radius (from trajectory direction): {ols_med:.4f}  "
      f"[quantifies whether trajectory direction is right]")
print(f"MLE sigma for posterior-mean trajectory: {sigma_mle_post:.4f}  "
      f"[should be ~{true_sigma_x} if trajectory is correct]")

print()
print("ALL DIAGNOSTICS COMPLETE")
