"""
Layer 2 current-model component diagnostics (pass 2).

Current free-sigma model only — no model variants. Diagnostics A-F:
  A: chain quality (Rhat, ESS, per-chain sigma_x and radius)
  B: phase error to true synthetic phase
  C: center cyclic amplitude and center/sigma_x correlations
  D: z/perpendicular compensation
  E: corrected projected radius (subtract normal*z before projecting)
  F: sigma_x correlation table across per-draw summary statistics

Usage:
    pixi run python docs/debug/scripts/diagnose_layer2_amplitude.py 2>&1 | tee /tmp/diag12.log
"""

import sys, os, time
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _repo_root)

import numpy as np
import arviz as az

# ---------------------------------------------------------------------------
# Synthetic data (identical to test_layer2.py)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(0)
fs = 100.0
n_cycles = 6
samples_per_cycle = 100
n_time = n_cycles * samples_per_cycle
t = np.arange(n_time) / fs
phase_true = 2 * np.pi * t
tilt = np.pi / 6
u = np.cos(phase_true)
v = np.sin(phase_true)
X = np.column_stack([u, v * np.cos(tilt), v * np.sin(tilt)])
X += rng.normal(scale=0.02, size=X.shape)

true_normal = np.array([0.0, -np.sin(tilt), np.cos(tilt)])
true_radius = 1.0
true_sigma_x = 0.02

print("=" * 70)
print("LAYER 2 CURRENT-MODEL COMPONENT DIAGNOSTICS")
print("=" * 70)
print(f"True normal  : {true_normal}")
print(f"True radius  : {true_radius}")
print(f"True sigma_x : {true_sigma_x}")
print()

# ---------------------------------------------------------------------------
# Run Layer 1 + Layer 2 (free sigma, no overrides)
# ---------------------------------------------------------------------------
from phase_coordinates.bayesian import (
    robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
    seed_boundary_indices, _fit_layer1, _fit_layer2, _numba_available,
)

R_X, xbar = robust_movement_scale(X)
ref = dominant_reference_signal(X)
T0 = estimate_dominant_period(ref, fs)
tau_idx = seed_boundary_indices(ref, fs, T0)
use_numba = _numba_available()

print(f"R_X = {R_X:.4f}   T0 = {T0:.4f}s   tau_idx = {tau_idx}")
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
post  = idata.posterior

def pmean(name):
    return post[name].mean(("chain", "draw")).values

def psamples(name):
    v = post[name].values
    return v.reshape(-1, *v.shape[2:])

n_chains   = int(post.sizes["chain"])
n_draws    = int(post.sizes["draw"])
n_samples  = n_chains * n_draws
t_fit      = layer2.time
n_time_fit = len(t_fit)

i0_fit = int(round(t_fit[0] * fs))
i1_fit = int(round(t_fit[-1] * fs))
X_fit  = X[i0_fit : i1_fit + 1]   # (n_time_fit, 3)
phase_true_fit = phase_true[i0_fit : i1_fit + 1]  # (n_time_fit,)

print(f"\nPosterior: {n_chains} chains × {n_draws} draws = {n_samples} samples")
print(f"Fit window: t=[{t_fit[0]:.3f}, {t_fit[-1]:.3f}]s   n_time_fit={n_time_fit}")
print()

# ---------------------------------------------------------------------------
# Pre-load per-draw arrays (used across multiple diagnostics)
# ---------------------------------------------------------------------------
sigma_x_s   = psamples("sigma_x")           # (S,)
rho_x_s     = psamples("rho_x")             # (S,)
radius_s    = psamples("radius")            # (S, T)
center_s    = psamples("center")            # (S, T, 3)
normal_s    = psamples("normal")            # (S, T, 3)
e1_s        = psamples("e1")               # (S, T, 3)
e2_s        = psamples("e2")               # (S, T, 3)
phase_s     = psamples("phase")            # (S, T)
z_s         = psamples("perp_deviation")   # (S, T)
pred_s      = psamples("predicted_trajectory")  # (S, T, 3)

r_med_s     = np.median(radius_s, axis=1)  # (S,)

# Cyclic and tangential unit vectors per draw
cos_p_s = np.cos(phase_s)   # (S, T)
sin_p_s = np.sin(phase_s)   # (S, T)
u_c_s = e1_s * cos_p_s[:, :, None] + e2_s * sin_p_s[:, :, None]  # (S, T, 3)
u_t_s = -e1_s * sin_p_s[:, :, None] + e2_s * cos_p_s[:, :, None] # (S, T, 3)

# Posterior-mean arrays (for residual decomposition at the mean)
phi_mean    = layer2.phase_mean             # (T,)
n_mean      = layer2.normal_mean            # (T, 3)
e1_mean     = layer2.e1_mean               # (T, 3)
e2_mean     = layer2.e2_mean               # (T, 3)
pred_mean   = pmean("predicted_trajectory") # (T, 3)
z_mean      = pmean("perp_deviation")       # (T,)
center_mean = pmean("center")              # (T, 3)
r_mean_t    = pmean("radius")              # (T,)

u_c_mean = e1_mean * np.cos(phi_mean)[:, None] + e2_mean * np.sin(phi_mean)[:, None]
u_t_mean = -e1_mean * np.sin(phi_mean)[:, None] + e2_mean * np.cos(phi_mean)[:, None]

# Per-draw residuals
resid_s = X_fit[None, :, :] - pred_s   # (S, T, 3)

# Cycle membership (for by-cycle breakdowns)
tau_mean = layer1.tau_mean
K = len(tau_mean)

# ---------------------------------------------------------------------------
# DIAGNOSTIC A: Chain quality
# ---------------------------------------------------------------------------
print("-" * 70)
print("DIAGNOSTIC A: Chain quality")
print("-" * 70)

sample_stats = idata["sample_stats"]
div_arr  = sample_stats["diverging"].values   # (chain, draw)
td_arr   = sample_stats["tree_depth"].values  # (chain, draw)
max_td   = 10  # PyMC default

print(f"\nDivergences per chain:")
for ch in range(n_chains):
    n_div = div_arr[ch].sum()
    td_hits = (td_arr[ch] >= max_td).sum()
    print(f"  Chain {ch}: {n_div} divergences,  {td_hits} max-treedepth hits (depth>={max_td})")
print(f"  Total  : {div_arr.sum()} divergences")

print(f"\nRhat and ESS (free variables):")
try:
    summ = az.summary(
        idata,
        var_names=["rho_x", "h_r_knots", "q_knots", "c2", "h_z_knots"],
    )
    for vname in ["rho_x", "h_r_knots", "q_knots", "c2", "h_z_knots"]:
        rows = summ[summ.index.str.startswith(vname)]
        if len(rows) == 0:
            continue
        rhat_col = "r_hat" if "r_hat" in rows.columns else "rhat"
        ess_col  = "ess_bulk" if "ess_bulk" in rows.columns else "ess"
        rhat_vals = rows[rhat_col].values
        ess_vals  = rows[ess_col].values
        print(f"  {vname:12s}  n_params={len(rows):3d}  "
              f"rhat=[{rhat_vals.min():.3f},{rhat_vals.max():.3f}]  "
              f"ess_bulk=[{ess_vals.min():.0f},{ess_vals.max():.0f}]")
except Exception as e:
    print(f"  az.summary failed: {e}")
    print("  Manual rhat for rho_x:")
    rho_chains = post["rho_x"].values   # (chain, draw)
    chain_means = rho_chains.mean(axis=1)
    B = n_draws * np.var(chain_means, ddof=1)
    W = np.mean(rho_chains.var(axis=1, ddof=1))
    var_hat = (n_draws - 1) / n_draws * W + B / n_draws
    rhat_rho = np.sqrt(var_hat / W) if W > 0 else float("nan")
    print(f"  rho_x manual rhat = {rhat_rho:.4f}")

print(f"\nPer-chain sigma_x (mean, median, 5%, 95%):")
for ch in range(n_chains):
    s = post["sigma_x"].values[ch]
    print(f"  Chain {ch}: mean={s.mean():.4f}  med={np.median(s):.4f}  "
          f"[{np.percentile(s,5):.4f}, {np.percentile(s,95):.4f}]")

print(f"\nPer-chain radius median-over-time (mean, median, 5%, 95%):")
for ch in range(n_chains):
    r = np.median(post["radius"].values[ch], axis=1)   # (draw,) median over time
    print(f"  Chain {ch}: mean={r.mean():.4f}  med={np.median(r):.4f}  "
          f"[{np.percentile(r,5):.4f}, {np.percentile(r,95):.4f}]")

# ---------------------------------------------------------------------------
# DIAGNOSTIC B: Phase error to true synthetic phase
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC B: Phase error to true synthetic phase")
print("-" * 70)

# Per-draw: best constant offset = mean(phase_est - phase_true) over time
phase_err_s    = phase_s - phase_true_fit[None, :]        # (S, T)
phase_offset_s = phase_err_s.mean(axis=1)                 # (S,) best constant offset
phase_err_corr_s = phase_err_s - phase_offset_s[:, None]  # (S, T) corrected errors

print(f"\nBest phase offset (per-draw mean error):")
print(f"  Mean over draws : {phase_offset_s.mean():.4f} rad")
print(f"  SD over draws   : {phase_offset_s.std():.4f} rad")
print(f"  Range           : [{phase_offset_s.min():.4f}, {phase_offset_s.max():.4f}] rad")

abs_err_s = np.abs(phase_err_corr_s)   # (S, T)
med_abs_err_s  = np.median(abs_err_s, axis=1)   # (S,)
p95_abs_err_s  = np.percentile(abs_err_s, 95, axis=1)  # (S,)

print(f"\nWrapped phase error (after removing best constant offset):")
print(f"  Median abs error — mean over draws: {med_abs_err_s.mean():.4f} rad  "
      f"sd: {med_abs_err_s.std():.4f}")
print(f"  95th pct abs error — mean: {p95_abs_err_s.mean():.4f} rad  "
      f"sd: {p95_abs_err_s.std():.4f}")

# Posterior-mean phase error
phi_err_mean = layer2.phase_mean - phase_true_fit
phi_offset   = phi_err_mean.mean()
phi_err_corr = phi_err_mean - phi_offset
print(f"\nPosterior-mean phase error (offset={phi_offset:.4f} rad):")
print(f"  Median abs: {np.median(np.abs(phi_err_corr)):.4f} rad")
print(f"  Max abs   : {np.max(np.abs(phi_err_corr)):.4f} rad")
print(f"  RMS       : {np.sqrt(np.mean(phi_err_corr**2)):.4f} rad")

# Phase error by cycle
print(f"\nPhase error by cycle (posterior mean, median abs within cycle):")
for k in range(K - 1):
    mask = (t_fit >= tau_mean[k]) & (t_fit < tau_mean[k + 1])
    if mask.sum() == 0:
        continue
    cycle_err = phi_err_corr[mask]
    print(f"  Cycle {k}: n={mask.sum():3d}  "
          f"median_abs={np.median(np.abs(cycle_err)):.4f}  "
          f"max_abs={np.max(np.abs(cycle_err)):.4f} rad")

# Phase error near boundaries (±5 samples from each tau_k)
print(f"\nPhase error near boundaries (|err| within ±5 samples of tau_k):")
for k in range(K):
    idx_k = int(round((tau_mean[k] - t_fit[0]) * fs))
    idx_lo = max(0, idx_k - 5)
    idx_hi = min(n_time_fit - 1, idx_k + 5)
    near_err = phi_err_corr[idx_lo : idx_hi + 1]
    if len(near_err) == 0:
        print(f"  tau_{k} (t={tau_mean[k]:.3f}s): outside fit window")
        continue
    print(f"  tau_{k} (t={tau_mean[k]:.3f}s): max|err|={np.max(np.abs(near_err)):.5f} rad")

# Correlation: |phase_error| vs |residual| and vs |r_proj_error| (use posterior mean)
resid_norm_mean = np.linalg.norm(X_fit - pred_mean, axis=-1)  # (T,)
corr_err_resid = np.corrcoef(np.abs(phi_err_corr), resid_norm_mean)[0, 1]
print(f"\nCorr(|phase_error|, |residual_norm|) [posterior mean]: {corr_err_resid:.4f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC C: Center cyclic amplitude
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC C: Center cyclic amplitude")
print("-" * 70)

# Per-draw
center_rms_s       = np.sqrt(np.mean(np.sum(center_s**2, axis=-1), axis=1))  # (S,)
center_cyc_proj_s  = np.sum(center_s * u_c_s, axis=-1)   # (S, T)
center_cyc_amp_s   = np.sqrt(np.mean(center_cyc_proj_s**2, axis=1))  # (S,) RMS

# Posterior-mean center stats
c_rms_mean = np.sqrt(np.mean(np.sum(center_mean**2, axis=-1)))
c_drift    = np.linalg.norm(center_mean[-1] - center_mean[0])
c_cyc_proj = np.sum(center_mean * u_c_mean, axis=-1)  # (T,)

print(f"\nPosterior-mean center:")
print(f"  center_rms         : {c_rms_mean:.5f}")
print(f"  center_drift (end-start): {c_drift:.5f}")
print(f"  center cyclic projection RMS: {np.sqrt(np.mean(c_cyc_proj**2)):.5f}")

# Center regression on cos(phi) and sin(phi) per coordinate
print(f"\nCenter regression on cos/sin(phase) [posterior mean]:")
A_reg = np.column_stack([np.cos(phi_mean), np.sin(phi_mean)])
for coord, name in enumerate(["x", "y", "z"]):
    coeffs, _, _, _ = np.linalg.lstsq(A_reg, center_mean[:, coord], rcond=None)
    amp = np.sqrt(coeffs[0]**2 + coeffs[1]**2)
    print(f"  center_{name}: amp={amp:.5f}  (cos_coef={coeffs[0]:.5f}, sin_coef={coeffs[1]:.5f})")
total_center_cyc_amp = np.sqrt(sum(
    np.linalg.lstsq(A_reg, center_mean[:, c], rcond=None)[0][:2] @
    np.linalg.lstsq(A_reg, center_mean[:, c], rcond=None)[0][:2]
    for c in range(3)
))
print(f"  total center cyclic amplitude : {total_center_cyc_amp:.5f}")
print(f"  / median radius               : {total_center_cyc_amp / np.median(r_mean_t):.5f}")

# Per-draw stats
print(f"\nPer-draw center_rms:       mean={center_rms_s.mean():.5f}  "
      f"sd={center_rms_s.std():.5f}  "
      f"[{np.percentile(center_rms_s,5):.5f}, {np.percentile(center_rms_s,95):.5f}]")
print(f"Per-draw center_cyc_amp:   mean={center_cyc_amp_s.mean():.5f}  "
      f"sd={center_cyc_amp_s.std():.5f}  "
      f"[{np.percentile(center_cyc_amp_s,5):.5f}, {np.percentile(center_cyc_amp_s,95):.5f}]")

corr_sig_crms = np.corrcoef(sigma_x_s, center_rms_s)[0, 1]
corr_sig_cca  = np.corrcoef(sigma_x_s, center_cyc_amp_s)[0, 1]
corr_r_crms   = np.corrcoef(r_med_s, center_rms_s)[0, 1]
corr_r_cca    = np.corrcoef(r_med_s, center_cyc_amp_s)[0, 1]
print(f"\nCorr(sigma_x, center_rms)        : {corr_sig_crms:.4f}")
print(f"Corr(sigma_x, center_cyc_amp)    : {corr_sig_cca:.4f}")
print(f"Corr(r_median, center_rms)       : {corr_r_crms:.4f}")
print(f"Corr(r_median, center_cyc_amp)   : {corr_r_cca:.4f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC D: z / perpendicular deviation compensation
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC D: z / perpendicular compensation")
print("-" * 70)

# Per-draw z stats
z_rms_s          = np.sqrt(np.mean(z_s**2, axis=1))          # (S,)
z_over_r_s       = np.abs(z_s) / (radius_s + 1e-12)          # (S, T)
med_zr_s         = np.median(z_over_r_s, axis=1)             # (S,)
frac_zr_25_s     = np.mean(z_over_r_s > 0.25, axis=1)        # (S,)
frac_zr_50_s     = np.mean(z_over_r_s > 0.50, axis=1)        # (S,)

print(f"\nPer-draw z statistics:")
print(f"  median |z|         : mean={np.median(np.abs(z_s), axis=1).mean():.4f}  "
      f"sd={np.median(np.abs(z_s), axis=1).std():.4f}")
print(f"  95th pct |z|       : mean={np.percentile(np.abs(z_s),95,axis=1).mean():.4f}")
print(f"  z_rms              : mean={z_rms_s.mean():.4f}  sd={z_rms_s.std():.4f}")
print(f"  median |z/r|       : mean={med_zr_s.mean():.4f}  sd={med_zr_s.std():.4f}")
print(f"  %% time |z/r|>0.25 : mean={frac_zr_25_s.mean()*100:.1f}%%")
print(f"  %% time |z/r|>0.50 : mean={frac_zr_50_s.mean()*100:.1f}%%")

# Residual decomposition per draw
resid_n_s = np.sum(resid_s * normal_s, axis=-1)   # (S, T)
resid_c_s = np.sum(resid_s * u_c_s,   axis=-1)   # (S, T)
resid_t_s = np.sum(resid_s * u_t_s,   axis=-1)   # (S, T)
resid_e1_s = np.sum(resid_s * e1_s,   axis=-1)   # (S, T)
resid_e2_s = np.sum(resid_s * e2_s,   axis=-1)   # (S, T)

rmse_tot_s = np.sqrt(np.mean(np.sum(resid_s**2, axis=-1), axis=1))  # (S,)
rmse_n_s   = np.sqrt(np.mean(resid_n_s**2, axis=1))
rmse_c_s   = np.sqrt(np.mean(resid_c_s**2, axis=1))
rmse_t_s   = np.sqrt(np.mean(resid_t_s**2, axis=1))

print(f"\nPer-draw residual RMS decomposition:")
print(f"  RMSE total    : mean={rmse_tot_s.mean():.4f}  sd={rmse_tot_s.std():.4f}  [true sigma={true_sigma_x}]")
print(f"  RMSE normal   : mean={rmse_n_s.mean():.4f}  sd={rmse_n_s.std():.4f}")
print(f"  RMSE cyclic   : mean={rmse_c_s.mean():.4f}  sd={rmse_c_s.std():.4f}")
print(f"  RMSE tangential: mean={rmse_t_s.mean():.4f}  sd={rmse_t_s.std():.4f}")

# Posterior-mean residual decomposition
resid_pm = X_fit - pred_mean
resid_n_pm = np.sum(resid_pm * n_mean,    axis=-1)
resid_c_pm = np.sum(resid_pm * u_c_mean,  axis=-1)
resid_t_pm = np.sum(resid_pm * u_t_mean,  axis=-1)
print(f"\nPosterior-mean residual RMS:")
print(f"  total={np.sqrt(np.mean(np.sum(resid_pm**2,axis=-1))):.4f}  "
      f"normal={np.sqrt(np.mean(resid_n_pm**2)):.4f}  "
      f"cyclic={np.sqrt(np.mean(resid_c_pm**2)):.4f}  "
      f"tangential={np.sqrt(np.mean(resid_t_pm**2)):.4f}")
print(f"  mean(resid_c) = {resid_c_pm.mean():.4f}  [expected r_est - r_true = "
      f"{np.median(r_mean_t) - true_radius:.4f}]")

corr_sig_z   = np.corrcoef(sigma_x_s, z_rms_s)[0, 1]
corr_sig_zr  = np.corrcoef(sigma_x_s, med_zr_s)[0, 1]
corr_r_z     = np.corrcoef(r_med_s, z_rms_s)[0, 1]
corr_r_zr    = np.corrcoef(r_med_s, med_zr_s)[0, 1]
print(f"\nCorr(sigma_x, z_rms)         : {corr_sig_z:.4f}")
print(f"Corr(sigma_x, median|z/r|)   : {corr_sig_zr:.4f}")
print(f"Corr(r_median, z_rms)        : {corr_r_z:.4f}")
print(f"Corr(r_median, median|z/r|)  : {corr_r_zr:.4f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC E: Corrected projected radius
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC E: Corrected projected radius (subtract normal*z)")
print("-" * 70)

# y[s,t] = X_fit[t] - center[s,t] - normal[s,t]*z[s,t]
y_s   = X_fit[None, :, :] - center_s - normal_s * z_s[:, :, None]  # (S, T, 3)
r_proj_s = np.sum(y_s * u_c_s, axis=-1)   # (S, T)
r_proj_med_s = np.median(r_proj_s, axis=1)  # (S,) median over time
r_proj_mean_s = r_proj_s.mean(axis=1)

print(f"\nCorrected projected radius (posterior median over draws):")
print(f"  median r_projected : {np.median(r_proj_med_s):.4f}  [true={true_radius}]")
print(f"  mean r_projected   : {np.median(r_proj_mean_s):.4f}")
print(f"  5%, 95%            : [{np.percentile(r_proj_med_s,5):.4f}, "
      f"{np.percentile(r_proj_med_s,95):.4f}]")
print(f"  SD over draws      : {r_proj_med_s.std():.4f}")

# Posterior-mean r_projected
y_pm     = X_fit - center_mean - n_mean * z_mean[:, None]   # (T, 3)
r_proj_pm = np.sum(y_pm * u_c_mean, axis=-1)                 # (T,)
print(f"\nPosterior-mean r_projected over time:")
print(f"  median={np.median(r_proj_pm):.4f}  mean={r_proj_pm.mean():.4f}  "
      f"sd={r_proj_pm.std():.4f}  [true={true_radius}]")
print(f"  5%, 95%: [{np.percentile(r_proj_pm,5):.4f}, {np.percentile(r_proj_pm,95):.4f}]")

# r_projected by cycle (posterior mean)
print(f"\nCorrected r_projected by cycle (posterior mean):")
for k in range(K - 1):
    mask = (t_fit >= tau_mean[k]) & (t_fit < tau_mean[k + 1])
    if mask.sum() == 0:
        continue
    rp = r_proj_pm[mask]
    print(f"  Cycle {k}: n={mask.sum():3d}  "
          f"median={np.median(rp):.4f}  mean={rp.mean():.4f}  sd={rp.std():.4f}")

# r_projected by phase bin (posterior mean, 8 bins)
print(f"\nCorrected r_projected by phase bin (posterior mean, 8 bins of [0,2pi)):")
phase_mod = phi_mean % (2 * np.pi)
bin_edges = np.linspace(0, 2 * np.pi, 9)
for b in range(8):
    mask = (phase_mod >= bin_edges[b]) & (phase_mod < bin_edges[b + 1])
    if mask.sum() == 0:
        continue
    rp = r_proj_pm[mask]
    print(f"  [{bin_edges[b]:.2f},{bin_edges[b+1]:.2f}): n={mask.sum():2d}  "
          f"median={np.median(rp):.4f}  mean={rp.mean():.4f}")

# Fitted radius (posterior mean) vs r_projected
corr_rproj_r = np.corrcoef(r_proj_pm, r_mean_t)[0, 1]
print(f"\nCorr(r_projected_t, fitted_radius_t) [posterior mean]: {corr_rproj_r:.4f}")

# Best scalar alpha: r_projected = alpha * fitted_radius
# alpha = sum(r_proj * r_fit) / sum(r_fit^2)
alpha_best = np.sum(r_proj_pm * r_mean_t) / np.sum(r_mean_t**2)
rmse_alpha1   = np.sqrt(np.mean((r_proj_pm - r_mean_t)**2))
rmse_alpha_best = np.sqrt(np.mean((r_proj_pm - alpha_best * r_mean_t)**2))
print(f"Best scalar alpha (r_proj ≈ alpha * r_fitted): {alpha_best:.4f}")
print(f"RMSE at alpha=1 (r_proj vs r_fitted)         : {rmse_alpha1:.4f}")
print(f"RMSE at best alpha                           : {rmse_alpha_best:.4f}")

# Projected radius error per draw
r_proj_err_s = np.abs(r_proj_med_s - true_radius)
print(f"\nProjected-radius error |median(r_proj) - 1.0|:")
print(f"  mean={r_proj_err_s.mean():.4f}  median={np.median(r_proj_err_s):.4f}  "
      f"sd={r_proj_err_s.std():.4f}")

# ---------------------------------------------------------------------------
# DIAGNOSTIC F: sigma_x correlation table
# ---------------------------------------------------------------------------
print()
print("-" * 70)
print("DIAGNOSTIC F: sigma_x correlation table (per-draw)")
print("-" * 70)

# Build per-draw summary table
log_sigma_x_s      = np.log(sigma_x_s)
radius_mean_t_s    = radius_s.mean(axis=1)           # (S,)
radius_sd_t_s      = radius_s.std(axis=1)            # (S,)
rmse_normal_s      = np.sqrt(np.mean(resid_n_s**2, axis=1))
rmse_cyclic_s      = rmse_c_s
rmse_tangential_s  = rmse_t_s
phase_err_med_s    = np.median(np.abs(phase_err_corr_s), axis=1)  # from Diag B
phase_err_95_s     = np.percentile(np.abs(phase_err_corr_s), 95, axis=1)
nrl_min_s          = np.linalg.norm(normal_s, axis=-1).min(axis=1)  # (S,) should be ≈1

row_names = [
    "sigma_x", "log_sigma_x", "radius_median", "radius_mean",
    "radius_sd_over_time", "RMSE", "RMSE_normal", "RMSE_cyclic",
    "RMSE_tangential", "center_rms", "center_cyclic_amp", "z_rms",
    "median_abs_z_over_radius", "phase_error_median", "phase_error_95",
    "projected_radius_median", "projected_radius_error",
    "normal_resultant_length_min",
]
row_data = np.column_stack([
    sigma_x_s, log_sigma_x_s, r_med_s, radius_mean_t_s,
    radius_sd_t_s, rmse_tot_s, rmse_normal_s, rmse_cyclic_s,
    rmse_tangential_s, center_rms_s, center_cyc_amp_s, z_rms_s,
    med_zr_s, phase_err_med_s, phase_err_95_s,
    r_proj_med_s, r_proj_err_s,
    nrl_min_s,
])  # (S, n_vars)

n_vars = len(row_names)
# Correlation matrix
corrmat = np.corrcoef(row_data.T)   # (n_vars, n_vars)

sigma_x_idx     = row_names.index("sigma_x")
log_sigma_x_idx = row_names.index("log_sigma_x")

def top_corr(target_idx, label, top_n=15):
    corrs = corrmat[target_idx]
    order = np.argsort(-np.abs(corrs))
    print(f"\nTop {top_n} |corr| with {label}:")
    for rank, j in enumerate(order[:top_n + 1]):
        if j == target_idx:
            continue
        print(f"  {rank+1:2d}. {row_names[j]:35s}  r={corrs[j]:+.4f}")

top_corr(sigma_x_idx,     "sigma_x")
top_corr(log_sigma_x_idx, "log(sigma_x)")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nFit quality (free sigma_x, seed=0):")
print(f"  Divergences     : {div_arr.sum()}")
print(f"  sigma_x         : {sigma_x_s.mean():.4f} mean  (true={true_sigma_x}, factor "
      f"{sigma_x_s.mean()/true_sigma_x:.1f}x too large)")
print(f"  radius median   : {np.median(r_med_s):.4f}  (true={true_radius})")
print(f"  normal cos_sim  : min={np.abs(layer2.normal_mean @ true_normal).min():.4f}  "
      f"med={np.median(np.abs(layer2.normal_mean @ true_normal)):.4f}")

print(f"\nKey component findings:")
print(f"  Ridge corr(sigma_x, r_median)           : {np.corrcoef(sigma_x_s,r_med_s)[0,1]:.4f}")
print(f"  Phase error RMS (posterior mean)         : {np.sqrt(np.mean(phi_err_corr**2)):.4f} rad")
print(f"  Center cyclic amplitude / radius         : {total_center_cyc_amp/np.median(r_mean_t):.5f}")
print(f"  z_rms (posterior mean)                   : {np.sqrt(np.mean(z_mean**2)):.4f}")
print(f"  Corrected r_projected (posterior mean)   : {np.median(r_proj_pm):.4f}  (true=1.0)")
print(f"  Best alpha (r_proj ≈ alpha*r_fitted)     : {alpha_best:.4f}")
print(f"  RMSE cyclic (per-draw mean)              : {rmse_c_s.mean():.4f}  (>> true sigma)")
print(f"  RMSE normal (per-draw mean)              : {rmse_n_s.mean():.4f}")

print()
print("ALL DIAGNOSTICS COMPLETE")
