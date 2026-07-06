"""
Two additional synthetic scenarios for the cycle-fixed geometry Layer 2 model.

Scenario 1: Clean complete cycles (no partial final cycle)
  Goal: test whether the residual sigma_x inflation in log 14 (0.052 vs 0.020)
  is from the short partial last cycle.

Scenario 2: Mild phase warp
  Goal: test whether linear-phase assumption is the next limitation.

Save output to docs/debug/logs/16_cycle_fixed_synthetic_suite.log
"""
import sys, os, time
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _repo_root)
import numpy as np


def _run_scenario(X, fs, tau_idx_hint, true_normal, true_sigma_x, true_radius,
                  scenario_name, random_seed=0):
    from phase_coordinates.bayesian import (
        robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
        seed_boundary_indices, _fit_layer1, _fit_layer2, _numba_available,
        interp_X_at_times,
    )

    print("\n" + "=" * 70)
    print(f"SCENARIO: {scenario_name}")
    print("=" * 70)

    R_X, xbar = robust_movement_scale(X)
    ref = dominant_reference_signal(X)
    T0 = estimate_dominant_period(ref, fs)
    if tau_idx_hint is not None:
        tau_idx = tau_idx_hint
    else:
        tau_idx = seed_boundary_indices(ref, fs, T0)
    print(f"tau_idx: {tau_idx.tolist()}")
    print(f"R_X={R_X:.4f}  T0={T0:.4f}s  K_cyc={len(tau_idx)-1}")

    use_numba = _numba_available()
    t0 = time.time()
    layer1 = _fit_layer1(
        X, fs, tau_idx, T0, R_X, xbar,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=random_seed, use_numba=use_numba,
    )
    t1 = time.time()
    print(f"Layer 1 took {t1 - t0:.1f}s")

    layer2 = _fit_layer2(
        X, fs, layer1, T0, R_X, n_velocity_knots=None,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=random_seed, use_numba=use_numba,
    )
    t2 = time.time()
    print(f"Layer 2 took {t2 - t1:.1f}s")

    tau_mean = layer1.tau_mean
    K = len(tau_mean)
    K_cyc = K - 1
    T_k = tau_mean[1:] - tau_mean[:-1]

    post = layer2.idata.posterior
    sample_stats = layer2.idata["sample_stats"]

    # ------------------------------------------------------------------
    # Divergences
    # ------------------------------------------------------------------
    div_per_chain = sample_stats["diverging"].values.sum(axis=1)
    td_per_chain = (sample_stats["tree_depth"].values >= 10).sum(axis=1)
    print(f"\nDivergences  : {div_per_chain.sum()}  ({div_per_chain.tolist()} per chain)")
    print(f"Max treedepth: {td_per_chain.sum()}  ({td_per_chain.tolist()} per chain)")

    # ------------------------------------------------------------------
    # sigma_x
    # ------------------------------------------------------------------
    sx = post["sigma_x"].values
    print(f"\nsigma_x  mean={sx.mean():.4f}  median={np.median(sx):.4f}"
          f"  5%={np.percentile(sx, 5):.4f}  95%={np.percentile(sx, 95):.4f}"
          f"  [true={true_sigma_x}]  factor={sx.mean()/true_sigma_x:.1f}x")

    # ------------------------------------------------------------------
    # R_k by cycle
    # ------------------------------------------------------------------
    R_k_post = post["R_k"].values  # (chains, draws, K_cyc)
    R_k_mean = R_k_post.mean(axis=(0, 1))
    R_k_sd = R_k_post.std(axis=(0, 1))
    print(f"\nR_k by cycle  [true={true_radius}]:")
    for k in range(K_cyc):
        print(f"  cycle {k}: mean={R_k_mean[k]:.4f}  sd={R_k_sd[k]:.4f}")
    r_k_collapsed = np.any(R_k_mean < 0.2 * true_radius)
    print(f"  R_k collapse (any mean < 0.2*true): {r_k_collapsed}")

    # ------------------------------------------------------------------
    # Center norm by cycle
    # ------------------------------------------------------------------
    c_k_post = post["c_k"].values  # (chains, draws, K_cyc, 3)
    c_k_mean = c_k_post.mean(axis=(0, 1))
    c_k_norm = np.linalg.norm(c_k_mean, axis=1)
    print(f"\nCenter norm by cycle  [true=0]:")
    for k in range(K_cyc):
        print(f"  cycle {k}: ||c_k||={c_k_norm[k]:.4f}")

    # ------------------------------------------------------------------
    # Normal cos_sim by cycle
    # ------------------------------------------------------------------
    n_k_post = post["n_k"].values  # (chains, draws, K_cyc, 3)
    n_k_mean = n_k_post.mean(axis=(0, 1))
    n_k_mean_normed = n_k_mean / (np.linalg.norm(n_k_mean, axis=1, keepdims=True) + 1e-12)
    signed_cos_per_cycle = n_k_mean_normed @ true_normal
    abs_cos_per_cycle = np.abs(signed_cos_per_cycle)
    print(f"\nNormal cos_sim by cycle  [true signed=+1.0]:")
    for k in range(K_cyc):
        print(f"  cycle {k}: signed={signed_cos_per_cycle[k]:.4f}  "
              f"abs={abs_cos_per_cycle[k]:.4f}")
    print(f"  signed: min={signed_cos_per_cycle.min():.4f}  "
          f"med={np.median(signed_cos_per_cycle):.4f}")
    print(f"  abs:    min={abs_cos_per_cycle.min():.4f}  "
          f"med={np.median(abs_cos_per_cycle):.4f}")

    # ------------------------------------------------------------------
    # Orientation score
    # ------------------------------------------------------------------
    x90_arr = interp_X_at_times(X, fs, tau_mean[:-1] + 0.25 * T_k)
    c_k_post_mean = post["c_k"].values.mean(axis=(0, 1))
    a90_post = x90_arr - c_k_post_mean
    a90_normed = a90_post / np.maximum(np.linalg.norm(a90_post, axis=1, keepdims=True), 1e-12)
    e2_k_post_mean = post["e2_k"].values.mean(axis=(0, 1))
    e2_k_normed = e2_k_post_mean / np.maximum(
        np.linalg.norm(e2_k_post_mean, axis=1, keepdims=True), 1e-12
    )
    orient_scores_post = np.sum(e2_k_normed * a90_normed, axis=1)
    print(f"\nOrientation score by cycle (e2 . normalize(a90))  [true>0, close to 1]:")
    for k in range(K_cyc):
        print(f"  cycle {k}: {orient_scores_post[k]:.4f}")

    # ------------------------------------------------------------------
    # z_rms
    # ------------------------------------------------------------------
    z_mean = layer2.perp_deviation_mean
    print(f"\nz_rms (posterior mean) : {np.sqrt(np.mean(z_mean**2)):.4f}  [true~0]")

    # ------------------------------------------------------------------
    # RMSE decomposition
    # ------------------------------------------------------------------
    pred_mean = layer2.predicted_trajectory_mean
    n_t = layer2.normal_mean
    e1_t = layer2.e1_mean
    e2_t = layer2.e2_mean
    phi_t = layer2.phase_mean
    r_t = layer2.radius_mean

    i0 = max(0, int(np.ceil(tau_mean[0] * fs)))
    i1 = min(X.shape[0] - 1, int(np.floor(tau_mean[-1] * fs)))
    X_fit = X[i0 : i1 + 1]
    resid = X_fit - pred_mean
    resid_normal = np.sum(resid * n_t, axis=1)
    cyclic_unit = e1_t * np.cos(phi_t)[:, None] + e2_t * np.sin(phi_t)[:, None]
    resid_cyclic = np.sum(resid * cyclic_unit, axis=1)
    tang_unit = -e1_t * np.sin(phi_t)[:, None] + e2_t * np.cos(phi_t)[:, None]
    resid_tang = np.sum(resid * tang_unit, axis=1)

    rmse_total = float(np.sqrt(np.mean(resid**2)))
    rmse_normal = float(np.sqrt(np.mean(resid_normal**2)))
    rmse_cyclic = float(np.sqrt(np.mean(resid_cyclic**2)))
    rmse_tang = float(np.sqrt(np.mean(resid_tang**2)))
    print(f"\nRMSE (posterior mean trajectory)  [true sigma={true_sigma_x}]:")
    print(f"  total={rmse_total:.4f}  normal={rmse_normal:.4f}"
          f"  cyclic={rmse_cyclic:.4f}  tangential={rmse_tang:.4f}")

    # ------------------------------------------------------------------
    # Phase monotonicity
    # ------------------------------------------------------------------
    dphi = np.diff(layer2.phase_mean)
    phase_monotone = bool(np.all(dphi >= -1e-6))
    print(f"\nPhase monotone: {phase_monotone}  (min dphi={dphi.min():.6f})")

    # ------------------------------------------------------------------
    # Frame orthonormality
    # ------------------------------------------------------------------
    e1_norm_range = (np.linalg.norm(e1_t, axis=1).min(), np.linalg.norm(e1_t, axis=1).max())
    e2_norm_range = (np.linalg.norm(e2_t, axis=1).min(), np.linalg.norm(e2_t, axis=1).max())
    e1_e2_dot_max = float(np.max(np.abs(np.sum(e1_t * e2_t, axis=1))))
    e1_n_dot_max = float(np.max(np.abs(np.sum(e1_t * n_t, axis=1))))
    e2_n_dot_max = float(np.max(np.abs(np.sum(e2_t * n_t, axis=1))))
    n_from_cross = np.cross(e1_t, e2_t)
    n_cross_max = float(np.max(np.abs(n_t - n_from_cross)))
    print(f"\nFrame orthonormality (n = cross(e1, e2)):")
    print(f"  ||e1|| range: [{e1_norm_range[0]:.5f}, {e1_norm_range[1]:.5f}]")
    print(f"  ||e2|| range: [{e2_norm_range[0]:.5f}, {e2_norm_range[1]:.5f}]")
    print(f"  max |e1·e2| : {e1_e2_dot_max:.2e}")
    print(f"  max |e1·n|  : {e1_n_dot_max:.2e}")
    print(f"  max |e2·n|  : {e2_n_dot_max:.2e}")
    print(f"  max |n - cross(e1,e2)|: {n_cross_max:.2e}")

    # ------------------------------------------------------------------
    # Hierarchical scales
    # ------------------------------------------------------------------
    sigma_c_post = post["sigma_c"].values.mean(axis=(0, 1))
    sigma_logR_post = float(post["sigma_log_R"].values.mean())
    print(f"\nHierarchical scales (posterior mean):")
    print(f"  sigma_c    : {sigma_c_post}")
    print(f"  sigma_log_R: {sigma_logR_post:.4f}")

    print(f"\n--- {scenario_name} SUMMARY ---")
    print(f"  Divergences  : {div_per_chain.sum()}")
    print(f"  sigma_x mean : {sx.mean():.4f}  [true={true_sigma_x}, factor {sx.mean()/true_sigma_x:.1f}x]")
    print(f"  R_k median   : {np.median(R_k_mean):.4f}  [true={true_radius}]")
    print(f"  R_k collapse : {r_k_collapsed}")
    print(f"  signed n cos : min={signed_cos_per_cycle.min():.4f}  med={np.median(signed_cos_per_cycle):.4f}")
    print(f"  abs n cos    : min={abs_cos_per_cycle.min():.4f}  med={np.median(abs_cos_per_cycle):.4f}")
    print(f"  orient scores: {np.round(orient_scores_post, 3).tolist()}")
    print(f"  RMSE total   : {rmse_total:.4f}  [true sigma={true_sigma_x}]")
    print(f"  RMSE cyclic  : {rmse_cyclic:.4f}  RMSE normal: {rmse_normal:.4f}")
    print(f"  z_rms        : {np.sqrt(np.mean(z_mean**2)):.4f}")
    print(f"  Phase monotone: {phase_monotone}")

    # Assertions
    assert np.median(abs_cos_per_cycle) > 0.95, \
        f"[{scenario_name}] abs normal cos_sim median {np.median(abs_cos_per_cycle):.3f} < 0.95"
    assert np.median(signed_cos_per_cycle) > 0.0, \
        f"[{scenario_name}] signed normal cos_sim median {np.median(signed_cos_per_cycle):.3f} <= 0"
    assert phase_monotone, f"[{scenario_name}] Phase is not monotone"
    assert 0.7 < np.median(R_k_mean) < 1.3, \
        f"[{scenario_name}] R_k median {np.median(R_k_mean):.3f} outside [0.7, 1.3]"
    assert np.median(np.abs(z_mean)) < 0.1, \
        f"[{scenario_name}] median |z| {np.median(np.abs(z_mean)):.3f} >= 0.1"
    assert np.allclose(np.linalg.norm(e1_t, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(e2_t, axis=1), 1.0, atol=1e-3)
    assert e1_e2_dot_max < 1e-3
    assert e1_n_dot_max < 1e-3
    assert e2_n_dot_max < 1e-3
    assert n_cross_max < 1e-3, \
        f"[{scenario_name}] max |n - cross(e1,e2)| = {n_cross_max:.2e} >= 1e-3"

    print(f"\nALL CHECKS PASSED: {scenario_name}")


def main():
    from phase_coordinates.bayesian import seed_boundary_indices

    # ======================================================================
    # Scenario 1: Clean complete cycles (no partial final cycle)
    # ======================================================================
    # Goal: test whether sigma_x inflation in log 14 (0.052 vs 0.020) is from
    # the short partial last cycle that seed_boundary_indices often creates.
    #
    # Strategy: generate n_cycles + 2 full cycles, then slice to cycles 1..n_cycles
    # so that the first and last boundaries are solid interior peaks.

    rng1 = np.random.default_rng(1)
    fs = 100.0
    n_cycles = 6
    n_per_cycle = 100
    n_extra = 2  # one extra cycle on each side

    n_time_full = (n_cycles + n_extra) * n_per_cycle
    t_full = np.arange(n_time_full) / fs
    phase_full = 2 * np.pi * t_full
    tilt = np.pi / 6
    X_full = np.column_stack([
        np.cos(phase_full),
        np.sin(phase_full) * np.cos(tilt),
        np.sin(phase_full) * np.sin(tilt),
    ])
    X_full += rng1.normal(scale=0.02, size=X_full.shape)

    # Slice to cycles 1 .. n_cycles  (skip first and last extra cycle)
    i_start = n_per_cycle      # start of cycle 1 (0-indexed extra at front)
    i_end = (n_cycles + 1) * n_per_cycle  # end of last cycle (exclusive)
    X1 = X_full[i_start:i_end]

    true_normal1 = np.array([0.0, -np.sin(tilt), np.cos(tilt)])
    true_sigma_x1 = 0.02
    true_radius1 = 1.0

    # Provide exact boundary indices: 0, 100, 200, ..., 600
    tau_idx1 = np.arange(0, n_cycles + 1) * n_per_cycle

    _run_scenario(
        X=X1,
        fs=fs,
        tau_idx_hint=tau_idx1,
        true_normal=true_normal1,
        true_sigma_x=true_sigma_x1,
        true_radius=true_radius1,
        scenario_name="Scenario 1: Clean complete cycles",
        random_seed=1,
    )

    # ======================================================================
    # Scenario 2: Mild phase warp
    # ======================================================================
    # Goal: test whether linear-phase assumption is the next limitation after
    # fixing the anchor frame.
    #
    # True phase: phi(t) = 2*pi*t + alpha * sin(2*pi*t),  alpha = 0.3
    # This is monotone since d phi/dt = 2*pi*(1 + alpha*cos(2*pi*t)) >= 2*pi*(1-0.3) > 0

    rng2 = np.random.default_rng(2)
    fs = 100.0
    n_cycles = 6
    n_per_cycle = 100
    n_time2 = n_cycles * n_per_cycle
    t2 = np.arange(n_time2) / fs
    alpha = 0.3
    phase_warp = 2 * np.pi * t2 + alpha * np.sin(2 * np.pi * t2)
    tilt2 = np.pi / 6
    X2 = np.column_stack([
        np.cos(phase_warp),
        np.sin(phase_warp) * np.cos(tilt2),
        np.sin(phase_warp) * np.sin(tilt2),
    ])
    X2 += rng2.normal(scale=0.02, size=X2.shape)

    true_normal2 = np.array([0.0, -np.sin(tilt2), np.cos(tilt2)])
    true_sigma_x2 = 0.02
    true_radius2 = 1.0

    _run_scenario(
        X=X2,
        fs=fs,
        tau_idx_hint=None,   # let seed_boundary_indices find them
        true_normal=true_normal2,
        true_sigma_x=true_sigma_x2,
        true_radius=true_radius2,
        scenario_name="Scenario 2: Mild phase warp (alpha=0.3)",
        random_seed=2,
    )

    print("\n" + "=" * 70)
    print("ALL SCENARIOS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
