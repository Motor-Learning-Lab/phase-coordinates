import sys, os, time
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _repo_root)
import numpy as np


def main():
    from phase_coordinates.bayesian import (
        robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
        seed_boundary_indices, _fit_layer1, _fit_layer2, _numba_available,
        normalize,
    )

    # Same fixed-plane synthetic data as test_layer2.py
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
    true_sigma_x = 0.02
    true_radius = 1.0

    R_X, xbar = robust_movement_scale(X)
    ref = dominant_reference_signal(X)
    T0 = estimate_dominant_period(ref, fs)
    tau_idx = seed_boundary_indices(ref, fs, T0)
    print("tau_idx", tau_idx)

    use_numba = _numba_available()
    t0 = time.time()
    layer1 = _fit_layer1(
        X, fs, tau_idx, T0, R_X, xbar,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=0, use_numba=use_numba,
    )
    t1 = time.time()
    print(f"Layer 1 took {t1 - t0:.1f}s")

    layer2 = _fit_layer2(
        X, fs, layer1, T0, R_X, n_velocity_knots=None,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=0, use_numba=use_numba,
    )
    t2 = time.time()
    print(f"Layer 2 took {t2 - t1:.1f}s")

    post = layer2.idata.posterior
    sample_stats = layer2.idata["sample_stats"]
    tau_mean = layer1.tau_mean
    K = len(tau_mean)
    K_cyc = K - 1

    cycle_idx_arr = np.searchsorted(tau_mean, layer2.time, side="right") - 1
    cycle_idx_arr = np.clip(cycle_idx_arr, 0, K_cyc - 1).astype(int)

    print("\n" + "=" * 70)
    print("LAYER 2 CYCLE-FIXED GEOMETRY DIAGNOSTICS")
    print("=" * 70)
    print(f"True normal  : {true_normal}")
    print(f"True radius  : {true_radius}")
    print(f"True sigma_x : {true_sigma_x}")
    print(f"R_X = {R_X:.4f}   T0 = {T0:.4f}s   K_cyc = {K_cyc}")

    # ------------------------------------------------------------------
    # Divergences and max treedepth
    # ------------------------------------------------------------------
    n_chains = sample_stats["diverging"].values.shape[0]
    n_draws = sample_stats["diverging"].values.shape[1]
    div_per_chain = sample_stats["diverging"].values.sum(axis=1)
    td_per_chain = (sample_stats["tree_depth"].values >= 10).sum(axis=1)
    print(f"\nDivergences  : {div_per_chain.sum()}  ({div_per_chain.tolist()} per chain)")
    print(f"Max treedepth: {td_per_chain.sum()}  ({td_per_chain.tolist()} per chain)")

    # ------------------------------------------------------------------
    # sigma_x
    # ------------------------------------------------------------------
    sx = post["sigma_x"].values  # (chains, draws)
    print(f"\nsigma_x  mean={sx.mean():.4f}  median={np.median(sx):.4f}"
          f"  5%={np.percentile(sx, 5):.4f}  95%={np.percentile(sx, 95):.4f}"
          f"  [true={true_sigma_x}]")

    # ------------------------------------------------------------------
    # R_k by cycle
    # ------------------------------------------------------------------
    R_k_post = post["R_k"].values  # (chains, draws, K_cyc)
    R_k_mean = R_k_post.mean(axis=(0, 1))
    R_k_sd = R_k_post.std(axis=(0, 1))
    print(f"\nR_k by cycle  [true={true_radius}]:")
    for k in range(K_cyc):
        print(f"  cycle {k}: mean={R_k_mean[k]:.4f}  sd={R_k_sd[k]:.4f}")

    # ------------------------------------------------------------------
    # Center norm by cycle
    # ------------------------------------------------------------------
    c_k_post = post["c_k"].values  # (chains, draws, K_cyc, 3)
    c_k_mean = c_k_post.mean(axis=(0, 1))  # (K_cyc, 3)
    c_k_norm = np.linalg.norm(c_k_mean, axis=1)
    print(f"\nCenter norm by cycle  [true=0]:")
    for k in range(K_cyc):
        print(f"  cycle {k}: ||c_k||={c_k_norm[k]:.4f}")

    # ------------------------------------------------------------------
    # Normal cos_sim by cycle
    # ------------------------------------------------------------------
    n_k_post = post["n_k"].values  # (chains, draws, K_cyc, 3)
    n_k_mean = n_k_post.mean(axis=(0, 1))  # (K_cyc, 3)
    n_k_mean_normed = n_k_mean / (np.linalg.norm(n_k_mean, axis=1, keepdims=True) + 1e-12)
    cos_sim_per_cycle = np.abs(n_k_mean_normed @ true_normal)
    print(f"\nNormal cos_sim by cycle  [true=1.0]:")
    for k in range(K_cyc):
        print(f"  cycle {k}: cos_sim={cos_sim_per_cycle[k]:.4f}")
    print(f"  min={cos_sim_per_cycle.min():.4f}  median={np.median(cos_sim_per_cycle):.4f}")

    # ------------------------------------------------------------------
    # z_rms
    # ------------------------------------------------------------------
    z_mean = layer2.perp_deviation_mean
    print(f"\nz_rms (posterior mean) : {np.sqrt(np.mean(z_mean**2)):.4f}  [true~0]")

    # ------------------------------------------------------------------
    # RMSE decomposition (posterior mean trajectory)
    # ------------------------------------------------------------------
    pred_mean = layer2.predicted_trajectory_mean
    n_t = layer2.normal_mean
    e1_t = layer2.e1_mean
    e2_t = layer2.e2_mean
    phi_t = layer2.phase_mean
    r_t = layer2.radius_mean
    n_time_fit = len(layer2.time)

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
    e1_n_dot_max = float(np.max(np.abs(np.sum(e1_t * n_t, axis=1))))
    e2_n_dot_max = float(np.max(np.abs(np.sum(e2_t * n_t, axis=1))))
    e2_cross_max = float(np.max(np.abs(e2_t - np.cross(n_t, e1_t))))
    print(f"\nFrame orthonormality:")
    print(f"  ||e1|| range: [{e1_norm_range[0]:.5f}, {e1_norm_range[1]:.5f}]")
    print(f"  ||e2|| range: [{e2_norm_range[0]:.5f}, {e2_norm_range[1]:.5f}]")
    print(f"  max |e1·n|  : {e1_n_dot_max:.2e}")
    print(f"  max |e2·n|  : {e2_n_dot_max:.2e}")
    print(f"  max |e2 - cross(n,e1)|: {e2_cross_max:.2e}")

    # ------------------------------------------------------------------
    # Hierarchical scales
    # ------------------------------------------------------------------
    sigma_c_post = post["sigma_c"].values.mean(axis=(0, 1))   # (3,)
    sigma_logR_post = float(post["sigma_log_R"].values.mean())
    sigma_n_post = float(post["sigma_n_angle"].values.mean())
    sigma_a_post = post["sigma_a"].values.mean(axis=(0, 1))   # (3,)
    print(f"\nHierarchical scales (posterior mean):")
    print(f"  sigma_c       : {sigma_c_post}")
    print(f"  sigma_log_R   : {sigma_logR_post:.4f}")
    print(f"  sigma_n_angle : {sigma_n_post:.4f}")
    print(f"  sigma_a       : {sigma_a_post}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Divergences  : {div_per_chain.sum()}")
    print(f"  sigma_x mean : {sx.mean():.4f}  [true={true_sigma_x},"
          f" factor {sx.mean()/true_sigma_x:.1f}x]")
    print(f"  R_k median   : {np.median(R_k_mean):.4f}  [true={true_radius}]")
    print(f"  normal cos_sim: min={cos_sim_per_cycle.min():.4f}"
          f"  med={np.median(cos_sim_per_cycle):.4f}")
    print(f"  RMSE total   : {rmse_total:.4f}  [true sigma={true_sigma_x}]")
    print(f"  RMSE cyclic  : {rmse_cyclic:.4f}  RMSE normal: {rmse_normal:.4f}")
    print(f"  z_rms        : {np.sqrt(np.mean(z_mean**2)):.4f}")

    # Assertions
    assert np.median(cos_sim_per_cycle) > 0.95, \
        f"Normal cos_sim median {np.median(cos_sim_per_cycle):.3f} < 0.95"
    assert phase_monotone, "Phase is not monotone"
    assert 0.7 < np.median(R_k_mean) < 1.3, \
        f"R_k median {np.median(R_k_mean):.3f} outside [0.7, 1.3]"
    assert np.median(np.abs(z_mean)) < 0.1, \
        f"median |z| {np.median(np.abs(z_mean)):.3f} >= 0.1"
    assert np.allclose(np.linalg.norm(e1_t, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(e2_t, axis=1), 1.0, atol=1e-3)
    assert e1_n_dot_max < 1e-3
    assert e2_n_dot_max < 1e-3
    assert e2_cross_max < 1e-3

    print("\nALL CYCLE-FIXED CHECKS PASSED")


if __name__ == "__main__":
    main()
