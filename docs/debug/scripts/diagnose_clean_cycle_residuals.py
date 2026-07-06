"""
Diagnostic for residual inflation in the clean complete-cycle synthetic scenario.

Runs Scenario 1 from test_cycle_fixed_synthetic_suite.py (same seed, same data),
then applies the diagnostics requested in the session prompt.

Save output to: docs/debug/logs/17_clean_cycle_residual_diagnostics.log
"""
import sys, os, time
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _repo_root)

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rmse_components(X_fit, pred, e1_t, e2_t, n_t, phi_t):
    resid = X_fit - pred
    resid_n = np.sum(resid * n_t, axis=1)
    cyclic_unit = e1_t * np.cos(phi_t)[:, None] + e2_t * np.sin(phi_t)[:, None]
    resid_c = np.sum(resid * cyclic_unit, axis=1)
    tang_unit = -e1_t * np.sin(phi_t)[:, None] + e2_t * np.cos(phi_t)[:, None]
    resid_t = np.sum(resid * tang_unit, axis=1)
    rmse_tot = float(np.sqrt(np.mean(resid ** 2)))
    rmse_nrm = float(np.sqrt(np.mean(resid_n ** 2)))
    rmse_cyc = float(np.sqrt(np.mean(resid_c ** 2)))
    rmse_tan = float(np.sqrt(np.mean(resid_t ** 2)))
    return rmse_tot, rmse_nrm, rmse_cyc, rmse_tan, resid, resid_n, resid_c, resid_t


def print_rmse(label, tot, nrm, cyc, tan):
    print(f"  {label}: total={tot:.4f}  normal={nrm:.4f}  "
          f"cyclic={cyc:.4f}  tangential={tan:.4f}")


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Build exact Scenario 1 dataset
# ---------------------------------------------------------------------------

def make_dataset():
    rng = np.random.default_rng(1)
    fs = 100.0
    n_cycles = 6
    n_per_cycle = 100
    n_extra = 2
    tilt = np.pi / 6
    n_time_full = (n_cycles + n_extra) * n_per_cycle
    t_full = np.arange(n_time_full) / fs
    phase_full = 2 * np.pi * t_full
    X_full = np.column_stack([
        np.cos(phase_full),
        np.sin(phase_full) * np.cos(tilt),
        np.sin(phase_full) * np.sin(tilt),
    ])
    X_full += rng.normal(scale=0.02, size=X_full.shape)
    i_start = n_per_cycle
    i_end = (n_cycles + 1) * n_per_cycle
    X = X_full[i_start:i_end]
    tau_idx_hint = np.arange(0, n_cycles + 1) * n_per_cycle
    true_normal = np.array([0.0, -np.sin(tilt), np.cos(tilt)])
    # True frame: e1=[1,0,0], e2=[0,cos(tilt),sin(tilt)], n=cross(e1,e2)
    true_e1 = np.array([1.0, 0.0, 0.0])
    true_e2 = np.array([0.0, np.cos(tilt), np.sin(tilt)])
    return X, fs, tau_idx_hint, tilt, true_normal, true_e1, true_e2


# ---------------------------------------------------------------------------
# Section 2: Deterministic baseline residuals A–D
# ---------------------------------------------------------------------------

def section2_baselines(X, fs, tau_idx_hint, tilt, true_normal, true_e1, true_e2):
    from phase_coordinates.bayesian import interp_X_at_times, _oriented_frame_from_anchors

    section("2. DETERMINISTIC BASELINE RESIDUALS (A–D)")

    fs_val = fs
    n_time = X.shape[0]
    K_cyc = len(tau_idx_hint) - 1
    tau_hint_s = tau_idx_hint / fs_val
    T_k = tau_hint_s[1:] - tau_hint_s[:-1]

    # Time array for X — clip i1 to last valid sample index
    i0 = int(tau_idx_hint[0])
    i1 = min(X.shape[0] - 1, int(tau_idx_hint[-1]))
    t_fit = np.arange(i0, i1 + 1) / fs_val
    X_fit = X[i0 : i1 + 1]
    n_fit = X_fit.shape[0]

    # Cycle membership using exact tau_hint
    cycle_idx = np.searchsorted(tau_hint_s, t_fit, side="right") - 1
    cycle_idx = np.clip(cycle_idx, 0, K_cyc - 1)

    # True linear phase using exact tau_hint
    tau_k_arr = tau_hint_s[cycle_idx]
    tau_kp1_arr = tau_hint_s[cycle_idx + 1]
    phi_true = (
        2 * np.pi * cycle_idx
        + 2 * np.pi * (t_fit - tau_k_arr) / (tau_kp1_arr - tau_k_arr)
    )

    # Expand true frame to per-time
    e1_true_t = np.tile(true_e1, (n_fit, 1))
    e2_true_t = np.tile(true_e2, (n_fit, 1))
    n_true_t = np.tile(true_normal, (n_fit, 1))
    c_zero_t = np.zeros((n_fit, 3))
    R1 = 1.0

    # --- A: true center=0, true frame, true phase ---
    pred_A = (
        c_zero_t
        + e1_true_t * (R1 * np.cos(phi_true))[:, None]
        + e2_true_t * (R1 * np.sin(phi_true))[:, None]
    )
    tot_A, nrm_A, cyc_A, tan_A, _, _, _, _ = rmse_components(
        X_fit, pred_A, e1_true_t, e2_true_t, n_true_t, phi_true
    )
    print_rmse("A (true c=0, true frame, true phase)", tot_A, nrm_A, cyc_A, tan_A)

    # --- B: true center=0, noisy anchors from exact tau_hint ---
    x0_b = interp_X_at_times(X, fs_val, tau_hint_s[:-1])
    x90_b = interp_X_at_times(X, fs_val, tau_hint_s[:-1] + 0.25 * T_k)
    c_zero_cyc = np.zeros((K_cyc, 3))
    _, _, e1_cyc_b, e2_cyc_b, n_cyc_b, _ = _oriented_frame_from_anchors(x0_b, x90_b, c_zero_cyc)
    e1_b_t = e1_cyc_b[cycle_idx]
    e2_b_t = e2_cyc_b[cycle_idx]
    n_b_t = n_cyc_b[cycle_idx]
    pred_B = (
        c_zero_t
        + e1_b_t * (R1 * np.cos(phi_true))[:, None]
        + e2_b_t * (R1 * np.sin(phi_true))[:, None]
    )
    tot_B, nrm_B, cyc_B, tan_B, _, _, _, _ = rmse_components(
        X_fit, pred_B, e1_b_t, e2_b_t, n_b_t, phi_true
    )
    print_rmse("B (true c=0, noisy anchors/exact tau)", tot_B, nrm_B, cyc_B, tan_B)

    # Anchor quality check
    cos_e1 = np.sum(e1_cyc_b * true_e1, axis=1)
    cos_e2 = np.sum(e2_cyc_b * true_e2, axis=1)
    cos_n = np.sum(n_cyc_b * true_normal, axis=1)
    print(f"  e1 cos_sim with true_e1: {cos_e1.min():.4f}–{cos_e1.max():.4f}")
    print(f"  e2 cos_sim with true_e2: {cos_e2.min():.4f}–{cos_e2.max():.4f}")
    print(f"  n  cos_sim with true_n : {cos_n.min():.4f}–{cos_n.max():.4f}")

    # --- C: per-cycle sample-mean center, noisy anchors from exact tau_hint ---
    c_cyc_c = np.array([X_fit[cycle_idx == k].mean(axis=0) for k in range(K_cyc)])
    x0_c = interp_X_at_times(X, fs_val, tau_hint_s[:-1])
    x90_c = interp_X_at_times(X, fs_val, tau_hint_s[:-1] + 0.25 * T_k)
    _, _, e1_cyc_c, e2_cyc_c, n_cyc_c, _ = _oriented_frame_from_anchors(x0_c, x90_c, c_cyc_c)
    e1_c_t = e1_cyc_c[cycle_idx]
    e2_c_t = e2_cyc_c[cycle_idx]
    n_c_t = n_cyc_c[cycle_idx]
    c_c_t = c_cyc_c[cycle_idx]
    pred_C = (
        c_c_t
        + e1_c_t * (R1 * np.cos(phi_true))[:, None]
        + e2_c_t * (R1 * np.sin(phi_true))[:, None]
    )
    tot_C, nrm_C, cyc_C, tan_C, _, _, _, _ = rmse_components(
        X_fit, pred_C, e1_c_t, e2_c_t, n_c_t, phi_true
    )
    print_rmse("C (sample-mean c, noisy anchors/exact tau)", tot_C, nrm_C, cyc_C, tan_C)
    print(f"  per-cycle center norms: "
          f"{np.round(np.linalg.norm(c_cyc_c, axis=1), 4).tolist()}")

    # --- D: per-cycle sample-mean center, true frame, true phase ---
    pred_D = (
        c_c_t
        + e1_true_t * (R1 * np.cos(phi_true))[:, None]
        + e2_true_t * (R1 * np.sin(phi_true))[:, None]
    )
    tot_D, nrm_D, cyc_D, tan_D, _, _, _, _ = rmse_components(
        X_fit, pred_D, e1_true_t, e2_true_t, n_true_t, phi_true
    )
    print_rmse("D (sample-mean c, true frame, true phase)", tot_D, nrm_D, cyc_D, tan_D)

    print()
    print("  Interpretation:")
    print(f"    A ~ noise floor: {tot_A:.4f}  (should be ~0.020)")
    print(f"    B–A gap (anchor error): {tot_B-tot_A:.4f}")
    print(f"    C–B gap (center-shift effect): {tot_C-tot_B:.4f}")
    print(f"    model sigma_x from log 16: 0.0449  (gap from A: {0.0449-tot_A:.4f})")

    return {
        "tau_hint_s": tau_hint_s, "T_k": T_k, "t_fit": t_fit,
        "X_fit": X_fit, "cycle_idx": cycle_idx, "phi_true": phi_true,
        "e1_true_t": e1_true_t, "e2_true_t": e2_true_t, "n_true_t": n_true_t,
        "c_cyc_samp": c_cyc_c,
    }


# ---------------------------------------------------------------------------
# Section 3: Layer 1 tau drift
# ---------------------------------------------------------------------------

def section3_layer1_drift(X, fs, tau_idx_hint, T0, R_X, xbar, layer1):
    section("3. LAYER 1 TAU DRIFT")

    tau_hint_s = tau_idx_hint / fs
    tau_mean = layer1.tau_mean
    tau_sd = layer1.tau_sd
    T_mean = layer1.period_mean

    K = len(tau_mean)
    K_cyc = K - 1

    print(f"  {'':>8}  {'tau_hint':>10}  {'tau_mean':>10}  {'drift_s':>9}  "
          f"{'drift_samp':>11}  {'tau_sd_s':>9}  {'tau_sd_samp':>11}")
    for k in range(K):
        drift_s = tau_mean[k] - tau_hint_s[k]
        drift_samp = drift_s * fs
        print(f"  tau[{k}]:  {tau_hint_s[k]:>10.4f}  {tau_mean[k]:>10.4f}  "
              f"{drift_s:>+9.4f}  {drift_samp:>+11.2f}  "
              f"{tau_sd[k]:>9.4f}  {tau_sd[k]*fs:>11.2f}")

    print()
    print(f"  {'':>8}  {'T_mean':>8}  {'T_hint':>8}  {'T_error_s':>10}  {'T_error_samp':>12}")
    T_hint = tau_hint_s[1:] - tau_hint_s[:-1]
    T_k = tau_mean[1:] - tau_mean[:-1]
    for k in range(K_cyc):
        err_s = T_k[k] - T_hint[k]
        print(f"  T[{k}]:   {T_k[k]:>8.4f}  {T_hint[k]:>8.4f}  "
              f"{err_s:>+10.4f}  {err_s*fs:>+12.2f}")

    drift_samples = (tau_mean - tau_hint_s) * fs
    print(f"\n  Max |drift| in samples: {np.max(np.abs(drift_samples)):.2f}")
    print(f"  RMS drift in samples  : {np.sqrt(np.mean(drift_samples**2)):.2f}")
    return tau_mean, tau_hint_s


# ---------------------------------------------------------------------------
# Section 4: Boundary-clustering likelihood at tau_hint vs tau_mean
# ---------------------------------------------------------------------------

def section4_boundary_likelihood(X, fs, tau_hint_s, tau_mean, layer1):
    from phase_coordinates.bayesian import interp_X_at_times

    section("4. BOUNDARY-CLUSTERING LIKELIHOOD: tau_hint vs tau_mean")

    post1 = layer1.idata.posterior
    mu_tau_mean = post1["mu_tau"].mean(("chain", "draw")).values   # (3,)
    rho_tau_mean = float(post1["rho_tau"].mean(("chain", "draw")).values)
    R_X_val = float(np.linalg.norm(
        interp_X_at_times(X, fs, tau_mean[:1]) - mu_tau_mean
    ))  # just for scale; use R_X from caller
    # Actually use the stored rho_tau to get sigma_tau_x
    # We need R_X — compute it
    from phase_coordinates.bayesian import robust_movement_scale
    R_X, _ = robust_movement_scale(X)
    sigma_tau_x = R_X * rho_tau_mean

    # Evaluate X at tau_hint and tau_mean
    X_tau_hint = interp_X_at_times(X, fs, tau_hint_s)
    X_tau_mean = interp_X_at_times(X, fs, tau_mean)

    def boundary_log_likelihood(X_tau, mu, sigma):
        """Sum of Normal(mu, sigma) log-prob over all boundary points."""
        resid = X_tau - mu[None, :]  # (K, 3)
        return float(-0.5 * np.sum((resid / sigma) ** 2))

    loglik_hint = boundary_log_likelihood(X_tau_hint, mu_tau_mean, sigma_tau_x)
    loglik_mean = boundary_log_likelihood(X_tau_mean, mu_tau_mean, sigma_tau_x)

    print(f"  Posterior mean mu_tau  : {mu_tau_mean}")
    print(f"  Posterior mean rho_tau : {rho_tau_mean:.4f}")
    print(f"  sigma_tau_x (R_X*rho)  : {sigma_tau_x:.4f}  (R_X={R_X:.4f})")
    print()
    print(f"  Log-lik at tau_hint    : {loglik_hint:.2f}")
    print(f"  Log-lik at tau_mean    : {loglik_mean:.2f}")
    print(f"  Difference (mean−hint) : {loglik_mean - loglik_hint:.2f}")
    if loglik_mean > loglik_hint:
        print("  => Layer 1 boundary clustering prefers tau_mean over tau_hint")
    else:
        print("  => tau_hint has higher boundary-cluster likelihood than tau_mean")

    # Also check: distances from mu_tau
    dist_hint = np.linalg.norm(X_tau_hint - mu_tau_mean[None, :], axis=1)
    dist_mean = np.linalg.norm(X_tau_mean - mu_tau_mean[None, :], axis=1)
    print()
    print(f"  ||X(tau_hint) - mu_tau|| per boundary: "
          f"{np.round(dist_hint, 4).tolist()}")
    print(f"  ||X(tau_mean) - mu_tau|| per boundary: "
          f"{np.round(dist_mean, 4).tolist()}")


# ---------------------------------------------------------------------------
# Section 5: Residuals using tau_hint vs tau_mean (E–H)
# ---------------------------------------------------------------------------

def section5_tau_comparison(X, fs, tau_hint_s, tau_mean, layer2):
    from phase_coordinates.bayesian import interp_X_at_times, _oriented_frame_from_anchors

    section("5. RESIDUALS: tau_hint vs tau_mean PHASE (E–H)")

    # Use Layer 2's own time window so z_pm length matches X_fit
    t_fit = layer2.time                                    # (n_fit,) — Layer 2's window
    n_fit = len(t_fit)
    l2_i0 = int(round(t_fit[0] * fs))
    l2_i1 = int(round(t_fit[-1] * fs))
    X_fit = X[l2_i0 : l2_i1 + 1]
    assert X_fit.shape[0] == n_fit, f"X_fit rows {X_fit.shape[0]} != n_fit {n_fit}"

    K_cyc = len(tau_mean) - 1

    # Cycle membership under tau_mean
    cycle_idx_M = np.searchsorted(tau_mean, t_fit, side="right") - 1
    cycle_idx_M = np.clip(cycle_idx_M, 0, K_cyc - 1)

    # Phase using tau_mean
    T_k_M = tau_mean[1:] - tau_mean[:-1]
    tau_k_M = tau_mean[cycle_idx_M]
    tau_kp1_M = tau_mean[cycle_idx_M + 1]
    phi_mean = (
        2 * np.pi * cycle_idx_M
        + 2 * np.pi * (t_fit - tau_k_M) / (tau_kp1_M - tau_k_M)
    )

    # Cycle membership under tau_hint — clamp to K_cyc cycles
    cycle_idx_H = np.searchsorted(tau_hint_s, t_fit, side="right") - 1
    cycle_idx_H = np.clip(cycle_idx_H, 0, K_cyc - 1)

    # Phase using tau_hint for t_fit window
    T_k_H = tau_hint_s[1:] - tau_hint_s[:-1]
    tau_k_H_arr = tau_hint_s[cycle_idx_H]
    tau_kp1_H_arr = tau_hint_s[cycle_idx_H + 1]
    phi_hint = (
        2 * np.pi * cycle_idx_H
        + 2 * np.pi * (t_fit - tau_k_H_arr) / (tau_kp1_H_arr - tau_k_H_arr)
    )

    # Posterior mean c_k, R_k from Layer 2
    post = layer2.idata.posterior
    c_k_pm = post["c_k"].mean(("chain", "draw")).values   # (K_cyc, 3)
    R_k_pm = post["R_k"].mean(("chain", "draw")).values   # (K_cyc,)
    z_pm = layer2.perp_deviation_mean                      # (n_fit,)

    # Fixed anchors recomputed at tau_mean (same as Layer 2 used)
    x0_M = interp_X_at_times(X, fs, tau_mean[:-1])
    x90_M = interp_X_at_times(X, fs, tau_mean[:-1] + 0.25 * T_k_M)
    _, _, e1_M, e2_M, n_M, _ = _oriented_frame_from_anchors(x0_M, x90_M, c_k_pm)

    # Anchors at tau_hint
    T_k_H = tau_hint_s[1:] - tau_hint_s[:-1]
    x0_H = interp_X_at_times(X, fs, tau_hint_s[:-1])
    x90_H = interp_X_at_times(X, fs, tau_hint_s[:-1] + 0.25 * T_k_H)
    _, _, e1_H, e2_H, n_H, _ = _oriented_frame_from_anchors(x0_H, x90_H, c_k_pm)

    # --- E: tau_mean, posterior c_k/R_k, posterior z_t ---
    e1_E_t = e1_M[cycle_idx_M]
    e2_E_t = e2_M[cycle_idx_M]
    n_E_t = n_M[cycle_idx_M]
    r_E_t = R_k_pm[cycle_idx_M]
    c_E_t = c_k_pm[cycle_idx_M]
    pred_E = (
        c_E_t
        + e1_E_t * (r_E_t * np.cos(phi_mean))[:, None]
        + e2_E_t * (r_E_t * np.sin(phi_mean))[:, None]
        + n_E_t * z_pm[:, None]
    )
    tot_E, nrm_E, cyc_E, tan_E, _, _, _, _ = rmse_components(
        X_fit, pred_E, e1_E_t, e2_E_t, n_E_t, phi_mean)
    print_rmse("E (tau_mean, post c_k/R_k, post z)", tot_E, nrm_E, cyc_E, tan_E)

    # --- F: tau_hint phase+anchors, posterior c_k/R_k, z=0 ---
    # Use tau_hint for phase and anchors; expand to per-time from cycle_idx_H
    # Restrict to samples where cycle_idx_H == cycle_idx_M (same cycles covered)
    e1_F_t = e1_H[cycle_idx_H]
    e2_F_t = e2_H[cycle_idx_H]
    n_F_t = n_H[cycle_idx_H]
    r_F_t = R_k_pm[cycle_idx_H]
    c_F_t = c_k_pm[cycle_idx_H]
    pred_F = (
        c_F_t
        + e1_F_t * (r_F_t * np.cos(phi_hint))[:, None]
        + e2_F_t * (r_F_t * np.sin(phi_hint))[:, None]
        + n_F_t * z_pm[:, None]
    )
    tot_F, nrm_F, cyc_F, tan_F, _, _, _, _ = rmse_components(
        X_fit, pred_F, e1_F_t, e2_F_t, n_F_t, phi_hint)
    print_rmse("F (tau_hint phase/anchors, post c_k/R_k, post z)", tot_F, nrm_F, cyc_F, tan_F)

    # --- G: tau_hint + c=0 + R=1 + z=0 ---
    e1_G_t = e1_H[cycle_idx_H]
    e2_G_t = e2_H[cycle_idx_H]
    n_G_t = n_H[cycle_idx_H]
    pred_G = (
        e1_G_t * np.cos(phi_hint)[:, None]
        + e2_G_t * np.sin(phi_hint)[:, None]
    )
    tot_G, nrm_G, cyc_G, tan_G, _, _, _, _ = rmse_components(
        X_fit, pred_G, e1_G_t, e2_G_t, n_G_t, phi_hint)
    print_rmse("G (tau_hint, c=0, R=1, z=0)", tot_G, nrm_G, cyc_G, tan_G)

    # --- H: tau_mean + c=0 + R=1 + z=0 ---
    e1_H_t2 = e1_M[cycle_idx_M]
    e2_H_t2 = e2_M[cycle_idx_M]
    n_H_t2 = n_M[cycle_idx_M]
    # anchors at tau_mean with c=0
    x0_H2 = interp_X_at_times(X, fs, tau_mean[:-1])
    x90_H2 = interp_X_at_times(X, fs, tau_mean[:-1] + 0.25 * T_k_M)
    c_zero_cyc = np.zeros((K_cyc, 3))
    _, _, e1_H2, e2_H2, n_H2, _ = _oriented_frame_from_anchors(x0_H2, x90_H2, c_zero_cyc)
    e1_H2_t = e1_H2[cycle_idx_M]
    e2_H2_t = e2_H2[cycle_idx_M]
    n_H2_t = n_H2[cycle_idx_M]
    pred_H = (
        e1_H2_t * np.cos(phi_mean)[:, None]
        + e2_H2_t * np.sin(phi_mean)[:, None]
    )
    tot_H, nrm_H2, cyc_H, tan_H, _, _, _, _ = rmse_components(
        X_fit, pred_H, e1_H2_t, e2_H2_t, n_H2_t, phi_mean)
    print_rmse("H (tau_mean, c=0, R=1, z=0)", tot_H, nrm_H2, cyc_H, tan_H)

    print()
    print("  tau_hint vs tau_mean effect on RMSE:")
    print(f"    F (tau_hint) vs E (tau_mean): {tot_F:.4f} vs {tot_E:.4f}  diff={tot_F-tot_E:+.4f}")
    print(f"    G (tau_hint+ideal) vs H (tau_mean+ideal): {tot_G:.4f} vs {tot_H:.4f}  diff={tot_G-tot_H:+.4f}")
    print()
    print("  Center-shift effect (c=0 vs posterior c_k):")
    print(f"    tau_hint: G={tot_G:.4f} vs F={tot_F:.4f}  diff (c=0 is better if neg): {tot_G-tot_F:+.4f}")
    print(f"    tau_mean: H={tot_H:.4f} vs E={tot_E:.4f}  diff (c=0 is better if neg): {tot_H-tot_E:+.4f}")

    return {
        "phi_mean": phi_mean, "phi_hint": phi_hint,
        "cycle_idx_M": cycle_idx_M, "cycle_idx_H": cycle_idx_H,
        "e1_M": e1_M, "e2_M": e2_M, "n_M": n_M,
        "c_k_pm": c_k_pm, "R_k_pm": R_k_pm,
        "T_k_M": T_k_M, "K_cyc": K_cyc,
        "t_fit": t_fit, "X_fit": X_fit,
    }


# ---------------------------------------------------------------------------
# Section 6: Best phase offset per cycle
# ---------------------------------------------------------------------------

def section6_phase_offset(sec5, layer2):
    section("6. BEST PHASE OFFSET PER CYCLE (grid search)")

    X_fit = sec5["X_fit"]
    phi_mean = sec5["phi_mean"]
    cycle_idx_M = sec5["cycle_idx_M"]
    e1_M = sec5["e1_M"]
    e2_M = sec5["e2_M"]
    n_M = sec5["n_M"]
    c_k_pm = sec5["c_k_pm"]
    R_k_pm = sec5["R_k_pm"]
    T_k_M = sec5["T_k_M"]
    K_cyc = sec5["K_cyc"]
    z_pm = layer2.perp_deviation_mean

    n_grid = 1001
    delta_grid = np.linspace(-0.5, 0.5, n_grid)  # radians

    print(f"  Grid: {n_grid} points in [-0.5, 0.5] rad")
    print()
    print(f"  {'cyc':>4}  {'best_delta_rad':>14}  {'best_delta_samp':>15}  "
          f"{'RMSE_cyc_before':>16}  {'RMSE_cyc_after':>15}  "
          f"{'RMSE_tang_before':>17}  {'RMSE_tang_after':>16}")

    total_tan_before = []
    total_tan_after = []
    total_cyc_before = []
    total_cyc_after = []

    for k in range(K_cyc):
        mask = cycle_idx_M == k
        if mask.sum() < 5:
            continue
        X_k = X_fit[mask]
        phi_k = phi_mean[mask]
        c_k = c_k_pm[k]
        R_k = R_k_pm[k]
        z_k = z_pm[mask]
        e1_k = e1_M[k]
        e2_k = e2_M[k]
        n_k = n_M[k]
        T_k = T_k_M[k]

        # Pre-compute residuals without phase offset for reference
        pred_k0 = (
            c_k[None, :]
            + e1_k[None, :] * (R_k * np.cos(phi_k))[:, None]
            + e2_k[None, :] * (R_k * np.sin(phi_k))[:, None]
            + n_k[None, :] * z_k[:, None]
        )
        resid0 = X_k - pred_k0
        cyc_unit0 = e1_k[None, :] * np.cos(phi_k)[:, None] + e2_k[None, :] * np.sin(phi_k)[:, None]
        tang_unit0 = -e1_k[None, :] * np.sin(phi_k)[:, None] + e2_k[None, :] * np.cos(phi_k)[:, None]
        rmse_cyc_before = float(np.sqrt(np.mean((resid0 * cyc_unit0).sum(axis=1) ** 2)))
        rmse_tan_before = float(np.sqrt(np.mean((resid0 * tang_unit0).sum(axis=1) ** 2)))

        # Grid search over delta
        best_rmse_tan = np.inf
        best_delta = 0.0
        for delta in delta_grid:
            phi_shifted = phi_k + delta
            pred_d = (
                c_k[None, :]
                + e1_k[None, :] * (R_k * np.cos(phi_shifted))[:, None]
                + e2_k[None, :] * (R_k * np.sin(phi_shifted))[:, None]
                + n_k[None, :] * z_k[:, None]
            )
            resid_d = X_k - pred_d
            tang_unit_d = (-e1_k[None, :] * np.sin(phi_shifted)[:, None]
                           + e2_k[None, :] * np.cos(phi_shifted)[:, None])
            rmse_tan_d = float(np.sqrt(np.mean((resid_d * tang_unit_d).sum(axis=1) ** 2)))
            if rmse_tan_d < best_rmse_tan:
                best_rmse_tan = rmse_tan_d
                best_delta = delta

        # Recompute cyclic RMSE at best delta
        phi_best = phi_k + best_delta
        pred_best = (
            c_k[None, :]
            + e1_k[None, :] * (R_k * np.cos(phi_best))[:, None]
            + e2_k[None, :] * (R_k * np.sin(phi_best))[:, None]
            + n_k[None, :] * z_k[:, None]
        )
        resid_best = X_k - pred_best
        cyc_unit_best = (e1_k[None, :] * np.cos(phi_best)[:, None]
                         + e2_k[None, :] * np.sin(phi_best)[:, None])
        tang_unit_best = (-e1_k[None, :] * np.sin(phi_best)[:, None]
                          + e2_k[None, :] * np.cos(phi_best)[:, None])
        rmse_cyc_after = float(np.sqrt(np.mean((resid_best * cyc_unit_best).sum(axis=1) ** 2)))
        rmse_tan_after = float(np.sqrt(np.mean((resid_best * tang_unit_best).sum(axis=1) ** 2)))

        n_k_samp = mask.sum()
        best_delta_samp = best_delta / (2 * np.pi) * n_k_samp

        print(f"  {k:>4}  {best_delta:>+14.4f}  {best_delta_samp:>+15.2f}  "
              f"{rmse_cyc_before:>16.4f}  {rmse_cyc_after:>15.4f}  "
              f"{rmse_tan_before:>17.4f}  {rmse_tan_after:>16.4f}")

        total_tan_before.append(rmse_tan_before)
        total_tan_after.append(rmse_tan_after)
        total_cyc_before.append(rmse_cyc_before)
        total_cyc_after.append(rmse_cyc_after)

    print()
    print(f"  Mean tangential RMSE before phase offset: {np.mean(total_tan_before):.4f}")
    print(f"  Mean tangential RMSE after  phase offset: {np.mean(total_tan_after):.4f}")
    print(f"  Mean cyclic    RMSE before phase offset: {np.mean(total_cyc_before):.4f}")
    print(f"  Mean cyclic    RMSE after  phase offset: {np.mean(total_cyc_after):.4f}")
    tang_reduction = (np.mean(total_tan_before) - np.mean(total_tan_after)) / np.mean(total_tan_before)
    print(f"  Tangential RMSE reduction by phase offset: {tang_reduction*100:.1f}%")


# ---------------------------------------------------------------------------
# Section 7: Per-cycle residual audit
# ---------------------------------------------------------------------------

def section7_per_cycle(tau_mean, tau_hint_s, layer2, sec5):
    section("7. PER-CYCLE RESIDUAL AUDIT")

    X_fit = sec5["X_fit"]
    phi_mean = sec5["phi_mean"]
    cycle_idx_M = sec5["cycle_idx_M"]
    e1_M = sec5["e1_M"]
    e2_M = sec5["e2_M"]
    n_M = sec5["n_M"]
    c_k_pm = sec5["c_k_pm"]
    R_k_pm = sec5["R_k_pm"]
    K_cyc = sec5["K_cyc"]
    z_pm = layer2.perp_deviation_mean

    print(f"  {'cyc':>4}  {'RMSE_tot':>9}  {'RMSE_nrm':>9}  {'RMSE_cyc':>9}  "
          f"{'RMSE_tan':>9}  {'c_norm':>7}  {'R_k':>7}  {'z_rms':>6}  "
          f"{'tau_start_err':>14}  {'tau_end_err':>12}  {'T_err_samp':>11}")

    for k in range(K_cyc):
        mask = cycle_idx_M == k
        if mask.sum() < 2:
            continue
        X_k = X_fit[mask]
        phi_k = phi_mean[mask]
        z_k = z_pm[mask]
        e1_k = e1_M[k]
        e2_k = e2_M[k]
        n_k = n_M[k]
        c_k = c_k_pm[k]
        R_k = R_k_pm[k]

        pred_k = (
            c_k[None, :]
            + e1_k[None, :] * (R_k * np.cos(phi_k))[:, None]
            + e2_k[None, :] * (R_k * np.sin(phi_k))[:, None]
            + n_k[None, :] * z_k[:, None]
        )
        rmse_tot, rmse_nrm, rmse_cyc, rmse_tan, _, _, _, _ = rmse_components(
            X_k, pred_k,
            np.tile(e1_k, (mask.sum(), 1)),
            np.tile(e2_k, (mask.sum(), 1)),
            np.tile(n_k, (mask.sum(), 1)),
            phi_k,
        )
        c_norm = float(np.linalg.norm(c_k))
        z_rms_k = float(np.sqrt(np.mean(z_k ** 2)))

        start_err = (tau_mean[k] - tau_hint_s[k]) * 100.0   # samples (fs=100)
        end_err = (tau_mean[k + 1] - tau_hint_s[k + 1]) * 100.0
        T_err = (tau_mean[k + 1] - tau_mean[k] - (tau_hint_s[k + 1] - tau_hint_s[k])) * 100.0

        print(f"  {k:>4}  {rmse_tot:>9.4f}  {rmse_nrm:>9.4f}  {rmse_cyc:>9.4f}  "
              f"{rmse_tan:>9.4f}  {c_norm:>7.4f}  {R_k:>7.4f}  {z_rms_k:>6.4f}  "
              f"{start_err:>+14.2f}  {end_err:>+12.2f}  {T_err:>+11.2f}")


# ---------------------------------------------------------------------------
# Section 8: Divergence localization
# ---------------------------------------------------------------------------

def section8_divergences(layer2):
    section("8. DIVERGENCE LOCALIZATION")

    idata = layer2.idata
    sample_stats = idata["sample_stats"]
    post = idata.posterior

    div = sample_stats["diverging"].values  # (chains, draws)
    n_div = int(div.sum())
    print(f"  Total divergences: {n_div}")

    if n_div == 0:
        print("  No divergences — skipping localization.")
        return

    div_flat = div.reshape(-1)
    n_total = len(div_flat)
    print(f"  Divergence rate: {n_div}/{n_total} = {n_div/n_total*100:.2f}%")

    # Flatten posterior samples
    def flat(name):
        v = post[name].values
        return v.reshape(-1, *v.shape[2:])

    sigma_x_flat = flat("sigma_x")
    R_k_flat = flat("R_k")
    c_k_flat = flat("c_k")      # (samples, K_cyc, 3)
    sigma_c_flat = flat("sigma_c")
    sigma_logR_flat = flat("sigma_log_R")

    c_norm_flat = np.linalg.norm(c_k_flat.mean(axis=1), axis=1)  # mean center norm

    def compare(name, vals_div, vals_ok):
        m_div = np.mean(vals_div) if len(vals_div) else float("nan")
        m_ok = np.mean(vals_ok) if len(vals_ok) else float("nan")
        print(f"  {name:30s}:  divergent mean={m_div:.4f}  non-div mean={m_ok:.4f}"
              f"  diff={m_div-m_ok:+.4f}")

    compare("sigma_x", sigma_x_flat[div_flat == 1], sigma_x_flat[div_flat == 0])
    compare("sigma_log_R", sigma_logR_flat[div_flat == 1], sigma_logR_flat[div_flat == 0])
    compare("mean c_k norm", c_norm_flat[div_flat == 1], c_norm_flat[div_flat == 0])

    # Per-cycle R_k
    for k in range(R_k_flat.shape[1]):
        compare(f"R_k[{k}]", R_k_flat[div_flat == 1, k], R_k_flat[div_flat == 0, k])

    # sigma_c components
    for d in range(sigma_c_flat.shape[1]):
        compare(f"sigma_c[{d}]", sigma_c_flat[div_flat == 1, d], sigma_c_flat[div_flat == 0, d])

    # h_z_knots spread
    h_z_flat = flat("h_z_knots")
    h_z_rms = np.sqrt(np.mean(h_z_flat ** 2, axis=1))
    compare("h_z_knots rms", h_z_rms[div_flat == 1], h_z_rms[div_flat == 0])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from phase_coordinates.bayesian import (
        robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
        _fit_layer1, _fit_layer2, _numba_available,
    )

    print("=" * 70)
    print("DIAGNOSTIC: Clean complete-cycle residual inflation")
    print("Reproducing Scenario 1 from test_cycle_fixed_synthetic_suite.py")
    print("seed=1, tune=400, draws=400, chains=2, target_accept=0.9")
    print("=" * 70)

    X, fs, tau_idx_hint, tilt, true_normal, true_e1, true_e2 = make_dataset()
    print(f"\nData shape: {X.shape}  fs={fs}  tau_idx_hint: {tau_idx_hint.tolist()}")
    print(f"True normal: {true_normal}")
    print(f"True e1:     {true_e1}")
    print(f"True e2:     {true_e2}")

    R_X, xbar = robust_movement_scale(X)
    ref = dominant_reference_signal(X)
    T0 = estimate_dominant_period(ref, fs)
    use_numba = _numba_available()

    print(f"\nR_X={R_X:.4f}  T0={T0:.4f}s  xbar={xbar}")

    # Section 2 (deterministic) — no sampling needed
    baselines = section2_baselines(X, fs, tau_idx_hint, tilt, true_normal, true_e1, true_e2)

    # Layer 1
    section("\nRunning Layer 1...")
    t0 = time.time()
    layer1 = _fit_layer1(
        X, fs, tau_idx_hint, T0, R_X, xbar,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=1, use_numba=use_numba,
    )
    print(f"Layer 1 took {time.time()-t0:.1f}s")

    tau_mean, tau_hint_s = section3_layer1_drift(
        X, fs, tau_idx_hint, T0, R_X, xbar, layer1)

    section4_boundary_likelihood(X, fs, tau_hint_s, tau_mean, layer1)

    # Layer 2
    section("\nRunning Layer 2...")
    t0 = time.time()
    layer2 = _fit_layer2(
        X, fs, layer1, T0, R_X, n_velocity_knots=None,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=1, use_numba=use_numba,
    )
    print(f"Layer 2 took {time.time()-t0:.1f}s")

    # Quick summary
    post = layer2.idata.posterior
    sx = post["sigma_x"].values
    div_total = int(layer2.idata["sample_stats"]["diverging"].values.sum())
    print(f"\n  sigma_x mean={sx.mean():.4f}  [true=0.020]  factor={sx.mean()/0.020:.1f}x")
    print(f"  Divergences: {div_total}")

    sec5 = section5_tau_comparison(X, fs, tau_hint_s, tau_mean, layer2)

    section6_phase_offset(sec5, layer2)

    section7_per_cycle(tau_mean, tau_hint_s, layer2, sec5)

    section8_divergences(layer2)

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    section("SUMMARY AND DIAGNOSIS")
    print()
    print("  Most likely causes of sigma_x inflation (ranked):")
    print()
    print("  See diagnostics above:")
    print("  1. If best_phase_offset diagnostic removes most tangential RMSE:")
    print("       => Layer 1 tau drift / phase misalignment is primary cause")
    print("  2. If tau_hint gives lower RMSE than tau_mean (G < H or F < E):")
    print("       => Layer 1 boundary sampling moves tau away from true boundaries")
    print("  3. If baseline B–A gap >> 0 and > 50% of total surplus RMSE:")
    print("       => Single-sample noisy anchor is contributing")
    print("  4. If center forced to 0 gives lower RMSE (G < F or H < E):")
    print("       => center compensation is harmful for clean data")
    print("  5. If divergences concentrate in c_k or z regions:")
    print("       => geometry of center/z coupling needs reparameterization")
    print()
    print("  Recommended next model change (do not implement — diagnostics only):")
    print("  If tau drift is primary: consider fixing tau to tau_hint in Layer 2,")
    print("  or adding a tighter tau prior / passing tau_hint directly to Layer 2.")


if __name__ == "__main__":
    main()
