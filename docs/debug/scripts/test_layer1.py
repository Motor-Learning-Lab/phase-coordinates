import sys, time
sys.path.insert(0, r"D:\Repositories\phase-coordinates")
import numpy as np


def main():
    from phase_coordinates.bayesian import (
        robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
        seed_boundary_indices, _fit_layer1, _numba_available,
    )

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

    R_X, xbar = robust_movement_scale(X)
    ref = dominant_reference_signal(X)
    T0 = estimate_dominant_period(ref, fs)
    tau_idx = seed_boundary_indices(ref, fs, T0)
    print("tau_idx", tau_idx, "T0", T0, "R_X", R_X)

    t0 = time.time()
    summary = _fit_layer1(
        X, fs, tau_idx, T0, R_X, xbar,
        draws=300, tune=300, chains=2, target_accept=0.9,
        random_seed=0, use_numba=_numba_available(),
    )
    print("Layer1 fit took", time.time() - t0, "s")

    true_normal = np.array([0, -np.sin(tilt), np.cos(tilt)])
    print("tau_mean", summary.tau_mean)
    print("tau_sd", summary.tau_sd)
    print("period_mean", summary.period_mean, "expected ~1.0")
    print("normal_mean", summary.normal_mean)
    cos_sim = np.abs(summary.normal_mean @ true_normal)
    print("cos_sim to true normal (per cycle):", cos_sim)
    assert np.all(cos_sim > 0.95)
    print("rho_tau_mean", summary.rho_tau_mean)
    print("normal_angular_sd", summary.normal_angular_sd)
    print("ALL LAYER1 CHECKS PASSED")


if __name__ == "__main__":
    main()
