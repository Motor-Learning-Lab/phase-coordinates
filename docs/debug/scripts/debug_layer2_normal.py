import sys, time
sys.path.insert(0, r"D:\Repositories\phase-coordinates")
import numpy as np


def main():
    from phase_coordinates.bayesian import (
        robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
        seed_boundary_indices, _fit_layer1, _fit_layer2, _numba_available,
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

    use_numba = _numba_available()
    layer1 = _fit_layer1(
        X, fs, tau_idx, T0, R_X, xbar,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=0, use_numba=use_numba,
    )
    layer2 = _fit_layer2(
        X, fs, layer1, T0, R_X, n_velocity_knots=None,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=0, use_numba=use_numba,
    )

    true_normal = np.array([0, -np.sin(tilt), np.cos(tilt)])
    cos_sim = np.abs(layer2.normal_mean @ true_normal)
    bad_idx = np.argsort(cos_sim)[:10]
    print("worst indices:", bad_idx)
    print("worst cos_sim:", cos_sim[bad_idx])
    print("worst times:", layer2.time[bad_idx])
    print("tau_mean:", layer1.tau_mean)
    print("normal at worst idx:", layer2.normal_mean[bad_idx])
    print("normal_angular_sd at worst idx:", layer2.normal_angular_sd[bad_idx])


if __name__ == "__main__":
    main()
