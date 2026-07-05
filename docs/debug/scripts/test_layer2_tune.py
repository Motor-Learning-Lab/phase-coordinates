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

    for n_vel, target_accept in [(6, 0.95), (6, 0.99)]:
        print(f"--- n_velocity_knots={n_vel}, target_accept={target_accept} ---")
        t0 = time.time()
        layer2 = _fit_layer2(
            X, fs, layer1, T0, R_X, n_velocity_knots=n_vel,
            draws=200, tune=200, chains=2, target_accept=target_accept,
            random_seed=0, use_numba=use_numba,
        )
        print("Layer2 took", time.time() - t0, "s")
        idata = layer2.idata
        div = int(idata.sample_stats["diverging"].sum())
        print("divergences:", div)


if __name__ == "__main__":
    main()
