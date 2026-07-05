import sys, time
sys.path.insert(0, r"D:\Repositories\phase-coordinates")
import numpy as np


def main():
    from phase_coordinates.bayesian import (
        robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
        seed_boundary_indices, _fit_layer1, _numba_available,
    )
    import arviz as az

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

    summary = _fit_layer1(
        X, fs, tau_idx, T0, R_X, xbar,
        draws=300, tune=300, chains=2, target_accept=0.9,
        random_seed=0, use_numba=_numba_available(),
    )
    idata = summary.idata
    print(az.summary(idata, var_names=["tau", "rho_tau", "mu_tau"]))
    n_post = idata.posterior["n"].values  # (chain, draw, K-1, 3)
    print("n shape", n_post.shape)
    for c in range(n_post.shape[0]):
        print("chain", c, "cycle1 mean n:", n_post[c, :, 1, :].mean(axis=0), "sd:", n_post[c,:,1,:].std(axis=0))
    div = idata.sample_stats["diverging"].values
    print("divergences per chain:", div.sum(axis=1))


if __name__ == "__main__":
    main()
