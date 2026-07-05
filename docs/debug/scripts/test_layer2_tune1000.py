"""
Variant of test_layer2.py with tune=1000 to give the mass matrix more time to
adapt to the amplitude parameter subspace. Motivation: the reparameterized
model (log 08) shows 0 divergences and normal artifact gone, but Chain 1 hits
max_treedepth and ESS < 100 for some amplitude parameters, leaving radius at
0.709 (expected ~1.0) and sigma_x at 0.467 (expected ~0.02). More tune steps
should let adapt_diag learn better per-parameter scales for radius/sigma_x.
"""
import sys, time
import os
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, _repo_root)
import numpy as np


def main():
    from phase_coordinates.bayesian import (
        robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
        seed_boundary_indices, _fit_layer1, _fit_layer2, _numba_available,
        construct_frame,
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
    print("tau_idx", tau_idx)

    use_numba = _numba_available()
    t0 = time.time()
    layer1 = _fit_layer1(
        X, fs, tau_idx, T0, R_X, xbar,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=0, use_numba=use_numba,
    )
    print("Layer1 took", time.time() - t0, "s")

    t0 = time.time()
    layer2 = _fit_layer2(
        X, fs, layer1, T0, R_X, n_velocity_knots=None,
        draws=400, tune=1000, chains=2, target_accept=0.9,
        random_seed=0, use_numba=use_numba,
    )
    print("Layer2 took", time.time() - t0, "s")

    true_normal = np.array([0, -np.sin(tilt), np.cos(tilt)])
    cos_sim = np.abs(layer2.normal_mean @ true_normal)
    print("normal cos_sim stats: min", cos_sim.min(), "median", np.median(cos_sim))
    assert np.median(cos_sim) > 0.95

    dphi = np.diff(layer2.phase_mean)
    print("phase monotonic:", np.all(dphi >= -1e-6), "min dphi", dphi.min())
    assert np.all(dphi >= -1e-6)

    print("radius stats: min", layer2.radius_mean.min(), "median", np.median(layer2.radius_mean), "expected ~1.0")
    assert np.all(layer2.radius_mean > 0)
    assert 0.7 < np.median(layer2.radius_mean) < 1.3

    print("perp deviation median abs:", np.median(np.abs(layer2.perp_deviation_mean)))
    assert np.median(np.abs(layer2.perp_deviation_mean)) < 0.1

    e1 = layer2.e1_mean
    e2 = layer2.e2_mean
    n = layer2.normal_mean
    print("e1 norms range:", np.linalg.norm(e1, axis=-1).min(), np.linalg.norm(e1, axis=-1).max())
    print("e2 norms range:", np.linalg.norm(e2, axis=-1).min(), np.linalg.norm(e2, axis=-1).max())
    print("e1.n dot max abs:", np.max(np.abs(np.sum(e1 * n, axis=-1))))
    print("e2.n dot max abs:", np.max(np.abs(np.sum(e2 * n, axis=-1))))
    print("e2 vs cross(n,e1) max diff:", np.max(np.abs(e2 - np.cross(n, e1))))
    assert np.allclose(np.linalg.norm(e1, axis=-1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(e2, axis=-1), 1.0, atol=1e-3)
    assert np.max(np.abs(np.sum(e1 * n, axis=-1))) < 1e-3
    assert np.max(np.abs(np.sum(e2 * n, axis=-1))) < 1e-3
    assert np.allclose(e2, np.cross(n, e1), atol=1e-3)

    # Tighter radius check: after more tuning, expect close to true value 1.0
    print("radius median (tight check):", np.median(layer2.radius_mean), "— want > 0.85")
    assert 0.85 < np.median(layer2.radius_mean) < 1.15, (
        f"Radius median {np.median(layer2.radius_mean):.3f} still far from expected ~1.0 "
        "even with tune=1000; amplitude parameters may need further investigation."
    )

    phase_mod = np.mod(layer2.phase_mean, 2 * np.pi)
    near_zero = phase_mod < 0.2
    print("fraction of time near phase 0:", near_zero.mean(), "count", near_zero.sum())

    print("sigma_x_mean", layer2.sigma_x_mean, "R_X", R_X, "— expected sigma_x ~ 0.02")
    assert layer2.sigma_x_mean < 0.10, (
        f"sigma_x_mean {layer2.sigma_x_mean:.3f} still large (> 0.10); "
        "amplitude model may not have converged even with tune=1000."
    )

    print("ALL LAYER2 TUNE1000 CHECKS PASSED")


if __name__ == "__main__":
    main()
