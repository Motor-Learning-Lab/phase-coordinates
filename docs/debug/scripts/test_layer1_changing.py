import sys, time
sys.path.insert(0, r"D:\Repositories\phase-coordinates")
import numpy as np


def rotation_matrix_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_changing_planes(n_cycles=6, samples_per_cycle=120, noise_std=0.01, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n_time = n_cycles * samples_per_cycle
    t = np.arange(n_time) / float(samples_per_cycle)
    phase_true = 2 * np.pi * t
    X = np.zeros((n_time, 3))
    true_normals = []
    for cyc in range(n_cycles):
        yaw = cyc * (np.pi / 6)
        tilt = cyc * (np.pi / 6)
        R = rotation_matrix_z(yaw) @ rotation_matrix_x(tilt)
        normal = R @ np.array([0.0, 0.0, 1.0])
        true_normals.append(normal)
        start = cyc * samples_per_cycle
        end = start + samples_per_cycle
        theta = phase_true[start:end] - phase_true[start]
        local = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))])
        X[start:end] = (R @ local.T).T
    X += rng.standard_normal(X.shape) * noise_std
    return X, phase_true, true_normals


def main():
    from phase_coordinates.bayesian import (
        robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
        seed_boundary_indices, _fit_layer1, _numba_available,
    )

    X, phase_true, true_normals = make_changing_planes()
    fs = 120.0

    R_X, xbar = robust_movement_scale(X)
    ref = dominant_reference_signal(X)
    T0 = estimate_dominant_period(ref, fs)
    tau_idx = seed_boundary_indices(ref, fs, T0)
    print("tau_idx", tau_idx, "T0", T0, "n cycles detected", len(tau_idx) - 1)

    t0 = time.time()
    summary = _fit_layer1(
        X, fs, tau_idx, T0, R_X, xbar,
        draws=400, tune=400, chains=2, target_accept=0.9,
        random_seed=0, use_numba=_numba_available(),
    )
    print("Layer1 fit took", time.time() - t0, "s")

    print("recovered normals:")
    print(summary.normal_mean)
    print("true normals:")
    print(np.array(true_normals))

    n_cyc_recovered = summary.normal_mean.shape[0]
    for k in range(n_cyc_recovered):
        # match against the closest true normal by index proportionally
        true_idx = min(k, len(true_normals) - 1)
        cos_sim = abs(np.dot(summary.normal_mean[k], true_normals[true_idx]))
        print(f"cycle {k}: cos_sim to true_normals[{true_idx}] = {cos_sim:.4f}")


if __name__ == "__main__":
    main()
