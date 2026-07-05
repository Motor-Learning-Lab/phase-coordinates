import sys
sys.path.insert(0, r"D:\Repositories\phase-coordinates")
import numpy as np
from phase_coordinates.bayesian import (
    robust_movement_scale, dominant_reference_signal, estimate_dominant_period,
    seed_boundary_indices, seed_cycle_centers, seed_cycle_normals,
    seed_boundary_vectors, normalize, construct_frame, cubic_spline_matrix,
    spline_eval, _linear_interp_matrix,
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
print("R_X", R_X, "xbar", xbar)
assert 0.5 < R_X < 1.5

ref = dominant_reference_signal(X)
T0 = estimate_dominant_period(ref, fs)
print("T0", T0, "expected ~1.0")
assert 0.8 < T0 < 1.2

tau_idx = seed_boundary_indices(ref, fs, T0)
print("tau_idx", tau_idx)
assert len(tau_idx) >= n_cycles - 1

centers = seed_cycle_centers(X, tau_idx)
normals = seed_cycle_normals(X, tau_idx)
true_normal = np.array([0, -np.sin(tilt), np.cos(tilt)])
print("normals[0]", normals[0], "true (up to sign)", true_normal)
for nk in normals:
    cos_sim = abs(np.dot(nk, true_normal))
    assert cos_sim > 0.9, cos_sim

a = seed_boundary_vectors(X, tau_idx, centers)
e1, e2, pnorm = construct_frame(normals, a)
print("e1 norms", np.linalg.norm(e1, axis=-1))
print("e2 norms", np.linalg.norm(e2, axis=-1))
print("e1.n dot", np.sum(e1*normals, axis=-1))
print("e2.n dot", np.sum(e2*normals, axis=-1))
print("e2 == cross(n,e1)?", np.allclose(e2, np.cross(normals, e1)))
assert np.allclose(np.linalg.norm(e1, axis=-1), 1.0, atol=1e-6)
assert np.allclose(np.linalg.norm(e2, axis=-1), 1.0, atol=1e-6)
assert np.allclose(np.sum(e1*normals, axis=-1), 0.0, atol=1e-6)
assert np.allclose(np.sum(e2*normals, axis=-1), 0.0, atol=1e-6)
assert np.allclose(e2, np.cross(normals, e1), atol=1e-6)

# spline matrix check against scipy CubicSpline directly
knot_x = tau_idx / fs
eval_x = t[tau_idx[0]:tau_idx[-1]+1]
B = cubic_spline_matrix(knot_x, eval_x)
knot_y = centers  # but len(knot_x)=len(tau_idx), centers has len(tau_idx)-1 -> pad
knot_y_padded = np.vstack([centers, centers[-1:]])
direct = spline_eval(knot_x, knot_y_padded, eval_x)
via_matrix = B @ knot_y_padded
print("spline matrix max abs diff:", np.max(np.abs(direct - via_matrix)))
assert np.allclose(direct, via_matrix, atol=1e-8)

# linear interp matrix check
L = _linear_interp_matrix(t, knot_x)
via_L = L @ X
direct_interp = np.column_stack([np.interp(knot_x, t, X[:, d]) for d in range(3)])
print("interp matrix max abs diff:", np.max(np.abs(via_L - direct_interp)))
assert np.allclose(via_L, direct_interp, atol=1e-8)

print("ALL UTILITY CHECKS PASSED")
