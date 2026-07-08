"""
Reusable geometry utilities used by both PCA and Bayesian pipelines.

Nothing here should depend on PyMC / PyTensor: these are pure numpy helpers.
"""

from __future__ import annotations

import numpy as np


def interp_X_at_times(X, fs, times):
    """
    Linearly interpolate rows of ``X`` at real-valued times.

    Parameters
    ----------
    X : ndarray, shape (n_time, n_features)
        Trajectory data on the fixed grid ``t_grid = arange(n_time) / fs``.
    fs : float
        Sampling rate in Hz.
    times : array-like, shape (m,)
        Query times in seconds.  Values outside ``[0, (n_time-1)/fs]`` are
        clamped to the edge values by :func:`numpy.interp`.

    Returns
    -------
    ndarray, shape (m, n_features)
        Trajectory linearly interpolated at ``times``.
    """
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    n_time = X.shape[0]
    t_grid = np.arange(n_time) / float(fs)
    return np.column_stack([np.interp(times, t_grid, X[:, d]) for d in range(X.shape[1])])


def oriented_frame_from_anchors(x0_arr, x90_arr, c_arr, eps: float = 1e-12):
    """
    Build a consistently oriented in-plane frame from two anchors and a center.

    For each cycle ``k``:

    ``a0  = x0 - c``,
    ``a90 = x90 - c``,
    ``e1  = normalize(a0)``,
    ``e2  = normalize(a90 - e1 * dot(a90, e1))``,
    ``n   = normalize(cross(e1, e2))``.

    Parameters
    ----------
    x0_arr : ndarray, shape (K, 3)
        Trajectory positions at the phase-zero time of each cycle.
    x90_arr : ndarray, shape (K, 3)
        Trajectory positions at the quarter-cycle time of each cycle.
    c_arr : ndarray, shape (K, 3)
        Cycle centers.
    eps : float
        Small guard used in normalization denominators.

    Returns
    -------
    e1, e2, n : ndarray, each shape (K, 3)
        Orthonormal in-plane basis and plane normal for each cycle.
    """
    x0_arr = np.asarray(x0_arr, dtype=float)
    x90_arr = np.asarray(x90_arr, dtype=float)
    c_arr = np.asarray(c_arr, dtype=float)

    a0 = x0_arr - c_arr
    a90 = x90_arr - c_arr

    e1 = a0 / np.maximum(np.linalg.norm(a0, axis=1, keepdims=True), eps)

    dot_a90_e1 = np.sum(a90 * e1, axis=1, keepdims=True)
    a90_orth = a90 - e1 * dot_a90_e1
    e2 = a90_orth / np.maximum(np.linalg.norm(a90_orth, axis=1, keepdims=True), eps)

    n = np.cross(e1, e2)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), eps)

    return e1, e2, n
