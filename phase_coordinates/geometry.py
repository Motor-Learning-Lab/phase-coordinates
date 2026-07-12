"""
Reusable geometry utilities used by both PCA and Bayesian pipelines.

Nothing here should depend on PyMC / PyTensor: these are pure numpy helpers.
"""

from __future__ import annotations

import numpy as np


class AnchorOutOfBoundsError(ValueError):
    """
    Raised by :func:`interp_X_at_times` when a query time falls outside the
    recorded data window and ``bounds_error=True``.

    Subclasses ``ValueError`` so existing code that catches the broader type
    keeps working, but gives callers that specifically mean "this
    candidate's anchor time simply isn't in the data" (an expected,
    per-candidate failure mode during a period/offset search -- see
    :func:`~phase_coordinates.scoring.find_epochs_by_geometric_score`) a way
    to catch *that* narrowly, distinct from an unrelated ``ValueError``
    (bad arguments, a real bug elsewhere) that should propagate instead of
    being silently treated as "this candidate didn't work out."
    """


def interp_X_at_times(X, fs, times, *, bounds_error: bool = True):
    """
    Linearly interpolate rows of ``X`` at real-valued times.

    Parameters
    ----------
    X : ndarray, shape (n_time, n_features)
        Trajectory data on the fixed grid ``t_grid = arange(n_time) / fs``.
    fs : float
        Sampling rate in Hz.
    times : array-like, shape (m,)
        Query times in seconds.
    bounds_error : bool
        If ``True`` (the default), raise :class:`AnchorOutOfBoundsError`
        when any query time falls outside ``[0, (n_time-1)/fs]`` rather
        than silently clamping to the edge value.  A candidate anchor time
        that lands outside the signal window is usually a sign of a bad
        candidate (e.g. a bad period/offset guess), and clamping would give
        it a plausible-looking but wrong value.  Pass ``bounds_error=False``
        for callers that genuinely intend extrapolation-by-clamping (e.g.
        posterior-mean boundary times that may drift slightly past the data
        edge).

    Returns
    -------
    ndarray, shape (m, n_features)
        Trajectory linearly interpolated at ``times``.

    Raises
    ------
    AnchorOutOfBoundsError
        If ``bounds_error=True`` and any query time is outside the data
        window. A subclass of ``ValueError``.
    """
    X = np.asarray(X, dtype=float)
    times = np.asarray(times, dtype=float)
    n_time = X.shape[0]
    t_grid = np.arange(n_time) / float(fs)

    if bounds_error and times.size:
        t_lo, t_hi = t_grid[0], t_grid[-1]
        eps = 1e-9 * max(1.0, t_hi - t_lo)
        out_of_bounds = (times < t_lo - eps) | (times > t_hi + eps)
        if np.any(out_of_bounds):
            bad = times[out_of_bounds]
            raise AnchorOutOfBoundsError(
                f"interp_X_at_times: {len(bad)} query time(s) fall outside "
                f"the signal window [{t_lo:.6g}, {t_hi:.6g}] seconds "
                f"(e.g. {bad[0]:.6g}). Pass bounds_error=False to clamp to "
                "the edge value instead."
            )

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
