"""
Per-cycle diagnostics for a :class:`CycleEpochs` assignment.

Diagnostics are *observational* — no cycle is excluded or filtered here, no
pass/fail decision is made.  A caller who wants to reject cycles must do so
on top of :func:`compute_cycle_quality`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .epochs import CycleEpochs
from .geometry import interp_X_at_times, oriented_frame_from_anchors


_DIAG_COLUMNS = [
    "cycle",
    "sample_start", "sample_stop",
    "time_start", "time_stop", "duration", "n_samples",
    "planarity_ratio",
    "pca_variance_ratio_1", "pca_variance_ratio_2", "pca_variance_ratio_3",
    "anchor_norm",
    "quarter_anchor_orth_norm", "quarter_anchor_orth_ratio",
    "oriented_normal_x", "oriented_normal_y", "oriented_normal_z",
    "orientation_score",
    "signed_orientation_score",
    "edge_valid",
]


def _resolve_X(X, columns):
    if isinstance(X, pd.DataFrame):
        if columns is not None:
            return X[columns].to_numpy(dtype=float)
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def compute_cycle_quality(
    X,
    epochs: CycleEpochs,
    *,
    sampling_rate_hz: float,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute per-cycle diagnostic quantities.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n_time, n_features)
        Multivariate trajectory.  Requires at least 3 features.
    epochs : CycleEpochs
        Cycle assignment from any identification stage.
    sampling_rate_hz : float
        Sampling rate in Hz.
    columns : list, optional
        Subset of columns to use when ``X`` is a DataFrame.

    Returns
    -------
    pandas.DataFrame
        One row per cycle.  See module docstring for the column list.

    Notes
    -----
    - ``planarity_ratio = 1 - PC3_var / total_var``.
    - ``anchor_norm``/``quarter_anchor_orth_*`` use trajectory positions at
      cycle start and quarter-cycle time, linearly interpolated.
    - ``orientation_score`` = dot(``n_aligned_k``, ``global_n_mean``), where
      ``n_aligned_k`` is the per-cycle normal *after* flipping it into the
      same hemisphere as the reference normal, and ``global_n_mean`` is the
      mean of the aligned normals.  Because every ``n_aligned_k`` is already
      sign-corrected, this value is always near ``+1`` (near ``0`` only
      means the cycle's plane is nearly orthogonal to the population
      direction) — it *cannot* go near ``-1``, so it cannot detect a cycle
      whose traversal direction is genuinely reversed.
    - ``signed_orientation_score`` = dot(``n_k``, ``global_n_mean``) using
      the *unaligned* per-cycle normal ``n_k``. A cycle traversed in the
      opposite direction from the rest of the population shows up here as a
      value near ``-1`` (rather than being masked by sign-alignment as in
      ``orientation_score``); use this column to detect traversal-direction
      reversals.
    - ``edge_valid`` is ``True`` iff the cycle boundaries lie inside the
      observed time window.
    """
    X_arr = _resolve_X(X, columns)
    if X_arr.ndim != 2:
        raise ValueError("X must have shape (n_time, n_features).")
    if X_arr.shape[1] < 3:
        raise ValueError("Need at least 3 features for diagnostics.")
    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")

    K = epochs.n_cycles
    if K == 0:
        return pd.DataFrame(columns=_DIAG_COLUMNS)

    n_time = X_arr.shape[0]
    t_end = (n_time - 1) / fs

    tau = epochs.tau
    duration = epochs.duration
    sample_start = epochs.sample_start
    sample_stop = epochs.sample_stop

    # Anchors (in first 3 dims)
    X3 = X_arr[:, :3]
    x0_arr = interp_X_at_times(X3, fs, tau[:-1])
    x90_arr = interp_X_at_times(X3, fs, tau[:-1] + 0.25 * duration)

    # Per-cycle centers (mean over samples in the cycle, in first 3 dims)
    centers3 = np.zeros((K, 3))
    for k in range(K):
        idx = np.where(epochs.cycle_index == k)[0]
        if idx.size > 0:
            centers3[k] = X3[idx].mean(axis=0)
        else:
            centers3[k] = 0.5 * (x0_arr[k] + x90_arr[k])

    e1_arr, e2_arr, n_arr = oriented_frame_from_anchors(x0_arr, x90_arr, centers3)

    # Global mean normal (sign-aligned): reflect any n whose dot with the
    # first non-degenerate normal is negative.
    ref = None
    for k in range(K):
        if np.linalg.norm(n_arr[k]) > 1e-9:
            ref = n_arr[k]
            break
    if ref is not None:
        signs = np.sign(n_arr @ ref)
        signs[signs == 0] = 1.0
        n_aligned = n_arr * signs[:, None]
        global_n_mean = n_aligned.mean(axis=0)
        gm_norm = np.linalg.norm(global_n_mean)
        if gm_norm > 0:
            global_n_mean = global_n_mean / gm_norm
        else:
            global_n_mean = ref / max(np.linalg.norm(ref), 1e-12)
    else:
        global_n_mean = np.array([0.0, 0.0, 1.0])
        n_aligned = n_arr.copy()

    # Assemble rows
    a0_arr = x0_arr - centers3
    a90_arr = x90_arr - centers3
    anchor_norms = np.linalg.norm(a0_arr, axis=1)
    a90_norms = np.linalg.norm(a90_arr, axis=1)

    a90_orth = a90_arr - e1_arr * np.sum(a90_arr * e1_arr, axis=1, keepdims=True)
    a90_orth_norms = np.linalg.norm(a90_orth, axis=1)
    eps = 1e-12
    q_orth_ratios = a90_orth_norms / np.maximum(a90_norms, eps)

    rows = []
    for k in range(K):
        idx = np.where(epochs.cycle_index == k)[0]
        n_k = int(idx.size)

        # PCA per cycle
        pcs = np.array([np.nan, np.nan, np.nan])
        planarity_k = float("nan")
        if n_k >= 3:
            X_k = X_arr[idx]
            pca = PCA(n_components=min(3, X_k.shape[1]))
            pca.fit(X_k - X_k.mean(axis=0))
            evr = pca.explained_variance_ratio_
            if len(evr) < 3:
                evr = np.concatenate([evr, np.zeros(3 - len(evr))])
            pcs = evr[:3].astype(float)
            planarity_k = float(1.0 - pcs[2])

        orient_k = float(np.dot(n_aligned[k], global_n_mean))
        signed_orient_k = float(np.dot(n_arr[k], global_n_mean))
        edge_valid_k = bool((tau[k] >= 0.0) and (tau[k + 1] <= t_end + 1e-12))

        rows.append({
            "cycle": k,
            "sample_start": int(sample_start[k]),
            "sample_stop": int(sample_stop[k]),
            "time_start": float(tau[k]),
            "time_stop": float(tau[k + 1]),
            "duration": float(duration[k]),
            "n_samples": n_k,
            "planarity_ratio": planarity_k,
            "pca_variance_ratio_1": float(pcs[0]),
            "pca_variance_ratio_2": float(pcs[1]),
            "pca_variance_ratio_3": float(pcs[2]),
            "anchor_norm": float(anchor_norms[k]),
            "quarter_anchor_orth_norm": float(a90_orth_norms[k]),
            "quarter_anchor_orth_ratio": float(q_orth_ratios[k]),
            "oriented_normal_x": float(n_arr[k, 0]),
            "oriented_normal_y": float(n_arr[k, 1]),
            "oriented_normal_z": float(n_arr[k, 2]),
            "orientation_score": orient_k,
            "signed_orientation_score": signed_orient_k,
            "edge_valid": edge_valid_k,
        })

    return pd.DataFrame(rows, columns=_DIAG_COLUMNS)
