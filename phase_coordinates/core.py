"""
Core PCA coordinate estimation and phase estimation utilities.

This module now has a single responsibility: given a
:class:`~phase_coordinates.epochs.CycleEpochs` assignment, fit a per-cycle
PCA and produce the shared ``samples`` / ``cycles`` DataFrames.  Phase
estimation (:func:`hilbert_phase`) lives here too but has no
coordinate-estimation responsibilities: callers build epochs from the phase
themselves and pass the epochs in.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import butter, sosfiltfilt, hilbert

from .epochs import CycleEpochs

# Minimum signal length required by sosfiltfilt with a 4th-order Butterworth
# filter (2 second-order sections → default padlen = 3 * 2 * 2 = 12).
_HILBERT_MIN_SAMPLES = 13

_PHASE_JUMP_THRESHOLD = -0.5
_NON_MONOTONIC_THRESHOLD = 0.003

# ---------------------------------------------------------------------------
# Shared output schema constants
# ---------------------------------------------------------------------------

SAMPLE_COLUMNS = [
    "sample_index",
    "time",
    "cycle",
    "phase",
    "phase_in_cycle",
    "u",
    "v",
    "radius",
    "theta",
    "theta_wrapped",
    "perp",
]

CYCLE_COLUMNS = [
    "cycle",
    "sample_start",
    "sample_stop",
    "time_start",
    "time_stop",
    "time_quarter",
    "duration",
    "center_x",
    "center_y",
    "center_z",
    "e1_x",
    "e1_y",
    "e1_z",
    "e2_x",
    "e2_y",
    "e2_z",
    "normal_x",
    "normal_y",
    "normal_z",
    "radius_mean",
    "radius_sd",
    "perp_mean",
    "perp_sd",
    "n_samples",
    "fit_ok",
]


def hilbert_phase(ref_signal, fs, f_range):
    """
    Get unwrapped Hilbert phase from a scalar reference signal.

    Parameters
    ----------
    ref_signal : array-like, shape (n_time,)
        Scalar 1-D time series.  Must contain only finite values.
    fs : float
        Sampling rate in Hz. Must be positive.
    f_range : tuple of float
        Bandpass frequency range ``(low, high)`` in Hz.

    Returns
    -------
    phase_unwrapped : numpy.ndarray
        Unwrapped instantaneous phase in radians.
    phase_wrapped : numpy.ndarray
        Wrapped instantaneous phase in radians.
    amplitude : numpy.ndarray
        Instantaneous amplitude of the analytic signal.
    """
    ref_signal = np.asarray(ref_signal, dtype=float)

    if ref_signal.ndim != 1:
        raise ValueError(
            f"ref_signal must be 1-D, got shape {ref_signal.shape}."
        )
    if not np.all(np.isfinite(ref_signal)):
        raise ValueError("ref_signal contains non-finite values (NaN or Inf).")
    if len(ref_signal) < _HILBERT_MIN_SAMPLES:
        raise ValueError(
            f"ref_signal is too short: need at least {_HILBERT_MIN_SAMPLES} "
            f"samples for the 4th-order bandpass filter, got {len(ref_signal)}."
        )

    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")

    f_range = tuple(f_range)
    if len(f_range) != 2:
        raise ValueError("f_range must be a length-2 sequence (low, high).")
    low, high = f_range
    if not (0 < low < high):
        raise ValueError(
            f"f_range must satisfy 0 < low < high, got ({low}, {high})."
        )
    if high >= fs / 2:
        raise ValueError(
            f"f_range high ({high} Hz) must be less than the Nyquist "
            f"frequency (fs/2 = {fs / 2} Hz)."
        )

    sos = butter(N=4, Wn=f_range, btype="bandpass", fs=fs, output="sos")

    try:
        x_filt = sosfiltfilt(sos, ref_signal)
    except ValueError as exc:
        raise ValueError(
            "ref_signal is too short for scipy.signal.sosfiltfilt with the "
            "chosen 4th-order bandpass filter. Use a longer signal, use a "
            "lower-order filter in the future, or provide a precomputed phase."
        ) from exc
    analytic = hilbert(x_filt)

    phase_wrapped = np.angle(analytic)
    phase_unwrapped = np.unwrap(phase_wrapped)

    # ---- warn on badly non-monotonic phase ----
    n = len(phase_unwrapped)
    trim = max(1, n // 10)
    if n - 2 * trim > 1:
        diffs = np.diff(phase_unwrapped[trim : n - trim])
        neg_fraction = np.sum(diffs < _PHASE_JUMP_THRESHOLD) / len(diffs)
        if neg_fraction > _NON_MONOTONIC_THRESHOLD:
            warnings.warn(
                "The unwrapped Hilbert phase has many large negative steps "
                f"({100 * neg_fraction:.1f}% of steps in the central region). "
                "The reference signal or frequency band may not define a "
                "reliable instantaneous phase.",
                UserWarning,
                stacklevel=2,
            )

    return phase_unwrapped, phase_wrapped, np.abs(analytic)


def fit_pca_phase_coordinates(
    X,
    *,
    epochs: CycleEpochs,
    columns=None,
    min_samples_per_cycle: int = 10,
):
    """
    Fit cycle-by-cycle PCA phase coordinates given a cycle assignment.

    This function does *only* coordinate estimation.  Phase estimation and
    cycle identification are separate pipeline stages that produce the
    :class:`~phase_coordinates.epochs.CycleEpochs` argument.

    Parameters
    ----------
    X : array-like or pandas.DataFrame, shape (n_time, n_features)
        Multivariate movement data.  Requires at least 3 features.
    epochs : CycleEpochs
        Cycle assignment.  Provides sample-to-cycle map, boundary times, and
        (optionally) phase / phase_in_cycle for the ``samples`` output.
    columns : list of str, optional
        Subset of columns to use when ``X`` is a DataFrame.
    min_samples_per_cycle : int
        Cycles with fewer valid samples are skipped (``fit_ok = False``).

    Returns
    -------
    samples : pandas.DataFrame
        One row per input time sample.  Columns: :data:`SAMPLE_COLUMNS`.
    cycles : pandas.DataFrame
        One row per fitted cycle.  Columns: :data:`CYCLE_COLUMNS`.
    details : dict
        Includes ``algorithm="pca"`` and a ``models`` dict keyed by cycle
        index, plus provenance from ``epochs``.
    """
    if not isinstance(epochs, CycleEpochs):
        raise TypeError(
            "fit_pca_phase_coordinates requires epochs=CycleEpochs; "
            f"got {type(epochs).__name__}."
        )

    # ---- input handling ----
    if isinstance(X, pd.DataFrame):
        index = X.index
        columns_used = columns if columns else list(X.columns)
        if columns:
            X_arr = X[columns].to_numpy(dtype=float)
        else:
            X_arr = X.to_numpy(dtype=float)
    else:
        index = None
        columns_used = columns
        X_arr = np.asarray(X, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError("X must have shape (n_time, n_features).")
    if X_arr.shape[1] < 3:
        raise ValueError("Need at least 3 features for local plane + perpendicular deviation.")

    n_time = X_arr.shape[0]
    n_features = X_arr.shape[1]

    if len(epochs.cycle_index) != n_time:
        raise ValueError(
            f"epochs.cycle_index has length {len(epochs.cycle_index)} but X has "
            f"{n_time} rows."
        )

    cycle_id = np.asarray(epochs.cycle_index, dtype=int)
    K = epochs.n_cycles

    # ---- phase columns (optional; from epochs) ----
    if epochs.phase is not None:
        phase = np.asarray(epochs.phase, dtype=float)
        if len(phase) != n_time:
            raise ValueError("epochs.phase length does not match X.")
    else:
        phase = np.full(n_time, np.nan)

    if epochs.phase_in_cycle is not None:
        phase_in_cycle = np.asarray(epochs.phase_in_cycle, dtype=float)
        if len(phase_in_cycle) != n_time:
            raise ValueError("epochs.phase_in_cycle length does not match X.")
    else:
        phase_in_cycle = np.full(n_time, np.nan)

    # ---- allocate outputs ----
    pc1 = np.full(n_time, np.nan)
    pc2 = np.full(n_time, np.nan)
    pc3 = np.full(n_time, np.nan)
    theta_local = np.full(n_time, np.nan)
    theta_local_wrapped = np.full(n_time, np.nan)
    radius_local = np.full(n_time, np.nan)
    perp_local = np.full(n_time, np.nan)

    models = {}
    previous_components = None

    # ---- local PCA per cycle ----
    n_samples_in_cycle = {}

    for cyc in range(K):
        idx = np.where(cycle_id == cyc)[0]
        n_samples_in_cycle[cyc] = int(len(idx))

        if len(idx) < min_samples_per_cycle:
            continue

        X_cyc = X_arr[idx]
        valid = np.all(np.isfinite(X_cyc), axis=1)
        if valid.sum() < min_samples_per_cycle:
            continue

        idx_valid = idx[valid]
        X_valid = X_cyc[valid]

        center = X_valid.mean(axis=0)

        pca = PCA(n_components=3)
        scores = pca.fit_transform(X_valid - center)
        components = pca.components_.copy()

        if previous_components is not None:
            for k in range(3):
                if np.dot(components[k], previous_components[k]) < 0:
                    components[k] *= -1
                    scores[:, k] *= -1
            if n_features == 3:
                if np.dot(np.cross(components[0], components[1]), components[2]) < 0:
                    components[2] *= -1
                    scores[:, 2] *= -1

        previous_components = components.copy()

        u = scores[:, 0]
        v = scores[:, 1]
        z = scores[:, 2]

        theta_w = np.arctan2(v, u)
        theta_u = np.unwrap(theta_w)

        pc1[idx_valid] = u
        pc2[idx_valid] = v
        pc3[idx_valid] = z
        theta_local[idx_valid] = theta_u
        theta_local_wrapped[idx_valid] = theta_w
        radius_local[idx_valid] = np.hypot(u, v)
        perp_local[idx_valid] = z

        models[int(cyc)] = {
            "pca": pca,
            "center": center,
            "components": components,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "indices": idx_valid,
        }

    # ---- build samples DataFrame ----
    sample_index = np.arange(n_time)
    time_arr = np.asarray(epochs.time, dtype=float)
    samples = pd.DataFrame(
        {
            "sample_index": sample_index,
            "time": time_arr,
            "cycle": cycle_id,
            "phase": phase,
            "phase_in_cycle": phase_in_cycle,
            "u": pc1,
            "v": pc2,
            "radius": radius_local,
            "theta": theta_local,
            "theta_wrapped": theta_local_wrapped,
            "perp": perp_local,
        },
        index=index,
    )

    # ---- build cycles DataFrame ----
    # One row per epoch cycle (0..K-1): cycles skipped by the
    # min_samples_per_cycle filter still appear, with fit_ok=False and NaN
    # geometry, so len(cycles) == epochs.n_cycles always holds and callers
    # can distinguish "fitted" from "never fitted".
    cycle_rows = []
    for cyc_k in range(K):
        t_start = float(epochs.tau[cyc_k])
        t_stop = float(epochs.tau[cyc_k + 1])
        duration = t_stop - t_start
        t_quarter = t_start + 0.25 * duration
        s_start = int(epochs.sample_start[cyc_k])
        s_stop = int(epochs.sample_stop[cyc_k])

        m = models.get(cyc_k)
        if m is None:
            cycle_rows.append({
                "cycle": cyc_k,
                "sample_start": s_start,
                "sample_stop": s_stop,
                "time_start": t_start,
                "time_stop": t_stop,
                "time_quarter": t_quarter,
                "duration": duration,
                "center_x": float("nan"),
                "center_y": float("nan"),
                "center_z": float("nan"),
                "e1_x": float("nan"),
                "e1_y": float("nan"),
                "e1_z": float("nan"),
                "e2_x": float("nan"),
                "e2_y": float("nan"),
                "e2_z": float("nan"),
                "normal_x": float("nan"),
                "normal_y": float("nan"),
                "normal_z": float("nan"),
                "radius_mean": float("nan"),
                "radius_sd": float("nan"),
                "perp_mean": float("nan"),
                "perp_sd": float("nan"),
                "n_samples": n_samples_in_cycle.get(cyc_k, 0),
                "fit_ok": False,
            })
            continue

        idx_k = m["indices"]
        center = m["center"]
        comps = m["components"]

        def _get_vec3(arr, row):
            v = np.zeros(3)
            if len(arr) > row:
                d = min(3, len(arr[row]))
                v[:d] = arr[row][:d]
            return v

        cx, cy, cz = (center[:3] if len(center) >= 3
                      else np.pad(center, (0, 3 - len(center))))
        e1 = _get_vec3(comps, 0)
        e2 = _get_vec3(comps, 1)
        normal = _get_vec3(comps, 2)

        r_vals = radius_local[idx_k]
        p_vals = perp_local[idx_k]

        cycle_rows.append({
            "cycle": cyc_k,
            "sample_start": s_start,
            "sample_stop": s_stop,
            "time_start": t_start,
            "time_stop": t_stop,
            "time_quarter": t_quarter,
            "duration": duration,
            "center_x": float(cx),
            "center_y": float(cy),
            "center_z": float(cz),
            "e1_x": float(e1[0]),
            "e1_y": float(e1[1]),
            "e1_z": float(e1[2]),
            "e2_x": float(e2[0]),
            "e2_y": float(e2[1]),
            "e2_z": float(e2[2]),
            "normal_x": float(normal[0]),
            "normal_y": float(normal[1]),
            "normal_z": float(normal[2]),
            "radius_mean": float(np.nanmean(r_vals)) if r_vals.size else float("nan"),
            "radius_sd": float(np.nanstd(r_vals)) if r_vals.size else float("nan"),
            "perp_mean": float(np.nanmean(p_vals)) if p_vals.size else float("nan"),
            "perp_sd": float(np.nanstd(p_vals)) if p_vals.size else float("nan"),
            "n_samples": int(len(idx_k)),
            "fit_ok": True,
        })

    if cycle_rows:
        cycles = pd.DataFrame(cycle_rows, columns=CYCLE_COLUMNS)
    else:
        cycles = pd.DataFrame(columns=CYCLE_COLUMNS)

    details = {
        "algorithm": "pca",
        "models": models,
        "epochs_source": epochs.source,
        "epochs_metadata": dict(epochs.metadata),
        "input_columns": columns_used,
        "warnings": [],
    }

    return samples, cycles, details


def reconstruct_phase_coordinates(samples, cycles):
    """
    Reconstruct trajectory from samples and cycles DataFrames.

    Returns np.ndarray of shape (n_time, 3). NaN rows where not reconstructable.
    """
    n_time = len(samples)
    X_hat = np.full((n_time, 3), np.nan)

    merged = samples[["sample_index", "cycle", "u", "v", "perp"]].merge(
        cycles[["cycle", "center_x", "center_y", "center_z",
                "e1_x", "e1_y", "e1_z",
                "e2_x", "e2_y", "e2_z",
                "normal_x", "normal_y", "normal_z", "fit_ok"]],
        on="cycle", how="left"
    )

    mask = (
        merged["fit_ok"].fillna(False) &
        merged["u"].notna() &
        merged["v"].notna() &
        merged["perp"].notna()
    )

    m = merged[mask]
    if len(m) == 0:
        return X_hat

    idx = m["sample_index"].values.astype(int)
    u = m["u"].values
    v = m["v"].values
    perp = m["perp"].values

    X_hat[idx, 0] = (m["center_x"].values + u * m["e1_x"].values
                     + v * m["e2_x"].values + perp * m["normal_x"].values)
    X_hat[idx, 1] = (m["center_y"].values + u * m["e1_y"].values
                     + v * m["e2_y"].values + perp * m["normal_y"].values)
    X_hat[idx, 2] = (m["center_z"].values + u * m["e1_z"].values
                     + v * m["e2_z"].values + perp * m["normal_z"].values)

    return X_hat
