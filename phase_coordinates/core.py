"""
Core functions for cycle-by-cycle PCA phase coordinates.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import butter, find_peaks, sosfiltfilt, hilbert, periodogram

# Minimum signal length required by sosfiltfilt with a 4th-order Butterworth
# filter (2 second-order sections → default padlen = 3 * 2 * 2 = 12).
_HILBERT_MIN_SAMPLES = 13

# A phase step more negative than this (radians) is counted as a "large
# backward jump".  A value of -0.5 rad is large relative to normal per-sample
# phase increments for typical movement frequencies but small relative to the
# π jumps that occur at amplitude nulls in multi-frequency or noisy signals.
_PHASE_JUMP_THRESHOLD = -0.5

# If the fraction of backward jumps in the central region of the signal
# exceeds this value, a UserWarning is issued.  0.3% is essentially zero for
# a clean single-frequency signal but is reliably exceeded by two-frequency
# beating signals (which produce π jumps at amplitude nulls, ~0.4%).
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


def dominant_reference_signal(X, *, y_axis=1):
    """Return a Y-oriented score along the dominant principal component.

    ``X`` is mean-centred before estimating its primary axis of variation.
    PCA component signs are otherwise arbitrary, so the axis is oriented to
    have a non-negative component along ``y_axis`` (the Y coordinate by
    default).  Consequently, a positive peak of the returned score denotes
    the positive-Y end of the dominant movement.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}.")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(
            "X must contain at least two samples and one feature to derive "
            "a dominant reference signal."
        )
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains non-finite values (NaN or Inf).")
    if not 0 <= y_axis < X.shape[1]:
        raise ValueError(
            f"y_axis must index a feature of X, got {y_axis} for shape {X.shape}."
        )

    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    axis = vt[0]
    if axis[y_axis] < 0:
        axis = -axis
    return Xc @ axis


def estimate_dominant_period(ref_signal, fs):
    """Estimate a scalar signal's dominant positive-frequency period."""
    ref_signal = np.asarray(ref_signal, dtype=float)
    freqs, power = periodogram(ref_signal, fs=fs)
    valid = freqs > 0
    if not np.any(valid):
        raise ValueError(
            "Cannot estimate a dominant frequency: signal is too short or "
            "has no positive-frequency content."
        )
    f0 = float(freqs[valid][np.argmax(power[valid])])
    if f0 <= 0:
        raise ValueError("Estimated dominant frequency is non-positive.")
    return 1.0 / f0


def _automatic_hilbert_band(ref_signal, fs):
    """Choose a Hilbert band centred on the dominant periodogram peak."""
    T0 = estimate_dominant_period(ref_signal, fs)
    low = 0.5 / T0
    high = min(2.0 / T0, np.nextafter(float(fs) / 2.0, 0.0))
    if low >= high:
        raise ValueError(
            "Cannot construct a valid Hilbert band around the estimated "
            "dominant frequency. Provide ref_signal and f_range explicitly."
        )
    return (low, high)


def _bandpass_reference_signal(ref_signal, fs, f_range):
    """Validate and band-pass a scalar reference signal."""
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
        return sosfiltfilt(sos, ref_signal)
    except ValueError as exc:
        raise ValueError(
            "ref_signal is too short for scipy.signal.sosfiltfilt with the "
            "chosen 4th-order bandpass filter. Use a longer signal, use a "
            "lower-order filter in the future, or provide a precomputed phase."
        ) from exc


def _first_positive_peak_index(ref_signal, fs, f_range):
    """Return the first interior positive peak of a filtered reference."""
    filtered = _bandpass_reference_signal(ref_signal, fs, f_range)
    peaks, _ = find_peaks(filtered)
    if len(peaks) == 0:
        raise ValueError(
            "Could not detect a positive peak in the band-passed dominant "
            "reference signal. Provide a longer / cleaner recording."
        )
    return int(peaks[0])


def hilbert_phase(ref_signal, fs, f_range):
    """
    Get unwrapped Hilbert phase from a scalar reference signal.

    Parameters
    ----------
    ref_signal : array-like, shape (n_time,)
        Scalar 1-D time series (e.g. one joint angle or marker coordinate).
        Must contain only finite values.
    fs : float
        Sampling rate in Hz. Must be positive.
    f_range : tuple of float
        Bandpass frequency range ``(low, high)`` in Hz.
        Must satisfy ``0 < low < high < fs / 2``.

    Returns
    -------
    phase_unwrapped : numpy.ndarray
        Unwrapped instantaneous phase in radians.
    phase_wrapped : numpy.ndarray
        Wrapped instantaneous phase in radians (range ``[-pi, pi]``).
    amplitude : numpy.ndarray
        Instantaneous amplitude (envelope) of the analytic signal.

    Raises
    ------
    ValueError
        If ``ref_signal`` is not 1-D, contains non-finite values, is too
        short for the filter, ``fs`` is not positive, or ``f_range`` is
        invalid.

    Warns
    -----
    UserWarning
        If the unwrapped phase has many large negative steps in the central
        region of the signal, indicating that the reference signal or
        frequency band may not define a reliable instantaneous phase.
    """
    x_filt = _bandpass_reference_signal(ref_signal, fs, f_range)
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
    phase=None,
    ref_signal=None,
    sampling_rate_hz=None,
    f_range=None,
    columns=None,
    min_samples_per_cycle=10,
):
    """
    Compute cycle-by-cycle PCA geometric coordinates.

    Each cycle gets its own PCA plane fitted to the movement data in that
    cycle. Within each cycle, the first two principal components span the
    "phase plane" and the third component captures deviation perpendicular
    to that plane.

    Parameters
    ----------
    X : array-like or pandas.DataFrame, shape (n_time, n_features)
        Multivariate movement data, e.g. x/y/z marker positions.
        Requires at least 3 features.
    phase : array-like, optional
        Precomputed *unwrapped* phase in radians. When supplied,
        ``ref_signal``, ``sampling_rate_hz``, and ``f_range`` are not needed.
    ref_signal : array-like, optional
        Scalar signal used to estimate Hilbert phase (e.g. one joint angle
        or one marker coordinate). If omitted along with ``phase``, the
        Y-oriented dominant principal-component score of ``X`` is used.
    sampling_rate_hz : float, optional
        Sampling rate in Hz. Required when phase is not supplied.
    f_range : tuple of float, optional
        Bandpass range for Hilbert-phase estimation, e.g. ``(0.5, 3.0)``.
        Required when ``ref_signal`` is supplied. With the automatic
        dominant reference, it defaults to ``(0.5 * f0, 2 * f0)`` around
        the reference's periodogram peak ``f0``.
    columns : list of str, optional
        Subset of columns to use when ``X`` is a :class:`pandas.DataFrame`.
        If ``None``, all columns are used.
    min_samples_per_cycle : int
        Cycles with fewer than this many valid samples are skipped.

    Returns
    -------
    samples : pandas.DataFrame
        One row per time point. Columns = SAMPLE_COLUMNS.
    cycles : pandas.DataFrame
        One row per fitted cycle. Columns = CYCLE_COLUMNS.
    details : dict
        Algorithm-specific information including the per-cycle PCA models.
    """
    # ---- input handling ----
    if isinstance(X, pd.DataFrame):
        index = X.index
        columns_used = columns if columns else list(X.columns)
        X_arr = X[columns_used].to_numpy(dtype=float) if columns else X.to_numpy(dtype=float)
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

    # ---- get phase ----
    phase_source = "provided"
    phase_zero = None
    if phase is None:
        if sampling_rate_hz is None:
            raise ValueError(
                "Provide either phase directly, or provide sampling_rate_hz "
                "with ref_signal and f_range."
            )
        if ref_signal is None:
            ref_signal = dominant_reference_signal(X_arr)
            if f_range is None:
                f_range = _automatic_hilbert_band(ref_signal, sampling_rate_hz)
            phase_source = "dominant_reference_hilbert"
        elif f_range is None:
            raise ValueError(
                "f_range is required when ref_signal is supplied. Omit both "
                "ref_signal and f_range to use the automatic dominant reference."
            )

        phase, phase_wrapped, amp_hilbert = hilbert_phase(
            ref_signal=ref_signal,
            fs=sampling_rate_hz,
            f_range=f_range,
        )
        if phase_source == "dominant_reference_hilbert":
            peak_index = _first_positive_peak_index(
                ref_signal, sampling_rate_hz, f_range
            )
            phase = phase - phase[peak_index]
            phase_wrapped = np.angle(np.exp(1j * phase))
            phase_zero = {
                "method": "first_positive_bandpassed_peak",
                "sample_index": peak_index,
            }
        else:
            phase_source = "hilbert"
    else:
        phase = np.asarray(phase, dtype=float)
        if phase.ndim != 1:
            raise ValueError(
                f"phase must be 1-D, got shape {phase.shape}."
            )
        if not np.all(np.isfinite(phase)):
            raise ValueError("phase contains non-finite values (NaN or Inf).")
        phase_wrapped = np.angle(np.exp(1j * phase))
        amp_hilbert = np.full(n_time, np.nan)

    if len(phase) != n_time:
        raise ValueError("phase/ref_signal must have the same length as X.")

    # ---- define cycles from unwrapped phase ----
    # An automatic reference is explicitly anchored at its first positive,
    # band-passed peak. Retain the existing first-sample convention for an
    # externally supplied or precomputed phase.
    phase0 = phase if phase_zero is not None else phase - phase[0]
    cycle_id = np.floor(phase0 / (2 * np.pi)).astype(int)
    phase_in_cycle = np.mod(phase0, 2 * np.pi)

    # ---- allocate outputs ----
    pc1 = np.full(n_time, np.nan)
    pc2 = np.full(n_time, np.nan)
    pc3 = np.full(n_time, np.nan)

    theta_local_wrapped = np.full(n_time, np.nan)
    theta_local = np.full(n_time, np.nan)
    radius_local = np.full(n_time, np.nan)
    perp_local = np.full(n_time, np.nan)

    models = {}

    previous_components = None

    # ---- local PCA per cycle ----
    for cyc in np.unique(cycle_id):
        idx = np.where(cycle_id == cyc)[0]

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

        # ---- align signs/orientation to previous cycle ----
        # PCA axes can flip sign arbitrarily; keep the local frame consistent.
        if previous_components is not None:
            for k in range(3):
                if np.dot(components[k], previous_components[k]) < 0:
                    components[k] *= -1
                    scores[:, k] *= -1

            # Keep right-handed orientation approximately consistent.
            # np.cross is only defined for 3-D vectors; skip this check
            # for higher-dimensional feature spaces.
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

        theta_local_wrapped[idx_valid] = theta_w
        theta_local[idx_valid] = theta_u
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
    if sampling_rate_hz is not None:
        time_arr = sample_index / float(sampling_rate_hz)
    else:
        time_arr = np.full(n_time, np.nan)

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
    cycle_rows = []
    for cyc_k, m in models.items():
        idx_k = m["indices"]
        s_start = int(idx_k.min())
        s_stop = int(idx_k.max()) + 1

        if sampling_rate_hz is not None:
            fs = float(sampling_rate_hz)
            t_start = s_start / fs
            t_stop = s_stop / fs
            duration = t_stop - t_start
            t_quarter = t_start + 0.25 * duration
        else:
            t_start = np.nan
            t_stop = np.nan
            duration = np.nan
            t_quarter = np.nan

        center = m["center"]
        comps = m["components"]

        def _get_vec3(arr, row):
            if len(arr) > row and len(arr[row]) >= 3:
                return arr[row][:3]
            v = np.zeros(3)
            if len(arr) > row:
                v[:len(arr[row])] = arr[row]
            return v

        cx, cy, cz = (center[:3] if len(center) >= 3 else np.pad(center, (0, 3 - len(center))))
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
            "radius_mean": float(np.nanmean(r_vals)),
            "radius_sd": float(np.nanstd(r_vals)),
            "perp_mean": float(np.nanmean(p_vals)),
            "perp_sd": float(np.nanstd(p_vals)),
            "n_samples": len(idx_k),
            "fit_ok": True,
        })

    if cycle_rows:
        cycles = pd.DataFrame(cycle_rows, columns=CYCLE_COLUMNS)
    else:
        cycles = pd.DataFrame(columns=CYCLE_COLUMNS)

    # ---- build details dict ----
    details = {
        "algorithm": "pca",
        "models": models,
        "phase_source": phase_source,
        "ref_signal": ref_signal if phase_source != "provided" else None,
        "phase_zero": phase_zero,
        "input_columns": columns_used,
        "amp_hilbert": amp_hilbert,
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
