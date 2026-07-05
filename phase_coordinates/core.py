"""
Core functions for cycle-by-cycle PCA phase coordinates.
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import butter, sosfiltfilt, hilbert

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
    ref_signal = np.asarray(ref_signal, dtype=float)

    # ---- input validation ----
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

    sos = butter(
        N=4,
        Wn=f_range,
        btype="bandpass",
        fs=fs,
        output="sos",
    )

    x_filt = sosfiltfilt(sos, ref_signal)
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


def cycle_by_cycle_pca_coordinates(
    X,
    ref_signal=None,
    phase=None,
    fs=None,
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

    .. note::
        **Recommended phase coordinate**: ``phase_in_cycle`` is the primary
        timing/phase coordinate to use for cross-cycle alignment and
        averaging. It is derived directly from the input phase estimate and
        is consistent across cycles.

        ``theta_local`` is the geometric angle in the local PCA plane and is
        useful for describing within-cycle geometry. However, it should be
        treated cautiously across cycles: because PCA axes can rotate, flip
        signs, or swap when PC1/PC2 variances are similar, ``theta_local``
        may be discontinuous or inconsistent between cycles.

    Parameters
    ----------
    X : array-like or pandas.DataFrame, shape (n_time, n_features)
        Multivariate movement data, e.g. x/y/z marker positions.
        Requires at least 3 features.
    ref_signal : array-like, optional
        Scalar signal used to estimate Hilbert phase (e.g. one joint angle
        or one marker coordinate). Required when ``phase`` is not supplied.
    phase : array-like, optional
        Precomputed *unwrapped* phase in radians. When supplied,
        ``ref_signal``, ``fs``, and ``f_range`` are not needed.
    fs : float, optional
        Sampling rate in Hz. Required when ``ref_signal`` is used.
    f_range : tuple of float, optional
        Bandpass range for Hilbert-phase estimation, e.g. ``(0.5, 3.0)``.
        Required when ``ref_signal`` is used.
    columns : list of str, optional
        Subset of columns to use when ``X`` is a :class:`pandas.DataFrame`.
        If ``None``, all columns are used.
    min_samples_per_cycle : int
        Cycles with fewer than this many valid samples are skipped.

    Returns
    -------
    coords : pandas.DataFrame
        One row per time point. Columns:

        ``cycle``
            Integer cycle index (floor of unwrapped phase / 2π).
        ``phase``
            Unwrapped phase in radians.
        ``phase_wrapped``
            Phase wrapped to ``[-π, π]``.
        ``phase_in_cycle``
            **Primary phase coordinate.** Phase within the current cycle
            (range ``[0, 2π)``). Use this for cross-cycle alignment and
            averaging.
        ``amp_hilbert``
            Hilbert amplitude (``NaN`` when phase is supplied directly).
        ``pc1_local``
            Score along the first local principal component.
        ``pc2_local``
            Score along the second local principal component.
        ``pc3_local``
            Score along the third local principal component.
        ``theta_local``
            Unwrapped angle in the local PCA plane (radians). Useful for
            within-cycle geometric description, but may be inconsistent
            across cycles if PCA axes rotate or flip between cycles.
        ``theta_local_wrapped``
            ``theta_local`` wrapped to ``[-π, π]``.
        ``radius_local``
            Radius in the local PCA plane (Euclidean distance from the
            cycle centre in the pc1–pc2 plane).
        ``perp_local``
            Signed deviation perpendicular to the local PCA plane (pc3
            score).

    models : dict
        Keyed by integer cycle index. Each value is a dict with:

        ``pca``
            Fitted :class:`sklearn.decomposition.PCA` object.
        ``center``
            Mean position of the cycle data (shape ``(n_features,)``).
        ``components``
            Sign-aligned PCA components (shape ``(3, n_features)``).
        ``explained_variance_ratio``
            Explained variance ratio for each component.
        ``indices``
            Integer *positional* indices (0-based) of the time points
            belonging to this cycle in the original array/DataFrame.

    Notes
    -----
    **Reconstruction accuracy**: for 3-feature input, reconstruction from
    ``(pc1_local, pc2_local, pc3_local)`` via the per-cycle ``center`` and
    ``components`` is exact (up to floating-point precision). For input with
    more than 3 features, only 3 principal components are retained, so
    reconstruction is generally approximate. The reconstruction error is
    small only when the data are locally near rank-3 within each cycle.
    """
    # ---- input handling ----
    if isinstance(X, pd.DataFrame):
        index = X.index
        X_arr = X[columns].to_numpy(dtype=float) if columns else X.to_numpy(dtype=float)
    else:
        index = None
        X_arr = np.asarray(X, dtype=float)

    if X_arr.ndim != 2:
        raise ValueError("X must have shape (n_time, n_features).")

    if X_arr.shape[1] < 3:
        raise ValueError("Need at least 3 features for local plane + perpendicular deviation.")

    n_time = X_arr.shape[0]
    n_features = X_arr.shape[1]

    # ---- get phase ----
    if phase is None:
        if ref_signal is None or fs is None or f_range is None:
            raise ValueError(
                "Provide either phase directly, or provide ref_signal, fs, and f_range."
            )

        phase, phase_wrapped, amp_hilbert = hilbert_phase(
            ref_signal=ref_signal,
            fs=fs,
            f_range=f_range,
        )
    else:
        phase = np.asarray(phase, dtype=float)
        phase_wrapped = np.angle(np.exp(1j * phase))
        amp_hilbert = np.full(n_time, np.nan)

    if len(phase) != n_time:
        raise ValueError("phase/ref_signal must have the same length as X.")

    # ---- define cycles from unwrapped phase ----
    phase0 = phase - phase[0]
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

    coords = pd.DataFrame(
        {
            "cycle": cycle_id,
            "phase": phase,
            "phase_wrapped": phase_wrapped,
            "phase_in_cycle": phase_in_cycle,
            "amp_hilbert": amp_hilbert,
            "pc1_local": pc1,
            "pc2_local": pc2,
            "pc3_local": pc3,
            "theta_local": theta_local,
            "theta_local_wrapped": theta_local_wrapped,
            "radius_local": radius_local,
            "perp_local": perp_local,
        },
        index=index,
    )

    return coords, models
