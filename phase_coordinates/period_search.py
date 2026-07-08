"""
Period candidate search for cycle identification.

This module *only* proposes candidate periods for a scalar reference signal.
It has no knowledge of cycle epochs, geometric scoring, or PCA — those live
downstream in :mod:`phase_coordinates.scoring`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks, periodogram


@dataclass
class PeriodCandidate:
    """
    A candidate period value with a normalized strength score.

    Parameters
    ----------
    period : float
        Candidate period in seconds.
    source : str
        Where the candidate came from (``"periodogram"``, ``"autocorrelation"``,
        or ``"harmonic:<parent_source>"``).
    score : float
        Strength of the candidate, normalized to ``[0, 1]``.  Larger is
        stronger.
    """

    period: float
    source: str
    score: float


def _resolve_period_bounds(fs, n_samples, max_period, min_period):
    """Return (min_period, max_period) with sensible defaults for a signal."""
    fs = float(fs)
    if min_period is None:
        min_period = 2.0 / fs
    if max_period is None:
        # Half the signal duration lets us see at least ~2 full cycles.
        max_period = (n_samples - 1) / fs / 2.0
    min_period = float(min_period)
    max_period = float(max_period)
    if max_period <= min_period:
        raise ValueError(
            f"max_period ({max_period}) must be > min_period ({min_period})."
        )
    return min_period, max_period


def period_candidates_from_periodogram(
    ref_signal,
    fs,
    *,
    max_period: Optional[float] = None,
    min_period: Optional[float] = None,
    n_candidates: int = 5,
) -> list:
    """
    Propose period candidates from local peaks of the periodogram.

    Parameters
    ----------
    ref_signal : array-like, shape (n_time,)
        Scalar reference signal.
    fs : float
        Sampling rate in Hz.
    max_period, min_period : float, optional
        Search-window bounds in seconds.  Default: ``[2/fs, (n-1)/(2*fs)]``.
    n_candidates : int
        Return up to this many candidates, sorted by score descending.

    Returns
    -------
    list of :class:`PeriodCandidate`
    """
    ref = np.asarray(ref_signal, dtype=float)
    if ref.ndim != 1:
        raise ValueError(f"ref_signal must be 1-D, got shape {ref.shape}.")
    if not np.all(np.isfinite(ref)):
        raise ValueError("ref_signal contains non-finite values.")
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")

    lo, hi = _resolve_period_bounds(fs, len(ref), max_period, min_period)
    f_min = 1.0 / hi
    f_max = 1.0 / lo

    freqs, power = periodogram(ref, fs=fs)
    valid = (freqs >= f_min) & (freqs <= f_max) & (freqs > 0)
    if not np.any(valid):
        return []

    f_valid = freqs[valid]
    p_valid = power[valid]

    # Peaks in the (restricted) power spectrum.  Fall back to argmax if
    # find_peaks returns nothing (e.g. monotonic power).
    peak_idx, _ = find_peaks(p_valid)
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(p_valid))])

    p_max = float(p_valid.max())
    if p_max <= 0:
        return []

    order = np.argsort(-p_valid[peak_idx])
    peak_idx = peak_idx[order][:n_candidates]

    return [
        PeriodCandidate(
            period=float(1.0 / f_valid[i]),
            source="periodogram",
            score=float(p_valid[i] / p_max),
        )
        for i in peak_idx
    ]


def period_candidates_from_autocorrelation(
    ref_signal,
    fs,
    *,
    max_period: Optional[float] = None,
    min_period: Optional[float] = None,
    n_candidates: int = 5,
) -> list:
    """
    Propose period candidates from peaks of the normalized autocorrelation.

    Parameters
    ----------
    ref_signal : array-like, shape (n_time,)
        Scalar reference signal.
    fs : float
        Sampling rate in Hz.
    max_period, min_period : float, optional
        Search-window bounds in seconds.
    n_candidates : int
        Return up to this many candidates, sorted by score descending.

    Returns
    -------
    list of :class:`PeriodCandidate`
    """
    ref = np.asarray(ref_signal, dtype=float)
    if ref.ndim != 1:
        raise ValueError(f"ref_signal must be 1-D, got shape {ref.shape}.")
    if not np.all(np.isfinite(ref)):
        raise ValueError("ref_signal contains non-finite values.")
    fs = float(fs)
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}.")

    lo, hi = _resolve_period_bounds(fs, len(ref), max_period, min_period)
    lag_min = max(1, int(np.floor(lo * fs)))
    lag_max = min(len(ref) - 1, int(np.ceil(hi * fs)))
    if lag_max <= lag_min:
        return []

    x = ref - ref.mean()
    n = len(x)
    # Full autocorrelation via numpy.correlate; normalize by variance at lag 0.
    ac = np.correlate(x, x, mode="full")[n - 1 :]
    if ac[0] <= 0:
        return []
    ac_norm = ac / ac[0]

    lags = np.arange(lag_min, lag_max + 1)
    ac_window = ac_norm[lag_min : lag_max + 1]
    peak_idx, _ = find_peaks(ac_window)
    if len(peak_idx) == 0:
        peak_idx = np.array([int(np.argmax(ac_window))])

    # Clip scores to [0, 1] — negative correlations are not usable.
    scores = np.clip(ac_window[peak_idx], 0.0, 1.0)
    order = np.argsort(-scores)
    peak_idx = peak_idx[order][:n_candidates]
    scores = scores[order][:n_candidates]

    return [
        PeriodCandidate(
            period=float(lags[i] / fs),
            source="autocorrelation",
            score=float(s),
        )
        for i, s in zip(peak_idx, scores)
    ]


def expand_period_harmonics(candidates, *, harmonics=(0.5, 1.0, 2.0)) -> list:
    """
    Add harmonic multipliers of each candidate.

    For every input candidate ``c``, output ``PeriodCandidate(c.period * h, ...)``
    for each ``h`` in ``harmonics``.  Duplicate periods within 5% of one
    another are merged (keeping the higher-scoring one).

    Parameters
    ----------
    candidates : list of :class:`PeriodCandidate`
    harmonics : tuple of float
        Multipliers to apply.  ``1.0`` should typically be included to keep
        the originals.

    Returns
    -------
    list of :class:`PeriodCandidate`
        Sorted by score descending.
    """
    expanded = []
    for c in candidates:
        for h in harmonics:
            if h == 1.0:
                expanded.append(
                    PeriodCandidate(
                        period=c.period, source=c.source, score=c.score
                    )
                )
            else:
                expanded.append(
                    PeriodCandidate(
                        period=c.period * float(h),
                        source=f"harmonic:{c.source}",
                        score=c.score,
                    )
                )

    # Deduplicate within 5% period difference, keeping the higher score.
    expanded.sort(key=lambda c: -c.score)
    kept = []
    for c in expanded:
        merge = False
        for prev in kept:
            if abs(c.period - prev.period) / max(prev.period, 1e-12) < 0.05:
                merge = True
                break
        if not merge:
            kept.append(c)

    return kept
