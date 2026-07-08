"""
Cycle epoch representation and construction.

A :class:`CycleEpochs` is the shared "who belongs to which cycle" contract
used by every downstream stage (coordinate estimation, diagnostics, and the
Bayesian model).  Cycle identification is a *separate* pipeline stage from
phase estimation, coordinate estimation, and diagnostics.

Currently supported constructors:

``identify_cycles_from_phase``
    Build cycles from an unwrapped phase signal.  Cycle boundaries are the
    zero-crossings of ``phase - phase_zero`` at multiples of ``2*pi``.

``epochs_from_boundary_indices``
    Build cycles from a set of integer sample indices marking boundaries.
    Phase information is unknown and left as ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class CycleEpochs:
    """
    Per-cycle boundary times and per-sample cycle assignments.

    Parameters
    ----------
    tau : ndarray, shape (K + 1,)
        Real-valued cycle boundary times, in seconds. ``tau[k]`` is the start
        of cycle ``k`` and ``tau[k + 1]`` is its end.  Must be strictly
        increasing.
    duration : ndarray, shape (K,)
        Cycle durations. Equal to ``tau[1:] - tau[:-1]``.
    cycle_index : ndarray, shape (n_time,), int
        Per-sample cycle assignment. Samples that fall outside the covered
        time window (before ``tau[0]`` or after ``tau[-1]``) are marked with
        ``-1``.
    phase : ndarray or None, shape (n_time,)
        Unwrapped phase in radians when known, otherwise ``None``.
    phase_in_cycle : ndarray or None, shape (n_time,)
        Phase within the current cycle, in ``[0, 2*pi)``, when known,
        otherwise ``None``.
    time : ndarray, shape (n_time,)
        Sample times in seconds (``sample_index / sampling_rate_hz``).
    source : str
        Free-form tag describing where the epochs came from
        (``"phase"``, ``"periodogram_peaks"``, ``"geometric_score"``, ...).
    metadata : dict
        Arbitrary metadata (algorithm parameters, sampling rate, provenance).
    """

    tau: np.ndarray
    duration: np.ndarray
    cycle_index: np.ndarray
    phase: Optional[np.ndarray]
    phase_in_cycle: Optional[np.ndarray]
    time: np.ndarray
    source: str
    metadata: dict = field(default_factory=dict)

    @property
    def n_cycles(self) -> int:
        """Number of cycles ``K``."""
        return int(len(self.tau) - 1)

    @property
    def sample_start(self) -> np.ndarray:
        """First sample index in each cycle, shape ``(K,)``.

        For cycles that contain at least one sample, this is the index of
        that first sample.  For empty cycles (no sample falls inside),
        ``sample_start == sample_stop`` and both equal the smallest sample
        index at-or-after ``tau[k]``.
        """
        K = self.n_cycles
        starts = np.zeros(K, dtype=int)
        for k in range(K):
            hit = np.where(self.cycle_index == k)[0]
            if hit.size > 0:
                starts[k] = int(hit[0])
            else:
                # Empty cycle: point at the smallest sample_index at-or-after
                # tau[k].  We only need this as a placeholder; length is
                # zero because sample_stop = sample_start.
                after = np.searchsorted(self.time, self.tau[k], side="left")
                starts[k] = int(min(after, len(self.time)))
        return starts

    @property
    def sample_stop(self) -> np.ndarray:
        """One past the last sample index in each cycle (Python slice), shape ``(K,)``."""
        K = self.n_cycles
        stops = np.zeros(K, dtype=int)
        starts = self.sample_start
        for k in range(K):
            hit = np.where(self.cycle_index == k)[0]
            if hit.size > 0:
                stops[k] = int(hit[-1]) + 1
            else:
                stops[k] = int(starts[k])
        return stops

    @property
    def time_start(self) -> np.ndarray:
        """Cycle start times, ``tau[:-1]`` (shape ``(K,)``)."""
        return self.tau[:-1]

    @property
    def time_stop(self) -> np.ndarray:
        """Cycle stop times, ``tau[1:]`` (shape ``(K,)``)."""
        return self.tau[1:]

    @property
    def time_quarter(self) -> np.ndarray:
        """Quarter-cycle times: ``tau[:-1] + 0.25 * duration`` (shape ``(K,)``)."""
        return self.tau[:-1] + 0.25 * self.duration


def identify_cycles_from_phase(
    phase: np.ndarray,
    *,
    sampling_rate_hz: Optional[float] = None,
    phase_zero: str = "first_sample",
) -> CycleEpochs:
    """
    Build :class:`CycleEpochs` from an unwrapped phase signal.

    Parameters
    ----------
    phase : ndarray, shape (n_time,)
        Unwrapped instantaneous phase in radians.
    sampling_rate_hz : float, optional
        Sampling rate in Hz.  Required to convert sample indices to seconds
        for the boundary times ``tau``.
    phase_zero : {"first_sample"}
        How to define phase zero.  Currently only ``"first_sample"`` is
        supported, which subtracts ``phase[0]`` so the first cycle starts at
        sample 0.

    Returns
    -------
    CycleEpochs
        Epochs with ``source="phase"``, ``phase = phase``, and
        ``phase_in_cycle`` filled in.

    Notes
    -----
    Boundary times ``tau[k]`` are the times where ``phase - phase[0]`` crosses
    ``k * 2*pi``, computed by linear interpolation between adjacent samples.
    This function does *not* call :func:`hilbert_phase` — callers who need
    that must run it first.
    """
    phase = np.asarray(phase, dtype=float)
    if phase.ndim != 1:
        raise ValueError(f"phase must be 1-D, got shape {phase.shape}.")
    if not np.all(np.isfinite(phase)):
        raise ValueError("phase contains non-finite values (NaN or Inf).")
    if len(phase) < 2:
        raise ValueError("phase must contain at least 2 samples.")
    if sampling_rate_hz is None:
        raise ValueError("sampling_rate_hz is required.")
    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")

    if phase_zero != "first_sample":
        raise ValueError(
            f"phase_zero={phase_zero!r} is not supported (only 'first_sample')."
        )

    n_time = len(phase)
    time = np.arange(n_time) / fs

    phase0 = phase - phase[0]
    cycle_index = np.floor(phase0 / (2 * np.pi)).astype(int)
    phase_in_cycle = np.mod(phase0, 2 * np.pi)

    # Find boundary times: where phase0 crosses k * 2*pi.  Under the
    # "first_sample" convention cycle 0 starts at sample 0 by construction.
    k_max = int(cycle_index.max())
    boundaries = [0.0]
    for k in range(1, k_max + 1):
        target = k * 2 * np.pi
        # find the first sample i where phase0[i] >= target
        idx = np.searchsorted(phase0, target, side="left")
        if idx <= 0 or idx >= n_time:
            # cycle boundary falls outside the observation window; stop
            break
        p_lo, p_hi = phase0[idx - 1], phase0[idx]
        if p_hi == p_lo:
            frac = 0.0
        else:
            frac = (target - p_lo) / (p_hi - p_lo)
        t_lo = (idx - 1) / fs
        t_hi = idx / fs
        boundaries.append(t_lo + frac * (t_hi - t_lo))
    tau = np.asarray(boundaries, dtype=float)
    duration = np.diff(tau)

    # Clip cycle_index for samples beyond the last boundary — they don't
    # belong to any *complete* cycle we can identify.
    K = int(len(tau) - 1)
    ci = cycle_index.copy()
    ci[ci >= K] = -1
    ci[ci < 0] = -1

    return CycleEpochs(
        tau=tau,
        duration=duration,
        cycle_index=ci,
        phase=phase,
        phase_in_cycle=phase_in_cycle,
        time=time,
        source="phase",
        metadata={"phase_zero": phase_zero, "sampling_rate_hz": fs},
    )


def epochs_from_boundary_indices(
    tau_idx: np.ndarray,
    *,
    sampling_rate_hz: float,
    n_time: int,
    source: str = "periodogram_peaks",
    metadata: Optional[dict] = None,
) -> CycleEpochs:
    """
    Build :class:`CycleEpochs` from integer sample indices of cycle boundaries.

    Parameters
    ----------
    tau_idx : array-like of int
        Sample indices of successive cycle boundaries, length ``K + 1``.
    sampling_rate_hz : float
        Sampling rate in Hz.  Used to convert indices to seconds.
    n_time : int
        Total number of samples in the underlying trajectory.
    source : str
        Free-form label (e.g. ``"periodogram_peaks"``, ``"seed_peaks"``).
    metadata : dict, optional
        Arbitrary metadata to attach to the epochs.

    Returns
    -------
    CycleEpochs
        Epochs with ``phase = None`` and ``phase_in_cycle = None``.

    Notes
    -----
    Samples before ``tau_idx[0]`` or at/after ``tau_idx[-1]`` (translated to
    seconds and compared to ``time``) get ``cycle_index = -1``.  Boundary
    indices must be strictly increasing.
    """
    tau_idx = np.asarray(tau_idx, dtype=int)
    if tau_idx.ndim != 1:
        raise ValueError("tau_idx must be 1-D.")
    if len(tau_idx) < 2:
        raise ValueError("Need at least 2 boundary indices (1 cycle).")
    if np.any(np.diff(tau_idx) <= 0):
        raise ValueError("tau_idx must be strictly increasing.")

    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")
    if n_time <= 0:
        raise ValueError(f"n_time must be positive, got {n_time}.")

    tau = tau_idx.astype(float) / fs
    duration = np.diff(tau)

    time = np.arange(n_time) / fs
    K = len(duration)
    # For each sample, which cycle does it belong to?  searchsorted with
    # side="right" gives k where tau[k] <= t < tau[k+1] (offset by one).
    ci = np.searchsorted(tau, time, side="right") - 1
    # Mask samples outside [tau[0], tau[-1])
    outside = (ci < 0) | (ci >= K) | (time < tau[0]) | (time >= tau[-1])
    ci[outside] = -1

    md = dict(metadata) if metadata else {}
    md.setdefault("sampling_rate_hz", fs)

    return CycleEpochs(
        tau=tau,
        duration=duration,
        cycle_index=ci.astype(int),
        phase=None,
        phase_in_cycle=None,
        time=time,
        source=source,
        metadata=md,
    )
