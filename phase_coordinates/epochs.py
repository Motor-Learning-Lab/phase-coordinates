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
        Unwrapped phase in radians, zero-referenced so that ``phase[0] == 0``,
        when known, otherwise ``None``.  ``phase_in_cycle`` is
        ``mod(phase, 2*pi)`` — the two fields share the same reference so
        that ``phase // (2*pi) == cycle_index`` holds wherever
        ``cycle_index >= 0``.
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

    def __post_init__(self):
        """Validate the invariants documented above.

        ``tau`` may have length 1 (zero cycles) — this represents a
        candidate window in which no complete cycle fits, which
        :func:`candidate_epochs_from_period_offset` legitimately produces
        during a period/offset search.
        """
        tau = np.asarray(self.tau)
        if tau.ndim != 1:
            raise ValueError(f"tau must be 1-D, got shape {tau.shape}.")
        if len(tau) < 1:
            raise ValueError("tau must have at least 1 element.")
        if not np.all(np.isfinite(tau)):
            raise ValueError("tau must be finite.")
        if len(tau) >= 2 and np.any(np.diff(tau) <= 0):
            raise ValueError("tau must be strictly increasing.")

        duration = np.asarray(self.duration)
        if not np.all(np.isfinite(duration)):
            raise ValueError("duration must be finite.")
        expected_duration = np.diff(tau)
        if duration.shape != expected_duration.shape:
            raise ValueError(
                f"duration has shape {duration.shape}, expected "
                f"{expected_duration.shape} (= diff(tau))."
            )
        if not np.allclose(duration, expected_duration, rtol=1e-8, atol=1e-10):
            raise ValueError("duration must equal diff(tau).")

        cycle_index = np.asarray(self.cycle_index)
        time = np.asarray(self.time)
        if not np.all(np.isfinite(time)):
            raise ValueError("time must be finite.")
        if cycle_index.shape != time.shape:
            raise ValueError(
                f"cycle_index has shape {cycle_index.shape} but time has "
                f"shape {time.shape}; they must match."
            )
        K = len(tau) - 1
        valid_cycle_index = (cycle_index == -1) | ((cycle_index >= 0) & (cycle_index < K))
        if not np.all(valid_cycle_index):
            bad = np.unique(cycle_index[~valid_cycle_index])
            raise ValueError(
                f"cycle_index must be in {{-1, 0, ..., {K - 1}}} (K={K} "
                f"cycles); got out-of-range value(s) {bad[:5].tolist()}."
            )

        if self.phase is not None:
            phase = np.asarray(self.phase)
            if phase.shape != time.shape:
                raise ValueError(
                    f"phase has shape {phase.shape} but time has shape "
                    f"{time.shape}; they must match."
                )
            if not np.all(np.isfinite(phase)):
                raise ValueError("phase must be finite.")

        if self.phase_in_cycle is not None:
            phase_in_cycle = np.asarray(self.phase_in_cycle)
            if phase_in_cycle.shape != time.shape:
                raise ValueError(
                    f"phase_in_cycle has shape {phase_in_cycle.shape} but "
                    f"time has shape {time.shape}; they must match."
                )
            if not np.all(np.isfinite(phase_in_cycle)):
                raise ValueError("phase_in_cycle must be finite.")

    @property
    def n_cycles(self) -> int:
        """Number of cycles ``K``."""
        return int(len(self.tau) - 1)

    def _compute_sample_bounds(self) -> "tuple[np.ndarray, np.ndarray]":
        """
        Single-pass, vectorized computation of ``(sample_start, sample_stop)``
        shared by both properties below, so retrieving either is
        ``O(N + K)`` rather than the ``O(K*N)`` of one full
        ``cycle_index`` scan per cycle.

        Exploits an invariant of every :class:`CycleEpochs` constructor:
        samples are time-ordered and ``tau`` is sorted, so a fixed set of
        half-open ``[tau[k], tau[k+1))`` intervals can only ever assign a
        *monotonic non-decreasing* sequence of cycle indices -- i.e.
        ``cycle_index`` always looks like ``[-1]*p + [0, 0, .., 1, 1, ..,
        K-1] + [-1]*s`` for some prefix/suffix length ``p``/``s`` (either
        may be zero). That means cycle boundaries within the valid ``0..K-1``
        region can be found with one ``np.diff`` pass instead of one
        ``np.where`` scan per cycle. A cycle can still be genuinely empty
        (no sample at all, e.g. a very short candidate cycle) anywhere in
        that region, including between two non-empty ones, not just at the
        edges -- handled the same way as before, via ``searchsorted`` on
        ``tau[k]``.
        """
        K = self.n_cycles
        starts = np.empty(K, dtype=int)
        stops = np.empty(K, dtype=int)

        valid_mask = self.cycle_index >= 0
        if np.any(valid_mask):
            first_valid = int(np.argmax(valid_mask))
            last_valid = len(valid_mask) - 1 - int(np.argmax(valid_mask[::-1]))
            valid_ci = self.cycle_index[first_valid : last_valid + 1]

            change_points = np.flatnonzero(np.diff(valid_ci)) + 1
            block_start_offsets = np.concatenate([[0], change_points])
            block_stop_offsets = np.concatenate([change_points, [len(valid_ci)]])
            block_values = valid_ci[block_start_offsets]

            starts.fill(-1)
            starts[block_values] = first_valid + block_start_offsets
            stops[block_values] = first_valid + block_stop_offsets
        else:
            starts.fill(-1)

        empty = starts < 0
        if np.any(empty):
            # Empty cycle: point at the smallest sample_index at-or-after
            # tau[k]. We only need this as a placeholder; length is zero
            # because sample_stop = sample_start.
            empty_k = np.flatnonzero(empty)
            after = np.searchsorted(self.time, self.tau[empty_k], side="left")
            after = np.minimum(after, len(self.time))
            starts[empty_k] = after
            stops[empty_k] = after

        return starts, stops

    @property
    def sample_start(self) -> np.ndarray:
        """First sample index in each cycle, shape ``(K,)``.

        For cycles that contain at least one sample, this is the index of
        that first sample.  For empty cycles (no sample falls inside),
        ``sample_start == sample_stop`` and both equal the smallest sample
        index at-or-after ``tau[k]``.

        Call once and reuse the result rather than indexing this property
        inside a per-cycle loop -- each access recomputes the full ``(K,)``
        array (``O(N + K)``; still cheap for one call, but ``O(K)`` calls in
        a loop would make it ``O(K*(N+K))`` again).
        """
        starts, _ = self._compute_sample_bounds()
        return starts

    @property
    def sample_stop(self) -> np.ndarray:
        """One past the last sample index in each cycle (Python slice), shape ``(K,)``.

        Call once and reuse the result -- see :attr:`sample_start`.
        """
        _, stops = self._compute_sample_bounds()
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


def cycle_index_from_tau(tau: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Assign each sample time to a half-open cycle interval
    ``[tau[k], tau[k+1))``, given monotonically increasing boundary times
    ``tau`` (shape ``(K+1,)``).

    This is the single shared implementation of the package-wide half-open
    boundary convention used by :func:`epochs_from_boundary_indices`,
    :func:`~phase_coordinates.scoring.candidate_epochs_from_period_offset`,
    and the Bayesian cycle-membership assignment in
    :func:`~phase_coordinates.bayesian.fit_bayesian_phase_coordinates` --
    factored out so all three derive cycle membership the same way instead
    of maintaining parallel (and easily subtly-inconsistent) copies of the
    same ``searchsorted`` logic.

    Parameters
    ----------
    tau : ndarray, shape (K + 1,)
        Strictly increasing boundary times in seconds. ``K = len(tau) - 1``
        may be 0 (no cycles), in which case every sample gets ``-1``.
    time : ndarray, shape (n_time,)
        Sample times in seconds.

    Returns
    -------
    ndarray, shape (n_time,), int
        Cycle index per sample, in ``{-1, 0, ..., K-1}``. ``-1`` for samples
        before ``tau[0]`` or at/after ``tau[-1]``.
    """
    tau = np.asarray(tau, dtype=float)
    time = np.asarray(time, dtype=float)
    K = len(tau) - 1
    if K <= 0:
        return np.full(time.shape, -1, dtype=int)
    ci = np.searchsorted(tau, time, side="right") - 1
    outside = (ci < 0) | (ci >= K) | (time < tau[0]) | (time >= tau[-1])
    ci[outside] = -1
    return ci.astype(int)


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
        Epochs with ``source="phase"``, ``phase = phase - phase[0]``
        (zero-referenced, so ``phase[0] == 0``), and ``phase_in_cycle``
        filled in.

    Notes
    -----
    Boundary times ``tau[k]`` are the times where ``phase - phase[0]`` crosses
    ``k * 2*pi``, computed by linear interpolation between adjacent samples.
    This function does *not* call :func:`hilbert_phase` — callers who need
    that must run it first.

    Raises
    ------
    ValueError
        If ``phase`` is not non-decreasing.  ``identify_cycles_from_phase``
        locates cycle boundaries with ``searchsorted``, which silently
        produces wrong results on a non-monotone signal; a reversal usually
        means the reference signal or frequency band used to compute
        ``phase`` does not define a reliable instantaneous phase.
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

    reversals = np.diff(phase0) < 0
    if np.any(reversals):
        first_idx = int(np.argmax(reversals)) + 1
        raise ValueError(
            f"phase is not monotonically non-decreasing: {int(reversals.sum())} "
            f"reversal(s) found, first at sample index {first_idx} "
            f"(phase drop of {phase0[first_idx - 1] - phase0[first_idx]:.4g} rad). "
            "identify_cycles_from_phase requires a monotone unwrapped phase; "
            "this usually means the reference signal or frequency band used "
            "to compute the phase does not define a reliable instantaneous "
            "phase (e.g. low SNR)."
        )

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
        phase=phase0,
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
    Cycles are half-open sample intervals ``[tau_k, tau_{k+1})``.  Samples
    before ``tau_idx[0]`` or at/after ``tau_idx[-1]`` (translated to seconds
    and compared to ``time``) get ``cycle_index = -1``.  ``tau_idx[-1]`` may
    equal ``n_time`` (one past the last valid sample index) to close the
    final cycle deterministically without requiring a real sample there;
    see :func:`~phase_coordinates.scoring.candidate_epochs_from_period_offset`
    for the same convention applied to regularly-spaced candidates.
    Boundary indices must be strictly increasing.
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
    ci = cycle_index_from_tau(tau, time)

    md = dict(metadata) if metadata else {}
    md.setdefault("sampling_rate_hz", fs)

    return CycleEpochs(
        tau=tau,
        duration=duration,
        cycle_index=ci,
        phase=None,
        phase_in_cycle=None,
        time=time,
        source=source,
        metadata=md,
    )
