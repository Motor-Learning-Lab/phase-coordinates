"""
Geometric-score cycle identification.

Given period candidates from :mod:`phase_coordinates.period_search`, this
module searches phase offsets for the (period, offset) that produces cycles
whose per-cycle geometry (planarity, anchor placement) is most consistent
with a near-circular motion in a locally near-planar embedding.

The output is always a :class:`~phase_coordinates.epochs.CycleEpochs` — the
same contract every downstream stage consumes.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .epochs import CycleEpochs, cycle_index_from_tau, epochs_from_boundary_indices
from .geometry import AnchorOutOfBoundsError, interp_X_at_times, oriented_frame_from_anchors


DEFAULT_SCORE_WEIGHTS = {
    "planarity": 0.5,
    "quarter_anchor_orth_ratio": 0.3,
    "anchor_norm": 0.2,
}

# Default "approximately one revolution" range for the winding validity
# check (see score_epoch_geometry / find_epochs_by_geometric_score). A
# starting point per the task that introduced it, sensitivity-checked
# against the validation suite in docs/debug/epoch_finder_validation_report.md
# rather than assumed correct.
DEFAULT_WINDING_VALID_MIN = 0.75
DEFAULT_WINDING_VALID_MAX = 1.25

# A sample is excluded from the winding calculation (and any transition
# touching it) if its in-plane radius from the cycle center is below this
# fraction of the cycle's own median radius -- close to the center, the
# angle atan2(v, u) is dominated by noise, not genuine rotation. This is a
# numerical-stability detail of *computing* the diagnostic, not a
# user-facing validity setting (see DEFAULT_WINDING_VALID_MIN/MAX above for
# that), so it stays a module constant rather than a keyword argument.
_WINDING_DEGENERATE_RADIUS_FRAC = 0.05

# If fewer than this fraction of a cycle's consecutive-sample transitions
# have a well-defined angle (both endpoints outside the degenerate-radius
# zone), the cycle's winding is left undefined (NaN) rather than computed
# from an unreliable minority of samples -- see _cycle_winding.
_WINDING_MIN_VALID_TRANSITION_FRACTION = 0.5


def _cycle_winding(X_k, center_k, pca, x_close=None):
    """
    Signed winding number (revolutions, in time order) of one cycle's
    samples, using the in-plane (PC1, PC2) coordinates of an *already
    fitted* per-cycle PCA (reused from :func:`score_epoch_geometry`'s
    planarity computation -- this only calls ``pca.transform``, an O(n)
    projection, not a new fit).

    For each consecutive pair of centered in-plane vectors (u_i, v_i) and
    (u_{i+1}, v_{i+1}), the signed angle between them is
    ``atan2(cross, dot)`` where ``cross = u_i*v_{i+1} - v_i*u_{i+1}`` and
    ``dot = u_i*u_{i+1} + v_i*v_{i+1}``; the winding number is the sum of
    these per-step angles divided by ``2*pi``.

    Samples whose in-plane radius is too small relative to the cycle's own
    median radius are excluded pairwise (their angle is unstable, not
    meaningful) -- see ``_WINDING_DEGENERATE_RADIUS_FRAC``. If that leaves
    too few well-defined transitions (``_WINDING_MIN_VALID_TRANSITION_FRACTION``),
    the winding is undefined (NaN): a cycle that is mostly degenerate gives
    no reliable evidence either way and must not be silently treated as a
    valid single lap.

    Parameters
    ----------
    x_close : ndarray, shape (D,), optional
        Trajectory position at the cycle's own end boundary (``tau[k+1]``),
        already interpolated by the caller. Summing only *consecutive
        recorded samples* systematically undercounts a genuine single lap:
        for ``n`` evenly-spaced samples spanning one clean revolution
        (first sample at the start boundary, last sample one slot short of
        the end boundary), that sum is ``(n-1)/n`` turns, not ``1`` -- a
        3-sample cycle would measure ``2/3 ~= 0.667``, below the default
        ``0.75`` validity floor, purely because the closing segment (last
        recorded sample -> the cycle's actual endpoint) was never included.
        When given, ``x_close`` is appended as one extra point after the
        last real sample, adding that closing transition. Pass ``None`` to
        omit (matches the old, undercounting behavior).

    Returns
    -------
    winding : float
        Signed winding number, or NaN if undefined.
    n_valid_transitions, n_total_transitions : int
        Diagnostic counts (e.g. for reporting how much of a cycle was
        usable). Includes the closing transition when ``x_close`` is given.
    """
    n_k = X_k.shape[0]
    n_total = max(0, n_k - 1)
    if n_k < 3:
        return float("nan"), 0, n_total

    scores = pca.transform(X_k - center_k)
    if x_close is not None:
        close_score = pca.transform((np.asarray(x_close, dtype=float) - center_k).reshape(1, -1))
        scores = np.vstack([scores, close_score])
        n_total += 1

    u, v = scores[:, 0], scores[:, 1]
    r = np.hypot(u, v)
    # Typical-radius scale from the real recorded samples only, so a
    # potentially-clamped x_close can't skew what counts as "degenerate".
    radius_scale = float(np.median(r[:n_k]))
    if not np.isfinite(radius_scale) or radius_scale < 1e-12:
        return float("nan"), 0, n_total

    valid_point = r >= _WINDING_DEGENERATE_RADIUS_FRAC * radius_scale
    valid_transition = valid_point[:-1] & valid_point[1:]
    n_valid = int(np.sum(valid_transition))

    if n_total == 0 or n_valid / n_total < _WINDING_MIN_VALID_TRANSITION_FRACTION:
        return float("nan"), n_valid, n_total

    cross = u[:-1] * v[1:] - v[:-1] * u[1:]
    dot = u[:-1] * u[1:] + v[:-1] * v[1:]
    delta_theta = np.arctan2(cross, dot)
    winding = float(np.sum(delta_theta[valid_transition]) / (2 * np.pi))
    return winding, n_valid, n_total


def _normalize_score_weights(weights: Optional[dict]) -> dict:
    """
    Merge ``weights`` with :data:`DEFAULT_SCORE_WEIGHTS` and rescale to sum
    to 1.

    Shared by :func:`score_epoch_geometry` (called once per candidate) and
    :func:`find_epochs_by_geometric_score` (called once upfront purely to
    validate -- see its docstring). Keeping one implementation means a bad
    ``weights`` argument is detected identically in both places, rather
    than risking a second, subtly different copy of the same rule.

    Raises
    ------
    ValueError
        If the merged weights don't sum to a positive value.
    """
    w = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        w.update(weights)
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError("Score weights must sum to a positive value.")
    return {k: v / total_w for k, v in w.items()}


def _resolve_X_and_columns(X, columns):
    if isinstance(X, pd.DataFrame):
        if columns is not None:
            X_arr = X[columns].to_numpy(dtype=float)
        else:
            X_arr = X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)
    if X_arr.ndim != 2:
        raise ValueError("X must have shape (n_time, n_features).")
    if X_arr.shape[1] != 3:
        # Geometric scoring requires exactly 3 dimensions: PCA planarity
        # would otherwise be computed over however many columns are given,
        # while anchor geometry only ever used the first 3 -- results would
        # silently depend on which columns happen to be first. Pass
        # columns=[...] to select exactly 3 out of a higher-dimensional
        # DataFrame if needed.
        raise ValueError(
            f"Geometric scoring requires exactly 3 features, got "
            f"{X_arr.shape[1]}. Select exactly 3 columns via columns=... "
            "if X has more (planarity and anchor geometry must agree on "
            "which 3 dimensions they're both computed from)."
        )
    return X_arr


def candidate_epochs_from_period_offset(
    period: float,
    offset: float,
    *,
    sampling_rate_hz: float,
    n_time: int,
    source: str = "geometric_score_candidate",
    metadata: Optional[dict] = None,
) -> CycleEpochs:
    """
    Build regularly-spaced :class:`CycleEpochs` from a (period, offset) pair.

    Boundary times are ``tau_k = offset + k * period`` for
    ``k = 0, 1, 2, ...`` while ``tau_k <= n_time / fs``.  Cycles are
    half-open sample intervals ``[tau_k, tau_{k+1})``, so the final boundary
    may land exactly one sample period past the last recorded sample
    (``n_time / fs``) without requiring data there — this matches
    :func:`~phase_coordinates.epochs.epochs_from_boundary_indices`, which
    accepts a closing boundary index of ``n_time``.  Only complete cycles
    fully contained in the observed window are kept — no clamping or
    extrapolation.  Anchor interpolation (:func:`score_epoch_geometry`) only
    ever queries times at ``tau[:-1]`` and interior quarter points, never at
    the closing boundary itself, so this relaxation never causes
    interpolation past the last real sample;
    :func:`~phase_coordinates.geometry.interp_X_at_times` still raises if a
    caller ever does query past the data window.

    Parameters
    ----------
    period : float
        Cycle period in seconds.  Must be positive.
    offset : float
        Time of the first boundary in seconds.  Must satisfy
        ``0 <= offset < period``.
    sampling_rate_hz : float
        Sampling rate in Hz.
    n_time : int
        Total number of samples in the underlying trajectory.
    source : str
        Free-form label for the :class:`CycleEpochs` ``source`` field.
    metadata : dict, optional
        Metadata to attach to the epochs.

    Returns
    -------
    CycleEpochs
        Epochs with ``phase = None`` and ``phase_in_cycle = None``.
    """
    period = float(period)
    offset = float(offset)
    fs = float(sampling_rate_hz)
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}.")
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")
    if n_time <= 0:
        raise ValueError(f"n_time must be positive, got {n_time}.")

    t_end = n_time / fs
    # tau_k = offset + k * period; keep k with tau_k <= t_end
    k_max = int(np.floor((t_end - offset) / period))
    if k_max < 1:
        # Not even one full cycle fits: return an epochs with zero cycles.
        tau = np.asarray([offset], dtype=float)
    else:
        tau = offset + np.arange(k_max + 1) * period
    duration = np.diff(tau)

    time = np.arange(n_time) / fs
    ci = cycle_index_from_tau(tau, time)

    md = dict(metadata) if metadata else {}
    md.setdefault("period", period)
    md.setdefault("offset", offset)
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


def score_epoch_geometry(
    X,
    epochs: CycleEpochs,
    *,
    sampling_rate_hz: float,
    columns: Optional[list] = None,
    weights: Optional[dict] = None,
    winding_valid_min: float = DEFAULT_WINDING_VALID_MIN,
    winding_valid_max: float = DEFAULT_WINDING_VALID_MAX,
) -> dict:
    """
    Score how "cycle-like" the geometry of each epoch is.

    For each cycle we compute:

    - **planarity** = ``PC1_var_ratio + PC2_var_ratio`` (the variance
      explained by the best-fit 2-D plane) from a per-cycle 3-component PCA
      over exactly 3 dimensions.  ``1`` means the cycle lives in a plane;
      ``0`` means it spreads isotropically.  Equivalent to ``1 - PC3_var /
      total_var`` only because ``X`` is fixed at exactly 3 dimensions here
      (see the dimensionality contract below) -- with more dimensions
      those would differ, since ``1 - PC3`` would then also count variance
      in PC4, PC5, ... that was never part of the fitted plane at all.
    - **anchor_norm** = ``||x0 - center||``, where ``x0`` is the trajectory
      interpolated at the cycle start.  Larger means the phase-zero anchor
      sits well away from the cycle center (as expected for a near-circular
      cycle of finite radius).
    - **quarter_anchor_orth_norm** = ``||a90_orth||`` and
      **quarter_anchor_orth_ratio** = ``||a90_orth|| / ||a90||``, where
      ``a90_orth`` is the component of the quarter-cycle anchor perpendicular
      to ``a0``.  Closer to 1 means the quarter-cycle anchor is truly
      "sideways" from the phase-zero anchor.
    - **winding** = signed revolutions completed by the cycle's samples in
      time order (see :func:`_cycle_winding`). None of ``planarity``,
      ``anchor_norm``, or ``quarter_anchor_orth_ratio`` can tell a single
      lap from an exact integer multiple of it (retracing the same planar
      loop looks identical on all three); winding is the diagnostic that
      can. It is *not* folded into ``total_score`` -- see
      :func:`find_epochs_by_geometric_score` for how it's used as a
      candidate validity filter instead.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n_time, 3)
        Multivariate trajectory.  Requires *exactly* 3 features -- pass
        ``columns=[...]`` to select 3 out of a higher-dimensional
        DataFrame.  Planarity (a per-cycle PCA over all of ``X``) and
        anchor geometry (interpolated directly from ``X``) must agree on
        which dimensions they're computed from; allowing more than 3 and
        silently using only the first 3 for anchors made results depend on
        column order.
    epochs : CycleEpochs
        Candidate cycle epochs.
    sampling_rate_hz : float
        Sampling rate in Hz (used for anchor interpolation).
    columns : list, optional
        Subset of columns to use when ``X`` is a DataFrame.
    weights : dict, optional
        Mapping used to combine the score components into ``total_score``.
        Missing keys fall back to :data:`DEFAULT_SCORE_WEIGHTS`.  Weights are
        rescaled to sum to 1.
    winding_valid_min, winding_valid_max : float
        Range of ``abs(winding)`` counted as "approximately one lap" when
        computing ``fraction_single_lap_cycles`` below. This function only
        *reports* the fraction; :func:`find_epochs_by_geometric_score` is
        where it becomes an accept/reject decision.

    Returns
    -------
    dict
        With keys ``total_score``, ``planarity``, ``anchor_norm``,
        ``quarter_anchor_orth_norm``, ``quarter_anchor_orth_ratio``,
        ``min_samples_per_cycle``, ``n_cycles``, ``fraction_samples_assigned``
        (fraction of input samples with ``cycle_index >= 0``),
        ``coverage_duration_fraction`` (``(tau[-1] - tau[0]) /
        (n_time / fs)``), ``winding_median_abs``, ``winding_min_abs``,
        ``winding_max_abs`` (per-cycle ``abs(winding)`` statistics, ``nan``
        if every cycle is winding-degenerate), ``fraction_single_lap_cycles``
        (fraction of *all* cycles -- including winding-degenerate ones,
        which never count as single-lap -- with ``abs(winding)`` in
        ``[winding_valid_min, winding_valid_max]``), and ``per_cycle`` (list
        of dicts, one per cycle, each including a signed ``winding``, and
        ``winding_n_valid_transitions`` / ``winding_n_total_transitions``).
        The coverage and winding metrics are report-only: they are not
        folded into ``total_score``, so two candidates with identical
        planarity and anchor geometry but very different coverage or
        winding score identically on ``total_score`` alone — inspect those
        columns separately.
    """
    X_arr = _resolve_X_and_columns(X, columns)
    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")
    n_time = X_arr.shape[0]

    w = _normalize_score_weights(weights)

    K = epochs.n_cycles
    if K == 0:
        return {
            "total_score": float("-inf"),
            "planarity": float("nan"),
            "anchor_norm": float("nan"),
            "quarter_anchor_orth_norm": float("nan"),
            "quarter_anchor_orth_ratio": float("nan"),
            "min_samples_per_cycle": 0,
            "n_cycles": 0,
            "fraction_samples_assigned": 0.0,
            "coverage_duration_fraction": 0.0,
            "winding_median_abs": float("nan"),
            "winding_min_abs": float("nan"),
            "winding_max_abs": float("nan"),
            "fraction_single_lap_cycles": 0.0,
            "per_cycle": [],
        }

    tau = epochs.tau
    duration = epochs.duration

    x0_arr = interp_X_at_times(X_arr, fs, tau[:-1])
    x90_arr = interp_X_at_times(X_arr, fs, tau[:-1] + 0.25 * duration)
    # Cycle end-boundary anchors, for closing the winding sum (see
    # _cycle_winding). bounds_error=False (unlike x0_arr/x90_arr above) is
    # deliberate here, not a loosening of the same check: every CycleEpochs
    # constructor in this package guarantees tau[-1] <= n_time/fs, at most
    # one sample period past the last real sample (the package-wide
    # half-open convention that lets a closing boundary land there without
    # requiring data at that exact instant -- see
    # candidate_epochs_from_period_offset). So the clamp this can trigger is
    # bounded to <= 1 sample period of linear extrapolation, not silent
    # extrapolation over an unbounded distance.
    x_end_arr = interp_X_at_times(X_arr, fs, tau[1:], bounds_error=False)

    per_cycle = []
    planarities = []
    anchor_norms = []
    q_orth_norms = []
    q_orth_ratios = []
    n_samples_per = []
    windings = []

    for k in range(K):
        idx = np.where(epochs.cycle_index == k)[0]
        n_k = len(idx)
        n_samples_per.append(n_k)

        # Per-cycle PCA + planarity. X_arr is always exactly 3 columns (see
        # _resolve_X_and_columns), so planarity and anchor geometry below
        # are always computed from the same 3 dimensions.
        if n_k >= 3:
            X_k = X_arr[idx]
            center_k = X_k.mean(axis=0)
            pca = PCA(n_components=3)
            pca.fit(X_k - center_k)
            evr = pca.explained_variance_ratio_
            # Variance explained by the best 2-D plane (PC1 + PC2), i.e. the
            # planar-loop fraction -- *not* "1 - PC3" (only equal to this
            # when evr sums to 1 across exactly 3 components, which it now
            # always does since X_arr is fixed at exactly 3 dimensions).
            planarity_k = float(evr[0] + evr[1])
            winding_k, n_valid_trans, n_total_trans = _cycle_winding(
                X_k, center_k, pca, x_close=x_end_arr[k],
            )
        else:
            X_k = X_arr[idx] if n_k > 0 else np.empty((0, 3))
            center_k = X_k.mean(axis=0) if n_k > 0 else np.zeros(3)
            planarity_k = float("nan")
            winding_k, n_valid_trans, n_total_trans = float("nan"), 0, max(0, n_k - 1)

        # Anchors
        a0 = x0_arr[k] - center_k
        a90 = x90_arr[k] - center_k
        anchor_norm_k = float(np.linalg.norm(a0))
        a90_norm = float(np.linalg.norm(a90))
        eps = 1e-12
        e1 = a0 / max(anchor_norm_k, eps)
        a90_orth = a90 - e1 * np.dot(a90, e1)
        q_orth_norm = float(np.linalg.norm(a90_orth))
        q_orth_ratio = float(q_orth_norm / max(a90_norm, eps))

        per_cycle.append({
            "cycle": k,
            "n_samples": int(n_k),
            "planarity": planarity_k,
            "anchor_norm": anchor_norm_k,
            "quarter_anchor_orth_norm": q_orth_norm,
            "quarter_anchor_orth_ratio": q_orth_ratio,
            "winding": winding_k,
            "winding_n_valid_transitions": n_valid_trans,
            "winding_n_total_transitions": n_total_trans,
        })
        planarities.append(planarity_k)
        anchor_norms.append(anchor_norm_k)
        q_orth_norms.append(q_orth_norm)
        q_orth_ratios.append(q_orth_ratio)
        windings.append(winding_k)

    def _nanmean(seq):
        arr = np.asarray(seq, dtype=float)
        if not np.any(np.isfinite(arr)):
            return float("nan")
        return float(np.nanmean(arr))

    planarity_mean = _nanmean(planarities)
    anchor_norm_mean = _nanmean(anchor_norms)
    q_orth_norm_mean = _nanmean(q_orth_norms)
    q_orth_ratio_mean = _nanmean(q_orth_ratios)

    # Combine into total_score.  anchor_norm is not naturally in [0, 1] — we
    # normalize by the movement scale (median distance from median), so that
    # a full-radius anchor scores ~1.
    med = np.median(X_arr, axis=0)
    dists = np.linalg.norm(X_arr - med, axis=1)
    R = float(np.median(dists))
    if not np.isfinite(R) or R < 1e-9:
        R = float(np.sqrt(np.mean(dists**2)))
    if not np.isfinite(R) or R < 1e-9:
        R = 1.0
    anchor_score = anchor_norm_mean / R if np.isfinite(anchor_norm_mean) else float("nan")
    anchor_score = float(np.clip(anchor_score, 0.0, 1.0)) if np.isfinite(anchor_score) else float("nan")

    components_for_total = {
        "planarity": planarity_mean,
        "quarter_anchor_orth_ratio": q_orth_ratio_mean,
        "anchor_norm": anchor_score,
    }
    if all(np.isfinite(v) for v in components_for_total.values()):
        total = sum(w.get(k, 0.0) * components_for_total[k] for k in components_for_total)
    else:
        total = float("-inf")

    fraction_samples_assigned = float(np.sum(epochs.cycle_index >= 0)) / n_time
    total_duration = n_time / fs
    coverage_duration_fraction = (
        float((tau[-1] - tau[0]) / total_duration) if total_duration > 0 else 0.0
    )

    windings_abs = np.abs(np.asarray(windings, dtype=float))
    if np.any(np.isfinite(windings_abs)):
        winding_median_abs = float(np.nanmedian(windings_abs))
        winding_min_abs = float(np.nanmin(windings_abs))
        winding_max_abs = float(np.nanmax(windings_abs))
    else:
        winding_median_abs = float("nan")
        winding_min_abs = float("nan")
        winding_max_abs = float("nan")
    # A winding-degenerate cycle (NaN) never counts as single-lap -- it's
    # excluded from the *numerator* here but not the denominator, so a
    # candidate cannot look more reliable than it is just by having more
    # cycles whose winding couldn't be computed at all.
    is_single_lap = (windings_abs >= winding_valid_min) & (windings_abs <= winding_valid_max)
    fraction_single_lap_cycles = float(np.sum(is_single_lap)) / K

    return {
        "total_score": float(total),
        "planarity": planarity_mean,
        "anchor_norm": anchor_norm_mean,
        "quarter_anchor_orth_norm": q_orth_norm_mean,
        "quarter_anchor_orth_ratio": q_orth_ratio_mean,
        "min_samples_per_cycle": int(min(n_samples_per)) if n_samples_per else 0,
        "n_cycles": K,
        "fraction_samples_assigned": fraction_samples_assigned,
        "coverage_duration_fraction": coverage_duration_fraction,
        "winding_median_abs": winding_median_abs,
        "winding_min_abs": winding_min_abs,
        "winding_max_abs": winding_max_abs,
        "fraction_single_lap_cycles": fraction_single_lap_cycles,
        "per_cycle": per_cycle,
    }


class _ScoredCandidate(NamedTuple):
    """One scored (period, offset) candidate, kept around only long enough
    to run selection -- see ``_select_best_candidate``."""
    n_cycles: int
    total_score: float
    period: float
    offset: float
    epochs: "CycleEpochs"
    meta: dict
    winding_valid: bool


def _select_best_candidate(
    candidate_pool: list, score_tolerance: float,
) -> Optional[_ScoredCandidate]:
    """
    Deterministic compound-ordering selection among already winding-valid
    (or, if ``require_winding_valid=False``, all scoreable) candidates.

    Priority, in order:

    1. Candidate must already be in ``candidate_pool`` (validity filtering
       -- winding validity, ``n_cycles >= 2``, scoreable at all -- happens
       in the caller, before this function ever sees a candidate).
    2. Candidate must fall within ``score_tolerance`` of the best
       ``total_score`` achieved by any candidate in the pool.
    3. Among those, prefer more complete cycles (``n_cycles``) -- see
       :func:`find_epochs_by_geometric_score`'s Notes for why this,
       specifically, is the tie-break ``score_tolerance`` exists to apply.
    4. Among equal cycle counts, prefer the higher ``total_score`` (not
       "whichever the search happened to visit first with that cycle
       count" -- the previous ``max(..., key=lambda c: c[0])`` picked
       Python's ``max`` first-occurrence winner on cycle count alone,
       which is input-order-dependent whenever two candidates tie on
       ``n_cycles`` but differ in score).
    5. If *still* tied (identical ``n_cycles`` and ``total_score``, e.g. a
       perfectly symmetric candidate shape with no true preferred phase
       offset), break the tie deterministically and reproducibly by
       preferring the smaller ``period``, then the smaller ``offset`` --
       arbitrary but fixed, so the result never depends on
       ``period_candidates``' input order or iteration order.

    Returns ``None`` if ``candidate_pool`` is empty.
    """
    if not candidate_pool:
        return None
    best_score = max(c.total_score for c in candidate_pool)
    threshold = best_score - score_tolerance
    qualified = [c for c in candidate_pool if c.total_score >= threshold]
    # qualified always contains at least the best-score candidate(s).
    return max(qualified, key=lambda c: (c.n_cycles, c.total_score, -c.period, -c.offset))


def find_epochs_by_geometric_score(
    X,
    sampling_rate_hz: float,
    *,
    period_candidates: list,
    n_phase_offsets: int = 64,
    columns: Optional[list] = None,
    weights: Optional[dict] = None,
    score_tolerance: float = 0.001,
    winding_valid_min: float = DEFAULT_WINDING_VALID_MIN,
    winding_valid_max: float = DEFAULT_WINDING_VALID_MAX,
    winding_min_fraction: float = 0.8,
    require_winding_valid: bool = True,
):
    """
    Search (period, offset) pairs for the epochs with the best geometric score.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n_time, 3)
        Multivariate trajectory.  Requires exactly 3 features -- see
        :func:`score_epoch_geometry`.
    sampling_rate_hz : float
        Sampling rate in Hz.
    period_candidates : list of :class:`PeriodCandidate`
        Periods to try (typically produced by
        :mod:`phase_coordinates.period_search`).
    n_phase_offsets : int
        Number of evenly-spaced offsets in ``[0, period)`` to try per period.
    columns : list, optional
        Subset of columns to use when ``X`` is a DataFrame.
    weights : dict, optional
        Weight overrides for :func:`score_epoch_geometry`.
    score_tolerance : float
        Candidate-*selection* tolerance, not a scoring change (see Notes).
        Absolute ``total_score`` tolerance below the best score achieved by
        any *winding-valid* candidate; among candidates within this
        tolerance of the best, selection uses a fully deterministic
        compound order (see :func:`_select_best_candidate`): most complete
        cycles first, then highest ``total_score``, then smaller period,
        then smaller offset -- so the result never depends on
        ``period_candidates``' input order, even when candidates are
        exactly tied.
    winding_valid_min, winding_valid_max : float
        Range of ``abs(winding)`` (see :func:`score_epoch_geometry`) counted
        as "approximately one lap" for a single cycle.
    winding_min_fraction : float
        A candidate is winding-valid if at least this fraction of *all* its
        cycles (winding-degenerate cycles included, and never counted as
        single-lap) have ``abs(winding)`` in
        ``[winding_valid_min, winding_valid_max]``. Chosen to tolerate an
        occasional noisy or degenerate cycle without being fooled by a
        systematic period error, which shows up in every cycle, not just a
        minority (see Notes).
    require_winding_valid : bool
        If ``True`` (default), only winding-valid candidates are eligible to
        win (see Notes for how this composes with ``score_tolerance``). If
        ``False``, the winding filter is skipped entirely and selection
        falls back to plain ``score_tolerance``-based selection over all
        scoreable candidates -- kept as an explicit opt-out for comparison,
        not because the filter is optional in normal use.

    Returns
    -------
    best_epochs : CycleEpochs
        The winning epochs, tagged with ``source="geometric_score"`` and
        with the winning period/offset in ``metadata``.
    candidate_table : pandas.DataFrame
        One row per scored (period, offset) pair -- including winding-invalid
        ones, so rejected candidates stay inspectable rather than
        disappearing from the report. Columns: ``period, offset, n_cycles,
        total_score, planarity, quarter_anchor_orth_ratio, anchor_norm,
        fraction_samples_assigned, min_samples_per_cycle,
        coverage_duration_fraction, candidate_source, winding_median_abs,
        winding_min_abs, winding_max_abs, fraction_single_lap_cycles,
        winding_valid``.  ``candidate_source`` is the originating
        :class:`~phase_coordinates.period_search.PeriodCandidate`'s
        ``source`` (e.g. ``"periodogram"``, ``"autocorrelation"``,
        ``"harmonic:periodogram"``) — a diagnostic/reporting field, not
        used in scoring.  ``winding_valid`` is the accept/reject decision
        actually used for selection (``fraction_single_lap_cycles >=
        winding_min_fraction``); the coverage and winding columns are
        otherwise report-only and are not folded into ``total_score``.

    Notes
    -----
    **The multi-lap / fractional-lap problem.** An exact integer multiple of
    the true period retraces the same physical loop multiple times per
    "cycle", which is geometrically indistinguishable from the true period
    itself (same planarity, same anchor placement) — so plain
    ``argmax(total_score)`` can pick a longer-period, fewer-cycle candidate
    over the true period by an arbitrarily thin, essentially arbitrary
    margin. A period that's a fraction of the true one has the complementary
    problem: each proposed cycle only covers part of a revolution. Neither
    is a defect in ``total_score`` as a measure of geometric quality
    (planarity and anchor placement genuinely don't distinguish these cases)
    — it's that nothing was checking whether a candidate cycle corresponds
    to *one* traversal of the loop at all.

    **Winding validity vs. score_tolerance: two separate questions.**
    ``winding_valid``/``winding_min_fraction`` answer "does this candidate
    represent approximately one traversal per cycle?" — a hard filter
    applied *before* selection, at the same tier as the existing
    ``epochs.n_cycles < 2`` exclusion. ``score_tolerance`` answers a
    different question: "among valid, geometrically near-tied candidates,
    how is the winner chosen?" It restricts attention to the near-tied top
    of the score distribution among *winding-valid* candidates only, then
    prefers more cycles within that band — resolving cases where two
    winding-valid candidates (e.g. different genuine phase offsets) are
    statistically tied on geometric quality. Both stay separate from
    ``total_score`` itself: it keeps measuring geometric quality, not an
    arbitrary mixture of unrelated objectives.

    **Why winding validity is a fraction-of-cycles rule, not "every cycle"
    or "any cycle".** Requiring *every* nondegenerate cycle to be in range
    is fragile to a single noisy cycle rejecting an otherwise-correct
    candidate. Requiring only *some* cycle to be in range would fail to
    reject genuine period errors, since those are systematic — they show up
    in every cycle of a candidate, not just one. A fraction threshold
    (default ``0.8``) is tolerant of the former and still rejects the
    latter: a true 2x/3x/0.5x period error produces a winding far from 1 in
    essentially *all* cycles, so any reasonable non-trivial fraction
    threshold catches it.

    **Why a raw-coverage filter was rejected in favor of both of the above.**
    A coarse coverage-first filter (e.g. "keep only candidates within X of
    the maximum ``n_cycles``") was considered for the multi-lap problem and
    rejected: an unrelated short-period, low-quality candidate can trivially
    achieve a very high ``n_cycles`` (dozens to hundreds, bounded only by
    :mod:`period_search`'s ``min_period`` floor) while scoring far worse, so
    gating on coverage first can *discard* the correct candidate in favor of
    a high-count spurious one. Winding validity targets the actual
    geometric signature of the problem instead (does this candidate retrace
    the same loop multiple times?), and ``score_tolerance`` gates on score
    first among already-valid candidates, so neither is fooled by a
    high-``n_cycles``, low-quality candidate the way a raw coverage filter
    would be.
    """
    # Validate every global argument once, up front, before the search loop
    # -- not because any individual call inside the loop wouldn't itself
    # eventually raise on a bad value, but because relying on that means a
    # bad argument (invalid weights, a nonsensical winding range) surfaces
    # only after that specific per-candidate try/except below decides
    # whether to propagate it, which is exactly the failure mode this
    # function used to have: a plain `except ValueError` around the
    # per-candidate score call caught *everything*, including config
    # errors, and reported them as "no candidate produced enough cycles" --
    # a real bug indistinguishable from an ordinary degenerate candidate.
    # Validating here means a bad global argument always raises immediately
    # and unambiguously, before any candidate is even attempted.
    X_arr = _resolve_X_and_columns(X, columns)
    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")
    if not period_candidates:
        raise ValueError("period_candidates is empty.")
    _normalize_score_weights(weights)  # raises if invalid; result unused here
    if winding_valid_min > winding_valid_max:
        raise ValueError(
            f"winding_valid_min ({winding_valid_min}) must not exceed "
            f"winding_valid_max ({winding_valid_max})."
        )
    if not (0.0 <= winding_min_fraction <= 1.0):
        raise ValueError(
            f"winding_min_fraction must be in [0, 1], got {winding_min_fraction}."
        )
    n_time = X_arr.shape[0]

    rows = []
    scored_candidates: list[_ScoredCandidate] = []

    for cand in period_candidates:
        period = float(cand.period)
        if period <= 0:
            continue
        offsets = np.linspace(0.0, period, n_phase_offsets, endpoint=False)
        for offset in offsets:
            offset = float(offset)
            epochs = candidate_epochs_from_period_offset(
                period, offset,
                sampling_rate_hz=fs, n_time=n_time,
                source="geometric_score_candidate",
                metadata={"candidate_source": cand.source, "candidate_score": cand.score},
            )
            if epochs.n_cycles < 2:
                continue
            try:
                score = score_epoch_geometry(
                    X_arr, epochs, sampling_rate_hz=fs, columns=None, weights=weights,
                    winding_valid_min=winding_valid_min, winding_valid_max=winding_valid_max,
                )
            except AnchorOutOfBoundsError:
                # A degenerate candidate (typically a harmonic-halved period
                # close to period_search's min_period floor) can produce a
                # cycle short enough that its quarter-cycle anchor time
                # falls outside the recorded data window --
                # interp_X_at_times(bounds_error=True) correctly refuses to
                # extrapolate there. Such a candidate simply cannot be
                # scored; skip it rather than let one bad candidate crash
                # the whole search. This does not change total_score for
                # any candidate that *can* be scored. Only this specific,
                # expected per-candidate failure is caught -- anything else
                # (bad weights, a real bug) propagates normally.
                continue
            winding_valid = score["fraction_single_lap_cycles"] >= winding_min_fraction
            rows.append({
                "period": period,
                "offset": offset,
                "n_cycles": score["n_cycles"],
                "total_score": score["total_score"],
                "planarity": score["planarity"],
                "quarter_anchor_orth_ratio": score["quarter_anchor_orth_ratio"],
                "anchor_norm": score["anchor_norm"],
                "fraction_samples_assigned": score["fraction_samples_assigned"],
                "min_samples_per_cycle": score["min_samples_per_cycle"],
                "coverage_duration_fraction": score["coverage_duration_fraction"],
                "candidate_source": cand.source,
                "winding_median_abs": score["winding_median_abs"],
                "winding_min_abs": score["winding_min_abs"],
                "winding_max_abs": score["winding_max_abs"],
                "fraction_single_lap_cycles": score["fraction_single_lap_cycles"],
                "winding_valid": winding_valid,
            })
            scored_candidates.append(_ScoredCandidate(
                n_cycles=score["n_cycles"],
                total_score=score["total_score"],
                period=period,
                offset=offset,
                epochs=epochs,
                meta={
                    "period": period,
                    "offset": offset,
                    "total_score": score["total_score"],
                    "candidate_source": cand.source,
                    "fraction_single_lap_cycles": score["fraction_single_lap_cycles"],
                },
                winding_valid=winding_valid,
            ))

    table = pd.DataFrame(
        rows,
        columns=[
            "period", "offset", "n_cycles", "total_score",
            "planarity", "quarter_anchor_orth_ratio", "anchor_norm",
            "fraction_samples_assigned", "min_samples_per_cycle",
            "coverage_duration_fraction", "candidate_source",
            "winding_median_abs", "winding_min_abs", "winding_max_abs",
            "fraction_single_lap_cycles", "winding_valid",
        ],
    )

    if require_winding_valid:
        candidate_pool = [c for c in scored_candidates if c.winding_valid]
    else:
        candidate_pool = scored_candidates

    winner = _select_best_candidate(candidate_pool, score_tolerance)
    best_epochs = winner.epochs if winner is not None else None
    best_meta = winner.meta if winner is not None else None

    if best_epochs is None:
        if scored_candidates:
            # Candidates existed and were scoreable, but none passed the
            # winding filter -- a different failure than "nothing scored at
            # all" below, so it gets its own message pointing at the actual
            # cause and the escape hatch.
            raise ValueError(
                "No (period, offset) candidate satisfied the winding "
                f"validity filter (fraction_single_lap_cycles >= "
                f"{winding_min_fraction} with abs(winding) in "
                f"[{winding_valid_min}, {winding_valid_max}]). All scored "
                "candidates appear to be multi-lap, fractional-lap, or too "
                "degenerate to evaluate reliably. Pass "
                "require_winding_valid=False to fall back to plain "
                "geometric-quality selection, or adjust the winding "
                "tolerance."
            )
        raise ValueError(
            "No (period, offset) candidate produced at least 2 complete cycles. "
            "Provide more/longer data or different period candidates."
        )

    # Retag the winning epochs and merge metadata.
    winner_meta = dict(best_epochs.metadata)
    winner_meta.update(best_meta or {})
    best_epochs = CycleEpochs(
        tau=best_epochs.tau,
        duration=best_epochs.duration,
        cycle_index=best_epochs.cycle_index,
        phase=None,
        phase_in_cycle=None,
        time=best_epochs.time,
        source="geometric_score",
        metadata=winner_meta,
    )

    return best_epochs, table
