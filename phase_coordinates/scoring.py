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

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .epochs import CycleEpochs, epochs_from_boundary_indices
from .geometry import interp_X_at_times, oriented_frame_from_anchors


DEFAULT_SCORE_WEIGHTS = {
    "planarity": 0.5,
    "quarter_anchor_orth_ratio": 0.3,
    "anchor_norm": 0.2,
}


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
    if X_arr.shape[1] < 3:
        raise ValueError("Need at least 3 features for planarity scoring.")
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
    K = len(duration)
    if K == 0:
        ci = np.full(n_time, -1, dtype=int)
    else:
        ci = np.searchsorted(tau, time, side="right") - 1
        outside = (ci < 0) | (ci >= K) | (time < tau[0]) | (time >= tau[-1])
        ci[outside] = -1

    md = dict(metadata) if metadata else {}
    md.setdefault("period", period)
    md.setdefault("offset", offset)
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


def score_epoch_geometry(
    X,
    epochs: CycleEpochs,
    *,
    sampling_rate_hz: float,
    columns: Optional[list] = None,
    weights: Optional[dict] = None,
) -> dict:
    """
    Score how "cycle-like" the geometry of each epoch is.

    For each cycle we compute:

    - **planarity** = ``1 - PC3_var / total_var`` from a per-cycle 3-component
      PCA.  ``1`` means the cycle lives in a plane; ``0`` means it spreads
      isotropically.
    - **anchor_norm** = ``||x0 - center||``, where ``x0`` is the trajectory
      interpolated at the cycle start.  Larger means the phase-zero anchor
      sits well away from the cycle center (as expected for a near-circular
      cycle of finite radius).
    - **quarter_anchor_orth_norm** = ``||a90_orth||`` and
      **quarter_anchor_orth_ratio** = ``||a90_orth|| / ||a90||``, where
      ``a90_orth`` is the component of the quarter-cycle anchor perpendicular
      to ``a0``.  Closer to 1 means the quarter-cycle anchor is truly
      "sideways" from the phase-zero anchor.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n_time, n_features)
        Multivariate trajectory.  Requires at least 3 features.
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

    Returns
    -------
    dict
        With keys ``total_score``, ``planarity``, ``anchor_norm``,
        ``quarter_anchor_orth_norm``, ``quarter_anchor_orth_ratio``,
        ``min_samples_per_cycle``, ``n_cycles``, ``fraction_samples_assigned``
        (fraction of input samples with ``cycle_index >= 0``),
        ``coverage_duration_fraction`` (``(tau[-1] - tau[0]) /
        (n_time / fs)``), and ``per_cycle`` (list of dicts, one per
        cycle).  The coverage metrics are report-only: they are not folded
        into ``total_score``, so two candidates with identical planarity and
        anchor geometry but very different coverage (e.g. 30% vs 90% of the
        recording assigned to cycles) score identically on ``total_score``
        alone — inspect the coverage columns separately.
    """
    X_arr = _resolve_X_and_columns(X, columns)
    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")
    n_time = X_arr.shape[0]

    w = dict(DEFAULT_SCORE_WEIGHTS)
    if weights:
        w.update(weights)
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError("Score weights must sum to a positive value.")
    w = {k: v / total_w for k, v in w.items()}

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
            "per_cycle": [],
        }

    tau = epochs.tau
    duration = epochs.duration

    x0_arr = interp_X_at_times(X_arr[:, :3], fs, tau[:-1])
    x90_arr = interp_X_at_times(X_arr[:, :3], fs, tau[:-1] + 0.25 * duration)

    per_cycle = []
    planarities = []
    anchor_norms = []
    q_orth_norms = []
    q_orth_ratios = []
    n_samples_per = []

    for k in range(K):
        idx = np.where(epochs.cycle_index == k)[0]
        n_k = len(idx)
        n_samples_per.append(n_k)

        # Per-cycle PCA + planarity
        if n_k >= 3:
            X_k = X_arr[idx]
            center_k = X_k.mean(axis=0)
            pca = PCA(n_components=min(3, X_k.shape[1]))
            pca.fit(X_k - center_k)
            evr = pca.explained_variance_ratio_
            if len(evr) < 3:
                evr = np.concatenate([evr, np.zeros(3 - len(evr))])
            planarity_k = float(1.0 - evr[2])
        else:
            X_k = X_arr[idx] if n_k > 0 else np.empty((0, X_arr.shape[1]))
            center_k = X_k.mean(axis=0) if n_k > 0 else np.zeros(X_arr.shape[1])
            planarity_k = float("nan")

        # Anchors (in the first 3 dimensions)
        c3 = center_k[:3] if len(center_k) >= 3 else np.pad(center_k, (0, 3 - len(center_k)))
        a0 = x0_arr[k] - c3
        a90 = x90_arr[k] - c3
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
        })
        planarities.append(planarity_k)
        anchor_norms.append(anchor_norm_k)
        q_orth_norms.append(q_orth_norm)
        q_orth_ratios.append(q_orth_ratio)

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
    med = np.median(X_arr[:, :3], axis=0)
    dists = np.linalg.norm(X_arr[:, :3] - med, axis=1)
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
        "per_cycle": per_cycle,
    }


def find_epochs_by_geometric_score(
    X,
    sampling_rate_hz: float,
    *,
    period_candidates: list,
    n_phase_offsets: int = 64,
    columns: Optional[list] = None,
    weights: Optional[dict] = None,
):
    """
    Search (period, offset) pairs for the epochs with the best geometric score.

    Parameters
    ----------
    X : array-like or DataFrame, shape (n_time, n_features)
        Multivariate trajectory.
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

    Returns
    -------
    best_epochs : CycleEpochs
        The winning epochs, tagged with ``source="geometric_score"`` and
        with the winning period/offset in ``metadata``.
    candidate_table : pandas.DataFrame
        One row per scored (period, offset) pair.  Columns: ``period,
        offset, n_cycles, total_score, planarity,
        quarter_anchor_orth_ratio, anchor_norm, fraction_samples_assigned,
        min_samples_per_cycle, coverage_duration_fraction``.  The coverage
        columns are report-only and are not folded into ``total_score``.
    """
    X_arr = _resolve_X_and_columns(X, columns)
    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")
    if not period_candidates:
        raise ValueError("period_candidates is empty.")
    n_time = X_arr.shape[0]

    rows = []
    best_score = float("-inf")
    best_epochs: Optional[CycleEpochs] = None
    best_meta = None

    for cand in period_candidates:
        period = float(cand.period)
        if period <= 0:
            continue
        offsets = np.linspace(0.0, period, n_phase_offsets, endpoint=False)
        for offset in offsets:
            epochs = candidate_epochs_from_period_offset(
                period, float(offset),
                sampling_rate_hz=fs, n_time=n_time,
                source="geometric_score_candidate",
                metadata={"candidate_source": cand.source, "candidate_score": cand.score},
            )
            if epochs.n_cycles < 2:
                continue
            score = score_epoch_geometry(
                X_arr, epochs, sampling_rate_hz=fs, columns=None, weights=weights,
            )
            rows.append({
                "period": period,
                "offset": float(offset),
                "n_cycles": score["n_cycles"],
                "total_score": score["total_score"],
                "planarity": score["planarity"],
                "quarter_anchor_orth_ratio": score["quarter_anchor_orth_ratio"],
                "anchor_norm": score["anchor_norm"],
                "fraction_samples_assigned": score["fraction_samples_assigned"],
                "min_samples_per_cycle": score["min_samples_per_cycle"],
                "coverage_duration_fraction": score["coverage_duration_fraction"],
            })
            if score["total_score"] > best_score:
                best_score = score["total_score"]
                best_epochs = epochs
                best_meta = {
                    "period": period,
                    "offset": float(offset),
                    "total_score": score["total_score"],
                    "candidate_source": cand.source,
                }

    table = pd.DataFrame(
        rows,
        columns=[
            "period", "offset", "n_cycles", "total_score",
            "planarity", "quarter_anchor_orth_ratio", "anchor_norm",
            "fraction_samples_assigned", "min_samples_per_cycle",
            "coverage_duration_fraction",
        ],
    )

    if best_epochs is None:
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
