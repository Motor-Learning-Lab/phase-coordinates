"""
phase_coordinates
=================

Composable four-stage pipeline for describing cyclic multivariate movement.

The pipeline separates four responsibilities:

1. **Phase estimation** — e.g. :func:`hilbert_phase` on a scalar reference
   signal.
2. **Cycle identification** — turn phase, boundary indices, or a
   geometric-score search into a :class:`CycleEpochs`.  Options:

   - :func:`identify_cycles_from_phase` (phase-based, primary path)
   - :func:`epochs_from_boundary_indices` (indices only)
   - :func:`candidate_epochs_from_period_offset` /
     :func:`find_epochs_by_geometric_score` (period-search based)

3. **Coordinate estimation** — fit per-cycle geometry given a
   :class:`CycleEpochs`.  Options:

   - :func:`fit_pca_phase_coordinates` (fast, deterministic)
   - :func:`fit_bayesian_phase_coordinates` (slow, uncertainty-aware)

4. **Diagnostics** — :func:`compute_cycle_quality` reports per-cycle
   planarity, anchor geometry, and orientation without filtering.

Two coordinate estimators share an identical output contract::

    samples, cycles, details = fit_pca_phase_coordinates(X, epochs=...)
    samples, cycles, details = fit_bayesian_phase_coordinates(X, sampling_rate_hz=...)
"""

from .epochs import (
    CycleEpochs,
    identify_cycles_from_phase,
    epochs_from_boundary_indices,
)
from .period_search import (
    PeriodCandidate,
    period_candidates_from_periodogram,
    period_candidates_from_autocorrelation,
    expand_period_harmonics,
)
from .scoring import (
    candidate_epochs_from_period_offset,
    score_epoch_geometry,
    find_epochs_by_geometric_score,
    DEFAULT_SCORE_WEIGHTS,
)
from .diagnostics import compute_cycle_quality
from .geometry import interp_X_at_times, oriented_frame_from_anchors
from .core import (
    hilbert_phase,
    fit_pca_phase_coordinates,
    reconstruct_phase_coordinates,
    SAMPLE_COLUMNS,
    CYCLE_COLUMNS,
)
from .bayesian import (
    fit_bayesian_phase_coordinates,
    dominant_reference_signal,
    estimate_dominant_period,
    seed_boundary_indices,
)

__all__ = [
    # Stage 1: phase
    "hilbert_phase",
    # Stage 2: cycle identification
    "CycleEpochs",
    "identify_cycles_from_phase",
    "epochs_from_boundary_indices",
    "PeriodCandidate",
    "period_candidates_from_periodogram",
    "period_candidates_from_autocorrelation",
    "expand_period_harmonics",
    "candidate_epochs_from_period_offset",
    "score_epoch_geometry",
    "find_epochs_by_geometric_score",
    "DEFAULT_SCORE_WEIGHTS",
    # Stage 3: coordinate estimation
    "fit_pca_phase_coordinates",
    "fit_bayesian_phase_coordinates",
    "reconstruct_phase_coordinates",
    # Stage 4: diagnostics
    "compute_cycle_quality",
    # Shared geometry helpers
    "interp_X_at_times",
    "oriented_frame_from_anchors",
    # Bayesian seed primitives
    "dominant_reference_signal",
    "estimate_dominant_period",
    "seed_boundary_indices",
    # Output schema
    "SAMPLE_COLUMNS",
    "CYCLE_COLUMNS",
]
