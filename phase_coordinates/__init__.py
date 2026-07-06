"""
phase_coordinates
=================

Experimental tools for describing cyclic multivariate movement using phase,
radius, and perpendicular deviation.

Two peer algorithms share an identical output contract:

    samples, cycles, details = fit_pca_phase_coordinates(X, ...)
    samples, cycles, details = fit_bayesian_phase_coordinates(X, ...)

Public API
----------
hilbert_phase
    Estimate instantaneous phase from a scalar reference signal via the
    Hilbert transform.
fit_pca_phase_coordinates
    Cycle-by-cycle PCA phase coordinates (fast; phase supplied or Hilbert).
fit_bayesian_phase_coordinates
    Bayesian two-layer phase-coordinate estimator (slow; requires
    ``pip install -e .[bayes]``).
reconstruct_phase_coordinates
    Reconstruct 3-D trajectory from samples and cycles DataFrames.
SAMPLE_COLUMNS
    Column names for the shared samples DataFrame.
CYCLE_COLUMNS
    Column names for the shared cycles DataFrame.
"""

from .core import (
    hilbert_phase,
    fit_pca_phase_coordinates,
    reconstruct_phase_coordinates,
    SAMPLE_COLUMNS,
    CYCLE_COLUMNS,
)

from .bayesian import fit_bayesian_phase_coordinates

__all__ = [
    "hilbert_phase",
    "fit_pca_phase_coordinates",
    "fit_bayesian_phase_coordinates",
    "reconstruct_phase_coordinates",
    "SAMPLE_COLUMNS",
    "CYCLE_COLUMNS",
]
