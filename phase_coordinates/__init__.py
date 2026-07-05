"""
phase_coordinates
=================

Cycle-by-cycle PCA phase coordinates for multivariate cyclic motion.

Public API
----------
hilbert_phase
    Estimate instantaneous phase from a scalar reference signal via the
    Hilbert transform.
cycle_by_cycle_pca_coordinates
    Compute cycle-by-cycle PCA geometric coordinates (phase, radius,
    perpendicular deviation) for multivariate movement data.
"""

from .core import hilbert_phase, cycle_by_cycle_pca_coordinates

__all__ = ["hilbert_phase", "cycle_by_cycle_pca_coordinates"]
