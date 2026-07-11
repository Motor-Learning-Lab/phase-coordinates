"""
Lightweight tests for the epoch-finder validation infrastructure
(docs/debug/scripts/validate_epoch_finders.py).

These test the *validation script's* helpers, not new package behavior:
synthetic generator shapes, the boundary-error metric, and that the
geometric-score path stays usable on clean data. No PyMC required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "docs" / "debug" / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from validate_epoch_finders import (  # noqa: E402
    SCENARIOS,
    boundary_errors_samples,
    run_geometric_score_method,
)


# ---------------------------------------------------------------------------
# 1. Synthetic generators return X, fs, true_tau with consistent shapes.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("make_dataset", SCENARIOS, ids=[s.__name__ for s in SCENARIOS])
def test_synthetic_generator_shapes(make_dataset):
    dataset = make_dataset()

    assert dataset.X.ndim == 2
    assert dataset.X.shape[1] == 3
    n_time = dataset.X.shape[0]
    assert np.all(np.isfinite(dataset.X))

    assert dataset.fs > 0
    assert dataset.true_period > 0
    assert isinstance(dataset.description, str) and len(dataset.description) > 0
    assert isinstance(dataset.name, str) and len(dataset.name) > 0

    assert dataset.true_tau.ndim == 1
    assert len(dataset.true_tau) >= 1
    assert np.all(np.isfinite(dataset.true_tau))
    if len(dataset.true_tau) >= 2:
        assert np.all(np.diff(dataset.true_tau) > 0), "true_tau must be strictly increasing"
    # Every true boundary must correspond to an actual recorded sample --
    # otherwise no empirical method could ever recover it (see
    # _true_boundaries docstring in the script).
    t_max = (n_time - 1) / dataset.fs
    assert dataset.true_tau[-1] <= t_max + 1e-9

    if dataset.true_phase is not None:
        assert dataset.true_phase.shape == (n_time,)
        assert np.all(np.isfinite(dataset.true_phase))
        assert np.all(np.diff(dataset.true_phase) >= 0), "true_phase must be non-decreasing"


# ---------------------------------------------------------------------------
# 2. Boundary error computation is correct on exact known epochs.
# ---------------------------------------------------------------------------

def test_boundary_errors_samples_exact_match():
    fs = 100.0
    true_tau = np.array([0.0, 1.0, 2.0, 3.0])
    pred_tau = np.array([0.0, 1.0, 2.0, 3.0])
    errors = boundary_errors_samples(true_tau, pred_tau, fs)
    np.testing.assert_allclose(errors, 0.0)


def test_boundary_errors_samples_known_offset():
    fs = 100.0
    true_tau = np.array([0.0, 1.0, 2.0])
    # Every predicted boundary is 0.05 s (5 samples) late.
    pred_tau = np.array([0.05, 1.05, 2.05])
    errors = boundary_errors_samples(true_tau, pred_tau, fs)
    np.testing.assert_allclose(errors, [5.0, 5.0, 5.0])


def test_boundary_errors_samples_nearest_neighbor_picks_closer_side():
    fs = 100.0
    true_tau = np.array([1.0])
    # 1.0 is 0.3 s after 0.7 and 0.6 s before 1.6 -- nearest is 0.7.
    pred_tau = np.array([0.7, 1.6])
    errors = boundary_errors_samples(true_tau, pred_tau, fs)
    np.testing.assert_allclose(errors, [30.0])


def test_boundary_errors_samples_empty_inputs():
    fs = 100.0
    assert len(boundary_errors_samples(np.array([]), np.array([1.0]), fs)) == 0
    assert len(boundary_errors_samples(np.array([1.0]), np.array([]), fs)) == 0


# ---------------------------------------------------------------------------
# 3. Geometric epoch finder recovers approximately 1 s period on clean
#    complete cycles.
# ---------------------------------------------------------------------------

def test_geometric_score_recovers_1s_period_on_clean_cycles():
    make_clean = next(s for s in SCENARIOS if s.__name__ == "make_clean_complete_cycles")
    dataset = make_clean()
    epochs, extra, table = run_geometric_score_method(dataset.X, dataset.fs)
    assert abs(extra["period_estimate"] - dataset.true_period) < 0.05
    assert epochs.n_cycles >= 2


# ---------------------------------------------------------------------------
# 4. Candidate table contains the source/coverage columns the validation
#    script and report depend on.
# ---------------------------------------------------------------------------

def test_candidate_table_has_diagnostic_columns():
    make_clean = next(s for s in SCENARIOS if s.__name__ == "make_clean_complete_cycles")
    dataset = make_clean()
    _, extra, table = run_geometric_score_method(dataset.X, dataset.fs)
    required = {
        "period", "offset", "n_cycles", "total_score",
        "coverage_duration_fraction", "fraction_samples_assigned",
        "candidate_source",
    }
    assert required <= set(table.columns)
    assert table["candidate_source"].notna().all()
