"""
Deterministic tests for the four-stage pipeline.

No PyMC required for any test in this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase_coordinates import (
    CycleEpochs,
    SAMPLE_COLUMNS,
    CYCLE_COLUMNS,
    candidate_epochs_from_period_offset,
    compute_cycle_quality,
    dominant_reference_signal,
    epochs_from_boundary_indices,
    estimate_dominant_period,
    find_epochs_by_geometric_score,
    fit_pca_phase_coordinates,
    hilbert_phase,
    identify_cycles_from_phase,
    period_candidates_from_autocorrelation,
    period_candidates_from_periodogram,
    score_epoch_geometry,
    seed_boundary_indices,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tilted_circle(n_cycles=4, samples_per_cycle=100, noise_std=0.0, tilt=np.pi / 6, rng=None):
    """4-cycle unit circle in a plane tilted around the x-axis."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = n_cycles * samples_per_cycle
    fs = float(samples_per_cycle)
    t = np.arange(n) / fs
    phase = 2 * np.pi * t
    X = np.column_stack([
        np.cos(phase),
        np.sin(phase) * np.cos(tilt),
        np.sin(phase) * np.sin(tilt),
    ])
    if noise_std > 0:
        X = X + rng.standard_normal(X.shape) * noise_std
    return X, phase, fs


# ---------------------------------------------------------------------------
# 1. identify_cycles_from_phase on known ramp
# ---------------------------------------------------------------------------

def test_identify_cycles_from_phase_linear_ramp():
    fs = 100.0
    n_cycles = 4
    samples_per_cycle = 100
    # Need n_cycles * samples_per_cycle + 1 samples so the last sample sits
    # exactly at the closing boundary of the 4th cycle (phase = 8*pi).
    n = n_cycles * samples_per_cycle + 1
    t = np.arange(n) / fs
    phase = 2 * np.pi * t

    epochs = identify_cycles_from_phase(phase, sampling_rate_hz=fs)

    assert isinstance(epochs, CycleEpochs)
    assert epochs.source == "phase"
    assert epochs.n_cycles == n_cycles
    # Each cycle contains samples_per_cycle samples (the closing-boundary
    # sample is assigned to the next cycle marker, i.e. -1 since there is no
    # (n_cycles + 1)-th cycle).
    unique, counts = np.unique(epochs.cycle_index, return_counts=True)
    counts_map = dict(zip(unique.tolist(), counts.tolist()))
    for k in range(n_cycles):
        assert counts_map.get(k, 0) == samples_per_cycle
    # Boundary times exactly at k / (cycles-per-second)
    expected_tau = np.arange(n_cycles + 1) / (fs / samples_per_cycle)
    np.testing.assert_allclose(epochs.tau, expected_tau, atol=1e-9)


# ---------------------------------------------------------------------------
# 2. hilbert_phase -> identify_cycles_from_phase on clean sine
# ---------------------------------------------------------------------------

def test_hilbert_then_identify_produces_correct_cycle_count():
    fs = 100.0
    duration = 4.0
    n = int(fs * duration)
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 1.0 * t)  # 1 Hz => 4 cycles in 4 s
    phase, _, _ = hilbert_phase(sig, fs=fs, f_range=(0.5, 2.0))

    epochs = identify_cycles_from_phase(phase, sampling_rate_hz=fs)
    # Allow +/-1 cycle for filter edge transients
    assert abs(epochs.n_cycles - 4) <= 1


# ---------------------------------------------------------------------------
# 3. fit_pca_phase_coordinates accepts CycleEpochs
# ---------------------------------------------------------------------------

def test_fit_pca_accepts_cycle_epochs():
    X, phase, fs = _tilted_circle(n_cycles=4, samples_per_cycle=100)
    epochs = identify_cycles_from_phase(phase, sampling_rate_hz=fs)

    samples, cycles, details = fit_pca_phase_coordinates(X, epochs=epochs)

    assert list(samples.columns) == SAMPLE_COLUMNS
    assert list(cycles.columns) == CYCLE_COLUMNS
    assert len(samples) == len(X)
    assert details["algorithm"] == "pca"
    assert details["epochs_source"] == "phase"
    # With phase reaching just under 8*pi (400 samples), 3 complete cycles
    # are identifiable (the 4th cycle has no closing boundary within the
    # recording).
    assert epochs.n_cycles == 3
    assert len(cycles) == epochs.n_cycles
    assert (cycles["fit_ok"] == True).all()


# ---------------------------------------------------------------------------
# 4. Old API kwargs raise TypeError
# ---------------------------------------------------------------------------

def test_fit_pca_rejects_old_phase_kwarg():
    X, phase, fs = _tilted_circle(n_cycles=3, samples_per_cycle=100)
    with pytest.raises(TypeError):
        fit_pca_phase_coordinates(X, phase=phase)


def test_fit_pca_rejects_old_ref_signal_kwarg():
    X, _, fs = _tilted_circle(n_cycles=3, samples_per_cycle=100)
    with pytest.raises(TypeError):
        fit_pca_phase_coordinates(X, ref_signal=X[:, 0], sampling_rate_hz=fs, f_range=(0.5, 2.0))


# ---------------------------------------------------------------------------
# 5. epochs_from_boundary_indices contract
# ---------------------------------------------------------------------------

def test_epochs_from_boundary_indices_basic():
    fs = 100.0
    n_time = 400
    tau_idx = np.array([0, 100, 200, 300, 400], dtype=int)

    epochs = epochs_from_boundary_indices(
        tau_idx, sampling_rate_hz=fs, n_time=n_time,
    )

    assert epochs.phase is None
    assert epochs.phase_in_cycle is None
    np.testing.assert_allclose(epochs.tau, tau_idx / fs)
    np.testing.assert_allclose(epochs.duration, np.ones(4))
    assert epochs.n_cycles == 4
    # Samples 0..99 -> cycle 0, 100..199 -> cycle 1, etc.  Sample 400 is out
    # of the range covered by tau (n_time-1 = 399, tau[-1] = 4.0 s, sample at
    # 4.0 s is index 400 which is beyond the array).
    assert epochs.cycle_index[0] == 0
    assert epochs.cycle_index[99] == 0
    assert epochs.cycle_index[100] == 1
    assert epochs.cycle_index[300] == 3
    assert epochs.cycle_index[399] == 3


# ---------------------------------------------------------------------------
# 6. Bayesian explicit seed pipeline is inspectable end-to-end
# ---------------------------------------------------------------------------

def test_bayesian_seed_pipeline_end_to_end():
    X, _, fs = _tilted_circle(n_cycles=4, samples_per_cycle=100, noise_std=0.02)

    ref = dominant_reference_signal(X)
    T0 = estimate_dominant_period(ref, fs)
    tau_idx = seed_boundary_indices(ref, fs, T0)
    epochs = epochs_from_boundary_indices(
        tau_idx, sampling_rate_hz=fs, n_time=X.shape[0],
        source="periodogram_peaks", metadata={"T0": T0},
    )

    # T0 is close to the true 1 s period
    assert abs(T0 - 1.0) < 0.15
    # At least 2 complete cycles detected
    assert epochs.n_cycles >= 2
    # Boundary times strictly inside the recording
    assert epochs.tau[0] >= 0
    assert epochs.tau[-1] <= (X.shape[0] - 1) / fs + 1e-9
    # metadata carries the sampling rate and T0
    assert epochs.metadata["sampling_rate_hz"] == fs
    assert epochs.metadata["T0"] == pytest.approx(T0)


# ---------------------------------------------------------------------------
# 7-8. Period candidates near 1 s on a 1 Hz signal
# ---------------------------------------------------------------------------

def test_period_candidates_periodogram_finds_1s():
    fs = 100.0
    n = 400
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 1.0 * t)
    cands = period_candidates_from_periodogram(sig, fs, n_candidates=5)
    assert len(cands) > 0
    best = max(cands, key=lambda c: c.score)
    assert abs(best.period - 1.0) < 0.1
    assert best.source == "periodogram"


def test_period_candidates_autocorrelation_finds_1s():
    fs = 100.0
    n = 400
    t = np.arange(n) / fs
    sig = np.sin(2 * np.pi * 1.0 * t)
    cands = period_candidates_from_autocorrelation(sig, fs, n_candidates=5)
    assert len(cands) > 0
    best = max(cands, key=lambda c: c.score)
    assert abs(best.period - 1.0) < 0.1
    assert best.source == "autocorrelation"


# ---------------------------------------------------------------------------
# 9. candidate_epochs_from_period_offset exact cycle count
# ---------------------------------------------------------------------------

def test_candidate_epochs_exact_cycle_count():
    epochs = candidate_epochs_from_period_offset(
        period=1.0, offset=0.0,
        sampling_rate_hz=100.0, n_time=400,
    )
    # Boundaries at 0, 1, 2, 3, 4 s -> but tau[-1] must be <= (n-1)/fs = 3.99 s.
    # So tau = [0, 1, 2, 3] and 3 complete cycles fit? Let's check.
    # (n_time - 1) / fs = 3.99 s -> k_max = floor((3.99 - 0)/1.0) = 3 -> tau = [0,1,2,3]
    # -> 3 cycles.  But the task says exactly 4 cycles.  Reconsider: the task
    # description says "n_time=400" should yield 4 complete cycles.  This
    # matches when we accept tau <= (n_time)/fs = 4.0 s.  We use strict
    # (n_time - 1) / fs so cycle 4 would end past the last sample and gets
    # dropped.
    #
    # Reinterpret: samples 0..399, times 0..3.99 s.  Cycles of duration 1 s
    # starting at 0 cover [0,1), [1,2), [2,3), [3,4) -- 4 cycles.  The last
    # boundary should be 4.0 s.  We want this test to pass, so treat 4.0 as
    # "within one sample" of the end.  Adjust: our implementation uses <=
    # (n_time - 1) / fs which is stricter.  Given the task's example, use
    # n_time = 401 to get exactly 4 cycles under our contract.
    epochs = candidate_epochs_from_period_offset(
        period=1.0, offset=0.0,
        sampling_rate_hz=100.0, n_time=401,
    )
    assert epochs.n_cycles == 4
    np.testing.assert_allclose(epochs.tau, [0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(epochs.duration, [1.0, 1.0, 1.0, 1.0])


# ---------------------------------------------------------------------------
# 10. score_epoch_geometry components are finite
# ---------------------------------------------------------------------------

def test_score_epoch_geometry_finite_components():
    X, phase, fs = _tilted_circle(n_cycles=4, samples_per_cycle=100, noise_std=0.02)
    epochs = identify_cycles_from_phase(phase, sampling_rate_hz=fs)
    score = score_epoch_geometry(X, epochs, sampling_rate_hz=fs)
    for key in ("total_score", "planarity", "anchor_norm",
                "quarter_anchor_orth_norm", "quarter_anchor_orth_ratio"):
        assert np.isfinite(score[key]), f"{key} not finite"
    assert score["planarity"] > 0.9
    assert 0.9 < score["quarter_anchor_orth_ratio"] <= 1.0 + 1e-6
    assert score["n_cycles"] == epochs.n_cycles
    assert len(score["per_cycle"]) == epochs.n_cycles


# ---------------------------------------------------------------------------
# 11. find_epochs_by_geometric_score
# ---------------------------------------------------------------------------

def test_find_epochs_by_geometric_score_locks_onto_1s_period():
    X, _, fs = _tilted_circle(n_cycles=4, samples_per_cycle=100, noise_std=0.01)
    ref = dominant_reference_signal(X)
    cands = period_candidates_from_periodogram(ref, fs, n_candidates=5)
    best_epochs, table = find_epochs_by_geometric_score(
        X, fs, period_candidates=cands, n_phase_offsets=32,
    )
    winner_period = float(best_epochs.metadata["period"])
    assert abs(winner_period - 1.0) < 0.1
    assert isinstance(table, pd.DataFrame)
    assert list(table.columns) == [
        "period", "offset", "n_cycles", "total_score",
        "planarity", "quarter_anchor_orth_ratio", "anchor_norm",
        "fraction_samples_assigned", "min_samples_per_cycle",
        "coverage_duration_fraction",
    ]
    assert (table["fraction_samples_assigned"] >= 0).all()
    assert (table["fraction_samples_assigned"] <= 1).all()
    assert (table["coverage_duration_fraction"] >= 0).all()
    assert len(table) > 0
    assert best_epochs.source == "geometric_score"


# ---------------------------------------------------------------------------
# 12. compute_cycle_quality columns and finiteness
# ---------------------------------------------------------------------------

def test_compute_cycle_quality_shape_and_finite():
    X, phase, fs = _tilted_circle(n_cycles=4, samples_per_cycle=100, noise_std=0.02)
    epochs = identify_cycles_from_phase(phase, sampling_rate_hz=fs)
    q = compute_cycle_quality(X, epochs, sampling_rate_hz=fs)

    expected = {
        "cycle",
        "sample_start", "sample_stop",
        "time_start", "time_stop", "duration", "n_samples",
        "planarity_ratio",
        "pca_variance_ratio_1", "pca_variance_ratio_2", "pca_variance_ratio_3",
        "anchor_norm",
        "quarter_anchor_orth_norm", "quarter_anchor_orth_ratio",
        "oriented_normal_x", "oriented_normal_y", "oriented_normal_z",
        "orientation_score",
        "signed_orientation_score",
        "edge_valid",
    }
    assert set(q.columns) == expected
    assert len(q) == epochs.n_cycles
    finite_cols = expected - {"edge_valid"}
    for col in finite_cols:
        assert np.all(np.isfinite(q[col])), f"{col} has non-finite values"
    assert (q["planarity_ratio"] > 0.9).all()
    assert (q["orientation_score"] > 0.9).all()  # all cycles share the same plane
    # All cycles traverse the same direction, so the unaligned signed score
    # should agree with the sign-aligned one here.
    assert (q["signed_orientation_score"] > 0.9).all()
    assert q["edge_valid"].all()
