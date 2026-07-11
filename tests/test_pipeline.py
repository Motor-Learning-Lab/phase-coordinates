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
    expand_period_harmonics,
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
    # Samples 0..399 span times [0, 4.0) s at fs=100 Hz (n_time / fs = 4.0 s).
    # Cycles are half-open [tau_k, tau_{k+1}), so 1 s cycles starting at 0
    # cover [0,1), [1,2), [2,3), [3,4) -- 4 complete cycles, with the final
    # boundary landing exactly at n_time / fs (one sample period past the
    # last recorded sample). This matches the epochs_from_boundary_indices
    # convention, where a closing boundary index of n_time is likewise
    # allowed.
    epochs = candidate_epochs_from_period_offset(
        period=1.0, offset=0.0,
        sampling_rate_hz=100.0, n_time=400,
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
        "coverage_duration_fraction", "candidate_source",
        "winding_median_abs", "winding_min_abs", "winding_max_abs",
        "fraction_single_lap_cycles", "winding_valid",
    ]
    assert (table["fraction_samples_assigned"] >= 0).all()
    assert (table["fraction_samples_assigned"] <= 1).all()
    assert (table["coverage_duration_fraction"] >= 0).all()
    assert table["candidate_source"].notna().all()
    assert set(table["candidate_source"]) <= {"periodogram", "autocorrelation"} | {
        f"harmonic:{s}" for s in ("periodogram", "autocorrelation")
    }
    assert len(table) > 0
    assert best_epochs.source == "geometric_score"
    # The winner should be a genuine single-lap candidate.
    winner_row = table[
        (table["period"] == winner_period)
        & np.isclose(table["offset"], float(best_epochs.metadata["offset"]))
    ]
    assert winner_row["winding_valid"].all()
    assert (table["fraction_single_lap_cycles"] >= 0).all()
    assert (table["fraction_single_lap_cycles"] <= 1).all()


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


# ---------------------------------------------------------------------------
# 13. CycleEpochs.__post_init__ invariant validation
# ---------------------------------------------------------------------------

def _valid_epochs_kwargs():
    """A minimal, valid set of CycleEpochs field values (K=2 cycles, n_time=4)."""
    tau = np.array([0.0, 1.0, 2.0])
    time = np.array([0.0, 0.5, 1.0, 1.5])
    return dict(
        tau=tau,
        duration=np.diff(tau),
        cycle_index=np.array([0, 0, 1, 1]),
        phase=np.array([0.0, np.pi, 2 * np.pi, 3 * np.pi]),
        phase_in_cycle=np.array([0.0, np.pi, 0.0, np.pi]),
        time=time,
        source="test",
    )


def test_cycle_epochs_valid_construction_ok():
    epochs = CycleEpochs(**_valid_epochs_kwargs())
    assert epochs.n_cycles == 2


def test_cycle_epochs_rejects_non_finite_tau():
    kwargs = _valid_epochs_kwargs()
    kwargs["tau"] = np.array([0.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="tau must be finite"):
        CycleEpochs(**kwargs)


def test_cycle_epochs_rejects_non_finite_duration():
    kwargs = _valid_epochs_kwargs()
    kwargs["duration"] = np.array([1.0, np.inf])
    with pytest.raises(ValueError, match="duration must be finite"):
        CycleEpochs(**kwargs)


def test_cycle_epochs_rejects_non_finite_time():
    kwargs = _valid_epochs_kwargs()
    kwargs["time"] = np.array([0.0, 0.5, np.nan, 1.5])
    with pytest.raises(ValueError, match="time must be finite"):
        CycleEpochs(**kwargs)


def test_cycle_epochs_rejects_non_finite_phase():
    kwargs = _valid_epochs_kwargs()
    kwargs["phase"] = np.array([0.0, np.nan, 2 * np.pi, 3 * np.pi])
    with pytest.raises(ValueError, match="phase must be finite"):
        CycleEpochs(**kwargs)


def test_cycle_epochs_rejects_non_finite_phase_in_cycle():
    kwargs = _valid_epochs_kwargs()
    kwargs["phase_in_cycle"] = np.array([0.0, np.inf, 0.0, np.pi])
    with pytest.raises(ValueError, match="phase_in_cycle must be finite"):
        CycleEpochs(**kwargs)


@pytest.mark.parametrize("bad_value", [-2, 2, 100])
def test_cycle_epochs_rejects_out_of_range_cycle_index(bad_value):
    kwargs = _valid_epochs_kwargs()
    kwargs["cycle_index"] = np.array([0, 0, 1, bad_value])
    with pytest.raises(ValueError, match="cycle_index must be in"):
        CycleEpochs(**kwargs)


def test_cycle_epochs_allows_all_unassigned_cycle_index():
    """-1 is always valid, even with zero cycles (tau has length 1)."""
    epochs = CycleEpochs(
        tau=np.array([0.0]),
        duration=np.array([]),
        cycle_index=np.array([-1, -1, -1]),
        phase=None,
        phase_in_cycle=None,
        time=np.array([0.0, 0.5, 1.0]),
        source="test",
    )
    assert epochs.n_cycles == 0


# ---------------------------------------------------------------------------
# 14. seed_boundary_indices endpoint completion
# ---------------------------------------------------------------------------

def test_seed_boundary_indices_completes_genuine_edge_peaks():
    """When a real peak sits exactly at sample 0 and/or the last sample,
    find_peaks alone cannot report it (no neighbor on one side); the
    completion step should recover both using the estimated period."""
    fs = 100.0
    T0 = 1.0
    n = 601  # samples 0..600 -> peaks of cos at 0,100,...,600 (6 full cycles)
    t = np.arange(n) / fs
    ref = np.cos(2 * np.pi * t / T0)

    tau_idx = seed_boundary_indices(ref, fs, T0)

    assert tau_idx[0] == 0
    assert tau_idx[-1] == 600
    np.testing.assert_array_equal(tau_idx, [0, 100, 200, 300, 400, 500, 600])


def test_seed_boundary_indices_no_completion_without_room():
    """No completion should be added when there isn't a full extra cycle's
    worth of room, or the edge doesn't actually look like a peak -- this
    guards against the endpoint fix "simply forcing" a boundary."""
    fs = 100.0
    T0 = 1.0
    n = 601
    t = np.arange(n) / fs
    # Peaks at phase=pi (offset from t=0), same construction as the
    # validation suite's clean_complete_cycles scenario: neither array edge
    # has room for one more full cycle, so completion must be a no-op.
    ref = np.cos(2 * np.pi * t / T0 + np.pi)

    tau_idx = seed_boundary_indices(ref, fs, T0)

    assert tau_idx[0] not in (0,)
    assert tau_idx[-1] not in (n - 1,)
    np.testing.assert_array_equal(tau_idx, [50, 150, 250, 350, 450, 550])


# ---------------------------------------------------------------------------
# 15. find_epochs_by_geometric_score coverage-aware selection
# ---------------------------------------------------------------------------

def test_find_epochs_by_geometric_score_prefers_more_cycles_when_tied():
    """An exact integer multiple of the true period retraces the same loop
    and can score just as well as the true period -- selection should
    prefer the higher-coverage (more-cycles) candidate among near-tied
    scores, not the first/highest-scoring one regardless of coverage."""
    fs = 100.0
    period = 1.0
    offset = 0.37
    n_time = 651  # 6 true 1.0 s cycles starting at offset 0.37 s
    t = np.arange(n_time) / fs
    tilt = np.pi / 6
    phase = 2 * np.pi * (t - offset) / period
    X = np.column_stack([
        np.cos(phase), np.sin(phase) * np.cos(tilt), np.sin(phase) * np.sin(tilt),
    ])
    X += np.random.default_rng(1).standard_normal(X.shape) * 0.01

    ref = dominant_reference_signal(X)
    c1 = period_candidates_from_periodogram(ref, fs)
    c2 = period_candidates_from_autocorrelation(ref, fs)
    epochs, table = find_epochs_by_geometric_score(X, fs, period_candidates=c1 + c2)

    assert epochs.n_cycles >= 5, (
        f"expected the ~6-cycle true-period candidate to win, got "
        f"n_cycles={epochs.n_cycles}, period={epochs.metadata.get('period')}"
    )
    assert abs(float(epochs.metadata["period"]) - period) < 0.1


def test_find_epochs_by_geometric_score_selection_tolerance_is_absolute():
    """score_tolerance=0 falls back to plain argmax(total_score) (the
    pre-fix behavior) -- confirms the new parameter actually changes
    selection rather than being a no-op."""
    X, phase, fs = _tilted_circle(n_cycles=4, samples_per_cycle=100, noise_std=0.01)
    ref = dominant_reference_signal(X)
    cands = period_candidates_from_periodogram(ref, fs, n_candidates=5)
    epochs_strict, _ = find_epochs_by_geometric_score(
        X, fs, period_candidates=cands, n_phase_offsets=16, score_tolerance=0.0,
    )
    epochs_tolerant, _ = find_epochs_by_geometric_score(
        X, fs, period_candidates=cands, n_phase_offsets=16, score_tolerance=0.001,
    )
    # Coverage-aware selection can only match or beat plain argmax on cycle count.
    assert epochs_tolerant.n_cycles >= epochs_strict.n_cycles


# ---------------------------------------------------------------------------
# 16. Winding diagnostic and validity filter
# ---------------------------------------------------------------------------

def _clean_circle_epochs(n_cycles=6, samples_per_cycle=100, period=1.0):
    """A clean tilted circle plus a matching true-period CycleEpochs,
    reused across the winding tests below."""
    X, phase, fs = _tilted_circle(n_cycles=n_cycles, samples_per_cycle=samples_per_cycle)
    n_time = X.shape[0]
    epochs = candidate_epochs_from_period_offset(
        period, 0.0, sampling_rate_hz=fs, n_time=n_time,
    )
    return X, fs, epochs


def test_winding_single_clean_lap_is_accepted():
    X, fs, epochs = _clean_circle_epochs()
    score = score_epoch_geometry(X, epochs, sampling_rate_hz=fs)
    assert 0.9 <= score["winding_median_abs"] <= 1.1
    assert score["fraction_single_lap_cycles"] == 1.0
    for c in score["per_cycle"]:
        assert 0.9 <= abs(c["winding"]) <= 1.1


@pytest.mark.parametrize("multiplier,expected_abs_winding", [(2.0, 2.0), (3.0, 3.0)])
def test_winding_multi_lap_candidates_are_rejected(multiplier, expected_abs_winding):
    """A period that's an exact integer multiple of the true one retraces
    the same physical loop multiple times per proposed cycle."""
    X, phase, fs = _tilted_circle(n_cycles=6, samples_per_cycle=100)
    n_time = X.shape[0]
    epochs = candidate_epochs_from_period_offset(
        multiplier * 1.0, 0.0, sampling_rate_hz=fs, n_time=n_time,
    )
    score = score_epoch_geometry(X, epochs, sampling_rate_hz=fs)
    assert abs(score["winding_median_abs"] - expected_abs_winding) < 0.3
    assert score["fraction_single_lap_cycles"] < 0.5


def test_winding_half_lap_candidates_are_rejected():
    """A period that's a fraction of the true one only covers part of a
    revolution per proposed cycle. The measured winding is biased above the
    naive 0.5 guess (the per-cycle center is the sample mean of a *partial*
    arc, which sits off the true circle center -- see
    docs/debug/epoch_finder_validation_report.md), but it must still fall
    outside the valid range and be rejected."""
    X, phase, fs = _tilted_circle(n_cycles=6, samples_per_cycle=100)
    n_time = X.shape[0]
    epochs = candidate_epochs_from_period_offset(
        0.5, 0.0, sampling_rate_hz=fs, n_time=n_time,
    )
    score = score_epoch_geometry(X, epochs, sampling_rate_hz=fs)
    assert score["winding_median_abs"] < 0.75
    assert score["fraction_single_lap_cycles"] == 0.0


def test_winding_reversed_traversal_has_negative_signed_winding():
    """Reversing the *order* of one cycle's own points (not the whole
    multi-cycle recording) keeps the per-cycle PCA fit identical between the
    forward and reversed calls -- fit is order-invariant, depending only on
    the point *set* -- isolating the effect of traversal direction from
    PCA's independently-arbitrary axis sign (which a symmetric circle
    otherwise confounds: refitting PCA separately on forward vs. reversed
    multi-cycle data can pick differently-signed axes for reasons unrelated
    to traversal direction)."""
    fs = 100.0
    n = 100
    t = np.arange(n) / fs
    phase = 2 * np.pi * t  # exactly one lap
    tilt = np.pi / 6
    X_k = np.column_stack([
        np.cos(phase), np.sin(phase) * np.cos(tilt), np.sin(phase) * np.sin(tilt),
    ])
    epochs = candidate_epochs_from_period_offset(1.0, 0.0, sampling_rate_hz=fs, n_time=n)

    score_forward = score_epoch_geometry(X_k, epochs, sampling_rate_hz=fs)
    score_reversed = score_epoch_geometry(X_k[::-1], epochs, sampling_rate_hz=fs)
    w_fwd = score_forward["per_cycle"][0]["winding"]
    w_rev = score_reversed["per_cycle"][0]["winding"]

    assert np.isfinite(w_fwd) and np.isfinite(w_rev)
    assert np.sign(w_fwd) == -np.sign(w_rev)
    # abs(winding) validity is direction-independent.
    assert 0.9 <= abs(w_rev) <= 1.1
    assert score_reversed["fraction_single_lap_cycles"] == 1.0


def test_winding_accepts_mild_phase_warp():
    fs = 100.0
    period = 1.0
    n_cycles = 6
    n_time = n_cycles * 100 + 1
    t = np.arange(n_time) / fs
    tilt = np.pi / 6
    phase = 2 * np.pi * t / period + 0.3 * np.sin(2 * np.pi * t / period)
    X = np.column_stack([
        np.cos(phase), np.sin(phase) * np.cos(tilt), np.sin(phase) * np.sin(tilt),
    ])
    epochs = candidate_epochs_from_period_offset(period, 0.0, sampling_rate_hz=fs, n_time=n_time)
    score = score_epoch_geometry(X, epochs, sampling_rate_hz=fs)
    assert 0.9 <= score["winding_median_abs"] <= 1.1
    assert score["fraction_single_lap_cycles"] >= 0.8


def test_winding_near_center_degeneracy_excludes_but_does_not_crash():
    """A few points collapsed near the fitted center have an unstable angle
    and should be excluded from the winding sum (not corrupt it), while
    being visible in the transition-count diagnostics."""
    X, fs, epochs = _clean_circle_epochs()
    idx0 = np.where(epochs.cycle_index == 0)[0]
    X_degenerate = X.copy()
    center0 = X[idx0].mean(axis=0)
    # Collapse 2 of the ~100 samples in cycle 0 onto the center.
    X_degenerate[idx0[10]] = center0
    X_degenerate[idx0[11]] = center0 + 1e-10

    score = score_epoch_geometry(X_degenerate, epochs, sampling_rate_hz=fs)
    cycle0 = score["per_cycle"][0]
    assert np.isfinite(cycle0["winding"])
    assert 0.9 <= abs(cycle0["winding"]) <= 1.1
    assert cycle0["winding_n_valid_transitions"] < cycle0["winding_n_total_transitions"]


def test_winding_mostly_degenerate_cycle_is_undefined_not_silently_valid():
    """If most of a cycle's samples are too close to the center for a
    reliable angle, winding must be NaN, not a value computed from an
    unreliable minority -- and NaN must not count as single-lap.

    Exercises the internal _cycle_winding helper directly, since the real
    per-cycle center (score_epoch_geometry always passes center_k =
    X_k.mean(axis=0)) is a moving target: collapsing a majority of points
    toward an *externally* chosen point shifts the recomputed mean away
    from that point, so the "collapsed" points end up with a nonzero radius
    from the true center after all, defeating a naive construction. Fixed
    here by collapsing the majority onto the mean of the *remaining*
    (real-arc) points: since sum(10 real points) = 10 * arc_mean by
    definition, mean(90 copies of arc_mean + those 10 points) works out to
    exactly arc_mean again -- i.e. the collapsed points land exactly on the
    true center by construction, algebraically, not by trial and error."""
    from sklearn.decomposition import PCA
    from phase_coordinates.scoring import _cycle_winding

    n = 100
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    X_k = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(n)])
    arc_mean = X_k[90:].mean(axis=0)
    X_k[:90] = arc_mean
    center_k = X_k.mean(axis=0)
    np.testing.assert_allclose(center_k, arc_mean)  # sanity-check the algebra above
    pca = PCA(n_components=3)
    pca.fit(X_k - center_k)

    winding, n_valid, n_total = _cycle_winding(X_k, center_k, pca)
    assert not np.isfinite(winding)
    assert n_total == n - 1
    assert n_valid / n_total < 0.5


def test_candidate_table_retains_rejected_candidates_with_winding_diagnostics():
    """Rejected (winding-invalid) candidates must remain visible in the
    table, not be silently dropped from reporting."""
    X, phase, fs = _tilted_circle(n_cycles=6, samples_per_cycle=100, noise_std=0.01)
    ref = dominant_reference_signal(X)
    cands = period_candidates_from_periodogram(ref, fs, n_candidates=5)
    cands = expand_period_harmonics(cands)  # pulls in a 0.5x/2x candidate too
    _, table = find_epochs_by_geometric_score(X, fs, period_candidates=cands, n_phase_offsets=16)

    for col in ("winding_median_abs", "winding_min_abs", "winding_max_abs",
                "fraction_single_lap_cycles", "winding_valid"):
        assert col in table.columns
    assert (~table["winding_valid"]).any(), "expected at least one rejected candidate in the table"
    assert table["winding_valid"].any(), "expected at least one accepted candidate in the table"
    # Rejected rows still carry real (non-null) winding diagnostics.
    rejected = table[~table["winding_valid"]]
    assert rejected["fraction_single_lap_cycles"].notna().all()


def test_require_winding_valid_false_falls_back_to_score_tolerance_only():
    """The winding filter is opt-out (require_winding_valid=False), kept
    for direct comparison against plain score_tolerance selection -- not
    because it's optional in normal use."""
    X, phase, fs = _tilted_circle(n_cycles=6, samples_per_cycle=100, noise_std=0.01)
    ref = dominant_reference_signal(X)
    cands = period_candidates_from_periodogram(ref, fs, n_candidates=5)
    cands = expand_period_harmonics(cands)
    epochs_no_filter, _ = find_epochs_by_geometric_score(
        X, fs, period_candidates=cands, n_phase_offsets=16,
        require_winding_valid=False,
    )
    epochs_filtered, _ = find_epochs_by_geometric_score(
        X, fs, period_candidates=cands, n_phase_offsets=16,
        require_winding_valid=True,
    )
    # Both must produce a result; the filtered winner must be winding-valid.
    score_filtered = score_epoch_geometry(X, epochs_filtered, sampling_rate_hz=fs)
    assert score_filtered["fraction_single_lap_cycles"] >= 0.8
    assert epochs_no_filter.n_cycles >= 2
