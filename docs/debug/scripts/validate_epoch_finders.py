"""
Validate the geometric-score epoch finder against the periodogram/peak-seed
path on controlled synthetic data.

This script does *not* touch the Bayesian model. It only compares two
deterministic ways of turning a 3-D trajectory into a
:class:`~phase_coordinates.epochs.CycleEpochs` object:

  A. periodogram + peak-seed path
       dominant_reference_signal -> estimate_dominant_period ->
       seed_boundary_indices -> epochs_from_boundary_indices

  B. geometric-score path
       dominant_reference_signal ->
       period_candidates_from_periodogram / period_candidates_from_autocorrelation ->
       expand_period_harmonics -> find_epochs_by_geometric_score

  C. (reference only) phase-ground-truth path
       identify_cycles_from_phase(true_phase, ...)
     C is an upper-bound sanity check, not part of the A-vs-B comparison:
     identify_cycles_from_phase always anchors cycle 0 to sample 0
     ("first_sample" convention), so in the phase_offset_start scenario its
     boundaries are intentionally offset from `true_tau` by construction,
     not because it "got the offset wrong".

Run with the project's pixi Python, from the repo root, with the repo root
on PYTHONPATH (the script also inserts it itself so it works regardless of
cwd or PYTHONPATH):

    <repo-root>/.pixi/envs/default/bin/python docs/debug/scripts/validate_epoch_finders.py

Writes two CSVs next to this script's report
(docs/debug/epoch_finder_metrics.csv, docs/debug/epoch_finder_top_candidates.csv)
and prints a summary to stdout.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from phase_coordinates import (  # noqa: E402
    CycleEpochs,
    candidate_epochs_from_period_offset,
    compute_cycle_quality,
    dominant_reference_signal,
    epochs_from_boundary_indices,
    estimate_dominant_period,
    expand_period_harmonics,
    find_epochs_by_geometric_score,
    identify_cycles_from_phase,
    period_candidates_from_autocorrelation,
    period_candidates_from_periodogram,
    score_epoch_geometry,
    seed_boundary_indices,
)


# ---------------------------------------------------------------------------
# Synthetic scenario generators
# ---------------------------------------------------------------------------

@dataclass
class SyntheticEpochDataset:
    X: np.ndarray
    fs: float
    true_tau: np.ndarray
    true_period: float
    description: str
    name: str
    true_phase: Optional[np.ndarray] = None


def _tilted_circle_position(phase: np.ndarray, *, tilt: float, radius_fn=None) -> np.ndarray:
    """Map instantaneous phase (radians) to a 3-D position on a tilted circle.

    ``radius_fn(phase)``, if given, modulates the in-plane radius as a
    function of phase (used for the harmonic-ambiguity scenario).
    """
    r = np.ones_like(phase) if radius_fn is None else radius_fn(phase)
    return np.column_stack([
        r * np.cos(phase),
        r * np.sin(phase) * np.cos(tilt),
        r * np.sin(phase) * np.sin(tilt),
    ])


def _true_boundaries(period: float, offset: float, n_time: int, fs: float) -> np.ndarray:
    """tau_k = offset + k*period for k=0,1,... while tau_k <= (n_time-1)/fs.

    Deliberately stricter than the package's own half-open convention (which
    allows a closing boundary up to n_time/fs with no real sample there --
    see epochs_from_boundary_indices / candidate_epochs_from_period_offset).
    For *ground truth* in this validation, a boundary only counts as "true"
    if there is an actual recorded sample at that time -- otherwise no
    empirical method (peak detection, phase-crossing) could ever recover it
    even in principle, and grading them against it would be an artifact of
    the test construction, not a real capability gap.
    """
    t_max = (n_time - 1) / fs
    k_max = int(np.floor((t_max - offset) / period))
    return offset + np.arange(k_max + 1) * period


def make_clean_complete_cycles(rng=None) -> SyntheticEpochDataset:
    fs = 100.0
    period = 1.0
    n_cycles = 6
    samples_per_cycle = int(round(period * fs))
    # +1 sample so the last true boundary (t = n_cycles*period) lands on an
    # actual recorded sample -- see _true_boundaries docstring.
    n_time = n_cycles * samples_per_cycle + 1
    t = np.arange(n_time) / fs
    phase = 2 * np.pi * t / period
    X = _tilted_circle_position(phase, tilt=np.pi / 6)
    true_tau = _true_boundaries(period, 0.0, n_time, fs)
    return SyntheticEpochDataset(
        X=X, fs=fs, true_tau=true_tau, true_period=period,
        description=(
            "Exact tilted circle, no noise, no offset, 6 complete 1.0 s "
            "cycles. Baseline sanity case: both methods should recover the "
            "true period and boundaries almost exactly."
        ),
        name="clean_complete_cycles",
        true_phase=phase,
    )


def make_noisy_complete_cycles(rng=None) -> SyntheticEpochDataset:
    rng = rng or np.random.default_rng(0)
    fs = 100.0
    period = 1.0
    n_cycles = 6
    samples_per_cycle = int(round(period * fs))
    n_time = n_cycles * samples_per_cycle + 1
    t = np.arange(n_time) / fs
    phase = 2 * np.pi * t / period
    X = _tilted_circle_position(phase, tilt=np.pi / 6)
    X = X + rng.standard_normal(X.shape) * 0.03
    true_tau = _true_boundaries(period, 0.0, n_time, fs)
    return SyntheticEpochDataset(
        X=X, fs=fs, true_tau=true_tau, true_period=period,
        description=(
            "Same as clean_complete_cycles plus isotropic Gaussian noise "
            "(std=0.03, ~3% of the unit radius). Tests robustness to "
            "measurement noise alone."
        ),
        name="noisy_complete_cycles",
        true_phase=phase,
    )


def make_phase_offset_start(rng=None) -> SyntheticEpochDataset:
    rng = rng or np.random.default_rng(1)
    fs = 100.0
    period = 1.0
    offset = 0.37
    duration = 6.5
    n_time = int(round(duration * fs)) + 1
    t = np.arange(n_time) / fs
    phase = 2 * np.pi * (t - offset) / period
    X = _tilted_circle_position(phase, tilt=np.pi / 6)
    X = X + rng.standard_normal(X.shape) * 0.01
    true_tau = _true_boundaries(period, offset, n_time, fs)
    return SyntheticEpochDataset(
        X=X, fs=fs, true_tau=true_tau, true_period=period,
        description=(
            f"Recording starts {offset:.2f} s before the first true cycle "
            "boundary (light noise, std=0.01). Tests whether a method finds "
            "the correct *phase offset*, not just the correct period. Note: "
            "the phase-ground-truth reference path always anchors cycle 0 "
            "to sample 0 regardless of true offset, so it is not expected "
            "to match true_tau here — see module docstring."
        ),
        name="phase_offset_start",
        true_phase=phase,
    )


def make_mild_phase_warp(rng=None) -> SyntheticEpochDataset:
    rng = rng or np.random.default_rng(2)
    fs = 100.0
    period = 1.0
    n_cycles = 6
    samples_per_cycle = int(round(period * fs))
    n_time = n_cycles * samples_per_cycle + 1
    t = np.arange(n_time) / fs
    warp_amplitude = 0.3  # radians; sin(.) vanishes exactly at cycle boundaries
    phase = 2 * np.pi * t / period + warp_amplitude * np.sin(2 * np.pi * t / period)
    X = _tilted_circle_position(phase, tilt=np.pi / 6)
    X = X + rng.standard_normal(X.shape) * 0.01
    # warp_amplitude * sin(2*pi*k) == 0 for integer k, so true boundaries
    # are unchanged from the unwarped case even though within-cycle angular
    # velocity is now non-uniform.
    true_tau = _true_boundaries(period, 0.0, n_time, fs)
    return SyntheticEpochDataset(
        X=X, fs=fs, true_tau=true_tau, true_period=period,
        description=(
            f"Angular velocity varies within each cycle (phase += "
            f"{warp_amplitude} * sin(2*pi*t/T)), but the warp term is "
            "exactly zero at every true cycle boundary, so true_tau is "
            "unchanged from the clean case. Tests robustness to "
            "non-constant angular velocity, which the geometric score's "
            "quarter-cycle anchor implicitly assumes is roughly constant."
        ),
        name="mild_phase_warp",
        true_phase=phase,
    )


def make_harmonic_ambiguity_candidate(rng=None) -> SyntheticEpochDataset:
    rng = rng or np.random.default_rng(3)
    fs = 100.0
    period = 1.0
    n_cycles = 6
    samples_per_cycle = int(round(period * fs))
    n_time = n_cycles * samples_per_cycle + 1
    t = np.arange(n_time) / fs
    phase = 2 * np.pi * t / period

    def radius_fn(p):
        # Period-T/2 radius modulation. Position is still genuinely
        # period-T (position(phase+pi) != position(phase) because the
        # angular cos/sin term flips sign but the radius term does not),
        # but a 1-D projection can carry strong energy at 2/T, which is
        # exactly the kind of signal that could fool a purely spectral
        # period estimator into preferring period/2.
        return 1.0 + 0.5 * np.cos(2 * p)

    X = _tilted_circle_position(phase, tilt=np.pi / 6, radius_fn=radius_fn)
    X = X + rng.standard_normal(X.shape) * 0.01
    true_tau = _true_boundaries(period, 0.0, n_time, fs)
    return SyntheticEpochDataset(
        X=X, fs=fs, true_tau=true_tau, true_period=period,
        description=(
            "Radius modulated at period T/2 (0.5*cos(2*phase)) superimposed "
            "on the true period-T angular motion. Tests whether period "
            "search / geometric scoring is fooled into locking onto a 0.5x "
            "or 2x harmonic of the true period."
        ),
        name="harmonic_ambiguity_candidate",
        true_phase=phase,
    )


SCENARIOS: list[Callable[[], SyntheticEpochDataset]] = [
    make_clean_complete_cycles,
    make_noisy_complete_cycles,
    make_phase_offset_start,
    make_mild_phase_warp,
    make_harmonic_ambiguity_candidate,
]


# ---------------------------------------------------------------------------
# Epoch finders under comparison
# ---------------------------------------------------------------------------

def run_peak_seed_method(X: np.ndarray, fs: float) -> tuple[CycleEpochs, dict]:
    """Method A: periodogram + peak seed path."""
    ref = dominant_reference_signal(X)
    T0 = estimate_dominant_period(ref, fs)
    tau_idx = seed_boundary_indices(ref, fs, T0)
    epochs = epochs_from_boundary_indices(
        tau_idx, sampling_rate_hz=fs, n_time=len(X),
        source="periodogram_peaks", metadata={"T0": T0},
    )
    return epochs, {"period_estimate": T0}


def run_geometric_score_method(
    X: np.ndarray, fs: float, *, n_phase_offsets: int = 64,
) -> tuple[CycleEpochs, dict, pd.DataFrame]:
    """Method B: geometric-score path."""
    ref = dominant_reference_signal(X)
    c1 = period_candidates_from_periodogram(ref, fs)
    c2 = period_candidates_from_autocorrelation(ref, fs)
    candidates = expand_period_harmonics(c1 + c2)
    epochs, candidate_table = find_epochs_by_geometric_score(
        X, fs, period_candidates=candidates, n_phase_offsets=n_phase_offsets,
    )
    period_estimate = float(epochs.metadata.get("period", np.nan))
    return epochs, {"period_estimate": period_estimate}, candidate_table


def run_phase_ground_truth_method(
    X: np.ndarray, fs: float, true_phase: np.ndarray,
) -> tuple[CycleEpochs, dict]:
    """Method C (reference only): epochs from the true instantaneous phase."""
    epochs = identify_cycles_from_phase(true_phase, sampling_rate_hz=fs)
    durations = epochs.duration
    period_estimate = float(np.median(durations)) if len(durations) else float("nan")
    return epochs, {"period_estimate": period_estimate}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def boundary_errors_samples(true_tau: np.ndarray, pred_tau: np.ndarray, fs: float) -> np.ndarray:
    """For each true boundary, absolute time error (in samples) to the nearest
    predicted boundary. Matching is nearest-time, independent in each
    direction (see compute_metrics for the count-mismatch caveat)."""
    if len(pred_tau) == 0 or len(true_tau) == 0:
        return np.array([])
    # searchsorted-based nearest neighbor
    idx = np.searchsorted(pred_tau, true_tau)
    idx = np.clip(idx, 1, len(pred_tau) - 1)
    left = pred_tau[idx - 1]
    right = pred_tau[idx]
    nearest = np.where(np.abs(true_tau - left) <= np.abs(true_tau - right), left, right)
    return np.abs(true_tau - nearest) * fs


def compute_metrics(
    dataset: SyntheticEpochDataset, method_name: str, epochs: CycleEpochs,
    extra: dict,
) -> dict:
    fs = dataset.fs
    true_tau = dataset.true_tau
    n_cycles_true = len(true_tau) - 1
    n_cycles_found = epochs.n_cycles

    score = score_epoch_geometry(dataset.X, epochs, sampling_rate_hz=fs)

    tau_errors = boundary_errors_samples(true_tau, epochs.tau, fs)
    duration_errors = (
        (epochs.duration - dataset.true_period) * fs if n_cycles_found > 0 else np.array([])
    )

    period_estimate = extra.get("period_estimate", np.nan)
    period_error = (
        period_estimate - dataset.true_period if np.isfinite(period_estimate) else np.nan
    )

    return {
        "scenario": dataset.name,
        "method": method_name,
        "n_cycles_true": n_cycles_true,
        "n_cycles_found": n_cycles_found,
        "coverage_fraction": (
            n_cycles_found / n_cycles_true if n_cycles_true > 0 else np.nan
        ),
        "coverage_duration_fraction": score["coverage_duration_fraction"],
        "fraction_samples_assigned": score["fraction_samples_assigned"],
        "period_estimate": period_estimate,
        "period_error": period_error,
        "median_duration_error_samples": (
            float(np.median(np.abs(duration_errors))) if len(duration_errors) else np.nan
        ),
        "max_duration_error_samples": (
            float(np.max(np.abs(duration_errors))) if len(duration_errors) else np.nan
        ),
        "median_tau_error_samples": (
            float(np.median(tau_errors)) if len(tau_errors) else np.nan
        ),
        "max_tau_error_samples": (
            float(np.max(tau_errors)) if len(tau_errors) else np.nan
        ),
        "planarity": score["planarity"],
        "anchor_norm": score["anchor_norm"],
        "quarter_anchor_orth_ratio": score["quarter_anchor_orth_ratio"],
        "min_samples_per_cycle": score["min_samples_per_cycle"],
    }


METRIC_COLUMNS = [
    "scenario", "method",
    "n_cycles_true", "n_cycles_found",
    "coverage_fraction", "coverage_duration_fraction", "fraction_samples_assigned",
    "period_estimate", "period_error",
    "median_duration_error_samples", "max_duration_error_samples",
    "median_tau_error_samples", "max_tau_error_samples",
    "planarity", "anchor_norm", "quarter_anchor_orth_ratio", "min_samples_per_cycle",
]


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def run_all(n_phase_offsets: int = 64) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run every scenario x method combination.

    Returns
    -------
    metrics_df : pandas.DataFrame
        One row per (scenario, method).
    top_candidates_df : pandas.DataFrame
        Top-10-by-total_score rows (per scenario) from the geometric-score
        candidate table.
    cycle_quality : dict
        (scenario, method) -> compute_cycle_quality() DataFrame, for
        anyone who wants the per-cycle detail behind the summary metrics.
    """
    metric_rows = []
    top_candidate_rows = []
    cycle_quality = {}
    failures = []

    def _record_failure(scenario: str, method: str, exc: Exception):
        msg = f"{scenario} / {method}: {type(exc).__name__}: {exc}"
        print(f"  FAILED: {msg}")
        failures.append({"scenario": scenario, "method": method, "error": msg})
        metric_rows.append({col: np.nan for col in METRIC_COLUMNS} | {
            "scenario": scenario, "method": method,
        })

    for make_dataset in SCENARIOS:
        dataset = make_dataset()
        print(f"=== {dataset.name} ===")
        print(dataset.description)

        # Method A
        try:
            epochs_a, extra_a = run_peak_seed_method(dataset.X, dataset.fs)
            metric_rows.append(compute_metrics(dataset, "peak_seed", epochs_a, extra_a))
            cycle_quality[(dataset.name, "peak_seed")] = compute_cycle_quality(
                dataset.X, epochs_a, sampling_rate_hz=dataset.fs,
            )
        except Exception as exc:  # noqa: BLE001 -- deliberately broad: report, don't hide
            _record_failure(dataset.name, "peak_seed", exc)

        # Method B
        try:
            epochs_b, extra_b, candidate_table = run_geometric_score_method(
                dataset.X, dataset.fs, n_phase_offsets=n_phase_offsets,
            )
            metric_rows.append(compute_metrics(dataset, "geometric_score", epochs_b, extra_b))
            cycle_quality[(dataset.name, "geometric_score")] = compute_cycle_quality(
                dataset.X, epochs_b, sampling_rate_hz=dataset.fs,
            )
            top10 = candidate_table.sort_values("total_score", ascending=False).head(10).copy()
            top10.insert(0, "scenario", dataset.name)
            top_candidate_rows.append(top10)
        except Exception as exc:  # noqa: BLE001
            _record_failure(dataset.name, "geometric_score", exc)

        # Method C (reference only)
        try:
            epochs_c, extra_c = run_phase_ground_truth_method(dataset.X, dataset.fs, dataset.true_phase)
            metric_rows.append(compute_metrics(dataset, "phase_ground_truth", epochs_c, extra_c))
            cycle_quality[(dataset.name, "phase_ground_truth")] = compute_cycle_quality(
                dataset.X, epochs_c, sampling_rate_hz=dataset.fs,
            )
        except Exception as exc:  # noqa: BLE001
            _record_failure(dataset.name, "phase_ground_truth", exc)

        print()

    if failures:
        print(f"=== {len(failures)} failure(s) — see 'error' rows in the metrics table ===")
        for f in failures:
            print(f"  {f['error']}")
        print()

    metrics_df = pd.DataFrame(metric_rows, columns=METRIC_COLUMNS)
    top_candidates_df = (
        pd.concat(top_candidate_rows, ignore_index=True)
        if top_candidate_rows else pd.DataFrame()
    )
    return metrics_df, top_candidates_df, cycle_quality


# ---------------------------------------------------------------------------
# Winding diagnostic checks: deliberate multi-lap / fractional-lap / reversal
# probes, direct against candidate_epochs_from_period_offset + score_epoch_geometry
# (bypassing period search entirely -- these test the winding *diagnostic*
# and *validity rule* themselves against known period errors, not the
# search that finds periods; period-search-driven multi-lap/fractional-lap
# behavior is already exercised indirectly through the SCENARIOS above,
# e.g. phase_offset_start).
# ---------------------------------------------------------------------------

WINDING_CHECK_COLUMNS = [
    "check", "description", "winding_median_abs", "winding_min_abs",
    "winding_max_abs", "expected_abs_winding_range", "expected_valid",
    "actually_valid", "passed",
]


def _winding_check(
    results: list, name: str, description: str, X: np.ndarray, fs: float,
    epochs: CycleEpochs, expected_abs_winding_range: tuple, expect_valid: bool,
    *, winding_min_fraction: float = 0.8,
):
    score = score_epoch_geometry(X, epochs, sampling_rate_hz=fs)
    winding_median_abs = score["winding_median_abs"]
    lo, hi = expected_abs_winding_range
    in_expected_range = (
        np.isfinite(winding_median_abs) and lo <= winding_median_abs <= hi
    )
    actually_valid = score["fraction_single_lap_cycles"] >= winding_min_fraction
    passed = in_expected_range and (actually_valid == expect_valid)
    results.append({
        "check": name,
        "description": description,
        "winding_median_abs": winding_median_abs,
        "winding_min_abs": score["winding_min_abs"],
        "winding_max_abs": score["winding_max_abs"],
        "expected_abs_winding_range": expected_abs_winding_range,
        "expected_valid": expect_valid,
        "actually_valid": actually_valid,
        "passed": passed,
    })
    return score


def run_winding_diagnostic_checks() -> pd.DataFrame:
    """
    Deliberate probes of the winding diagnostic in both directions of
    period error, plus mild-warp tolerance and traversal-reversal, using
    candidate epochs built directly at known period multiples/fractions of
    the true period (not from period search -- these test the diagnostic's
    math and the validity rule, independent of whether period search would
    ever actually propose these periods).
    """
    results: list = []

    base = make_clean_complete_cycles()
    n_time = len(base.X)

    # 1. True-period candidate: winding ~ 1, must be valid.
    epochs_true = candidate_epochs_from_period_offset(
        base.true_period, 0.0, sampling_rate_hz=base.fs, n_time=n_time,
    )
    _winding_check(
        results, "true_period", "period = 1x true period",
        base.X, base.fs, epochs_true, (0.9, 1.1), True,
    )

    # 2. Multi-lap candidates: 2x and 3x the true period, winding ~ 2 / ~3,
    #    must be invalid (retracing the same loop multiple times per cycle).
    epochs_2x = candidate_epochs_from_period_offset(
        2 * base.true_period, 0.0, sampling_rate_hz=base.fs, n_time=n_time,
    )
    _winding_check(
        results, "multi_lap_2x", "period = 2x true period",
        base.X, base.fs, epochs_2x, (1.8, 2.2), False,
    )

    epochs_3x = candidate_epochs_from_period_offset(
        3 * base.true_period, 0.0, sampling_rate_hz=base.fs, n_time=n_time,
    )
    _winding_check(
        results, "multi_lap_3x", "period = 3x true period",
        base.X, base.fs, epochs_3x, (2.8, 3.2), False,
    )

    # 3. Fractional-lap candidate: 0.5x the true period, must be invalid
    #    (each proposed cycle covers only half a revolution). The expected
    #    winding is NOT ~0.5, though -- verified analytically and
    #    empirically (see docs/debug/epoch_finder_validation_report.md):
    #    the per-cycle center is the sample mean of that cycle's own
    #    points, reused from the existing planarity computation. Averaged
    #    over a *partial* arc, that mean sits off-center (a half-arc's
    #    centroid is 2/pi from the true circle center along the symmetry
    #    axis, not at the center), which inflates the apparent angular
    #    sweep measured around it. A full or multi-lap candidate doesn't
    #    have this bias -- averaging over one or more *complete*
    #    revolutions of a symmetric shape correctly cancels to the true
    #    center. Measured ~0.673 here (analytic continuous limit ~0.680);
    #    still well outside DEFAULT_WINDING_VALID_MIN (0.75), so the
    #    validity decision is unaffected -- but this is a real reason not
    #    to lower winding_valid_min much below its default without
    #    re-checking against this bias.
    epochs_half = candidate_epochs_from_period_offset(
        0.5 * base.true_period, 0.0, sampling_rate_hz=base.fs, n_time=n_time,
    )
    _winding_check(
        results, "fractional_lap_0.5x", "period = 0.5x true period",
        base.X, base.fs, epochs_half, (0.60, 0.75), False,
    )

    # 4. mild_phase_warp: true-period candidate must remain valid despite
    #    non-constant angular velocity within each cycle.
    warp = make_mild_phase_warp()
    epochs_warp = candidate_epochs_from_period_offset(
        warp.true_period, 0.0, sampling_rate_hz=warp.fs, n_time=len(warp.X),
    )
    _winding_check(
        results, "mild_phase_warp", "true period, non-constant angular velocity",
        warp.X, warp.fs, epochs_warp, (0.9, 1.1), True,
    )

    # 5. Traversal reversal: reverse the clean dataset in time (same loop,
    #    opposite direction). Signed winding should flip sign; abs(winding)
    #    should remain valid.
    X_reversed = base.X[::-1]
    epochs_reversed = candidate_epochs_from_period_offset(
        base.true_period, 0.0, sampling_rate_hz=base.fs, n_time=n_time,
    )
    score_forward = score_epoch_geometry(base.X, epochs_true, sampling_rate_hz=base.fs)
    score_reversed = _winding_check(
        results, "traversal_reversal", "same loop, reversed time order",
        X_reversed, base.fs, epochs_reversed, (0.9, 1.1), True,
    )
    signed_forward = [c["winding"] for c in score_forward["per_cycle"] if np.isfinite(c["winding"])]
    signed_reversed = [c["winding"] for c in score_reversed["per_cycle"] if np.isfinite(c["winding"])]
    sign_flipped = (
        len(signed_forward) > 0 and len(signed_reversed) > 0
        and np.sign(np.median(signed_forward)) == -np.sign(np.median(signed_reversed))
    )
    results.append({
        "check": "traversal_reversal_sign",
        "description": "signed winding flips sign under time reversal",
        "winding_median_abs": float("nan"),
        "winding_min_abs": float("nan"),
        "winding_max_abs": float("nan"),
        "expected_abs_winding_range": None,
        "expected_valid": True,
        "actually_valid": bool(sign_flipped),
        "passed": bool(sign_flipped),
    })

    return pd.DataFrame(results, columns=WINDING_CHECK_COLUMNS)


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    metrics_df, top_candidates_df, _ = run_all()
    winding_checks_df = run_winding_diagnostic_checks()

    out_dir = Path(__file__).resolve().parents[1]  # docs/debug/
    metrics_path = out_dir / "epoch_finder_metrics.csv"
    candidates_path = out_dir / "epoch_finder_top_candidates.csv"
    winding_checks_path = out_dir / "epoch_finder_winding_checks.csv"
    metrics_df.to_csv(metrics_path, index=False)
    top_candidates_df.to_csv(candidates_path, index=False)
    winding_checks_df.to_csv(winding_checks_path, index=False)

    print("=== Metrics summary (all scenarios x methods) ===")
    print(metrics_df.to_string(index=False))
    print()
    print("=== Winding diagnostic checks ===")
    print(winding_checks_df.to_string(index=False))
    if not winding_checks_df["passed"].all():
        failed = winding_checks_df.loc[~winding_checks_df["passed"], "check"].tolist()
        print(f"\n!!! {len(failed)} winding check(s) FAILED: {failed}")
    print()
    print(f"Wrote {metrics_path}")
    print(f"Wrote {candidates_path}")
    print(f"Wrote {winding_checks_path}")


if __name__ == "__main__":
    main()
