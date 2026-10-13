"""
Helpers for the issue #3 workup (inconsistent cycle-start alignment).

Three things live here:

1. Loading: movement runs for the subjects attached to the issue, built with
   the same rules as Noam's ``labeled_analysis.py`` (positive windows unioned
   per source file, gaps <= 15 frames bridged, runs >= 3 s kept).
2. ``segment_cycles``: a stand-alone segmentation in which each change made
   for the issue is a separate switch, so the changes can be turned on one at
   a time.  With every switch off it is the package as it was at ``d83b30d``
   (it reproduced that version exactly, cycle for cycle, before the package
   was changed).  With every switch on it must equal the package as it is
   now; the notebook checks that.
3. Metrics that do not depend on any phase model, so the variants can be
   compared fairly.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, periodogram

from phase_coordinates import (
    dominant_reference_axis,
    estimate_dominant_period,
    fit_pca_phase_coordinates,
    hilbert_phase,
)
from phase_coordinates.core import _bandpass_reference_signal

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DATA_FILE = HERE / "data" / "hand_root_xyz.npz"
WINDOW_FILES = {
    skill: REPO_ROOT / "test_data" / "repetitive_data" / f"window_predictions_{skill}.csv"
    for skill in ("walk", "jump", "climb")
}
FIGURES_DIR = HERE / "figures"

FS_HZ = 60.0
SKILLS = ("walk", "jump", "climb")
HANDS = ("r_hand", "l_hand")
MAX_GAP_TO_BRIDGE_FRAMES = 15
MIN_RUN_DURATION_SEC = 3.0
NORMALIZED_CYCLE_POINTS = 100

RUN_KEYS = ["subject", "skill", "day", "part", "source_file", "movement_run_id", "hand"]
CONDITION_KEYS = ["subject", "skill", "day", "part", "hand"]

# Column order inside each array of the data extract.
_XYZ_COLUMNS = {"root": slice(0, 3), "r_hand": slice(3, 6), "l_hand": slice(6, 9)}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_position_extract():
    """Return ``{source_file: (n_frames, 9) array}`` of root / r_hand / l_hand xyz."""
    with np.load(DATA_FILE) as archive:
        return {name: archive[name].astype(float) for name in archive.files}


def _bridge_gaps(mask, max_gap_frames):
    mask = mask.copy()
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = changes[::2], changes[1::2]
    for left_end, right_start in zip(ends[:-1], starts[1:]):
        if 0 < right_start - left_end <= max_gap_frames:
            mask[left_end:right_start] = True
    return mask


def _mask_to_runs(mask, min_run_frames):
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2]) if b - a >= min_run_frames]


def build_runs(positions):
    """One row per continuous movement run, for every source file in ``positions``."""
    min_run_frames = round(MIN_RUN_DURATION_SEC * FS_HZ)
    rows = []
    for skill, path in WINDOW_FILES.items():
        windows = pd.read_csv(path)
        windows = windows[windows["prediction"].astype(int).eq(1)]
        windows = windows[windows["source_file"].isin(positions)]
        for source_file, group in windows.groupby("source_file"):
            n_frames = len(positions[source_file])
            mask = np.zeros(n_frames, dtype=bool)
            for row in group.itertuples():
                # end_frame is inclusive in the prediction tables.
                mask[max(0, int(row.start_frame)):min(n_frames, int(row.end_frame) + 1)] = True
            bridged = _bridge_gaps(mask, MAX_GAP_TO_BRIDGE_FRAMES)
            first = group.iloc[0]
            for run_id, (start, end) in enumerate(_mask_to_runs(bridged, min_run_frames), 1):
                rows.append({
                    "subject": f"{int(first['subject']):03d}",
                    "skill": skill,
                    "day": int(first["day"]),
                    "part": str(first["session_part"]).strip().lower()[0],
                    "source_file": source_file,
                    "movement_run_id": run_id,
                    "run_start": start,
                    "run_end": end,
                    "run_n_frames": end - start,
                })
    runs = pd.DataFrame(rows)
    return runs.sort_values(
        ["subject", "skill", "day", "part", "source_file", "movement_run_id"]
    ).reset_index(drop=True)


def run_xyz(positions, run, hand):
    """Hand xyz relative to root xyz for one run, shape ``(n_frames, 3)``."""
    block = positions[run["source_file"]][int(run["run_start"]):int(run["run_end"])]
    return block[:, _XYZ_COLUMNS[hand]] - block[:, _XYZ_COLUMNS["root"]]


def run_key(run, hand):
    return tuple(run[k] for k in RUN_KEYS[:-1]) + (hand,)


# ---------------------------------------------------------------------------
# Reference signal: axis and dominant frequency
# ---------------------------------------------------------------------------

def dominant_axis(X_list):
    """Y-oriented dominant axis of one array or of several pooled, per-array-centred arrays."""
    return dominant_reference_axis(X_list)


def periodogram_frequency(ref, fs=FS_HZ):
    """The package's rule before the fix: largest periodogram bin at any positive frequency."""
    freqs, power = periodogram(ref, fs=fs)
    valid = freqs > 0
    return float(freqs[valid][np.argmax(power[valid])])


def _detrended_spectrum(ref, fs, nfft):
    ref = np.asarray(ref, dtype=float)
    t = np.arange(len(ref))
    ref = ref - np.polyval(np.polyfit(t, ref, 1), t)
    return periodogram(ref, fs=fs, window="hann", nfft=max(nfft, len(ref)))


def robust_frequency(ref, fs=FS_HZ):
    """The package's rule after the fix (``estimate_dominant_period``)."""
    return 1.0 / estimate_dominant_period(ref, fs)


def pooled_frequency(refs, fs=FS_HZ, min_cycles=3.0, f_max=5.0, nfft=4096):
    """
    One dominant frequency for several records of the same movement.

    The detrended, tapered spectra are summed weighted by record length, and
    the largest value is taken among frequencies that fit at least
    ``min_cycles`` cycles into every record and lie below ``f_max``.
    """
    total = None
    f_low = 0.0
    for ref in refs:
        freqs, power = _detrended_spectrum(ref, fs, nfft)
        total = power * len(ref) if total is None else total + power * len(ref)
        f_low = max(f_low, min_cycles * fs / len(ref))
    candidates = (freqs >= f_low) & (freqs <= f_max)
    if not candidates.any():
        raise ValueError("Records too short for the requested min_cycles / f_max.")
    return float(freqs[candidates][np.argmax(total[candidates])])


def shared_reference(X_list, fs=FS_HZ):
    """One axis and one dominant frequency for several runs of the same movement."""
    axis = dominant_axis(X_list)
    refs = [(X - X.mean(axis=0)) @ axis for X in X_list]
    # The lower search limit comes from the shortest run; use the median run
    # length instead so that one short run does not raise it for everyone.
    median_len = int(np.median([len(r) for r in refs]))
    usable = [r for r in refs if len(r) >= median_len] or refs
    return axis, pooled_frequency(usable, fs)


# ---------------------------------------------------------------------------
# Segmentation prototype
# ---------------------------------------------------------------------------

def segment_cycles(
    X,
    fs=FS_HZ,
    *,
    anchor="absolute",
    frequency="robust",
    axis=None,
    f0=None,
):
    """
    Segment one run into cycles.

    Parameters
    ----------
    anchor : {"first_peak", "absolute"}
        ``"first_peak"`` is the package's rule before the fix: subtract the phase at
        the first peak of the band-passed reference, so one event fixes phase
        zero for the whole run.  ``"absolute"`` uses the Hilbert phase as it
        is: phase zero is the peak of the band-passed reference in every cycle.
    frequency : {"periodogram", "robust"}
        How the dominant frequency is estimated when ``f0`` is not given.
    axis, f0 : optional
        A reference axis and dominant frequency estimated elsewhere, e.g. from
        all runs of a condition with ``shared_reference``.

    Returns
    -------
    dict with ``cycles`` (one row per cycle, including ``complete`` and the
    per-cycle quality columns), ``phase`` (unwrapped, anchored), ``ref``,
    ``filtered``, ``amplitude``, ``band``, ``axis``, ``f0`` and
    ``anchor_error`` (radians; Hilbert phase at the sample used as the anchor,
    0 for ``anchor="absolute"``).
    """
    X = np.asarray(X, dtype=float)
    if axis is None:
        axis = dominant_axis(X)
    ref = (X - X.mean(axis=0)) @ axis
    if f0 is None:
        f0 = periodogram_frequency(ref, fs) if frequency == "periodogram" else robust_frequency(ref, fs)
    band = (0.5 * f0, min(2.0 * f0, np.nextafter(fs / 2.0, 0.0)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        phase, _, amplitude = hilbert_phase(ref, fs, band)
    filtered = _bandpass_reference_signal(ref, fs, band)

    n = len(X)
    anchor_error = 0.0
    if anchor == "first_peak":
        peaks, _ = find_peaks(filtered)
        anchor_error = float(np.angle(np.exp(1j * phase[peaks[0]])))
        phase = phase - phase[peaks[0]]
        monotone = phase
        step = 0.0
    elif anchor == "absolute":
        # A cycle is the stretch between two successive upward crossings of a
        # multiple of 2*pi.  Taking the running maximum first makes every
        # cycle one contiguous block even where the phase briefly runs back.
        monotone = np.maximum.accumulate(phase)
        # An end cycle counts as complete when its outer sample is within one
        # typical phase step of the boundary.
        step = float(np.median(np.abs(np.diff(monotone))))
    else:
        raise ValueError(f"Unknown anchor {anchor!r}.")
    cycle_id = np.floor(monotone / (2 * np.pi)).astype(int)
    in_cycle = np.mod(monotone, 2 * np.pi)
    backward = np.r_[False, np.diff(phase) < 0]

    rows = []
    for cyc in np.unique(cycle_id):
        idx = np.flatnonzero(cycle_id == cyc)
        if len(idx) < 10:  # the package's min_samples_per_cycle default
            continue
        start, stop = int(idx.min()), int(idx.max()) + 1
        if anchor == "absolute":
            complete = (start > 0 or in_cycle[start] <= step) and (
                stop < n or 2 * np.pi - in_cycle[stop - 1] <= step
            )
        else:
            complete = start > 0 and stop < n
        rows.append({
            "cycle": int(cyc),
            "sample_start": start,
            "sample_stop": stop,
            "n_samples": len(idx),
            "duration": (stop - start) / fs,
            "contiguous": len(idx) == stop - start,
            "complete": bool(complete),
            "amplitude": float(amplitude[idx].mean()),
            "backward_steps": int(backward[idx].sum()),
        })
    cycles = pd.DataFrame(rows)
    if len(cycles):
        inner = cycles[cycles["complete"]] if cycles["complete"].any() else cycles
        cycles["duration_ratio"] = cycles["duration"] / inner["duration"].median()
        cycles["amplitude_ratio"] = cycles["amplitude"] / inner["amplitude"].median()
    return {
        "cycles": cycles, "phase": phase, "ref": ref, "filtered": filtered,
        "amplitude": amplitude, "band": band, "axis": axis, "f0": f0,
        "anchor_error": anchor_error,
    }


def package_cycles(X, fs=FS_HZ, ref_axis=None):
    """Cycles exactly as the installed ``fit_pca_phase_coordinates`` returns them."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        samples, cycles, details = fit_pca_phase_coordinates(
            X, sampling_rate_hz=fs, ref_axis=ref_axis
        )
    return samples, cycles, details


VARIANTS = {
    # label: (segment_cycles keyword arguments, keep only complete cycles, what the condition's runs share)
    "A. before the fix": (dict(anchor="first_peak", frequency="periodogram"), False, ()),
    "B. A + drop partial cycles": (dict(anchor="first_peak", frequency="periodogram"), True, ()),
    "C. B + phase zero at every peak": (dict(anchor="absolute", frequency="periodogram"), True, ()),
    "D. C + robust frequency": (dict(anchor="absolute", frequency="robust"), True, ()),
    "E. D + shared axis": (dict(anchor="absolute", frequency="robust"), True, ("axis",)),
}
# Tried and not recommended; kept so the notebook can show why.
REJECTED_VARIANTS = {
    "E + shared frequency": (dict(anchor="absolute", frequency="robust"), True, ("axis", "f0")),
}


def segment_all(positions, runs, variant):
    """Apply one variant to every run and hand; return ``(cycle_table, fits)``."""
    kwargs, complete_only, shared = {**VARIANTS, **REJECTED_VARIANTS}[variant]
    fits, tables = {}, []
    for hand in HANDS:
        for _, group in runs.groupby(["subject", "skill", "day", "part"]):
            X_list = [run_xyz(positions, run, hand) for _, run in group.iterrows()]
            extra = {}
            if shared:
                reference = dict(zip(("axis", "f0"), shared_reference(X_list)))
                extra = {name: reference[name] for name in shared}
            for (_, run), X in zip(group.iterrows(), X_list):
                fit = segment_cycles(X, **kwargs, **extra)
                key = run_key(run, hand)
                fits[key] = fit
                table = fit["cycles"].copy()
                if complete_only and len(table):
                    table = table[table["complete"]]
                for name, value in zip(RUN_KEYS, key):
                    table[name] = value
                tables.append(table)
    return pd.concat(tables, ignore_index=True), fits


# ---------------------------------------------------------------------------
# Model-free metrics
# ---------------------------------------------------------------------------

def resample(values, n_points=NORMALIZED_CYCLE_POINTS):
    values = np.asarray(values, dtype=float)
    old = np.linspace(0, 1, len(values))
    new = np.linspace(0, 1, n_points)
    if values.ndim == 1:
        return np.interp(new, old, values)
    return np.column_stack([np.interp(new, old, values[:, k]) for k in range(values.shape[1])])


def cycle_trajectories(positions, runs, cycle_table, *, centre="run"):
    """
    Time-normalised 3-D trajectory of every cycle, shape ``(n_cycles, 100, 3)``.

    ``centre="run"`` subtracts the mean position of the whole run, so where a
    cycle starts in space is preserved.  ``centre="cycle"`` subtracts each
    cycle's own mean, as in Noam's common-plane QC figure.
    """
    run_lookup = {run_key(run, hand)[:-1]: run for _, run in runs.iterrows() for hand in HANDS[:1]}
    cache, out = {}, []
    for row in cycle_table.itertuples():
        key = tuple(getattr(row, k) for k in RUN_KEYS)
        if key not in cache:
            cache[key] = run_xyz(positions, run_lookup[key[:-1]], row.hand)
        X = cache[key]
        segment = X[row.sample_start:row.sample_stop]
        origin = X.mean(axis=0) if centre == "run" else segment.mean(axis=0)
        out.append(resample(segment - origin))
    return np.stack(out) if out else np.empty((0, NORMALIZED_CYCLE_POINTS, 3))


def start_lags(trajectories, n_iterations=3):
    """
    How far each cycle's start is from the common start, in radians of a cycle.

    Every time-normalised cycle is treated as a closed loop and slid along
    itself until it best matches the condition's mean loop (largest 3-D
    cross-correlation).  The slide that was needed is the cycle's lag.  The
    mean loop is re-estimated from the aligned cycles a few times.  No phase
    model is involved: only the shapes of the cycles are compared.  Lags are
    reported relative to their circular mean over the cycles passed in.
    """
    centred = trajectories - trajectories.mean(axis=1, keepdims=True)
    n_points = centred.shape[1]
    spectra = np.fft.rfft(centred, axis=1)
    lags = np.zeros(len(centred), dtype=int)
    for _ in range(n_iterations):
        aligned = np.stack([np.roll(c, -lag, axis=0) for c, lag in zip(centred, lags)])
        template = np.fft.rfft(aligned.mean(axis=0), axis=0)
        correlation = np.fft.irfft(spectra * np.conj(template)[None], n=n_points, axis=1).sum(axis=2)
        lags = correlation.argmax(axis=1)
    # A cycle that had to be slid forward started early, so flip the sign:
    # positive lag = the cycle starts later than the common start.
    angle = -2 * np.pi * lags / n_points
    return wrap(angle - circular_mean(angle))


def condition_metrics(positions, runs, cycle_table):
    """
    Per condition (subject x skill x day x part x hand):

    lag_sd_deg, misaligned
        From ``start_lags``: circular SD of the start lags in degrees, and the
        fraction of cycles whose start is more than 45 degrees (an eighth of a
        cycle) from the common start.

    start_scatter
        RMS distance of the cycle-start positions from their centroid, divided
        by the RMS size of the movement (run-centred).  0 means every cycle
        starts at the same point in space; ~1.4 is what random start points on
        the trajectory give.
    shape_rms
        Noam's variability measure: RMS deviation of each time-normalised
        hand-Y cycle from the condition mean cycle, median over cycles, divided
        by the median cycle amplitude.
    odd_duration
        Fraction of cycles shorter than 0.6x or longer than 1.5x the condition
        median duration.
    """
    rows = []
    for keys, group in cycle_table.groupby(CONDITION_KEYS):
        if len(group) < 3:
            continue
        traj = cycle_trajectories(positions, runs, group)
        starts = traj[:, 0, :]
        size = np.sqrt((traj ** 2).sum(axis=2).mean())
        scatter = np.sqrt(((starts - starts.mean(axis=0)) ** 2).sum(axis=1).mean()) / size
        y = traj[:, :, 1]
        rms = np.sqrt(((y - y.mean(axis=0)) ** 2).mean(axis=1))
        amplitude = np.ptp(y, axis=1)
        duration = group["duration"].to_numpy()
        ratio = duration / np.median(duration)
        lags = start_lags(traj)
        rows.append({
            **dict(zip(CONDITION_KEYS, keys)),
            "n_cycles": len(group),
            "lag_sd_deg": float(np.degrees(np.sqrt(-2 * np.log(resultant_length(lags))))),
            "misaligned": float((np.abs(lags) > np.pi / 4).mean()),
            "start_scatter": float(scatter),
            "shape_rms": float(np.median(rms) / np.median(amplitude)),
            "odd_duration": float(((ratio < 0.6) | (ratio > 1.5)).mean()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Circular statistics
# ---------------------------------------------------------------------------

def wrap(angle):
    return np.angle(np.exp(1j * np.asarray(angle, dtype=float)))


def resultant_length(angle):
    return float(np.abs(np.mean(np.exp(1j * np.asarray(angle, dtype=float)))))


def circular_mean(angle):
    return float(np.angle(np.mean(np.exp(1j * np.asarray(angle, dtype=float)))))


# ---------------------------------------------------------------------------
# Workaround for the package before the fix (public functions only)
# ---------------------------------------------------------------------------

def fit_with_absolute_phase_zero(X, fs=FS_HZ):
    """
    Get peak-anchored, complete cycles from the package as it was before the fix.

    Not needed with the package on this branch, which does this itself; kept
    for anyone still on ``main`` at ``d83b30d``.

    The reference and band are the ones the package would choose by itself.
    The Hilbert phase is computed with the public ``hilbert_phase`` and the
    record is trimmed to run from the first to the last upward crossing of a
    multiple of 2*pi.  Passing that trimmed phase as ``phase=`` makes the
    package's "phase zero at the first sample" rule coincide with the peak of
    the reference, and leaves no partial cycle at either end.  The phase is
    also made non-decreasing so that every cycle is one contiguous block.

    Returns ``(samples, cycles, details, offset)``; add ``offset`` to
    ``sample_start`` / ``sample_stop`` to index the untrimmed run.
    """
    from phase_coordinates import dominant_reference_signal, estimate_dominant_period

    X = np.asarray(X, dtype=float)
    ref = dominant_reference_signal(X)
    period = estimate_dominant_period(ref, fs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        phase, _, _ = hilbert_phase(ref, fs, (0.5 / period, 2.0 / period))
    phase = np.maximum.accumulate(phase)
    turn = np.floor(phase / (2 * np.pi))
    crossings = np.flatnonzero(np.diff(turn) > 0) + 1
    if len(crossings) < 2:
        raise ValueError("Fewer than one complete cycle in this run.")
    first, last = int(crossings[0]), int(crossings[-1])
    trimmed = phase[first:last].copy()
    # The first kept sample is just past the crossing; pin it to the crossing
    # value so that the package's own "phase - phase[0]" is the absolute phase.
    trimmed[0] = 2 * np.pi * turn[first]
    samples, cycles, details = fit_pca_phase_coordinates(
        X[first:last], phase=trimmed, sampling_rate_hz=fs
    )
    return samples, cycles, details, first


def add_start_lags(positions, runs, cycle_table):
    """Add ``lag`` (radians, see ``start_lags``) to a cycle table, condition by condition."""
    table = cycle_table.copy()
    table["lag"] = np.nan
    for _, group in table.groupby(CONDITION_KEYS):
        if len(group) >= 3:
            table.loc[group.index, "lag"] = start_lags(cycle_trajectories(positions, runs, group))
    return table


def qc_flag(cycles):
    """The per-cycle screen suggested in the report (True = look before using)."""
    return (
        ~cycles["complete"]
        | (cycles["duration_ratio"] < 0.6) | (cycles["duration_ratio"] > 1.5)
        | (cycles["amplitude_ratio"] < 0.5)
        | (cycles["backward_steps"] > 0)
    )
