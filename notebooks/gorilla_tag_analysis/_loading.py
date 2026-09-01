"""
Shared block-loading helpers for the gorilla-tag extended analysis.

A "block" is one contiguous movement run: one row of the runs table built by
``build_runs_from_positive_windows`` (adapted from Noam's
``Labeled_data_analyses.ipynb``), i.e. one (subject, day, part, skill,
source_file, movement_run_id). Disjoint runs from the same file are never
concatenated into one signal -- that would fabricate phase discontinuities
at the seam.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_DATA_DIR = _REPO_ROOT / "test_data"

if str(_TEST_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DATA_DIR))

# config_paths.py runs `PATHS = get_project_paths()` at import time, so the
# env var must be set before the first import.
os.environ.setdefault("GORILLA_TAG_PROJECT_ROOT", str(_TEST_DATA_DIR))

import config_paths  # noqa: E402
from names_format import Repetitive  # noqa: E402  (imported for downstream reuse/validation)

PATHS = config_paths.get_project_paths()
REPETITIVE_PROCESSED_DIR = PATHS.repetitive_processed_data_dir
REPETITIVE_DATA_DIR = PATHS.repetitive_data_dir

FS_HZ = 60.0

SKILL_FROM_TASK = {"w": "walk", "j": "jump", "c": "climb"}
REPETITIVE_WINDOW_FILES = {
    "walk": REPETITIVE_DATA_DIR / "window_predictions_walk.csv",
    "jump": REPETITIVE_DATA_DIR / "window_predictions_jump.csv",
    "climb": REPETITIVE_DATA_DIR / "window_predictions_climb.csv",
}


# ---------------------------------------------------------------------------
# Adapted verbatim (logic unchanged) from Noam's Labeled_data_analyses.ipynb
# ---------------------------------------------------------------------------

def normalize_repetitive_subject(value):
    return f"{int(float(value)):03d}"


def normalize_part_code(value):
    value = str(value).strip().lower()
    base = value.split(".", 1)[0]
    return {"start": "s", "middle": "m", "end": "e"}.get(base, base)


def repetitive_pos_path(row):
    source_file = row.get("source_file")
    if pd.isna(source_file) or not str(source_file).strip():
        raise ValueError("Missing source_file; exact repetitive file resolution is required")
    return REPETITIVE_PROCESSED_DIR / f"{str(source_file).strip()}_pos_aligned.csv"


def load_position_file(path, file_cache=None):
    path = Path(path)
    if file_cache is not None:
        if path not in file_cache:
            file_cache[path] = pd.read_csv(path, low_memory=False)
        return file_cache[path]
    return pd.read_csv(path, low_memory=False)


def bridge_boolean_mask_gaps(mask, max_gap_frames=15):
    """Fill false gaps no longer than `max_gap_frames` between true runs."""
    mask = np.asarray(mask, dtype=bool).copy()
    if max_gap_frames <= 0 or not mask.any():
        return mask
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = changes[::2], changes[1::2]
    for left_end, right_start in zip(ends[:-1], starts[1:]):
        if 0 < right_start - left_end <= max_gap_frames:
            mask[left_end:right_start] = True
    return mask


def mask_to_runs(mask, min_run_frames=1):
    """Convert a boolean mask to exclusive-end runs meeting a minimum length."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2]) if b - a >= min_run_frames]


def count_overlapping_windows(group):
    ordered = group.sort_values(["start", "end"])
    max_end, overlaps = -1, 0
    for row in ordered.itertuples():
        if int(row.start) < max_end:
            overlaps += 1
        max_end = max(max_end, int(row.end))
    return overlaps


def build_runs_from_positive_windows(rep_df, parameters):
    """Union overlapping positive windows by exact source file, bridge short
    gaps, and return run/QC tables. Logic identical to Noam's version; only
    the path/column wiring differs (repetitive_pos_path uses config_paths)."""
    selected = rep_df.copy()
    wanted = {(int(day), normalize_part_code(part)) for day, part in parameters["day_part_map"].items()}
    selected = selected[selected.apply(lambda row: (int(row["day"]), normalize_part_code(row["part"])) in wanted, axis=1)]
    if parameters.get("subjects"):
        selected = selected[selected["subject_str"].isin([normalize_repetitive_subject(x) for x in parameters["subjects"]])]
    selected = selected[selected["skill"].isin(parameters["skills"])]
    run_rows, qc_rows, skipped = [], [], []
    file_cache = {}
    for keys, group in selected.groupby(["subject_str", "day", "part", "skill", "source_file"], dropna=False):
        subject, day, part, skill, source_file = keys
        path = repetitive_pos_path(group.iloc[0])
        base = {"subject": subject, "day": int(day), "part": normalize_part_code(part), "skill": skill,
                "source_file": source_file, "file": path.name}
        if not path.exists():
            qc_rows.append({**base, "file_exists": False, "n_positive_windows": len(group)})
            skipped.append({**base, "error": "missing_file"})
            continue
        position_df = load_position_file(path, file_cache)
        n_frames = len(position_df)
        mask = np.zeros(n_frames, dtype=bool)
        truncated = invalid = 0
        for row in group.itertuples():
            start, end = max(0, int(row.start)), min(n_frames, int(row.end))
            truncated += int(start != int(row.start) or end != int(row.end))
            if end <= start:
                invalid += 1
            else:
                mask[start:end] = True
        raw_runs = mask_to_runs(mask)
        bridged = bridge_boolean_mask_gaps(mask, parameters["max_gap_to_bridge_frames"])
        min_frames = round(parameters["min_run_duration_sec"] * FS_HZ)
        runs = mask_to_runs(bridged, min_frames)
        for run_id, (start, end) in enumerate(runs, 1):
            run_rows.append({**base, "path": str(path), "movement_run_id": run_id, "run_start": start,
                             "run_end": end, "run_n_frames": end - start, "run_duration_sec": (end - start) / FS_HZ})
        qc_rows.append({**base, "file_exists": True, "n_file_frames": n_frames,
                        "n_positive_windows": len(group), "n_positive_frames_union": int(mask.sum()),
                        "n_raw_runs": len(raw_runs), "n_movement_runs": len(runs),
                        "n_truncated_windows": truncated, "n_invalid_windows": invalid,
                        "overlapping_positive_windows": count_overlapping_windows(group),
                        "max_gap_to_bridge_frames": parameters["max_gap_to_bridge_frames"], "min_run_frames": min_frames})
    run_columns = ["subject", "day", "part", "skill", "source_file", "file", "path", "movement_run_id",
                   "run_start", "run_end", "run_n_frames", "run_duration_sec"]
    runs_df = pd.DataFrame(run_rows, columns=run_columns)
    if len(runs_df):
        runs_df = runs_df.sort_values(["subject", "skill", "day", "part", "file", "movement_run_id"]).reset_index(drop=True)
    return runs_df, pd.DataFrame(qc_rows), pd.DataFrame(skipped)


# ---------------------------------------------------------------------------
# New: rep_df construction (Noam's own cell wasn't captured, built directly
# from the raw window_predictions_{walk,jump,climb}.csv files -- their
# columns are: subject, day, session_part, task, source_file, window_id,
# start_frame, end_frame, probability, prediction)
# ---------------------------------------------------------------------------

DEFAULT_RUN_PARAMS = {
    "day_part_map": {1: "s", 3: "e"},
    "subjects": ("002",),
    "skills": ("walk", "jump", "climb"),
    "max_gap_to_bridge_frames": 15,
    "min_run_duration_sec": 3.0,
}


def _load_rep_df():
    frames = []
    for skill, path in REPETITIVE_WINDOW_FILES.items():
        df = pd.read_csv(path)
        df = df[df["prediction"] == 1].copy()
        df["subject_str"] = df["subject"].map(normalize_repetitive_subject)
        df["part"] = df["session_part"].map(normalize_part_code)
        df["skill"] = skill
        df = df.rename(columns={"start_frame": "start", "end_frame": "end"})
        frames.append(df[["subject_str", "day", "part", "skill", "source_file", "start", "end"]])
    return pd.concat(frames, ignore_index=True)


_REP_DF_CACHE = None
_RUNS_CACHE = None


def _get_runs(parameters=None):
    global _REP_DF_CACHE, _RUNS_CACHE
    parameters = parameters or DEFAULT_RUN_PARAMS
    if _REP_DF_CACHE is None:
        _REP_DF_CACHE = _load_rep_df()
    if _RUNS_CACHE is None:
        runs_df, qc_df, skipped_df = build_runs_from_positive_windows(_REP_DF_CACHE, parameters)
        _RUNS_CACHE = (runs_df, qc_df, skipped_df)
    return _RUNS_CACHE


def list_available_blocks(subject=None, parameters=None):
    """Return the runs table (one row per contiguous movement block)."""
    runs_df, _, _ = _get_runs(parameters)
    if subject is not None:
        runs_df = runs_df[runs_df["subject"] == normalize_repetitive_subject(subject)]
    return runs_df.reset_index(drop=True)


def get_qc_table(parameters=None):
    _, qc_df, _ = _get_runs(parameters)
    return qc_df


def load_block(subject, day, part, skill, run_index=0, hand="r_hand", parameters=None):
    """
    Load one full contiguous block (not truncated).

    NOTE: for a given (subject, day, part, skill) there are typically MANY
    disjoint positive-movement runs (bouts of a few to ~30s each), not one
    long continuous block -- see PROGRESS.md. `run_index` selects among them
    (0 = chronologically first); pass run_index="longest" to get the longest
    run for that condition instead. Use `list_available_blocks` to see all
    of them (e.g. for within-condition variability across bouts).

    Returns (X, columns_used, meta) where X has shape (n, 3) = hand xyz
    relative to root, columns_used = [f"{hand}.x", ...], and meta is the
    matching row of the runs table as a dict (includes run_start/run_end/
    run_duration_sec/path/source_file/...).
    """
    part = normalize_part_code(part)
    subject = normalize_repetitive_subject(subject)
    runs_df = list_available_blocks(subject=subject, parameters=parameters)
    match = runs_df[
        (runs_df["day"] == int(day))
        & (runs_df["part"] == part)
        & (runs_df["skill"] == skill)
    ].reset_index(drop=True)
    if match.empty:
        raise ValueError(f"No block found for subject={subject} day={day} part={part} skill={skill}")
    if run_index == "longest":
        row = match.loc[match["run_duration_sec"].idxmax()]
    else:
        if run_index >= len(match):
            raise ValueError(
                f"Only {len(match)} run(s) available for subject={subject} day={day} "
                f"part={part} skill={skill}; run_index={run_index} out of range"
            )
        row = match.iloc[run_index]
    position_df = load_position_file(row["path"])
    seg = position_df.iloc[int(row["run_start"]):int(row["run_end"])]
    cols = [f"{hand}.x", f"{hand}.y", f"{hand}.z"]
    root_cols = ["root.x", "root.y", "root.z"]
    X = seg[cols].to_numpy(dtype=float) - seg[root_cols].to_numpy(dtype=float)
    return X, cols, row.to_dict()


def hand_y_relative_to_root(subject, day, part, skill, run_index=0, hand="r_hand", parameters=None):
    """Scalar reference signal in Noam's convention: hand.y - root.y, for the full block."""
    X, cols, meta = load_block(subject, day, part, skill, run_index=run_index, hand=hand, parameters=parameters)
    y_idx = cols.index(f"{hand}.y")
    return X[:, y_idx], meta
