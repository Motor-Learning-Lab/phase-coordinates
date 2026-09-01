# --- Imports ---
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks, savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# --- Module Constants: Analysis Version and Subject Lists ---
ANALYSIS_VERSION = "labeled_data_v2"
FREE_SUBJECTS = ["110", "111", "112", "113", "114", "115", "116"]
REPETITIVE_SUBJECTS = ["002", "004", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015"]
SKILLS = ["walk", "jump", "climb"]
DAYS = [1, 3]
PARTS = ["s", "m", "e"]
HANDS = ["r_hand", "l_hand"]
DAY_PART_FOR_FINAL = {1: "s", 3: "e"}
FS_HZ = 60.0
DT = 1.0 / FS_HZ
RANDOM_SEED = 20260818


# --- Module Constants: Labels, Colors, and Markers ---
SKILL_LABELS = {"walk": "Walking", "jump": "Jumping", "climb": "Climbing"}
SKILL_COLORS = {"walk": "#1b9e77", "jump": "#d95f02", "climb": "#7570b3"}
HAND_COLORS = {"r_hand": "#D55E00", "l_hand": "#6A5ACD"}
HAND_LABELS = {"r_hand": "Right hand", "l_hand": "Left hand"}
DAY_LABELS = {1: "Day 1", 3: "Day 3"}
PART_LABELS = {"s": "start", "m": "middle", "e": "end"}
DAY_MARKERS = {1: "o", 3: "s"}
HAND_MARKERS = {"r_hand": "o", "l_hand": "^"}
DAY_LINESTYLES = {1: "--", 3: "-"}
GROUP_MARKERS = {"free": "o", "repetitive": "s"}
GROUP_LABELS = {"free": "Free practice", "repetitive": "Repetitive practice"}


# --- Module Constants: File Paths and Data Directories ---
# Note: paths object is from config_paths module and assumed to be in scope
# FREE_LABELS_PATH = paths.data_dir / "free_practice_labled_frames.csv"
# REPETITIVE_WINDOW_FILES = {
#     skill: paths.repetitive_data_dir / f"window_predictions_{skill}.csv"
#     for skill in SKILLS
# }
# FREE_PROCESSED_DIR = paths.processed_data_dir
# REPETITIVE_PROCESSED_DIR = paths.repetitive_processed_data_dir
# ANALYSIS_ROOT = paths.img_dir / "labeled_data_analysis"

REPETITIVE_WINDOW_FILES = {
    skill: f"window_predictions_{skill}.csv"
    for skill in SKILLS
}


# --- Module Constants: OUTPUT_DIRS Structure ---
# OUTPUT_DIRS = {}
# for analysis in ["trajectories", "pca", "smoothness", "cycles", "covariance"]:
#     base = ANALYSIS_ROOT / analysis
#     OUTPUT_DIRS[analysis] = {
#         "root": base,
#         "cache": base / "cache",
#         "tables": base / "tables",
#         "qc": base / "figures" / "qc",
#         "report": base / "figures" / "report",
#     }
# for mapping in OUTPUT_DIRS.values():
#     for directory in mapping.values():
#         directory.mkdir(parents=True, exist_ok=True)
# (ANALYSIS_ROOT / "tables").mkdir(parents=True, exist_ok=True)


# --- Module Constants: Position Points ---
POSITION_POINTS = [
    "root", "head", "neck_1", "neck_2",
    "torso_1", "torso_2", "torso_3", "torso_4", "torso_5", "torso_6", "torso_7",
    "l_shoulder", "r_shoulder", "l_up_arm", "r_up_arm", "l_low_arm", "r_low_arm",
    "l_hand", "r_hand",
]
POSITION_FEATURES = [f"{point}.{axis}" for point in POSITION_POINTS for axis in "xyz"]
SMOOTHNESS_POINTS = [
    "root", "head", "l_shoulder", "r_shoulder", "l_up_arm", "r_up_arm",
    "l_low_arm", "r_low_arm", "l_hand", "r_hand",
    "torso_1", "torso_2", "torso_3", "torso_4", "torso_5", "torso_6", "torso_7",
]


# --- build_subject_color_table ---
def build_subject_color_table():
    configured = [("free", s) for s in FREE_SUBJECTS] + [("repetitive", s) for s in REPETITIVE_SUBJECTS]
    palette = list(plt.get_cmap("tab20").colors)
    if len(configured) > len(palette):
        palette = [plt.get_cmap("turbo")(x) for x in np.linspace(0.03, 0.97, len(configured))]
    rows = []
    for index, ((group, subject), rgba) in enumerate(zip(configured, palette)):
        rows.append({
            "canonical_subject": f"{group}:{subject}", "group": group, "subject": subject,
            "color": plt.matplotlib.colors.to_hex(rgba), "palette_index": index,
        })
    return pd.DataFrame(rows)


SUBJECT_COLOR_TABLE = build_subject_color_table()
SUBJECT_COLOR_MAP = SUBJECT_COLOR_TABLE.set_index("canonical_subject")["color"].to_dict()

RESERVED_NON_SUBJECT_COLORS = {
    plt.matplotlib.colors.to_hex(color)
    for color in [
        *SKILL_COLORS.values(), *HAND_COLORS.values(),
        "#4daf4a", "#e41a1c", "#4C78A8", "#F58518", "#E45756",
        "#B279A2", "#9D755D", "#FF9DA6", "black", "0.25", "0.6", "0.65",
    ]
}


# --- canonical_subject ---
def canonical_subject(subject, group="repetitive"):
    """Return the stable `(group, subject)` display key used by the color system."""
    subject = str(subject)
    if group == "repetitive":
        subject = f"{int(float(subject)):03d}"
    elif group == "free":
        subject = str(int(float(subject)))
    return f"{group}:{subject}"


# --- get_subject_color ---
def get_subject_color(subject, group="repetitive"):
    """Return the deterministic color for one configured subject."""
    key = canonical_subject(subject, group)
    if key not in SUBJECT_COLOR_MAP:
        raise KeyError(f"Subject is not configured: {key}")
    return SUBJECT_COLOR_MAP[key]


# --- normalize_skill_label ---
def normalize_skill_label(label):
    return str(label).strip().lower()


# --- normalize_repetitive_subject ---
def normalize_repetitive_subject(value):
    return f"{int(float(value)):03d}"


# --- normalize_free_subject ---
def normalize_free_subject(value):
    return str(int(float(value)))


# --- normalize_part_code ---
def normalize_part_code(value):
    value = str(value).strip().lower()
    base = value.split(".", 1)[0]
    return {"start": "s", "middle": "m", "end": "e"}.get(base, base)


# --- parse_free_session ---
def parse_free_session(session):
    if pd.isna(session):
        raise ValueError("session is missing")
    value = str(session).strip()
    if "." in value:
        base, part = value.split(".", 1)
        return str(int(float(base))), (part if part and part != "0" else None)
    return str(int(float(value))), None


# --- free_pos_path ---
def free_pos_path(row):
    session, part = parse_free_session(row["session"])
    name = Current.csv_name(
        normalize_free_subject(row["subject"]), str(int(row["day"])), session,
        kind="pos", aligned=True, part=part, validate=False,
    )
    return FREE_PROCESSED_DIR / name


# --- repetitive_pos_path ---
def repetitive_pos_path(row):
    source_file = row.get("source_file")
    if pd.isna(source_file) or not str(source_file).strip():
        raise ValueError("Missing source_file; exact repetitive file resolution is required")
    return REPETITIVE_PROCESSED_DIR / f"{str(source_file).strip()}_pos_aligned.csv"


# --- segment_path ---
def segment_path(group, row):
    if group == "free":
        return free_pos_path(row)
    if group == "repetitive":
        return repetitive_pos_path(row)
    raise ValueError(f"Unknown group: {group}")


# --- load_position_file ---
def load_position_file(path, file_cache=None):
    path = Path(path)
    if file_cache is not None:
        if path not in file_cache:
            file_cache[path] = pd.read_csv(path, low_memory=False)
        return file_cache[path]
    return pd.read_csv(path, low_memory=False)


# --- load_segment ---
def load_segment(group, row, columns=None, file_cache=None):
    """Load one labeled segment; returns `(data, diagnostics)` with an exclusive end index."""
    path = segment_path(group, row)
    diag = {"group": group, "subject": row.get("subject_str", row.get("subject")),
            "skill": row.get("skill"), "day": row.get("day"), "file": path.name, "error": ""}
    if not path.exists():
        diag["error"] = "missing_file"
        return None, diag
    frame_df = load_position_file(path, file_cache=file_cache)
    requested_start, requested_end = int(row["start"]), int(row["end"])
    start, end = max(0, requested_start), min(len(frame_df), requested_end)
    diag.update({"requested_frames": requested_end - requested_start, "loaded_frames": max(0, end - start),
                 "truncated": bool(start != requested_start or end != requested_end)})
    if end <= start:
        diag["error"] = "invalid_or_empty_window"
        return None, diag
    segment = frame_df.iloc[start:end].copy()
    if columns is not None:
        missing = [column for column in columns if column not in segment.columns]
        if missing:
            diag["error"] = "missing_columns:" + ",".join(missing[:8])
            return None, diag
        segment = segment[list(columns)].copy()
    return segment.reset_index(drop=True), diag


# --- file_fingerprint ---
def file_fingerprint(path):
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {"path": str(path.resolve()), "exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


# --- cache_signature ---
def cache_signature(analysis, parameters, inputs):
    payload = {"analysis": analysis, "version": ANALYSIS_VERSION, "parameters": parameters,
               "inputs": [file_fingerprint(path) for path in inputs]}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


# --- atomic_pickle_dump ---
def atomic_pickle_dump(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent, suffix=".tmp") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


# --- atomic_npz ---
def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent, suffix=".tmp") as handle:
        np.savez_compressed(handle, **arrays)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


# --- atomic_json ---
def atomic_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, suffix=".tmp", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


# --- cache_is_valid ---
def cache_is_valid(manifest_path, expected_signature):
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        return manifest.get("signature") == expected_signature
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False


# --- write_manifest ---
def write_manifest(path, analysis, signature, payload, outputs):
    atomic_json({"analysis": analysis, "analysis_version": ANALYSIS_VERSION, "signature": signature,
                 "created_utc": datetime.now(timezone.utc).isoformat(), "compatibility": payload,
                 "outputs": [str(Path(item)) for item in outputs]}, path)


# --- save_figure ---
def save_figure(fig, directory, stem, pdf=True):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{ANALYSIS_VERSION}_{stem}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    outputs = [png]
    if pdf:
        pdf_path = directory / f"{ANALYSIS_VERSION}_{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        outputs.append(pdf_path)
    return outputs


# --- equal_limits ---
def equal_limits(*series, padding_fraction=0.08, positive=False, log_scale=False):
    """Return shared limits, optionally padding in log10 space for positive log axes."""
    values = np.concatenate([np.asarray(value, dtype=float).ravel() for value in series])
    values = values[np.isfinite(values)]
    if positive or log_scale:
        values = values[values > 0]
    if not len(values):
        return None
    if log_scale:
        log_values = np.log10(values)
        lo, hi = float(log_values.min()), float(log_values.max())
        pad = padding_fraction * (hi - lo if hi > lo else 1.0)
        return 10 ** (lo - pad), 10 ** (hi + pad)
    lo, hi = float(values.min()), float(values.max())
    pad = padding_fraction * (hi - lo if hi > lo else max(abs(hi), 1.0))
    return lo - pad, hi + pad


# --- referenced_position_paths ---
def referenced_position_paths(label_df, group):
    """Return unique aligned-position paths referenced by a label table."""
    return sorted({segment_path(group, row) for _, row in label_df.iterrows()}, key=str)


# --- bridge_boolean_mask_gaps ---
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


# --- mask_to_runs ---
def mask_to_runs(mask, min_run_frames=1):
    """Convert a boolean mask to exclusive-end runs meeting a minimum length."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.r_[False, mask, False]
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(a), int(b)) for a, b in zip(changes[::2], changes[1::2]) if b - a >= min_run_frames]


# --- count_overlapping_windows ---
def count_overlapping_windows(group):
    ordered = group.sort_values(["start", "end"])
    max_end, overlaps = -1, 0
    for row in ordered.itertuples():
        if int(row.start) < max_end:
            overlaps += 1
        max_end = max(max_end, int(row.end))
    return overlaps


# --- build_runs_from_positive_windows ---
def build_runs_from_positive_windows(rep_df, parameters):
    """Union overlapping positive windows by exact source file, bridge short gaps, and return run/QC tables."""
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


# --- load_free_labeled_frames ---
def load_free_labeled_frames():
    df = pd.read_csv(FREE_LABELS_PATH)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")].copy()
    df["subject_str"] = df["subject"].map(normalize_free_subject)
    df["skill"] = df["label"].map(normalize_skill_label)
    df["day"] = df["day"].astype(int)
    return df[df["subject_str"].isin(FREE_SUBJECTS) & df["skill"].isin(SKILLS) & df["day"].isin(DAYS)].reset_index(drop=True)


# --- TRAJECTORY_PARAMS ---
TRAJECTORY_PARAMS = {
    "subject": "015",
    "day_part_map": {1: "s", 3: "e"},
    "skills": tuple(SKILLS),
    "joints": ("root", "head", "l_hand", "r_hand"),
    "step": 1,
    "max_gap_to_bridge_frames": 15,
    "min_run_duration_sec": 3.0,
}
