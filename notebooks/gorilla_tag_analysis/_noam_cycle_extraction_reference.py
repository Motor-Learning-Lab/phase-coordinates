# --- CYCLE_PARAMS ---
CYCLE_PARAMS = {
    "sampling_frequency_hz": FS_HZ, "max_gap_to_bridge_frames": 15, "min_run_duration_sec": 3.0,
    "normalized_cycle_points": 100, "min_cycles_per_subject_hand": 3, "candidate_diagnostics_version": 1,
    "subjects": tuple(REPETITIVE_SUBJECTS), "skills": tuple(SKILLS), "hands": tuple(HANDS),
    "day_part_map": {1: "s", 3: "e"},
    "detection": {
        "walk": {"event_type_by_hand": {"r_hand": "peaks", "l_hand": "troughs"}, "smooth_window": 9, "smooth_polyorder": 3,
                 "prominence_sd_factor": 0.10, "min_event_distance_sec": 0.20, "min_cycle_duration_sec": 0.50,
                 "max_cycle_duration_sec": 3.10, "min_cycle_amplitude": 10},
        "jump": {"event_type_by_hand": {"r_hand": "peaks", "l_hand": "peaks"}, "smooth_window": 9, "smooth_polyorder": 3,
                 "prominence_sd_factor": 0.10, "min_event_distance_sec": 0.20, "min_cycle_duration_sec": 0.40,
                 "max_cycle_duration_sec": 1.50, "min_cycle_amplitude": None},
        "climb": {"event_type_by_hand": {"r_hand": "peaks", "l_hand": "peaks"}, "smooth_window": 9, "smooth_polyorder": 3,
                  "prominence_sd_factor": 0.10, "min_event_distance_sec": 0.20, "min_cycle_duration_sec": 0.40,
                  "max_cycle_duration_sec": 1.70, "min_cycle_amplitude": None},
    },
}

# --- compute_hand_y_relative_to_root ---
def compute_hand_y_relative_to_root(frame, hand):
    required = [f"{hand}.y", "root.y"]
    missing = [column for column in required if column not in frame]
    if missing: raise ValueError(f"Missing columns: {missing}")
    return (pd.to_numeric(frame[required[0]], errors="coerce") - pd.to_numeric(frame[required[1]], errors="coerce")).to_numpy(dtype=float)

# --- smooth_signal ---
def smooth_signal(values, window_length, polyorder):
    values = np.asarray(values, dtype=float)
    if len(values) < 5: return values.copy()
    window_length = int(window_length) + (int(window_length) % 2 == 0)
    maximum = len(values) if len(values) % 2 else len(values) - 1
    window_length = min(window_length, maximum)
    if window_length <= polyorder or window_length < 5: return values.copy()
    return savgol_filter(values, window_length, polyorder)

# --- extract_cycles_from_signal ---
def extract_cycles_from_signal(raw, skill, hand, parameters):
    config = parameters["detection"][skill]; fps = parameters["sampling_frequency_hz"]
    smooth = smooth_signal(raw, config["smooth_window"], config["smooth_polyorder"])
    event_type = config["event_type_by_hand"][hand]
    prominence = config["prominence_sd_factor"] * np.nanstd(smooth)
    distance = max(1, round(config["min_event_distance_sec"] * fps))
    events, _ = find_peaks(smooth if event_type == "peaks" else -smooth, distance=distance, prominence=prominence)
    rejected = {"too_short": 0, "too_long": 0, "nan": 0, "low_amplitude": 0, "resample_failed": 0}
    cycles, candidates = [], []
    for cycle_id, (start, end) in enumerate(zip(events[:-1], events[1:]), 1):
        frames = int(end - start); duration = frames / fps
        candidate = {"cycle_id_in_run": cycle_id, "start_idx_in_run": int(start), "end_idx_in_run": int(end),
                     "duration_frames": frames, "duration_sec": duration, "rejection_reason": "accepted"}
        if duration < config["min_cycle_duration_sec"]:
            rejected["too_short"] += 1; candidate["rejection_reason"] = "too_short"; candidates.append(candidate); continue
        if duration > config["max_cycle_duration_sec"]:
            rejected["too_long"] += 1; candidate["rejection_reason"] = "too_long"; candidates.append(candidate); continue
        raw_cycle, smooth_cycle = raw[start:end + 1], smooth[start:end + 1]
        if not np.isfinite(smooth_cycle).all():
            rejected["nan"] += 1; candidate["rejection_reason"] = "nan"; candidates.append(candidate); continue
        amplitude = float(np.ptp(smooth_cycle))
        candidate["amplitude"] = amplitude
        if config["min_cycle_amplitude"] is not None and amplitude < config["min_cycle_amplitude"]:
            rejected["low_amplitude"] += 1; candidate["rejection_reason"] = "low_amplitude"; candidates.append(candidate); continue
        normalized = np.interp(np.linspace(0, 1, parameters["normalized_cycle_points"]), np.linspace(0, 1, len(smooth_cycle)), smooth_cycle)
        if not np.isfinite(normalized).all():
            rejected["resample_failed"] += 1; candidate["rejection_reason"] = "resample_failed"; candidates.append(candidate); continue
        candidates.append(candidate)
        cycles.append({"cycle_id_in_run": cycle_id, "start_idx_in_run": int(start), "end_idx_in_run": int(end),
                       "duration_frames": frames, "duration_sec": duration, "amplitude": amplitude,
                       "event_type": event_type, "y_cycle_raw": raw_cycle, "y_cycle_smooth": smooth_cycle, "y_cycle_norm": normalized})
    return cycles, candidates, smooth, events, prominence, rejected

# --- extract_cycles_from_runs ---
def extract_cycles_from_runs(runs, parameters):
    cycles, candidates, diagnostics, skipped = [], [], [], []
    file_cache = {}
    for run in runs.itertuples():
        try:
            frame = load_position_file(run.path, file_cache).iloc[run.run_start:run.run_end].copy()
        except Exception as exc:
            skipped.append({"subject": run.subject, "skill": run.skill, "day": run.day, "part": run.part, "file": run.file, "error": str(exc)}); continue
        for hand in parameters["hands"]:
            try:
                raw = compute_hand_y_relative_to_root(frame, hand)
                extracted, candidate_records, smooth, events, prominence, rejected = extract_cycles_from_signal(raw, run.skill, hand, parameters)
                error = ""
            except Exception as exc:
                extracted, candidate_records, smooth, events, prominence, rejected = [], [], np.array([]), np.array([]), np.nan, {}
                error = str(exc); skipped.append({"subject": run.subject, "skill": run.skill, "day": run.day, "part": run.part, "file": run.file, "hand": hand, "error": error})
            diagnostic = {"subject": run.subject, "day": run.day, "part": run.part, "skill": run.skill, "file": run.file,
                          "movement_run_id": run.movement_run_id, "hand": hand, "n_events": len(events), "n_cycles": len(extracted),
                          "candidate_cycles": max(0, len(events)-1), "used_prominence": prominence, "error": error,
                          **{f"rejected_{key}": value for key, value in rejected.items()}}
            diagnostics.append(diagnostic)
            for candidate in candidate_records:
                candidates.append({"subject": run.subject, "day": int(run.day), "part": run.part, "skill": run.skill, "file": run.file,
                                   "movement_run_id": int(run.movement_run_id), "hand": hand, **candidate})
            for cycle in extracted:
                cycles.append({"subject": run.subject, "day": int(run.day), "part": run.part, "skill": run.skill, "file": run.file,
                               "path": run.path, "movement_run_id": int(run.movement_run_id), "run_start": int(run.run_start),
                               "run_end": int(run.run_end), "hand": hand,
                               "start_idx_in_file": int(run.run_start + cycle["start_idx_in_run"]),
                               "end_idx_in_file": int(run.run_start + cycle["end_idx_in_run"]), **cycle})
    return pd.DataFrame(cycles), pd.DataFrame(candidates), pd.DataFrame(diagnostics), pd.DataFrame(skipped)

# --- summarize_subject_cycles ---
def summarize_subject_cycles(cycles, parameters):
    summaries, means = [], []
    if cycles.empty: return pd.DataFrame(), pd.DataFrame()
    for keys, group in cycles.groupby(["subject", "day", "skill", "part", "hand"]):
        if len(group) < parameters["min_cycles_per_subject_hand"]: continue
        matrix = np.vstack(group.y_cycle_norm); mean = np.nanmean(matrix, axis=0)
        sd = np.nanstd(matrix, axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1])
        rms_per_cycle = np.sqrt(np.nanmean((matrix - mean) ** 2, axis=1)); mean_rms = float(np.nanmean(rms_per_cycle))
        median_amplitude = float(group.amplitude.median())
        base = dict(zip(["subject", "day", "skill", "part", "hand"], keys))
        summaries.append({**base, "n_cycles": len(group), "n_movement_runs_with_cycles": group.movement_run_id.nunique(),
                          "median_duration_sec": float(group.duration_sec.median()), "mean_duration_sec": float(group.duration_sec.mean()),
                          "median_amplitude": median_amplitude, "mean_amplitude": float(group.amplitude.mean()),
                          "mean_rms_deviation": mean_rms,
                          "normalized_rms_deviation": mean_rms / median_amplitude if median_amplitude > 1e-9 else np.nan})
        means.append({**base, "n_cycles": len(group), "mean_cycle": mean, "sd_cycle": sd,
                      "sem_cycle": sd / np.sqrt(len(group)), "normalized_time": np.linspace(0, 100, matrix.shape[1])})
    return pd.DataFrame(summaries), pd.DataFrame(means)
