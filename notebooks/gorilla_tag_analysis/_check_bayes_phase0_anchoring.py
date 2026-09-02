import sys, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

sys.path.insert(0, str(Path.cwd()))
import _loading as L
from phase_coordinates.bayesian import (
    dominant_reference_signal, estimate_dominant_period, seed_boundary_indices,
    interp_X_at_times, robust_movement_scale,
)

FS_HZ = 60.0
CYCLE_PARAMS = {
    "sampling_frequency_hz": FS_HZ,
    "detection": {
        "walk": {"event_type_by_hand": {"r_hand": "peaks", "l_hand": "troughs"}, "smooth_window": 9, "smooth_polyorder": 3,
                 "prominence_sd_factor": 0.10, "min_event_distance_sec": 0.20},
        "jump": {"event_type_by_hand": {"r_hand": "peaks", "l_hand": "peaks"}, "smooth_window": 9, "smooth_polyorder": 3,
                 "prominence_sd_factor": 0.10, "min_event_distance_sec": 0.20},
        "climb": {"event_type_by_hand": {"r_hand": "peaks", "l_hand": "peaks"}, "smooth_window": 9, "smooth_polyorder": 3,
                  "prominence_sd_factor": 0.10, "min_event_distance_sec": 0.20},
    },
}

def smooth_signal(values, window_length, polyorder):
    values = np.asarray(values, dtype=float)
    if len(values) < 5: return values.copy()
    window_length = int(window_length) + (int(window_length) % 2 == 0)
    maximum = len(values) if len(values) % 2 else len(values) - 1
    window_length = min(window_length, maximum)
    if window_length <= polyorder or window_length < 5: return values.copy()
    return savgol_filter(values, window_length, polyorder)

def noam_events(raw, skill, hand):
    config = CYCLE_PARAMS["detection"][skill]
    smooth = smooth_signal(raw, config["smooth_window"], config["smooth_polyorder"])
    event_type = config["event_type_by_hand"][hand]
    prominence = config["prominence_sd_factor"] * np.nanstd(smooth)
    distance = max(1, round(config["min_event_distance_sec"] * FS_HZ))
    events, _ = find_peaks(smooth if event_type == "peaks" else -smooth, distance=distance, prominence=prominence)
    return events

with open("cache/bayes_results.pkl", "rb") as f:
    BAYES = pickle.load(f)

def nn_offset_stats(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.array([])
    return np.min(np.abs(a[:, None] - b[None, :]), axis=1)

rows = []
for key, entry in BAYES.items():
    subject, day, part, skill, hand = key
    meta = entry["meta"]
    cycles = entry["cycles"]
    if cycles.empty:
        continue

    X, cols, meta2 = L.load_block(subject, day, part, skill, run_index="longest", hand=hand)

    ref = dominant_reference_signal(X)
    T0 = estimate_dominant_period(ref, FS_HZ)
    tau_idx = seed_boundary_indices(ref, FS_HZ, T0)
    tau_hat = tau_idx / FS_HZ

    tau_mean = np.array(sorted(cycles["time_start"].tolist() + [cycles["time_stop"].iloc[-1]]))

    raw, _ = L.hand_y_relative_to_root(subject, day, part, skill, run_index="longest", hand=hand)
    noam_ev = noam_events(raw, skill, hand) / FS_HZ

    off_seed_vs_mean = nn_offset_stats(tau_hat, tau_mean)
    off_mean_vs_noam = nn_offset_stats(tau_mean, noam_ev)
    off_seed_vs_noam = nn_offset_stats(tau_hat, noam_ev)

    R_X, xbar = robust_movement_scale(X)
    X_at_seed = interp_X_at_times(X, FS_HZ, tau_hat)
    X_at_mean = interp_X_at_times(X, FS_HZ, tau_mean)
    spread_seed = np.linalg.norm(X_at_seed - X_at_seed.mean(axis=0), axis=1).std()
    spread_mean = np.linalg.norm(X_at_mean - X_at_mean.mean(axis=0), axis=1).std()

    rows.append({
        "skill": skill, "day": day, "hand": hand,
        "T0": round(T0, 3), "n_seed": len(tau_hat), "n_post": len(tau_mean), "n_noam": len(noam_ev),
        "med_seed_vs_postmean_s": round(np.median(off_seed_vs_mean), 3) if len(off_seed_vs_mean) else np.nan,
        "med_postmean_vs_noam_s": round(np.median(off_mean_vs_noam), 3) if len(off_mean_vs_noam) else np.nan,
        "med_seed_vs_noam_s": round(np.median(off_seed_vs_noam), 3) if len(off_seed_vs_noam) else np.nan,
        "spread_seed": round(spread_seed, 3), "spread_postmean": round(spread_mean, 3),
        "R_X": round(R_X, 2),
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(df.to_string(index=False))
