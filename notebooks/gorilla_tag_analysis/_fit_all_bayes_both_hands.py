"""Fit all 6 primary blocks x 2 hands with the FIXED fit_bayesian_phase_coordinates,
combine into one cache file for the 3-way comparison notebook."""
import sys, time, pickle
sys.path.insert(0, ".")
from _loading import load_block
from phase_coordinates import fit_bayesian_phase_coordinates

SPECS = [
    (1, "s", "walk"), (3, "e", "walk"),
    (1, "s", "jump"), (3, "e", "jump"),
    (1, "s", "climb"), (3, "e", "climb"),
]
HANDS = ["r_hand", "l_hand"]

results = {}
for day, part, skill in SPECS:
    for hand in HANDS:
        tag = f"{skill}_day{day}{part}_{hand[0]}"
        t0 = time.time()
        X, cols, meta = load_block(subject="002", day=day, part=part, skill=skill,
                                    run_index="longest", hand=hand)
        try:
            samples, cycles, details = fit_bayesian_phase_coordinates(
                X, sampling_rate_hz=60, draws=300, tune=300, chains=2, random_seed=0,
            )
            elapsed = time.time() - t0
            results[("002", day, part, skill, hand)] = {
                "meta": meta, "samples": samples, "cycles": cycles,
                "diagnostics": details.get("diagnostics"), "elapsed_sec": elapsed,
                "X_shape": X.shape, "error": None,
            }
            print(f"[{tag}] SUCCESS elapsed={elapsed:.1f}s n_cycles={len(cycles)}", flush=True)
        except Exception as exc:
            elapsed = time.time() - t0
            results[("002", day, part, skill, hand)] = {
                "meta": meta, "error": repr(exc), "elapsed_sec": elapsed, "X_shape": X.shape,
            }
            print(f"[{tag}] FAILED elapsed={elapsed:.1f}s error={exc!r}", flush=True)
        # save incrementally so progress is visible/recoverable
        with open("cache/bayes_results_both_hands.pkl", "wb") as f:
            pickle.dump(results, f)

print("ALL DONE", flush=True)
