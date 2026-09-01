"""Stage 6: fit one block with fit_bayesian_phase_coordinates, save results+timing to a pickle.
Run under `pixi run -e bayes python _bayes_fit_block.py <key_tag> <subject> <day> <part> <skill> <source_file> <run_id> <hand>`.
"""
import sys
import time
import pickle
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from _loading import load_block

from phase_coordinates import fit_bayesian_phase_coordinates

key_tag = sys.argv[1]
subject, day, part, skill, hand = sys.argv[2:7]
day = int(day)
draws = int(sys.argv[7]) if len(sys.argv) > 7 else 300
tune = int(sys.argv[8]) if len(sys.argv) > 8 else 300
chains = int(sys.argv[9]) if len(sys.argv) > 9 else 2

t0 = time.time()
X, cols, meta = load_block(subject=subject, day=day, part=part, skill=skill, run_index="longest", hand=hand)
print(f"[{key_tag}] loaded block: X.shape={X.shape}, cols={cols}, meta={meta}", flush=True)

try:
    samples, cycles, details = fit_bayesian_phase_coordinates(
        X, sampling_rate_hz=60, draws=300, tune=300, chains=2, random_seed=0,
    )
    elapsed = time.time() - t0
    out = {
        "key": (subject, day, part, skill, hand),
        "meta": meta,
        "samples": samples,
        "cycles": cycles,
        "diagnostics": details.get("diagnostics"),
        "elapsed_sec": elapsed,
        "X_shape": X.shape,
        "error": None,
    }
    print(f"[{key_tag}] SUCCESS elapsed={elapsed:.1f}s n_cycles={len(cycles)}", flush=True)
except Exception as exc:
    elapsed = time.time() - t0
    out = {
        "key": (subject, day, part, skill, hand),
        "meta": meta,
        "error": repr(exc),
        "elapsed_sec": elapsed,
        "X_shape": X.shape,
    }
    print(f"[{key_tag}] FAILED elapsed={elapsed:.1f}s error={exc!r}", flush=True)

with open(f"cache/bayes_{key_tag}.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"[{key_tag}] saved to cache/bayes_{key_tag}.pkl", flush=True)
