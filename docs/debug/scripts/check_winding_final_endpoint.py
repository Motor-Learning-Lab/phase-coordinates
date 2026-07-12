import sys
from pathlib import Path
sys.path.insert(0, str(Path("/home/opher/Repositories/phase-coordinates/.claude/worktrees/agent-a81757e9539da220c")))

import numpy as np
from phase_coordinates.scoring import candidate_epochs_from_period_offset, score_epoch_geometry

fs = 100.0
period = 1.0
n_cycles = 4
n_time = n_cycles * 100  # no trailing cushion sample -- tau[-1] lands exactly
                          # one sample period past the last recorded sample
t = np.arange(n_time) / fs
theta = 2 * np.pi * t / period
tilt = np.pi / 6
X = np.column_stack([
    np.cos(theta), np.sin(theta) * np.cos(tilt), np.sin(theta) * np.sin(tilt),
])

epochs = candidate_epochs_from_period_offset(period, 0.0, sampling_rate_hz=fs, n_time=n_time)
print("tau:", epochs.tau)
print("last sample time:", (n_time - 1) / fs, " tau[-1]:", epochs.tau[-1])

score = score_epoch_geometry(X, epochs, sampling_rate_hz=fs)
print("winding per cycle:", [c["winding"] for c in score["per_cycle"]])
print("winding_median_abs:", score["winding_median_abs"])
print("winding_min_abs:", score["winding_min_abs"])
print("fraction_single_lap_cycles:", score["fraction_single_lap_cycles"])

# Expected undercount if the closing anchor were still clamped (old bug):
# last recorded sample IS x_close, so the closing segment contributes ~0,
# reproducing the same (n-1)/n undercount Fix 4 targeted, applied only to
# the final cycle: (100-1)/100 = 0.99.
print("old-bug expected (clamped) winding for last cycle: ~0.99")
