# phase-coordinates

Cycle-by-cycle PCA phase coordinates for multivariate cyclic motion data.

Given a time series of 3-D (or higher-dimensional) movement data and an
estimate of the instantaneous phase, this library fits a local PCA plane to
each cycle and returns, for every time point:

| Output column | Meaning |
|---|---|
| `cycle` | Integer cycle index |
| `phase` / `phase_wrapped` / `phase_in_cycle` | Phase in radians (unwrapped, wrapped, within-cycle) |
| `radius_local` | Distance from the cycle centre in the local PCA plane |
| `theta_local` / `theta_local_wrapped` | Angle in the local PCA plane (unwrapped / wrapped) |
| `pc1_local`, `pc2_local`, `pc3_local` | Scores along each local principal component |
| `perp_local` | Signed deviation **perpendicular** to the local phase plane (`pc3_local`) |
| `amp_hilbert` | Hilbert amplitude of the reference signal (when phase is estimated internally) |

The per-cycle PCA models are also returned so that the original data can be
reconstructed exactly from the coordinates.

## Installation

```bash
pip install .
```

For development (includes pytest, Jupyter, and matplotlib):

```bash
pip install ".[dev]"
```

**Requirements:** Python ≥ 3.9, NumPy, pandas, scikit-learn, SciPy.

## Quick start

### Using a pre-computed phase

```python
import numpy as np
from phase_coordinates import cycle_by_cycle_pca_coordinates

# X: (n_time, n_features) array of movement data  –  at least 3 features
# phase_unwrapped: unwrapped phase in radians, same length as X
coords, models = cycle_by_cycle_pca_coordinates(X, phase=phase_unwrapped)

print(coords[["cycle", "phase_in_cycle", "radius_local", "perp_local"]])
```

### Estimating phase from a reference signal (Hilbert transform)

```python
from phase_coordinates import cycle_by_cycle_pca_coordinates

coords, models = cycle_by_cycle_pca_coordinates(
    X,
    ref_signal=X[:, 0],   # scalar reference signal, e.g. one marker coordinate
    fs=100.0,             # sampling rate in Hz
    f_range=(0.5, 3.0),   # bandpass range in Hz
)
```

### Estimating phase directly

```python
from phase_coordinates import hilbert_phase

phase_unwrapped, phase_wrapped, amplitude = hilbert_phase(
    ref_signal, fs=100.0, f_range=(0.5, 3.0)
)
```

### Reconstructing the original data from PCA coordinates

```python
import numpy as np

X_rec = np.full_like(X, np.nan)

for cyc, m in models.items():
    idx    = m["indices"]
    center = m["center"]      # (n_features,)
    comps  = m["components"]  # (3, n_features)
    p1 = coords.loc[idx, "pc1_local"].to_numpy()
    p2 = coords.loc[idx, "pc2_local"].to_numpy()
    p3 = coords.loc[idx, "pc3_local"].to_numpy()
    X_rec[idx] = (
        p1[:, None] * comps[0]
        + p2[:, None] * comps[1]
        + p3[:, None] * comps[2]
        + center
    )
```

## Repository layout

```
phase_coordinates/      Python package
    __init__.py         Public API exports
    core.py             hilbert_phase, cycle_by_cycle_pca_coordinates
tests/
    test_phase_coordinates.py   pytest test suite (32 tests)
notebooks/
    demo.ipynb          End-to-end demo notebook
pyproject.toml          Project metadata and build configuration
```

## Demo notebook

`notebooks/demo.ipynb` walks through:

1. Generating synthetic 3-D noisy cyclic data (a tilted circle with Gaussian noise).
2. Running `cycle_by_cycle_pca_coordinates` to recover the cycle-by-cycle phase
   plane, per-time-point radius, angle, and perpendicular deviation.
3. Reconstructing the original noisy data from the PCA coordinates and verifying
   that the reconstruction error is at machine-precision level.

## Running the tests

```bash
python -m pytest tests/ -v
```

## API reference

### `hilbert_phase(ref_signal, fs, f_range)`

Estimates instantaneous phase from a scalar reference signal using a
zero-phase 4th-order Butterworth bandpass filter followed by the Hilbert
transform.

**Returns** `(phase_unwrapped, phase_wrapped, amplitude)` — all NumPy arrays
of shape `(n_time,)`.

---

### `cycle_by_cycle_pca_coordinates(X, *, ref_signal=None, phase=None, fs=None, f_range=None, columns=None, min_samples_per_cycle=10)`

Fits a PCA plane to each cycle and computes geometric coordinates for every
time point.

**`X`** — `(n_time, n_features)` array-like or `pandas.DataFrame` with at
least 3 features.

**Phase input** — supply *either* a pre-computed unwrapped `phase` array *or*
a `ref_signal` together with `fs` and `f_range`.

**Returns** `(coords, models)`:

- `coords` — `pandas.DataFrame` with one row per time point (columns listed in
  the table at the top of this README).
- `models` — `dict` keyed by cycle index. Each value contains the fitted `pca`
  object, cycle `center`, sign-aligned `components`, `explained_variance_ratio`,
  and the time `indices` belonging to that cycle.
