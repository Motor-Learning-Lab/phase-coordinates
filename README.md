# phase-coordinates

Cycle-by-cycle PCA phase coordinates for multivariate cyclic motion data.

Given a time series of 3-D (or higher-dimensional) movement data and an
estimate of the instantaneous phase, this library fits a local PCA plane to
each cycle and returns, for every time point:

| Output column | Meaning |
|---|---|
| `cycle` | Integer cycle index |
| **`phase_in_cycle`** | **Primary phase coordinate.** Phase within the current cycle (range `[0, 2π)`). Use this for cross-cycle alignment and averaging. |
| `phase` / `phase_wrapped` | Unwrapped / wrapped global phase in radians |
| **`radius_local`** | **Distance from the cycle centre** in the local PCA plane |
| **`perp_local`** | **Signed deviation perpendicular** to the local phase plane (`= pc3_local`) |
| `pc1_local`, `pc2_local`, `pc3_local` | Scores along each local principal component |
| `theta_local` / `theta_local_wrapped` | Geometric angle in the local PCA plane (see note below) |
| `amp_hilbert` | Hilbert amplitude of the reference signal (when phase is estimated internally) |

> **Note on `theta_local`.**  `theta_local` is the geometric angle in the
> local PCA plane and is useful for describing within-cycle geometry.
> However, it should be treated **cautiously across cycles**: because PCA
> axes can rotate, flip signs, or swap when PC1/PC2 variances are similar,
> `theta_local` may be discontinuous between cycles.  Use `phase_in_cycle`
> for reliable cross-cycle alignment.

> **Reconstruction accuracy.**  For **3-feature** input, reconstruction
> from `(pc1_local, pc2_local, pc3_local)` together with the per-cycle
> `center` and `components` is exact up to floating-point precision (the
> three PCs span the full feature space).  For input with **more than 3
> features**, only 3 principal components are retained, so reconstruction
> is generally approximate and the residual depends on how much variance
> the remaining components capture.

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

# X: (n_time, n_features) array of movement data — at least 3 features
# phase_unwrapped: unwrapped phase in radians, same length as X
coords, models = cycle_by_cycle_pca_coordinates(X, phase=phase_unwrapped)

# Primary coordinates of interest:
print(coords[["cycle", "phase_in_cycle", "radius_local", "perp_local"]])
```

### Estimating phase from a reference signal (Hilbert transform)

```python
from phase_coordinates import cycle_by_cycle_pca_coordinates

coords, models = cycle_by_cycle_pca_coordinates(
    X,
    ref_signal=X[:, 0],   # scalar reference signal, e.g. one marker coordinate
    fs=100.0,             # sampling rate in Hz
    f_range=(0.5, 3.0),   # bandpass range in Hz: 0 < low < high < fs/2
)
```

### Estimating phase directly

```python
from phase_coordinates import hilbert_phase

phase_unwrapped, phase_wrapped, amplitude = hilbert_phase(
    ref_signal, fs=100.0, f_range=(0.5, 3.0)
)
```

`hilbert_phase` validates inputs and will raise a `ValueError` if the signal
is non-finite, too short, or if `fs`/`f_range` are invalid.  It will also
emit a `UserWarning` when the Hilbert phase shows many large negative steps
(suggesting the reference signal or frequency band may not define a reliable
instantaneous phase).

### Reconstructing the original data from PCA coordinates

`models[cyc]["indices"]` always contains **positional integer indices**.
Use `coords.iloc[idx]` (not `coords.loc[idx]`) so that reconstruction works
correctly even when the DataFrame has a non-default index (e.g. time stamps).

For **3-feature** data the reconstruction is exact; for **>3 features** it is
approximate (limited to the first 3 PCs).

```python
import numpy as np

X_rec = np.full_like(X, np.nan)

for cyc, m in models.items():
    idx    = m["indices"]          # positional integer indices
    center = m["center"]           # (n_features,)
    comps  = m["components"]       # (3, n_features)
    p1 = coords.iloc[idx]["pc1_local"].to_numpy()
    p2 = coords.iloc[idx]["pc2_local"].to_numpy()
    p3 = coords.iloc[idx]["pc3_local"].to_numpy()
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
    test_phase_coordinates.py   pytest test suite (46 tests)
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
   that the reconstruction error is at machine-precision level (3-D case).

## Running the tests

```bash
python -m pytest tests/ -v
```

## Scientific framing

This package implements a **cycle-by-cycle local PCA plane**, not a
continuously varying plane.  Each cycle's plane is fitted independently to the
data from that cycle.

| Coordinate | Scientific meaning |
|---|---|
| `phase_in_cycle` | Timing / phase reference for cross-cycle alignment and ensemble averaging |
| `radius_local` | Movement amplitude (distance from local cycle centre in the local plane) |
| `perp_local` | Out-of-plane deviation for that cycle's plane |
| `theta_local` | Geometric angle in the local PCA plane — useful within a cycle, but treat with caution across cycles due to potential PCA axis rotation/flipping |

## API reference

### `hilbert_phase(ref_signal, fs, f_range)`

Estimates instantaneous phase from a 1-D scalar reference signal using a
zero-phase 4th-order Butterworth bandpass filter followed by the Hilbert
transform.

**Raises `ValueError`** if `ref_signal` is not 1-D, contains non-finite
values, is too short for the filter, `fs ≤ 0`, or `f_range` does not satisfy
`0 < low < high < fs/2`.

**Warns (`UserWarning`)** when the unwrapped phase has many large negative
steps in the central region, suggesting an unreliable phase estimate.

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
  the table at the top of this README).  The recommended primary coordinates
  are `phase_in_cycle`, `radius_local`, and `perp_local`.
- `models` — `dict` keyed by cycle index. Each value contains the fitted `pca`
  object, cycle `center`, sign-aligned `components`, `explained_variance_ratio`,
  and the positional time `indices` belonging to that cycle.
