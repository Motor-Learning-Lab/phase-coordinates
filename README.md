# phase-coordinates

phase-coordinates provides experimental tools for describing cyclic multivariate movement using phase, radius, and perpendicular deviation.

Two peer algorithms are provided with a shared output interface. Both are experimental.

## Installation

For local use (clone the repo first):

```bash
pip install -e .
```

For the Bayesian algorithm:

```bash
pip install -e ".[bayes]"
```

## Quick start

### PCA algorithm

```python
import numpy as np
from phase_coordinates import fit_pca_phase_coordinates, reconstruct_phase_coordinates

# X: (n_time, 3+) array of movement data
# phase: unwrapped phase in radians (or use ref_signal + sampling_rate_hz + f_range)
samples, cycles, details = fit_pca_phase_coordinates(X, phase=phase)
X_hat = reconstruct_phase_coordinates(samples, cycles)
```

### Bayesian algorithm

```python
from phase_coordinates import fit_bayesian_phase_coordinates, reconstruct_phase_coordinates

# X: (n_time, 3) array; phase is estimated from the data
samples, cycles, details = fit_bayesian_phase_coordinates(X, sampling_rate_hz=100.0)
X_hat = reconstruct_phase_coordinates(samples, cycles)
```

## Shared outputs

Both algorithms return `(samples, cycles, details)` with identical schemas for the first two outputs.

### samples DataFrame

One row per input time sample. Columns:

| Column | Description |
|---|---|
| `sample_index` | Integer sample index (0-based) |
| `time` | Time in seconds (NaN if `sampling_rate_hz` not provided for PCA) |
| `cycle` | Integer cycle index |
| `phase` | Unwrapped phase in radians |
| `phase_in_cycle` | Phase within the current cycle, range `[0, 2π)` |
| `u` | Score along the first in-plane axis |
| `v` | Score along the second in-plane axis |
| `radius` | In-plane radius (distance from cycle centre) |
| `theta` | Geometric angle in the local plane (radians) |
| `theta_wrapped` | `theta` wrapped to `[-π, π]` |
| `perp` | Signed deviation perpendicular to the local plane |

### cycles DataFrame

One row per fitted cycle. Columns:

| Column | Description |
|---|---|
| `cycle` | Integer cycle index |
| `sample_start` | First sample index of the cycle |
| `sample_stop` | One-past-last sample index (Python slice convention) |
| `time_start` | Cycle start time in seconds |
| `time_stop` | Cycle stop time in seconds |
| `time_quarter` | Time at 25% of the cycle |
| `duration` | Cycle duration in seconds |
| `center_x/y/z` | Cycle centre position in 3-D |
| `e1_x/y/z` | First in-plane axis direction |
| `e2_x/y/z` | Second in-plane axis direction |
| `normal_x/y/z` | Normal to the local plane |
| `radius_mean` | Mean in-plane radius for this cycle |
| `radius_sd` | Standard deviation of in-plane radius |
| `perp_mean` | Mean perpendicular deviation |
| `perp_sd` | Standard deviation of perpendicular deviation |
| `n_samples` | Number of samples in this cycle |
| `fit_ok` | True if the cycle was successfully fitted |

### Reconstruction helper

```python
X_hat = reconstruct_phase_coordinates(samples, cycles)
# Returns np.ndarray of shape (n_time, 3).
# NaN rows where reconstruction is not possible (outside fitted window or unfitted cycles).
```

`reconstruct_phase_coordinates` always returns a **(n_time, 3)** array. For 3-D input the reconstruction is exact to floating-point precision. For PCA inputs with more than 3 features, only the 3-PC projection is reconstructed; use `details["models"]` to reconstruct in the original feature space.

## Algorithm 1: fit_pca_phase_coordinates

```python
samples, cycles, details = fit_pca_phase_coordinates(
    X,
    *,
    phase=None,          # pre-computed unwrapped phase
    ref_signal=None,     # scalar reference for Hilbert phase estimation
    sampling_rate_hz=None,
    f_range=None,        # bandpass (low, high) in Hz for Hilbert estimation
    columns=None,        # subset of DataFrame columns to use
    min_samples_per_cycle=10,
)
```

**Assumptions:** Each cycle lies approximately in a plane (the PCA plane). The PCA plane is fitted independently per cycle, so the plane can change across cycles.

**details dict:**
- `algorithm`: `"pca"`
- `models`: per-cycle dict with `pca`, `center`, `components`, `explained_variance_ratio`, `indices`
- `phase_source`: `"provided"` or `"hilbert"`
- `amp_hilbert`: Hilbert amplitude array (NaN if phase was supplied directly)
- `warnings`: list of any collected warnings

## Algorithm 2: fit_bayesian_phase_coordinates

```python
samples, cycles, details = fit_bayesian_phase_coordinates(
    X,
    *,
    sampling_rate_hz,    # required
    columns=None,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.9,
    random_seed=None,
    return_report=False,
)
```

**Assumptions:** 3-D data only. Phase is estimated jointly with geometry using MCMC. The cycle-fixed frame uses an oriented basis derived from cycle-boundary anchor points.

**details dict:**
- `algorithm`: `"bayesian"`
- `diagnostics`: dict with convergence/quality diagnostics
- `uncertainty`: dict with posterior standard deviations
- `sampling_metadata`: MCMC settings used
- `report` (if `return_report=True`): layer1 and layer2 ArviZ InferenceData objects

## Which algorithm?

| | `fit_pca_phase_coordinates` | `fit_bayesian_phase_coordinates` |
|---|---|---|
| Geometry model | Local PCA plane per cycle | Cycle-fixed oriented frame |
| Speed | Fast | Slow (MCMC) |
| Phase input | Supplied or Hilbert | Estimated from data |
| Dimensions | 3-D or higher | 3-D only |
| Uncertainty | None | Posterior uncertainty |
| Known limitations | PCA axes may flip/rotate between cycles | Endpoint boundary drift; linear phase within cycle; MCMC runtime |

## Notebooks

- `notebooks/pca_phase_coordinates_demo.ipynb`
- `notebooks/bayesian_phase_coordinates_demo.ipynb`

## Known limitations

**fit_pca_phase_coordinates:**
- PCA axes can flip sign or rotate between cycles when PC1/PC2 variances are similar, making `theta` inconsistent across cycles. Use `phase_in_cycle` for cross-cycle alignment.
- Cycle boundaries are anchored to the first sample of the recording, not to external behavioural events.
- For input with more than 3 features, only 3 principal components are retained; reconstruction is approximate.

**fit_bayesian_phase_coordinates:**
- Endpoint boundary drift: the first and last cycle boundaries can drift 3-5 samples from the true cycle start/end, inflating residuals in the first and last cycles.
- Linear phase within cycle: the model assumes linear phase within each cycle, which inflates sigma_x when within-cycle speed is non-uniform.
- MCMC runtime: a 400-sample, 4-cycle run takes ~10-60 seconds; real data with more cycles and draws will take longer.
- 3-D only.

## API reference

- `hilbert_phase(ref_signal, fs, f_range)` → `(phase_unwrapped, phase_wrapped, amplitude)`
- `fit_pca_phase_coordinates(X, ...)` → `(samples, cycles, details)`
- `fit_bayesian_phase_coordinates(X, *, sampling_rate_hz, ...)` → `(samples, cycles, details)`
- `reconstruct_phase_coordinates(samples, cycles)` → `np.ndarray (n_time, 3)`
- `SAMPLE_COLUMNS` — list of column names for the samples DataFrame
- `CYCLE_COLUMNS` — list of column names for the cycles DataFrame
