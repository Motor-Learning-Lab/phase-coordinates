# Claude Implementation Prompt: Bayesian Two-Layer Phase Coordinate Estimator

You are working inside a local clone of:

`https://github.com/Motor-Learning-Lab/phase-coordinates.git`

The current package already contains a deterministic cycle-by-cycle PCA estimator. Your task is to add a separate Bayesian/probabilistic two-layer estimator next to the existing estimator. Do not replace the current public API or current deterministic behavior.

## Branch and workflow

1. First inspect the current repository state.
2. Make sure you are not working on `main`.
3. If the branch `bayesian-estimation-spec` already exists locally or remotely, use it. Otherwise create a new branch from `copilot/create-hilbert-phase-function`:

```bash
git fetch origin
git checkout -b bayesian-estimation-spec origin/copilot/create-hilbert-phase-function
```

4. Read `docs/bayesian_two_layer_spec.md` before implementing.
5. Work incrementally with small commits.
6. Keep the deterministic estimator and its tests passing.
7. Add new tests for the Bayesian/probabilistic estimator.

## High-level task

Add a new Bayesian/probabilistic two-layer estimator that estimates cyclic movement coordinates with uncertainty.

The design is:

```text
Layer 1: coarse cycle model
    estimate cycle frequency
    estimate cycle boundaries tau_k
    estimate cycle centers c_k
    estimate cycle normals n_k
    estimate boundary directions a_k = X(tau_k) - c_k
    preserve uncertainty

Layer 2: instantaneous model
    use Layer 1 posterior summaries as priors
    infer smoothly varying center c(t)
    infer smoothly varying normal n(t)
    infer monotone phase phi(t)
    infer phase velocity omega(t)
    infer radius r(t)
    infer perpendicular deviation z(t)
```

The instantaneous observation model is:

$$
X_t \sim \mathcal{N}\left(c(t) + e_1(t)r(t)\cos\phi(t) + e_2(t)r(t)\sin\phi(t) + n(t)z(t), \sigma_x^2 I\right)
$$

where:

$$
e_1(t) = \frac{\left(I - n(t)n(t)^\top\right)a(t)}{\left\lVert \left(I - n(t)n(t)^\top\right)a(t) \right\rVert}
$$

and:

$$
e_2(t) = n(t) \times e_1(t)
$$

## API requirements

Add a new public function. Suggested name:

```python
bayesian_phase_coordinates(...)
```

or:

```python
fit_bayesian_phase_coordinates(...)
```

The new function should live next to the current estimator, but not inside the current deterministic function. A reasonable structure is:

```text
phase_coordinates/
    __init__.py
    core.py
    bayesian.py
```

Export the new function from `phase_coordinates/__init__.py` only if it can be imported without forcing users to install heavy optional dependencies. If PyMC/ArviZ are optional, do lazy imports inside the Bayesian function.

The deterministic function `cycle_by_cycle_pca_coordinates` must remain unchanged except for any harmless shared utility refactoring.

## Dependency policy

The current package is lightweight. Do not make PyMC and ArviZ mandatory dependencies unless absolutely necessary.

Prefer adding an optional dependency group in `pyproject.toml`:

```toml
[project.optional-dependencies]
bayes = [
    "pymc",
    "arviz",
]
```

If PyMC is unavailable and the user calls the Bayesian function, raise a clear `ImportError` explaining how to install the optional dependencies:

```bash
pip install -e .[bayes]
```

## Output design

Create lightweight result containers, preferably dataclasses, for:

```text
BayesianPhaseResult
    estimates
    uncertainty
    diagnostics
    bayesian_report optional
```

The user should be able to call:

```python
result = fit_bayesian_phase_coordinates(X, sampling_rate_hz=100.0, return_report=False)
```

and access:

```python
result.estimates
result.uncertainty
result.diagnostics
```

If `return_report=True`, include the full posterior report, such as an ArviZ `InferenceData`. If `return_report=False`, do not retain heavy posterior objects after computing summaries.

## Layer 1 coarse model specification

Use the specification in `docs/bayesian_two_layer_spec.md`.

Important equations to implement or approximate:

Characteristic movement scale:

$$
\bar{x} = \operatorname{median}_t X_t
$$

$$
R_X = \operatorname{median}_t \lVert X_t - \bar{x} \rVert
$$

Fallback:

$$
R_X = \sqrt{\frac{1}{N}\sum_t \lVert X_t - \bar{x} \rVert^2}
$$

Cycle duration prior:

$$
\log T_k \sim \mathcal{N}(\log T_0, 0.15^2)
$$

Boundary prior:

$$
\tau_k \sim \mathcal{N}(\hat{\tau}_k, (0.075 T_0)^2)
$$

Boundary clustering likelihood:

$$
X(\tau_k) \sim \mathcal{N}(\mu_\tau, \Sigma_\tau)
$$

with:

$$
\sigma_\tau^{(x)} = R_X \rho_\tau
$$

$$
\rho_\tau \sim \operatorname{LogNormal}(\log 0.10, 0.5^2)
$$

Cycle center:

$$
c_k \sim \mathcal{N}(\hat{c}_k, (0.25 R_X)^2 I)
$$

$$
c_k - c_{k-1} \sim \mathcal{N}(0, (0.10 R_X)^2 I)
$$

Normal parameterization:

$$
n_k = \frac{u_k}{\lVert u_k \rVert}
$$

with:

$$
u_k \sim \mathcal{N}(\hat{n}_k, 0.20^2 I)
$$

and cycle-to-cycle angular smoothness:

$$
\cos^{-1}\left(\left|n_k^\top n_{k-1}\right|\right) \sim \operatorname{HalfNormal}(0.10)
$$

Boundary direction:

$$
a_k = X(\tau_k) - c_k
$$

Do not project `a_k` into the plane in Layer 1.

## Layer 2 instantaneous model specification

Use Layer 1 posterior summaries with uncertainty padding:

$$
\sigma_{\text{Layer2 prior}} = 1.5\sigma_{\text{Layer1 posterior}}
$$

Use floors:

$$
\sigma_{\tau,k}^{(2)} = \max\left(1.5s_{\tau,k}^{(1)}, 0.01T_k\right)
$$

$$
\sigma_{c,k}^{(2)} = \max\left(1.5s_{c,k}^{(1)}, 0.02R_X\right)
$$

$$
\sigma_{u,k}^{(2)} = \max\left(1.5s_{u,k}^{(1)}, 0.03\right)
$$

Use cubic spline interpolation for speed and smoothness:

$$
c(t)=\operatorname{CubicSpline}(\tau_k, c_k^{(2)})(t)
$$

$$
u(t)=\operatorname{CubicSpline}(\tau_k, u_k^{(2)})(t)
$$

$$
n(t)=\frac{u(t)}{\lVert u(t) \rVert}
$$

$$
a(t)=\operatorname{CubicSpline}(\tau_k, a_k^{(2)})(t)
$$

Use a positive phase velocity spline:

$$
\omega(t)=\exp(g(t))
$$

$$
\phi_t = \phi_{t-1}+\omega_t\Delta t
$$

with boundary phase constraints:

$$
\phi(\tau_k)-2\pi k \sim \mathcal{N}(0, 0.15^2)
$$

Use:

$$
g_j \sim \mathcal{N}(\log \omega_0, 0.20^2)
$$

and:

$$
g_j-g_{j-1}\sim\mathcal{N}(0,0.15^2)
$$

Radius:

$$
r(t)=\exp(h_r(t))
$$

Perpendicular deviation:

$$
z(t)=h_z(t)
$$

Observation noise:

$$
\sigma_x = R_X\rho_x
$$

$$
\rho_x\sim\operatorname{LogNormal}(\log0.03,0.5^2)
$$

## Implementation strategy

Implement in stages. Do not attempt a huge monolithic model first.

### Stage 1: utilities and deterministic seeds

Add utilities for:

- robust movement scale computation
- initial frequency estimate
- initial boundary estimate
- initial cycle centers
- initial cycle normals using existing cycle-wise PCA logic
- boundary vector `a_k = X(tau_k) - c_k`
- cubic spline interpolation helpers
- unit normalization helpers
- frame construction from `n(t)` and `a(t)`

### Stage 2: result containers

Create dataclasses such as:

```python
@dataclass
class BayesianPhaseEstimates:
    phase: Any
    phase_velocity: Any
    radius: Any
    perp: Any
    center: Any
    normal: Any
    e1: Any
    e2: Any
    cycle_boundaries: Any
    cycle_centers: Any
    cycle_normals: Any

@dataclass
class BayesianPhaseUncertainty:
    ...

@dataclass
class BayesianPhaseDiagnostics:
    warnings: list[str]
    failures: list[str]
    metrics: dict[str, float]

@dataclass
class BayesianPhaseResult:
    estimates: BayesianPhaseEstimates
    uncertainty: BayesianPhaseUncertainty
    diagnostics: BayesianPhaseDiagnostics
    bayesian_report: Any | None = None
```

Adapt exact field names as appropriate.

### Stage 3: lightweight prototype

If full PyMC sampling is too heavy for the first pass, implement a deterministic initialization plus uncertainty approximation, but keep the API and internal structure ready for PyMC.

However, prefer implementing at least a small PyMC model for Layer 1 if feasible.

### Stage 4: diagnostics

Implement diagnostic checks from the spec:

- boundary posterior multimodality
- boundary cloud spread relative to `R_X`
- projection of `a(t)` into plane near zero
- normal posterior dominated by prior
- large `|z| / r`
- center drift relative to `R_X`
- phase velocity degeneracy
- observation noise relative to `R_X`
- monotone phase sanity check

Use initial thresholds from the spec.

### Stage 5: tests

Add tests with synthetic data.

Required test cases:

1. Import test:
   - existing API still imports
   - new Bayesian API imports or gives a clear optional-dependency error

2. Synthetic stable plane:
   - circular/elliptical 3D trajectory in one fixed plane
   - recovered normal is close to true normal
   - recovered phase is monotone
   - recovered radius is positive
   - perpendicular deviation is small

3. Synthetic changing cycle-level planes:
   - each cycle has a different known normal
   - Layer 1 initial/coarse estimates recover normals up to sign

4. Boundary convention test:
   - synthetic cycles have a repeatable phase-zero point
   - inferred/initialized boundaries cluster near true phase-zero times

5. Frame construction test:
   - `e1`, `e2`, `n` are unit vectors
   - `e1` and `e2` are perpendicular to `n`
   - `e2 = cross(n, e1)` within tolerance

6. Diagnostic test:
   - create a case where `a(t)` is nearly parallel to `n(t)` and confirm a warning or failure is reported

Run:

```bash
python -m pytest tests/ -v
```

## Constraints

- Do not remove or break `hilbert_phase`.
- Do not remove or break `cycle_by_cycle_pca_coordinates`.
- Do not make PyMC a mandatory dependency unless there is a strong reason.
- Keep the new Bayesian estimator separate.
- Keep formulas in documentation written with `$...$` or `$$...$$` math delimiters.
- Add documentation explaining that the current deterministic estimator is cycle-fixed while the new Bayesian estimator is two-layer and uncertainty-aware.

## Deliverables

At the end, provide:

1. Summary of files changed.
2. Summary of the implemented public API.
3. Tests run and results.
4. Any limitations or parts intentionally left as prototype.
5. Any modeling choices that differ from `docs/bayesian_two_layer_spec.md` and why.
