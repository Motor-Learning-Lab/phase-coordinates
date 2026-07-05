# Claude Implementation Prompt: Bayesian Two-Layer Phase Coordinate Estimator

You are working inside a local clone of:

```text
https://github.com/Motor-Learning-Lab/phase-coordinates.git
```

Read `docs/bayesian_two_layer_spec.md` before implementing.

## Branch workflow

Use a branch based on `main`.

```bash
git fetch origin
git checkout main
git pull --ff-only
git checkout -b bayesian-two-layer-estimator
```

If the branch already exists locally, switch to it and rebase/merge from `main` only if needed.

## Goal

Add a new Bayesian/probabilistic estimator next to the current deterministic cycle-fixed estimator.

Do **not** replace or break:

```python
hilbert_phase
cycle_by_cycle_pca_coordinates
```

The new estimator should be separate. Suggested public function name:

```python
fit_bayesian_phase_coordinates
```

or:

```python
bayesian_phase_coordinates
```

A reasonable file structure is:

```text
phase_coordinates/
    __init__.py
    core.py
    bayesian.py
```

If Bayesian dependencies are optional, use lazy imports inside `bayesian.py`.

## Dependency policy

Do not make PyMC or ArviZ mandatory dependencies for users of the existing deterministic estimator.

Prefer adding an optional dependency group:

```toml
[project.optional-dependencies]
bayes = [
    "pymc",
    "arviz",
]
```

If the user calls the Bayesian estimator without optional dependencies installed, raise a clear `ImportError` explaining:

```bash
pip install -e .[bayes]
```

## Model design

Implement a two-layer approach.

Layer 1 coarse model estimates:

```text
frequency / period
cycle boundaries tau_k
cycle centers c_k
cycle normals n_k
boundary directions a_k = X(tau_k) - c_k
posterior uncertainty for those estimates
```

Layer 2 instantaneous model uses Layer 1 posterior summaries as priors and estimates:

```text
phase phi(t)
phase velocity omega(t)
center c(t)
normal n(t)
in-plane axes e1(t), e2(t)
radius r(t)
perpendicular deviation z(t)
```

The main observation model is:

$$
X_t\sim\mathcal{N}\left(c(t)+e_1(t)r(t)\cos\phi(t)+e_2(t)r(t)\sin\phi(t)+n(t)z(t),\sigma_x^2I\right)
$$

The in-plane frame is boundary-anchored:

$$
e_1(t)=\frac{(I-n(t)n(t)^\top)a(t)}{\left\lVert (I-n(t)n(t)^\top)a(t)\right\rVert}
$$

$$
e_2(t)=n(t)\times e_1(t)
$$

Do not introduce a free in-plane spin parameter in the first implementation.

## Priors and defaults

Use the defaults in `docs/bayesian_two_layer_spec.md`, especially:

```text
log cycle duration SD:                  0.15
boundary timing SD:                     0.075 * T0
boundary spatial scatter:               LogNormal(log 0.10, 0.5) * R_X
cycle center prior SD:                  0.25 * R_X
cycle center change SD:                 0.10 * R_X
normal unconstrained-vector SD:          0.20
normal cycle-to-cycle angle SD:          0.10 rad
Layer 2 posterior uncertainty padding:  1.5
Layer 2 tau uncertainty floor:           0.01 * T_k
Layer 2 center uncertainty floor:        0.02 * R_X
Layer 2 normal-vector uncertainty floor: 0.03
phase boundary constraint SD:            0.15 rad
phase velocity log-knot SD:              0.20
phase velocity smoothness SD:            0.15
```

## Parameterizations

### Normal

Use a normalized unconstrained vector:

$$
n=\frac{u}{\lVert u\rVert}
$$

For instantaneous normals, spline $u(t)$ and then normalize:

$$
u(t)=\operatorname{CubicSpline}(\tau_k,u_k)(t)
$$

$$
n(t)=\frac{u(t)}{\lVert u(t)\rVert}
$$

### Phase

Use a positive phase-velocity spline:

$$
\omega(t)=\exp(g(t))
$$

and cumulative phase:

$$
\phi_t=\phi_{t-1}+\omega_t\Delta t
$$

This should make phase monotone by construction.

### Radius and perpendicular deviation

Use:

$$
r(t)=\exp(h_r(t))
$$

and:

$$
z(t)=h_z(t)
$$

with smooth spline priors or an equivalent smooth parameterization.

## Results API

Create result containers, preferably dataclasses, with three levels:

```text
BayesianPhaseResult
    estimates
    uncertainty
    diagnostics
    bayesian_report optional
```

The user should be able to call something like:

```python
result = fit_bayesian_phase_coordinates(
    X,
    sampling_rate_hz=100.0,
    return_report=False,
)

result.estimates
result.uncertainty
result.diagnostics
```

If `return_report=True`, keep the full posterior object, such as an ArviZ `InferenceData`. If `return_report=False`, discard heavy posterior draws after producing summaries.

## Diagnostics

Implement diagnostics from the spec, including:

- boundary posterior multimodality
- boundary cloud spread relative to `R_X`
- projection of `a(t)` into the plane near zero
- normal estimates dominated by prior
- large `abs(z) / r`
- center drift relative to `R_X`
- phase velocity degeneracy
- observation noise relative to `R_X`
- phase monotonicity sanity check

Diagnostics should distinguish hard failures from warnings.

## Tests

Add tests without breaking the existing suite.

Required synthetic tests:

1. Existing API still imports and current deterministic tests still pass.
2. New Bayesian API imports or gives a clear optional-dependency error.
3. Frame construction test:
   - `e1`, `e2`, and `n` are unit vectors.
   - `e1` and `e2` are perpendicular to `n`.
   - `e2 = cross(n, e1)` within tolerance.
4. Synthetic fixed-plane trajectory:
   - recovered normal close to the true normal up to sign.
   - phase monotone.
   - radius positive.
   - perpendicular deviation small.
5. Synthetic cycle-level changing planes:
   - coarse normals recover known per-cycle normals up to sign.
6. Boundary convention test:
   - phase-zero points cluster near the known synthetic boundary event.
7. Diagnostic test:
   - create a case where `a(t)` is nearly parallel to `n(t)` and confirm a warning or failure.

Run:

```bash
python -m pytest tests/ -v
```

## Implementation strategy

Proceed in stages.

1. Add utility functions:
   - robust movement scale
   - deterministic frequency / boundary seeds
   - cycle centers
   - cycle normals
   - boundary vectors
   - cubic spline helpers
   - vector normalization
   - frame construction
2. Add result dataclasses.
3. Add a lightweight Layer 1 prototype. If full PyMC sampling is too heavy, implement deterministic seeds plus uncertainty approximations first, but keep the structure ready for sampling.
4. Add Layer 2 priors from Layer 1 summaries using the $1.5$ padding rule.
5. Add positive phase-velocity spline.
6. Add observation model and posterior summaries.
7. Add diagnostics.
8. Add tests.

## Deliverables

When finished, report:

1. Files changed.
2. New public API.
3. Tests run and results.
4. Known limitations.
5. Any deviations from `docs/bayesian_two_layer_spec.md` and why.
