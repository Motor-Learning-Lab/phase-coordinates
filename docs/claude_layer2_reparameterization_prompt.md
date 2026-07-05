# Claude Prompt: Fix Layer 2 Convergence by Reparameterizing Normals and Phase

You are working inside a local clone of:

```text
https://github.com/Motor-Learning-Lab/phase-coordinates.git
```

Use the branch:

```text
bayesian-two-layer-estimator
```

Do not start from scratch. The current implementation, progress notes, and debug artifacts are already committed on this branch.

## Read first

Read these files before editing code:

```text
docs/bayesian_two_layer_spec.md
docs/PROGRESS.md
docs/debug/README.md
phase_coordinates/bayesian.py
```

The spec has been updated with the current reparameterization plan. Follow the updated spec, not the older Layer 2 implementation currently in `bayesian.py`.

## Problem summary

Layer 1 is basically working. The open problem is Layer 2.

The current Layer 2 model has two coupled issues:

1. The visible bad artifact is a localized normal-direction failure. The old parameterization splines raw unconstrained 3D vectors and then normalizes:

$$
u(t)=\operatorname{CubicSpline}(u_k)(t)
$$

$$
n(t)=\frac{u(t)}{\lVert u(t)\rVert}
$$

If adjacent `u2` knots drift toward opposite signs or incompatible directions, the spline can pass near zero. Normalizing a near-zero vector produces an arbitrary local normal direction. This matches the observed narrow bad window in the debug logs.

2. The phase-velocity submodel creates difficult NUTS geometry. The old implementation uses:

$$
\omega(t)=\exp(g(t))
$$

$$
\phi_t=\phi_{t-1}+\omega_t\Delta t
$$

plus a tight soft boundary potential:

$$
\phi(\tau_k)-2\pi k\sim\mathcal{N}(0,0.15^2)
$$

This forces a few spline coefficients through an exponential and cumulative sum to hit multiple tight constraints. It is likely creating banana-shaped posterior geometry and divergences.

## Your goal

Reparameterize Layer 2 so that:

1. Instantaneous normals use tangent-plane deviations around Layer 1 posterior mean normals, not raw unconstrained 3D vector splines.
2. Phase satisfies cycle boundaries by construction, not via the tight `phase_boundary` soft potential.
3. The existing public API remains intact.
4. Layer 1 remains working.
5. Existing debug scripts are updated or supplemented so the old failure mode can be tested directly.

## Do not do these things

- Do not replace `hilbert_phase`.
- Do not replace or break `cycle_by_cycle_pca_coordinates`.
- Do not discard the two-layer architecture.
- Do not silently fall back to Laplace/MAP unless explicitly asked later.
- Do not treat this as mere sampler tuning. First fix the parameterization.
- Do not use equations with `\(` or `\)`. Use `$...$` or `$$...$$` in Markdown files.

## Step 1: Add tangent-basis utilities

Add utilities to `phase_coordinates/bayesian.py` or a helper section in that file.

Needed functionality:

```python
align_normal_signs(normals) -> normals_aligned
orthonormal_tangent_basis(normals) -> Q
```

`align_normal_signs` should flip signs so adjacent normals have positive dot product:

$$
m_j^\top m_{j-1} > 0
$$

`orthonormal_tangent_basis` should return `Q` with shape `(n_knots, 3, 2)` such that:

$$
Q_j^\top Q_j = I
$$

$$
Q_j^\top m_j = 0
$$

Use a stable deterministic construction. For each normal, choose the coordinate axis least aligned with the normal as a reference vector, project it into the tangent plane, normalize it, and take the cross product for the second tangent basis vector.

Add pure NumPy tests or scratch checks for:

```text
basis vectors have unit norm
basis vectors are orthogonal to normal
basis vectors are mutually orthogonal
basis construction works near coordinate axes
sign alignment flips only when needed
```

## Step 2: Replace Layer 2 normal parameterization

In `_fit_layer2`, replace the current raw-vector Layer 2 normal prior:

```python
u2 = pm.Normal("u2", mu=u_mean_p, sigma=sigma_u2, shape=u_mean_p.shape)
u_t = pt.dot(B_const, u2)
n_t = u_t / sqrt(sum(u_t**2))
```

with tangent-plane deviations.

Let `m_p` be the padded, sign-aligned Layer 1 posterior mean normals. Let `Q_p` be their tangent bases.

Use Layer 1 angular posterior SD, not componentwise `u_sd`, for the Layer 2 normal prior scale:

$$
\sigma_{\theta,j}^{(2)}=\max(1.5s_{\theta,j}^{(1)},0.03)
$$

Sample:

$$
\delta_j\sim\mathcal{N}(0,(\sigma_{\theta,j}^{(2)})^2I_2)
$$

Then:

$$
\tilde{n}_j=m_j+Q_j\delta_j
$$

$$
n_j^{(2)}=\frac{\tilde{n}_j}{\lVert \tilde{n}_j\rVert}
$$

In PyTensor, this means `delta_n` has shape `(n_knots, 2)`, `Q_const` has shape `(n_knots, 3, 2)`, and the tangent displacement can be computed with an appropriate batched sum/einsum.

Then spline the normal knots and renormalize:

$$
\bar{n}(t)=\operatorname{CubicSpline}(\tau_j,n_j^{(2)})(t)
$$

$$
n(t)=\frac{\bar{n}(t)}{\lVert\bar{n}(t)\rVert}
$$

Keep the deterministic name `normal` for the instantaneous normal so downstream result extraction remains mostly intact.

## Step 3: Add Layer 2 normal smoothness

Add a smoothness potential across adjacent Layer 2 normal knots:

$$
\log p(n_j^{(2)},n_{j-1}^{(2)}) \propto -\frac{1-n_j^{(2)\top}n_{j-1}^{(2)}}{\sigma_{\Delta n}^2}
$$

Start with:

$$
\sigma_{\Delta n}=0.10
$$

Because normal signs are explicitly aligned, do not use absolute value in this Layer 2 smoothness term unless there is evidence sign ambiguity has returned.

## Step 4: Add a floor for Layer 2 boundary-direction prior SD

The first implementation had no floor for `sigma_a2`. Add:

$$
\sigma_{a,k}^{(2)}=\max(1.5s_{a,k}^{(1)},0.02R_X)
$$

This prevents Layer 2 from becoming overconfident in `a2` because Layer 1 happened to produce tiny componentwise SDs.

## Step 5: Replace phase boundary potential with structural boundary satisfaction

Remove or bypass the old Layer 2 `phase_boundary` potential:

```python
phi_at_tau = dot(Btau_const, phi_t)
pm.Potential("phase_boundary", ...)
```

Replace with a boundary-normalized positive phase-speed model.

Within each cycle, define:

$$
s(t)=\exp(q(t))
$$

For $\tau_k\le t\le\tau_{k+1}$:

$$
\phi(t)=2\pi k+2\pi\frac{\int_{\tau_k}^{t}s(v)\,dv}{\int_{\tau_k}^{\tau_{k+1}}s(v)\,dv}
$$

This guarantees:

$$
\phi(\tau_k)=2\pi k
$$

$$
\phi(\tau_{k+1})=2\pi(k+1)
$$

and phase is monotone because $s(t)>0$.

A discrete implementation is fine. Since Layer 2 currently uses fixed `tau_mean` knot locations and a fitted time window `t_fit`, you can compute a fixed cycle index for each time sample from `tau_mean`. For each cycle:

$$
w_i=\exp(q_i)
$$

$$
S_{k,i}=\sum_{j\in k,\ j<i}w_j\Delta t
$$

$$
S_{k,\mathrm{total}}=\sum_{j\in k}w_j\Delta t
$$

$$
\phi_i=2\pi k+2\pi\frac{S_{k,i}}{S_{k,\mathrm{total}}}
$$

Use a spline or low-rank basis for `q(t)`. Keep it centered, because the per-cycle normalization makes an unconstrained global offset in `q(t)` weakly identified or irrelevant.

Good starting prior:

$$
q_j\sim\mathcal{N}(0,0.20^2)
$$

and smoothness:

$$
q_j-q_{j-1}\sim\mathcal{N}(0,0.15^2)
$$

You may keep the deterministic output name `phase_velocity`, but compute it consistently from the final phase if necessary. A practical discrete approximation is:

```python
phase_velocity = diff(phase) / dt
```

with suitable padding to length `n_time`.

## Step 6: Add diagnostics for misleading normal summaries

The old run showed that `normal_angular_sd` did not flag the bad region. Add a diagnostic based on the raw posterior mean resultant length:

$$
\bar{n}_{\mathrm{raw}}(t)=\mathbb{E}[n(t)\mid X]
$$

Warn if:

$$
\lVert\bar{n}_{\mathrm{raw}}(t)\rVert < 0.80
$$

The reported point estimate may still be:

$$
\frac{\bar{n}_{\mathrm{raw}}(t)}{\lVert\bar{n}_{\mathrm{raw}}(t)\rVert}
$$

but the resultant length should be exposed in uncertainty or diagnostics so users can see whether the normalized mean is trustworthy.

## Step 7: Run targeted tests before broad tuning

Do not start with a giant sampler-tuning sweep. Run targeted checks.

First, add or update cheap non-MCMC tests:

```text
normal tangent basis is correct
normal sign alignment is correct
boundary-normalized phase is monotone
boundary-normalized phase exactly hits 2*pi*k at boundaries
phase construction is invariant to adding a constant offset to q within each cycle, or else q is explicitly centered
frame construction remains orthonormal
```

Then run the existing debug scripts:

```bash
/c/Users/User/miniforge3/python.exe docs/debug/scripts/test_layer2.py
/c/Users/User/miniforge3/python.exe docs/debug/scripts/debug_layer2_normal.py
```

Use the interpreter documented in `docs/PROGRESS.md`, not the default `python` on PATH.

## Success criteria

The reparameterization is successful if, on the synthetic fixed-plane Layer 2 debug case:

```text
post-tuning divergences: 0 preferred; very small number only if explained and not associated with artifacts
normal cos_sim min: > 0.95, preferably > 0.99
normal cos_sim median: > 0.99
phase monotonic: true
radius median: roughly 1.0
median abs perp deviation: < 0.1
projection/frame checks: pass
normal resultant length: not low in any localized region
```

If divergences remain after this reparameterization, report exactly which submodel still appears responsible. Do not silently switch to Laplace/MAP fallback unless explicitly instructed.

## Update documentation

After code changes, update:

```text
docs/PROGRESS.md
docs/debug/README.md
```

Record:

```text
what changed
which scripts were run
runtime
divergence counts
normal cos_sim min/median
whether the old localized artifact disappeared
remaining limitations
```

## Final response expected from Claude

When done, report:

1. Files changed.
2. Exact reparameterizations implemented.
3. Tests/scripts run and results.
4. Whether the old localized normal artifact is gone.
5. Whether divergences are gone or reduced.
6. Any remaining model concerns.
