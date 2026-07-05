# Bayesian Two-Layer Phase Coordinate Estimation Specification

This document specifies a Bayesian extension to the existing `phase_coordinates` package. The Bayesian estimator should live next to the deterministic cycle-fixed estimator. It should not replace `hilbert_phase` or `cycle_by_cycle_pca_coordinates`.

The target use case is 3D rhythmic movement data where we want posterior estimates of phase, radius, smoothly varying cycle-plane normal, center, and perpendicular deviation.

This version of the specification reflects the current debugging conclusion from the Layer 2 convergence work:

1. The coarse cycle model remains useful and mostly unchanged.
2. The original Layer 2 normal model, which splined raw unconstrained 3D vectors and then normalized them, is too fragile.
3. The original Layer 2 phase model, which used a positive velocity spline plus a tight soft boundary potential, creates difficult NUTS geometry.
4. The next implementation should use tangent-plane normal deviations and a phase parameterization that satisfies cycle boundaries by construction.

All equations below use `$...$` or `$$...$$` math delimiters so they render correctly in Markdown contexts used by this project.

## Core idea

Use two models rather than one monolithic model.

1. **Layer 1: coarse cycle model** estimates dominant frequency, cycle boundaries, cycle centers, cycle-level normals, and boundary reference directions, while keeping posterior uncertainty.
2. **Layer 2: instantaneous model** uses the coarse posterior summaries as priors for smoothly varying instantaneous quantities.

Layer 1 estimates cycle-level objects:

$$
\tau_k,\quad T_k,\quad c_k,\quad n_k,\quad a_k
$$

Layer 2 estimates time-level objects:

$$
\phi(t),\quad \omega(t),\quad c(t),\quad n(t),\quad e_1(t),\quad e_2(t),\quad r(t),\quad z(t)
$$

## Notation

Let $X_t \in \mathbb{R}^3$ be the observed trajectory at time $t$.

- $\tau_k$: cycle boundary / phase-zero time for cycle $k$.
- $T_k = \tau_{k+1}-\tau_k$: cycle duration.
- $c_k \in \mathbb{R}^3$: cycle center, fixed within cycle $k$ in the coarse model.
- $c(t) \in \mathbb{R}^3$: smooth instantaneous center.
- $n_k \in S^2$: cycle-level unit plane normal.
- $n(t) \in S^2$: smooth instantaneous unit plane normal.
- $a_k = X(\tau_k)-c_k$: boundary-anchored reference direction.
- $a(t)$: cubic spline interpolation of $a_k$.
- $e_1(t)$: in-plane x-axis, defined by projecting $a(t)$ into the instantaneous plane.
- $e_2(t)=n(t)\times e_1(t)$: in-plane y-axis.
- $\phi(t)$: monotone instantaneous phase.
- $\omega(t)=d\phi/dt$: positive instantaneous phase velocity.
- $r(t)>0$: instantaneous in-plane radius.
- $z(t)$: instantaneous perpendicular deviation.

## Data-derived scale

Compute a robust characteristic movement scale from the data:

$$
\bar{x}=\operatorname{median}_t X_t
$$

$$
R_X=\operatorname{median}_t \lVert X_t-\bar{x}\rVert
$$

If $R_X$ is too small, use an RMS fallback:

$$
R_X=\sqrt{\frac{1}{N}\sum_t \lVert X_t-\bar{x}\rVert^2}
$$

All spatial priors should be expressed relative to $R_X$.

## Layer 1: coarse cycle model

Layer 1 can continue to use the existing implementation strategy unless debugging shows otherwise.

### Frequency and duration

Estimate a crude dominant frequency $f_0$ using periodogram, autocorrelation, Hilbert phase, or peak/event detection. Define:

$$
T_0=\frac{1}{f_0}
$$

Use a log-duration prior:

$$
\log T_k \sim \mathcal{N}(\log T_0,0.15^2)
$$

### Boundary times

Let $\hat{\tau}_k$ be initial candidate boundaries. Use:

$$
\tau_k \sim \mathcal{N}(\hat{\tau}_k,(0.075T_0)^2)
$$

Use $0.10T_0$ instead of $0.075T_0$ if the initial boundary detector is weak.

### Boundary clustering likelihood

The phase-zero boundary is identified by spatial reproducibility. Model:

$$
X(\tau_k) \sim \mathcal{N}(\mu_\tau,\Sigma_\tau)
$$

Start with an isotropic boundary cloud:

$$
\Sigma_\tau = (\sigma_\tau^{(x)})^2I
$$

Use a dimensionless scatter parameter:

$$
\sigma_\tau^{(x)}=R_X\rho_\tau
$$

$$
\rho_\tau\sim\operatorname{LogNormal}(\log 0.10,0.5^2)
$$

This says the boundary point is expected to repeat within about 10% of movement radius, while allowing smaller and larger scatter.

### Cycle center

Compute an initial center estimate $\hat{c}_k$ per cycle, for example:

$$
\hat{c}_k = \operatorname{mean}_{t\in k}X_t
$$

Use:

$$
c_k \sim \mathcal{N}(\hat{c}_k,(0.25R_X)^2I)
$$

and smoothness across cycles:

$$
c_k-c_{k-1}\sim\mathcal{N}(0,(0.10R_X)^2I)
$$

If the recording has large drift, increase $0.10R_X$ to $0.20R_X$.

### Cycle normal

Layer 1 may use a normalized unconstrained vector. Let $u_k\in\mathbb{R}^3$:

$$
n_k=\frac{u_k}{\lVert u_k\rVert}
$$

Initialize $\hat{n}_k$ from cycle-wise PCA. Use:

$$
u_k\sim\mathcal{N}(\hat{n}_k,0.20^2I)
$$

Add angular smoothness. The literal prior was:

$$
\cos^{-1}\left(\left|n_k^\top n_{k-1}\right|\right)\sim\operatorname{HalfNormal}(0.10)
$$

In implementation, avoid the literal `arccos` potential because it has a gradient singularity near perfect alignment. Use the already implemented smooth small-angle proxy:

$$
\log p(n_k,n_{k-1}) \propto -\frac{1-|n_k^\top n_{k-1}|}{\sigma_n^2}
$$

with:

$$
\sigma_n = 0.10
$$

### Boundary direction

Define:

$$
a_k=X(\tau_k)-c_k
$$

Do **not** project $a_k$ into the cycle plane in the coarse model. Project once in the instantaneous model.

### Layer 1 outputs

Layer 1 should return posterior summaries for:

- $\tau_k$
- $T_k$
- $c_k$
- $u_k$ and/or $n_k$
- $a_k$
- $\mu_\tau$
- $\Sigma_\tau$ or $\sigma_\tau^{(x)}$
- posterior angular SD for each $n_k$

## Layer 2: instantaneous model

Layer 2 uses Layer 1 posterior summaries as priors. This is a modular Bayesian handoff rather than a single joint model.

### Uncertainty propagation

Use a padding factor of $1.5$:

$$
\theta^{(2)}\sim\mathcal{N}\left(\mathbb{E}[\theta^{(1)}\mid X],\left(1.5\operatorname{SD}[\theta^{(1)}\mid X]\right)^2\right)
$$

For vectors:

$$
v^{(2)}\sim\mathcal{N}\left(\mathbb{E}[v^{(1)}\mid X],1.5^2\operatorname{Cov}[v^{(1)}\mid X]\right)
$$

If full covariance is inconvenient, use componentwise posterior SDs for ordinary Euclidean quantities such as center and boundary direction.

Use uncertainty floors:

$$
\sigma_{\tau,k}^{(2)}=\max(1.5s_{\tau,k}^{(1)},0.01T_k)
$$

$$
\sigma_{c,k}^{(2)}=\max(1.5s_{c,k}^{(1)},0.02R_X)
$$

For Layer 2 normals, do **not** use componentwise $u$ SD as the main uncertainty scale. Use Layer 1 angular posterior SD instead:

$$
\sigma_{\theta,k}^{(2)}=\max(1.5s_{\theta,k}^{(1)},0.03)
$$

where $s_{\theta,k}^{(1)}$ is the Layer 1 posterior angular SD for $n_k$.

## Layer 2 normal parameterization: tangent-plane deviations

The previous Layer 2 parameterization splined raw unconstrained 3D vectors and then normalized:

$$
u(t)=\operatorname{CubicSpline}(u_k)(t)
$$

$$
n(t)=\frac{u(t)}{\lVert u(t)\rVert}
$$

This is fragile. If adjacent $u_k$ knots drift toward opposite signs or incompatible directions, the spline can pass near zero:

$$
\lVert u(t)\rVert \approx 0
$$

Then $u(t)/\lVert u(t)\rVert$ becomes arbitrary, producing localized but confident-looking normal artifacts.

Replace this with tangent-plane normal deviations around the Layer 1 posterior mean normals.

Let $m_j$ be the padded Layer 1 posterior mean normal at Layer 2 normal knot $j$. Ensure signs are aligned so adjacent $m_j$ satisfy:

$$
m_j^\top m_{j-1} > 0
$$

Construct a deterministic orthonormal tangent basis $Q_j\in\mathbb{R}^{3\times 2}$ such that:

$$
Q_j^\top Q_j=I
$$

$$
Q_j^\top m_j=0
$$

Sample two-dimensional angular deviations:

$$
\delta_j\sim\mathcal{N}(0,(\sigma_{\theta,j}^{(2)})^2I_2)
$$

Then define Layer 2 normal knots by:

$$
\tilde{n}_j=m_j+Q_j\delta_j
$$

$$
n_j^{(2)}=\frac{\tilde{n}_j}{\lVert\tilde{n}_j\rVert}
$$

This removes the meaningless radial degree of freedom in $u_j$ and makes the prior scale genuinely angular.

Add Layer 2 normal smoothness across adjacent normal knots:

$$
\log p(n_j^{(2)},n_{j-1}^{(2)}) \propto -\frac{1-n_j^{(2)\top}n_{j-1}^{(2)}}{\sigma_{\Delta n}^2}
$$

Use:

$$
\sigma_{\Delta n}=0.05\text{ to }0.10
$$

Start with:

$$
\sigma_{\Delta n}=0.10
$$

Because signs are explicitly aligned, do not use absolute value in this Layer 2 smoothness term unless later evidence shows sign ambiguity has returned.

For instantaneous normals, spline the normal knots themselves and then renormalize:

$$
\bar{n}(t)=\operatorname{CubicSpline}(\tau_j,n_j^{(2)})(t)
$$

$$
n(t)=\frac{\bar{n}(t)}{\lVert\bar{n}(t)\rVert}
$$

This is still not a perfect geodesic spline on $S^2$, but it avoids the worst failure mode: raw unconstrained $u(t)$ passing through zero.

## Layer 2 center and boundary direction splines

Center remains an ordinary Euclidean spline:

$$
c(t)=\operatorname{CubicSpline}(\tau_j,c_j^{(2)})(t)
$$

Boundary direction remains an ordinary Euclidean spline:

$$
a(t)=\operatorname{CubicSpline}(\tau_j,a_j^{(2)})(t)
$$

Add a floor for boundary-direction uncertainty, which was missing in the first implementation:

$$
\sigma_{a,k}^{(2)}=\max(1.5s_{a,k}^{(1)},0.02R_X)
$$

## Boundary-anchored coordinate frame

Define the in-plane x-axis by projecting $a(t)$ into the instantaneous plane:

$$
e_1(t)=\frac{(I-n(t)n(t)^\top)a(t)}{\left\lVert (I-n(t)n(t)^\top)a(t)\right\rVert}
$$

Then:

$$
e_2(t)=n(t)\times e_1(t)
$$

This fixes the in-plane gauge and avoids a separate free in-plane spin parameter.

## Layer 2 phase parameterization: satisfy boundaries by construction

The previous implementation used:

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

This creates difficult NUTS geometry because a small number of spline coefficients must satisfy several tight nonlinear cumulative constraints.

Replace it with a boundary-normalized positive phase-speed model. Within each cycle, define a positive unnormalized speed:

$$
s(t)=\exp(q(t))
$$

For $\tau_k\le t\le\tau_{k+1}$, define:

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

In discrete implementation, for samples $i$ inside cycle $k$:

$$
w_i=\exp(q_i)
$$

$$
S_{k,i}=\sum_{j\in k,\ j<i} w_j\Delta t
$$

$$
S_{k,\mathrm{total}}=\sum_{j\in k}w_j\Delta t
$$

$$
\phi_i=2\pi k+2\pi\frac{S_{k,i}}{S_{k,\mathrm{total}}}
$$

This removes the `phase_boundary` potential from Layer 2. A weak prior on $q(t)$ and smoothness of $q(t)$ controls within-cycle acceleration and deceleration.

Use a spline or low-rank basis for $q(t)$, with mean-zero or weakly centered deviations so the normalization, not the absolute offset of $q(t)$, determines the cycle duration.

For example:

$$
q_j\sim\mathcal{N}(0,0.20^2)
$$

and:

$$
q_j-q_{j-1}\sim\mathcal{N}(0,0.15^2)
$$

Because the per-cycle normalization removes the absolute scale of $s(t)$, avoid adding unidentified global offsets to $q(t)$ unless they are explicitly constrained.

## Radius and perpendicular deviation

Radius is positive:

$$
r(t)=\exp(h_r(t))
$$

Perpendicular deviation is unconstrained but should be smooth and zero-centered:

$$
z(t)=h_z(t)
$$

Continue to use smooth spline priors for $h_r(t)$ and $h_z(t)$.

## Observation model

The instantaneous observation model is:

$$
X_t\sim\mathcal{N}\left(c(t)+e_1(t)r(t)\cos\phi(t)+e_2(t)r(t)\sin\phi(t)+n(t)z(t),\sigma_x^2I\right)
$$

Use:

$$
\sigma_x=R_X\rho_x
$$

$$
\rho_x\sim\operatorname{LogNormal}(\log0.03,0.5^2)
$$

## Return values

Return three layers of output.

### Estimates

Posterior mean or median values for practical analysis:

- cycle boundaries
- cycle centers
- cycle normals
- boundary directions
- instantaneous phase
- instantaneous phase velocity
- instantaneous center
- instantaneous normal
- $e_1(t)$ and $e_2(t)$
- radius
- perpendicular deviation
- predicted trajectory

### Uncertainty

Credible intervals or posterior SDs for key quantities.

For normals, report at least:

- angular SD around the posterior mean direction
- raw mean resultant length or equivalent diagnostic showing whether the mean normal is stable before normalization

The mean resultant length diagnostic is important because normalizing a small posterior mean vector can produce a confident-looking but misleading direction.

### Bayesian report

Optional full posterior object, such as an ArviZ `InferenceData`. Only retain this if `return_report=True`; otherwise compute summaries and discard heavy posterior draws.

## Diagnostics

### Hard failures

Boundary multimodality:

- fail if two posterior modes each have mass greater than $0.20$ and are separated by more than $0.20T_k$.

Boundary cloud too large:

$$
\rho_\tau=\frac{\sqrt{\operatorname{tr}(\Sigma_\tau)}}{R_X}
$$

- warn if $\rho_\tau>0.25$.
- fail if $\rho_\tau>0.40$.

Projection failure:

$$
p_t=\left\lVert (I-n(t)n(t)^\top)a(t)\right\rVert
$$

- warn if $p_t/\lVert a(t)\rVert<0.20$.
- fail if $p_t/\lVert a(t)\rVert<0.10$.

### Warnings

Normal dominated by prior:

Warn if the posterior angular shift from the prior is less than $0.25$ of the prior angular SD and the posterior SD remains larger than $0.75$ of the prior SD.

Low normal mean resultant length:

Let $\bar{n}_{\mathrm{raw}}(t)$ be the unnormalized posterior mean of normal samples:

$$
\bar{n}_{\mathrm{raw}}(t)=\mathbb{E}[n(t)\mid X]
$$

Warn if:

$$
\lVert\bar{n}_{\mathrm{raw}}(t)\rVert < 0.80
$$

This catches cases where the normalized reported mean direction may be misleading.

Large perpendicular deviation:

$$
\rho_z(t)=\frac{|z(t)|}{r(t)+\epsilon}
$$

- warn if $\operatorname{median}_t\rho_z(t)>0.25$.
- warn strongly if $\operatorname{median}_t\rho_z(t)>0.50$.
- warn if more than 10% of samples have $\rho_z(t)>0.50$.

Center drift:

$$
D_c=\operatorname{range}_t\lVert c(t)-\bar{c}\rVert
$$

- warn if $D_c/R_X>0.25$.
- warn strongly if $D_c/R_X>0.50$.

Phase velocity degeneracy:

$$
\omega_{95}/\omega_5
$$

- warn if $\omega_{95}/\omega_5>3$.
- warn strongly if $\omega_{95}/\omega_5>5$.

Observation noise:

- warn if $\sigma_x/R_X>0.10$.
- warn strongly if $\sigma_x/R_X>0.25$.

Phase monotonicity should be guaranteed by construction, but still include a sanity check.

Divergences and treedepth:

- hard-fail if post-tuning divergences are nonzero in final test configurations.
- warn if any chain reaches maximum treedepth.
- report `r_hat` and ESS summaries for final Bayesian reports.

## Implementation order for the current reparameterization pass

1. Add tangent-basis utilities for normals.
2. Replace Layer 2 `u2` raw-vector spline with tangent-plane normal deviations.
3. Add Layer 2 normal-knot smoothness.
4. Add a floor for Layer 2 boundary-direction prior SD.
5. Add normal mean-resultant-length uncertainty diagnostic.
6. Replace the soft `phase_boundary` potential with boundary-normalized positive phase by construction.
7. Re-run the existing debug scripts.
8. Only then tune NUTS settings such as `target_accept` or treedepth.

The current best hypothesis is that the visible normal artifact is primarily caused by the old Layer 2 normal parameterization, while the divergences are primarily driven by the old soft-boundary phase model. The reparameterization should address both rather than treating the problem as mere sampler tuning.
