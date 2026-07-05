# Bayesian Two-Layer Phase Coordinate Estimation Specification

This document specifies a Bayesian extension to the existing `phase_coordinates` package. The new estimator should live next to the current deterministic cycle-fixed estimator; it should not replace `hilbert_phase` or `cycle_by_cycle_pca_coordinates`.

The target use case is 3D rhythmic movement data where we want posterior estimates of phase, radius, smoothly varying cycle-plane normal, center, and perpendicular deviation.

## Core idea

Use two models rather than one monolithic model.

1. **Coarse cycle model**: estimate dominant frequency, cycle boundaries, cycle centers, cycle-level normals, and boundary reference directions, while keeping posterior uncertainty.
2. **Instantaneous model**: use the coarse posterior summaries as priors for smoothly varying instantaneous quantities.

The coarse model estimates cycle-level objects:

$$
\tau_k,\quad T_k,\quad c_k,\quad n_k,\quad a_k
$$

The instantaneous model estimates time-level objects:

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

### Frequency and duration

Estimate a crude dominant frequency $f_0$ using periodogram, autocorrelation, Hilbert phase, or peak/event detection. Define:

$$
T_0=\frac{1}{f_0}
$$

Use a log-duration prior:

$$
\log T_k \sim \mathcal{N}(\log T_0,0.15^2)
$$

This makes duration positive and allows roughly 25--35% variation across cycles before the prior becomes strongly skeptical.

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

Use a normalized unconstrained vector. Let $u_k\in\mathbb{R}^3$:

$$
n_k=\frac{u_k}{\lVert u_k\rVert}
$$

Initialize $\hat{n}_k$ from cycle-wise PCA. Use:

$$
u_k\sim\mathcal{N}(\hat{n}_k,0.20^2I)
$$

Add angular smoothness:

$$
\cos^{-1}\left(\left|n_k^\top n_{k-1}\right|\right)\sim\operatorname{HalfNormal}(0.10)
$$

The absolute value handles normal sign ambiguity.

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

## Layer 2: instantaneous model

Layer 2 uses the Layer 1 posterior summaries as priors. This is a modular Bayesian handoff rather than a single joint model.

### Uncertainty propagation

Use a padding factor of $1.5$:

$$
\theta^{(2)}\sim\mathcal{N}\left(\mathbb{E}[\theta^{(1)}\mid X],\left(1.5\operatorname{SD}[\theta^{(1)}\mid X]\right)^2\right)
$$

For vectors:

$$
v^{(2)}\sim\mathcal{N}\left(\mathbb{E}[v^{(1)}\mid X],1.5^2\operatorname{Cov}[v^{(1)}\mid X]\right)
$$

If full covariance is inconvenient, use componentwise posterior SDs.

Use uncertainty floors:

$$
\sigma_{\tau,k}^{(2)}=\max(1.5s_{\tau,k}^{(1)},0.01T_k)
$$

$$
\sigma_{c,k}^{(2)}=\max(1.5s_{c,k}^{(1)},0.02R_X)
$$

$$
\sigma_{u,k}^{(2)}=\max(1.5s_{u,k}^{(1)},0.03)
$$

### Cubic splines

Use cubic splines for speed and smoothness. Spline unconstrained quantities and then transform.

Center:

$$
c(t)=\operatorname{CubicSpline}(\tau_k,c_k^{(2)})(t)
$$

Normal:

$$
u(t)=\operatorname{CubicSpline}(\tau_k,u_k^{(2)})(t)
$$

$$
n(t)=\frac{u(t)}{\lVert u(t)\rVert}
$$

Boundary direction:

$$
a(t)=\operatorname{CubicSpline}(\tau_k,a_k^{(2)})(t)
$$

### Boundary-anchored coordinate frame

Define the in-plane x-axis by projecting $a(t)$ into the instantaneous plane:

$$
e_1(t)=\frac{(I-n(t)n(t)^\top)a(t)}{\left\lVert (I-n(t)n(t)^\top)a(t)\right\rVert}
$$

Then:

$$
e_2(t)=n(t)\times e_1(t)
$$

This fixes the in-plane gauge and avoids a separate free in-plane spin parameter.

### Positive phase-velocity spline

Let $g(t)$ be a cubic spline with a moderate number of knots. Define:

$$
\omega(t)=\exp(g(t))
$$

Then:

$$
\phi(t)=\phi_0+\int_0^t\omega(s)\,ds
$$

Discrete implementation:

$$
\Delta\phi_t=\omega_t\Delta t
$$

$$
\phi_t=\phi_{t-1}+\Delta\phi_t
$$

This makes phase monotone by construction.

Use:

$$
g_j\sim\mathcal{N}(\log\omega_0,0.20^2)
$$

where:

$$
\omega_0=\frac{2\pi}{T_0}
$$

and:

$$
g_j-g_{j-1}\sim\mathcal{N}(0,0.15^2)
$$

Boundary phase constraints:

$$
\phi(\tau_k)-2\pi k\sim\mathcal{N}(0,0.15^2)
$$

### Radius and perpendicular deviation

Radius is positive:

$$
r(t)=\exp(h_r(t))
$$

Perpendicular deviation is unconstrained but should be smooth and zero-centered:

$$
z(t)=h_z(t)
$$

### Observation model

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

Phase monotonicity should be guaranteed by positive increments, but still include a sanity check.

## Implementation order

1. Robust scale computation and deterministic seeds.
2. Layer 1 coarse summaries or coarse Bayesian prototype.
3. Result dataclasses for estimates, uncertainty, diagnostics, and optional report.
4. Layer 2 priors from Layer 1 summaries using the $1.5$ padding rule.
5. Cubic spline helpers and positive phase velocity spline.
6. Frame construction from $n(t)$ and $a(t)$.
7. Observation model.
8. Synthetic tests with known ground truth.
