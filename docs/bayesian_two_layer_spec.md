# Bayesian Two-Layer Phase Coordinate Estimation Specification

This document specifies a Bayesian extension to the existing cycle-by-cycle PCA phase-coordinate estimator in `phase_coordinates`. The goal is to add a separate Bayesian estimation function next to the current deterministic cycle-fixed estimator, not to replace it.

The current estimator fits a fixed local PCA plane per cycle. The Bayesian extension should use a two-layer approach:

1. A coarse cycle model estimates cycle frequency, cycle boundaries, cycle centers, boundary directions, and cycle-level normals, while preserving posterior uncertainty.
2. An instantaneous model uses the coarse posterior estimates and uncertainties as priors to infer smoothly varying phase, radius, center, normal, and perpendicular deviation at each time point.

The method is designed for 3D cyclic movement data, though some pieces may generalize to higher-dimensional data later.

## Notation

Let $X_t \in \mathbb{R}^3$ be the observed trajectory at time index $t$.

Let $k$ index cycles.

Key variables:

- $\tau_k$: phase-zero boundary time for cycle $k$.
- $T_k = \tau_{k+1} - \tau_k$: cycle duration.
- $c_k \in \mathbb{R}^3$: coarse center for cycle $k$, fixed within that cycle.
- $c(t) \in \mathbb{R}^3$: smoothly varying instantaneous center.
- $n_k \in S^2$: coarse cycle plane normal for cycle $k$.
- $n(t) \in S^2$: smoothly varying instantaneous plane normal.
- $a_k = X(\tau_k) - c_k$: boundary-anchored reference direction for cycle $k$.
- $a(t)$: smooth interpolation of $a_k$.
- $e_1(t)$: instantaneous in-plane x-axis.
- $e_2(t)$: instantaneous in-plane y-axis.
- $\phi(t)$: instantaneous phase.
- $\omega(t) = d\phi/dt$: instantaneous phase velocity.
- $r(t) > 0$: instantaneous in-plane radius.
- $z(t)$: instantaneous perpendicular deviation.
- $\sigma_x$: observation noise scale.

The characteristic movement scale is computed from the data:

$$
\bar{x} = \operatorname{median}_t X_t
$$

$$
d_t = \lVert X_t - \bar{x} \rVert
$$

$$
R_X = \operatorname{median}_t d_t
$$

If $R_X$ is very small, use an RMS radius fallback:

$$
R_X = \sqrt{\frac{1}{N}\sum_t \lVert X_t - \bar{x} \rVert^2}
$$

## Layer 1: Coarse Cycle Model

The coarse cycle model estimates cycle-level objects and their posterior uncertainty.

### Initial frequency and timing priors

Estimate an initial dominant frequency $f_0$ from the data using a periodogram, autocorrelation, Hilbert phase, or another robust method. Define:

$$
T_0 = \frac{1}{f_0}
$$

Use a log-duration prior:

$$
\log T_k \sim \mathcal{N}(\log T_0, 0.15^2)
$$

This allows substantial but not arbitrary cycle-to-cycle duration variation.

Given initial candidate boundaries $\hat{\tau}_k$, use:

$$
\tau_k \sim \mathcal{N}(\hat{\tau}_k, (0.075 T_0)^2)
$$

If the initial boundary detector is weak, this coefficient may be increased to $0.10 T_0$.

### Boundary clustering likelihood

The phase-zero boundary should be the phase at which the 3D boundary points are most reproducible across cycles. This is expressed as:

$$
X(\tau_k) \sim \mathcal{N}(\mu_\tau, \Sigma_\tau)
$$

Start with an isotropic boundary cloud:

$$
\Sigma_\tau = \left(\sigma_\tau^{(x)}\right)^2 I
$$

Use a scale-relative prior:

$$
\sigma_\tau^{(x)} = R_X \rho_\tau
$$

$$
\rho_\tau \sim \operatorname{LogNormal}(\log 0.10, 0.5^2)
$$

Interpretation: phase-zero boundary points are expected to cluster within roughly $10\%$ of the movement radius, with meaningful prior mass from a few percent to roughly $25\%$.

This spatial term is not a separate timing-error model. Instead, $\tau_k$ is latent, and the likelihood favors boundary times that make $X(\tau_k)$ cluster across cycles while respecting the duration priors.

### Cycle center prior

Compute a crude initial center $\hat{c}_k$, for example from samples assigned to cycle $k$:

$$
\hat{c}_k = \operatorname{mean}_{t \in k} X_t
$$

Use:

$$
c_k \sim \mathcal{N}(\hat{c}_k, (0.25 R_X)^2 I)
$$

and a smooth cycle-to-cycle prior:

$$
c_k - c_{k-1} \sim \mathcal{N}(0, (0.10 R_X)^2 I)
$$

If the trajectory is expected to drift substantially through space, the $0.10 R_X$ coefficient may be increased to $0.20 R_X$.

### Coarse normal prior

Use a normalized unconstrained vector parameterization. Let $u_k \in \mathbb{R}^3$ and:

$$
n_k = \frac{u_k}{\lVert u_k \rVert}
$$

Initialize $\hat{n}_k$ using cycle-wise PCA. Then use:

$$
u_k \sim \mathcal{N}(\hat{n}_k, 0.20^2 I)
$$

Smoothness across cycle normals should be expressed in angular terms when practical:

$$
\cos^{-1}\left(\left|n_k^\top n_{k-1}\right|\right) \sim \operatorname{HalfNormal}(0.10)
$$

The absolute value handles the sign ambiguity of plane normals.

### Boundary direction

Define the boundary reference vector as:

$$
a_k = X(\tau_k) - c_k
$$

Do not project $a_k$ into the cycle plane in Layer 1. Projection is performed once in the instantaneous model using the instantaneous normal.

### Layer 1 outputs

The coarse model should return posterior estimates and uncertainties for:

- $\tau_k$
- $T_k$
- $c_k$
- $u_k$ and/or $n_k$
- $a_k$
- $\mu_\tau$
- $\Sigma_\tau$ or $\sigma_\tau^{(x)}$

## Layer 2: Instantaneous Model

The instantaneous model uses Layer 1 posterior summaries as priors. It estimates smoothly varying instantaneous quantities.

### Modular uncertainty propagation

For scalar quantities, use:

$$
\theta^{(2)} \sim \mathcal{N}\left(\mathbb{E}[\theta^{(1)} \mid X], \left(1.5\operatorname{SD}[\theta^{(1)} \mid X]\right)^2\right)
$$

For vector quantities:

$$
v^{(2)} \sim \mathcal{N}\left(\mathbb{E}[v^{(1)} \mid X], 1.5^2\operatorname{Cov}[v^{(1)} \mid X]\right)
$$

If full covariance propagation is too cumbersome, use componentwise posterior standard deviations.

For positive quantities, propagate uncertainty on the log scale:

$$
\log q^{(2)} \sim \mathcal{N}\left(\mathbb{E}[\log q^{(1)} \mid X], \left(1.5\operatorname{SD}[\log q^{(1)} \mid X]\right)^2\right)
$$

Use uncertainty floors to avoid overconfident Layer 1 handoff:

$$
\sigma_{\tau,k}^{(2)} = \max\left(1.5s_{\tau,k}^{(1)}, 0.01T_k\right)
$$

$$
\sigma_{c,k}^{(2)} = \max\left(1.5s_{c,k}^{(1)}, 0.02R_X\right)
$$

$$
\sigma_{u,k}^{(2)} = \max\left(1.5s_{u,k}^{(1)}, 0.03\right)
$$

### Spline interpolation

Use cubic splines for speed, smoothness, and exact interpolation at knot values.

Spline unconstrained quantities, then transform as needed.

For centers:

$$
c(t) = \operatorname{CubicSpline}(\tau_k, c_k^{(2)})(t)
$$

For normals:

$$
u(t) = \operatorname{CubicSpline}(\tau_k, u_k^{(2)})(t)
$$

$$
n(t) = \frac{u(t)}{\lVert u(t) \rVert}
$$

For the boundary reference vector:

$$
a(t) = \operatorname{CubicSpline}(\tau_k, a_k^{(2)})(t)
$$

### Boundary-anchored frame

Define the in-plane x-axis by projecting $a(t)$ into the instantaneous plane:

$$
e_1(t) = \frac{\left(I - n(t)n(t)^\top\right)a(t)}{\left\lVert \left(I - n(t)n(t)^\top\right)a(t) \right\rVert}
$$

Define:

$$
e_2(t) = n(t) \times e_1(t)
$$

This fixes the in-plane gauge. It avoids a free in-plane spin parameter and reduces confounding between frame rotation and phase.

### Positive phase-velocity spline

Model phase velocity as a positive spline. Let $g(t)$ be a cubic spline with a moderate number of knots, and define:

$$
\omega(t) = \exp(g(t))
$$

Then:

$$
\phi(t) = \phi_0 + \int_0^t \omega(s)\,ds
$$

Discrete implementation:

$$
\Delta \phi_t = \omega_t \Delta t
$$

$$
\phi_t = \phi_{t-1} + \Delta \phi_t
$$

This makes phase monotonic by construction.

For phase-velocity spline knots $g_j$, use:

$$
g_j \sim \mathcal{N}(\log \omega_0, 0.20^2)
$$

where:

$$
\omega_0 = \frac{2\pi}{T_0}
$$

Add smoothness:

$$
g_j - g_{j-1} \sim \mathcal{N}(0, 0.15^2)
$$

Boundary phase constraints:

$$
\phi(\tau_k) - 2\pi k \sim \mathcal{N}(0, 0.15^2)
$$

### Radius and perpendicular deviation

Use a positive radius:

$$
r(t) = \exp(h_r(t))
$$

where $h_r(t)$ is a cubic spline or smooth latent function.

Use a zero-centered perpendicular deviation:

$$
z(t) = h_z(t)
$$

where $h_z(t)$ is a smooth latent function with a prior favoring smaller magnitude than the radius.

### Observation model

The instantaneous observation model is:

$$
X_t \sim \mathcal{N}\left(c(t) + e_1(t)r(t)\cos\phi(t) + e_2(t)r(t)\sin\phi(t) + n(t)z(t), \sigma_x^2 I\right)
$$

Use a scale-relative prior for observation noise:

$$
\sigma_x = R_X \rho_x
$$

$$
\rho_x \sim \operatorname{LogNormal}(\log 0.03, 0.5^2)
$$

This says the model should usually explain the trajectory to within a few percent of movement radius, but allows larger residuals.

## Returned Artifacts

Return three levels of output.

### 1. Estimates

Tidy arrays or DataFrames with posterior mean or median estimates for practical analysis:

- $\tau_k$
- $c_k$
- $n_k$
- $a_k$
- $\phi(t)$
- $\omega(t)$
- $c(t)$
- $n(t)$
- $e_1(t)$
- $e_2(t)$
- $r(t)$
- $z(t)$
- model-predicted trajectory

### 2. Uncertainty

Summaries such as posterior SDs and credible intervals:

- posterior SDs for cycle characteristics
- credible intervals for $\phi(t)$, $r(t)$, $z(t)$, $n(t)$, and $c(t)$
- diagnostic scalar summaries

### 3. Full Bayesian report

Optional full posterior object, for example an ArviZ `InferenceData` object. This should only be retained if requested, e.g. `return_report=True`.

If `return_report=False`, discard full posterior draws after producing summaries.

## Diagnostics

Diagnostics should be classified as hard failures or warnings.

### Hard failures

Boundary posterior multimodality:

- fail if two posterior modes each have mass greater than $0.20$ and are separated by more than $0.20T_k$.

Boundary phase does not produce a reproducible 3D boundary point:

$$
\rho_\tau = \frac{\sqrt{\operatorname{tr}(\Sigma_\tau)}}{R_X}
$$

- warn if $\rho_\tau > 0.25$.
- fail if $\rho_\tau > 0.40$.

Projection of $a(t)$ into the plane is near zero:

$$
p_t = \left\lVert \left(I - n(t)n(t)^\top\right)a(t) \right\rVert
$$

- warn if $p_t/\lVert a(t) \rVert < 0.20$.
- fail if $p_t/\lVert a(t) \rVert < 0.10$.

### Warnings

Normal changes dominated by the prior:

Let $\Delta n$ be the angular shift between prior and posterior means, and let $\sigma_{n,\text{prior}}$ be the prior angular SD.

Warn if:

$$
\Delta n < 0.25\sigma_{n,\text{prior}}
$$

and:

$$
\sigma_{n,\text{posterior}} > 0.75\sigma_{n,\text{prior}}
$$

Large perpendicular deviation relative to radius:

$$
\rho_z(t) = \frac{|z(t)|}{r(t) + \epsilon}
$$

- warn if $\operatorname{median}_t \rho_z(t) > 0.25$.
- warn strongly if $\operatorname{median}_t \rho_z(t) > 0.50$.
- warn if more than $10\%$ of samples have $\rho_z(t) > 0.50$.

Center drift stealing cyclic structure:

$$
D_c = \operatorname{range}_t \lVert c(t) - \bar{c} \rVert
$$

- warn if $D_c/R_X > 0.25$.
- warn strongly if $D_c/R_X > 0.50$.

Phase velocity degeneracy:

$$
\frac{\omega_{95}}{\omega_5}
$$

- warn if $\omega_{95}/\omega_5 > 3$.
- warn strongly if $\omega_{95}/\omega_5 > 5$.

Observation noise too large:

- warn if $\sigma_x/R_X > 0.10$.
- warn strongly if $\sigma_x/R_X > 0.25$.

Phase monotonicity should be guaranteed by the positive phase increment parameterization, but still include a sanity check for implementation bugs.

## Implementation Notes

The first implementation should prioritize a stable, testable prototype over maximal modeling elegance.

Recommended implementation order:

1. Implement robust scale computation and initial deterministic seeds.
2. Implement Layer 1 coarse model or a deterministic-plus-uncertainty approximation if full Bayesian sampling is too heavy.
3. Define result dataclasses for estimates, uncertainty, diagnostics, and optional full Bayesian report.
4. Implement Layer 2 priors from Layer 1 summaries with $1.5$ padding and uncertainty floors.
5. Implement cubic spline helpers and positive phase velocity spline.
6. Implement frame construction from $n(t)$ and $a(t)$.
7. Implement the observation model.
8. Add tests with synthetic data where ground truth is known.

The public API should be separate from the existing deterministic function. Do not replace `cycle_by_cycle_pca_coordinates`.
