"""
Bayesian two-layer phase-coordinate estimator.

Implements the model described in ``docs/bayesian_two_layer_spec.md``: a
coarse cycle-level model (Layer 1) estimating boundary times, cycle centers,
and cycle normals with posterior uncertainty, followed by an instantaneous
model (Layer 2) that uses the Layer 1 posterior summaries as priors for
smoothly varying phase, center, normal, radius, and perpendicular deviation.

This module is independent of :mod:`phase_coordinates.core` and does not
replace :func:`phase_coordinates.core.hilbert_phase` or
:func:`phase_coordinates.core.cycle_by_cycle_pca_coordinates`.

PyMC and ArviZ are optional dependencies. They are imported lazily so that
importing this module (and the rest of ``phase_coordinates``) never requires
them. Install with ``pip install -e .[bayes]`` to use
:func:`fit_bayesian_phase_coordinates`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks, periodogram

_BAYES_INSTALL_HINT = (
    "fit_bayesian_phase_coordinates() requires the optional 'pymc' and "
    "'arviz' dependencies, which are not installed.\n\n"
    "Install them with:\n\n"
    "    pip install -e .[bayes]\n"
)


def _import_pymc():
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - exercised via mock/skip
        raise ImportError(_BAYES_INSTALL_HINT) from exc
    return pm


def _import_pytensor_tensor():
    try:
        import pytensor.tensor as pt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_BAYES_INSTALL_HINT) from exc
    return pt


def _import_arviz():
    try:
        import arviz as az
    except ImportError as exc:  # pragma: no cover
        raise ImportError(_BAYES_INSTALL_HINT) from exc
    return az


def _numba_available() -> bool:
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True




# ---------------------------------------------------------------------------
# Data-derived scale (spec: "Data-derived scale")
# ---------------------------------------------------------------------------

def robust_movement_scale(X):
    """
    Compute a robust characteristic movement scale ``R_X`` and center ``xbar``.

    ``R_X`` is the median distance of ``X`` from its median point, falling
    back to an RMS distance if the median is degenerate (e.g. because more
    than half the samples sit exactly at the median point).
    """
    X = np.asarray(X, dtype=float)
    xbar = np.median(X, axis=0)
    dist = np.linalg.norm(X - xbar, axis=1)
    R_X = float(np.median(dist))
    if not np.isfinite(R_X) or R_X < 1e-9:
        R_X = float(np.sqrt(np.mean(dist**2)))
    if not np.isfinite(R_X) or R_X < 1e-9:
        R_X = 1.0
    return R_X, xbar


# ---------------------------------------------------------------------------
# Deterministic seeds (spec: "Frequency and duration", "Boundary times")
# ---------------------------------------------------------------------------

def dominant_reference_signal(X):
    """Top principal-component score series of mean-centered ``X``."""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[0]


def estimate_dominant_period(ref_signal, fs):
    """Estimate the dominant period ``T0`` of a scalar signal via periodogram."""
    ref_signal = np.asarray(ref_signal, dtype=float)
    freqs, power = periodogram(ref_signal, fs=fs)
    valid = freqs > 0
    if not np.any(valid):
        raise ValueError(
            "Cannot estimate a dominant frequency: signal is too short or "
            "has no positive-frequency content."
        )
    f0 = float(freqs[valid][np.argmax(power[valid])])
    if f0 <= 0:
        raise ValueError("Estimated dominant frequency is non-positive.")
    return 1.0 / f0


def seed_boundary_indices(ref_signal, fs, T0):
    """
    Detect candidate cycle-boundary sample indices as positive peaks of a
    reference signal, spaced at roughly the dominant period.

    Returns integer sample indices ``tau_idx`` (length ``K``), defining
    ``K - 1`` candidate cycles.
    """
    distance = max(1, int(0.6 * T0 * fs))
    peaks, _ = find_peaks(ref_signal, distance=distance)
    if len(peaks) < 3:
        raise ValueError(
            "Could not detect at least 3 boundary events (>= 2 complete "
            "cycles) from the data. Provide a longer / cleaner recording."
        )
    return peaks.astype(int)


def seed_cycle_centers(X, tau_idx):
    """Per-cycle mean position, one row per cycle between consecutive boundaries."""
    X = np.asarray(X, dtype=float)
    return np.array(
        [X[tau_idx[k] : tau_idx[k + 1]].mean(axis=0) for k in range(len(tau_idx) - 1)]
    )


def seed_cycle_normals(X, tau_idx):
    """
    Per-cycle unit plane normal (least-variance direction via SVD), with signs
    made continuous from one cycle to the next.
    """
    X = np.asarray(X, dtype=float)
    normals = []
    prev = None
    for k in range(len(tau_idx) - 1):
        seg = X[tau_idx[k] : tau_idx[k + 1]]
        c = seg.mean(axis=0)
        _, _, vt = np.linalg.svd(seg - c, full_matrices=False)
        n = vt[-1]
        if prev is not None and np.dot(n, prev) < 0:
            n = -n
        normals.append(n)
        prev = n
    return np.array(normals)


def seed_boundary_vectors(X, tau_idx, cycle_centers):
    """Boundary-anchored reference direction ``a_k = X(tau_k) - c_k`` per cycle."""
    X = np.asarray(X, dtype=float)
    return X[tau_idx[:-1]] - cycle_centers


# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------

def normalize(v, axis=-1, eps=1e-12):
    """Normalize vectors along ``axis`` to unit length."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.clip(norm, eps, None)


def align_normal_signs(normals):
    """
    Flip signs of consecutive normals so adjacent pairs have positive dot
    product. This removes the hemispheric sign ambiguity from Layer 1 normal
    estimates before using them as reference directions in Layer 2.
    """
    normals = np.array(normals, dtype=float)
    aligned = normals.copy()
    for j in range(1, len(aligned)):
        if np.dot(aligned[j], aligned[j - 1]) < 0:
            aligned[j] = -aligned[j]
    return aligned


def orthonormal_tangent_basis(normals, eps=1e-12):
    """
    Return an orthonormal tangent basis ``Q`` with shape ``(n, 3, 2)`` such
    that for each row ``j``:

    - ``Q[j].T @ Q[j] == I_2``
    - ``Q[j].T @ normals[j] == 0`` (columns lie in tangent plane of normal j)

    Construction: for each normal, choose the coordinate axis least aligned
    with it, project that axis into the tangent plane, normalise to get ``t1``,
    then take the cross product with the normal for ``t2``.
    """
    normals = normalize(np.asarray(normals, dtype=float))
    n_knots = len(normals)
    Q = np.zeros((n_knots, 3, 2))
    for j in range(n_knots):
        m = normals[j]
        # Choose the coordinate axis most perpendicular to m
        i_min = int(np.argmin(np.abs(m)))
        ref = np.zeros(3)
        ref[i_min] = 1.0
        t1 = ref - np.dot(ref, m) * m
        t1 = t1 / max(np.linalg.norm(t1), eps)
        t2 = np.cross(m, t1)
        t2 = t2 / max(np.linalg.norm(t2), eps)
        Q[j, :, 0] = t1
        Q[j, :, 1] = t2
    return Q


def construct_frame(n, a, eps=1e-12):
    """
    Build the boundary-anchored in-plane frame ``(e1, e2)`` from unit normal(s)
    ``n`` and boundary direction(s) ``a`` (spec: "Boundary-anchored coordinate
    frame").

    Returns ``(e1, e2, projection_norm)`` where ``projection_norm`` is
    ``||(I - n n^T) a||``, useful for the projection-failure diagnostic.
    """
    n = normalize(np.asarray(n, dtype=float), axis=-1, eps=eps)
    a = np.asarray(a, dtype=float)
    proj = a - n * np.sum(n * a, axis=-1, keepdims=True)
    projection_norm = np.linalg.norm(proj, axis=-1)
    e1 = normalize(proj, axis=-1, eps=eps)
    e2 = np.cross(n, e1)
    return e1, e2, projection_norm


# ---------------------------------------------------------------------------
# Cubic spline helpers
# ---------------------------------------------------------------------------

def cubic_spline_matrix(knot_x, eval_x, bc_type="natural"):
    """
    Build the fixed linear operator ``B`` (shape ``(len(eval_x), len(knot_x))``)
    such that ``CubicSpline(knot_x, y, bc_type=bc_type)(eval_x) == B @ y`` for
    any knot values ``y``.

    Because a natural cubic spline is a linear function of its knot values
    when the knot locations are fixed, this lets spline evaluation appear as
    a plain matrix multiply inside a PyMC model (fully differentiable, no
    special PyTensor Op needed).
    """
    knot_x = np.asarray(knot_x, dtype=float)
    eval_x = np.asarray(eval_x, dtype=float)
    n_knots = len(knot_x)
    B = np.empty((len(eval_x), n_knots))
    eye = np.eye(n_knots)
    for j in range(n_knots):
        cs = CubicSpline(knot_x, eye[j], bc_type=bc_type, extrapolate=True)
        B[:, j] = cs(eval_x)
    return B


def spline_eval(knot_x, knot_y, eval_x, bc_type="natural"):
    """Evaluate a (possibly vector-valued) natural cubic spline at ``eval_x``."""
    cs = CubicSpline(knot_x, knot_y, axis=0, bc_type=bc_type, extrapolate=True)
    return cs(np.asarray(eval_x, dtype=float))


def _linear_interp_matrix(t_grid, eval_t):
    """
    Fixed linear-interpolation operator ``L`` (shape ``(len(eval_t), len(t_grid))``)
    such that ``L @ y`` linearly interpolates samples ``y`` (defined on
    ``t_grid``) at query points ``eval_t``. Used where both the source grid
    and the query points are constants (not model parameters).
    """
    t_grid = np.asarray(t_grid, dtype=float)
    eval_t = np.asarray(eval_t, dtype=float)
    n = len(t_grid)
    L = np.zeros((len(eval_t), n))
    for i, t in enumerate(eval_t):
        j = np.searchsorted(t_grid, t)
        j = int(np.clip(j, 1, n - 1))
        t0, t1 = t_grid[j - 1], t_grid[j]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        w = float(np.clip(w, 0.0, 1.0))
        L[i, j - 1] = 1.0 - w
        L[i, j] = w
    return L



# ---------------------------------------------------------------------------
# Result containers (spec: "Return values")
# ---------------------------------------------------------------------------

@dataclass
class BayesianPhaseEstimates:
    """Posterior mean point estimates (spec: "Estimates")."""

    # Cycle-level (Layer 1)
    tau: np.ndarray                  # (K,) boundary times, seconds
    period: np.ndarray               # (K-1,) cycle durations T_k, seconds
    cycle_center: np.ndarray         # (K-1, 3)
    cycle_normal: np.ndarray         # (K-1, 3)
    boundary_direction: np.ndarray   # (K-1, 3) a_k

    # Instantaneous (Layer 2), one row per time sample in the fitted window
    time: np.ndarray                 # (n_time,) seconds
    phase: np.ndarray                # (n_time,)
    phase_velocity: np.ndarray       # (n_time,)
    center: np.ndarray               # (n_time, 3)
    normal: np.ndarray               # (n_time, 3)
    e1: np.ndarray                   # (n_time, 3)
    e2: np.ndarray                   # (n_time, 3)
    radius: np.ndarray               # (n_time,)
    perp_deviation: np.ndarray       # (n_time,)
    predicted_trajectory: np.ndarray  # (n_time, 3)


@dataclass
class BayesianPhaseUncertainty:
    """Posterior SDs / credible half-widths for key quantities (spec: "Uncertainty")."""

    tau_sd: np.ndarray                 # (K,)
    period_sd: np.ndarray              # (K-1,)
    cycle_center_sd: np.ndarray        # (K-1, 3)
    cycle_normal_angular_sd: np.ndarray  # (K-1,) radians
    boundary_direction_sd: np.ndarray  # (K-1, 3)

    phase_sd: np.ndarray               # (n_time,)
    phase_velocity_sd: np.ndarray      # (n_time,)
    center_sd: np.ndarray              # (n_time, 3)
    normal_angular_sd: np.ndarray      # (n_time,) radians
    radius_sd: np.ndarray              # (n_time,)
    perp_deviation_sd: np.ndarray      # (n_time,)

    observation_noise_sd: float        # sigma_x posterior mean


@dataclass
class BayesianPhaseDiagnostics:
    """
    Diagnostics from the spec's "Diagnostics" section. ``failures`` are hard
    failures; ``warnings`` includes both plain and "strongly" flagged
    warnings (text-prefixed with ``STRONG:``).
    """

    failures: list
    warnings: list

    boundary_multimodal: bool
    rho_tau: float                    # sqrt(tr(Sigma_tau)) / R_X
    projection_ratio: np.ndarray      # p_t / ||a(t)|| per time sample
    normal_prior_dominated: list      # cycle indices flagged
    normal_resultant_length: np.ndarray  # ||E[n(t)|X]|| (n_time,); low values flag misleading means
    rho_z_median: float
    center_drift_ratio: float         # D_c / R_X
    omega_ratio: float                # omega_95 / omega_5
    sigma_x_over_RX: float
    phase_monotonic: bool

    @property
    def ok(self) -> bool:
        """``True`` if there are no hard failures."""
        return len(self.failures) == 0


@dataclass
class BayesianPhaseResult:
    """Top-level result of :func:`fit_bayesian_phase_coordinates`."""

    estimates: BayesianPhaseEstimates
    uncertainty: BayesianPhaseUncertainty
    diagnostics: BayesianPhaseDiagnostics
    bayesian_report: Optional[Any] = None


def _pt_interp_at(t_grid_const, X_grid_const, tau, n_grid, pt):
    """
    Differentiable linear interpolation of a fixed data grid ``X_grid_const``
    (defined at fixed times ``t_grid_const``) evaluated at PyTensor query
    times ``tau``. Used for the Layer 1 boundary-clustering likelihood, where
    the query points (``tau_k``) are themselves latent variables.
    """
    idx = pt.extra_ops.searchsorted(t_grid_const, tau)
    idx = pt.clip(idx, 1, n_grid - 1)
    t0 = t_grid_const[idx - 1]
    t1 = t_grid_const[idx]
    w = pt.clip((tau - t0) / (t1 - t0), 0.0, 1.0)
    X0 = X_grid_const[idx - 1]
    X1 = X_grid_const[idx]
    return X0 + w[:, None] * (X1 - X0)


# ---------------------------------------------------------------------------
# Layer 1: coarse cycle model
# ---------------------------------------------------------------------------

# Prior/likelihood defaults from docs/bayesian_two_layer_spec.md
_LOG_DURATION_SD = 0.15
_BOUNDARY_TIMING_SD_FRAC = 0.075          # * T0
_BOUNDARY_SCATTER_LOGNORMAL_MU = np.log(0.10)
_BOUNDARY_SCATTER_LOGNORMAL_SD = 0.5
_CENTER_PRIOR_SD_FRAC = 0.25              # * R_X
_CENTER_CHANGE_SD_FRAC = 0.10             # * R_X
_NORMAL_VECTOR_SD = 0.20
_NORMAL_ANGLE_SD = 0.10                   # radians, HalfNormal

_LAYER2_PADDING = 1.5
_LAYER2_TAU_FLOOR_FRAC = 0.01             # * T_k
_LAYER2_CENTER_FLOOR_FRAC = 0.02          # * R_X
_LAYER2_NORMAL_FLOOR = 0.03

_OBS_NOISE_LOGNORMAL_MU = np.log(0.03)
_OBS_NOISE_LOGNORMAL_SD = 0.5     # reverted: 0.3 caused divergences; correct-mode initvals now prevent wrong-mode escape

_PHASE_VELOCITY_LOGKNOT_SD = 0.20
_PHASE_VELOCITY_SMOOTHNESS_SD = 0.15
_PHASE_BOUNDARY_SD = 0.15                 # kept for reference; no longer used in Layer 2
_LAYER2_BOUNDARY_DIR_FLOOR_FRAC = 0.02   # * R_X, floor for sigma_a2 (spec §"Floor for sigma_a2")
_LAYER2_NORMAL2_SMOOTHNESS_SD = 0.10     # sigma_Delta_n in spec §"Layer 2 normal smoothness"


@dataclass
class _Layer1Summary:
    """Internal posterior summary handed from Layer 1 to Layer 2."""

    tau_mean: np.ndarray
    tau_sd: np.ndarray
    period_mean: np.ndarray
    period_sd: np.ndarray
    center_mean: np.ndarray
    center_sd: np.ndarray
    u_hat: np.ndarray
    u_mean: np.ndarray
    u_sd: np.ndarray
    normal_mean: np.ndarray
    normal_angular_sd: np.ndarray
    boundary_direction_mean: np.ndarray
    boundary_direction_sd: np.ndarray
    rho_tau_mean: float
    idata: Any


def _sample_kwargs(draws, tune, chains, target_accept, random_seed, use_numba, initvals=None):
    # Seeds (tau_hat, c_hat, u_hat, ...) are informative and already near the
    # mode. Explicit initvals start every chain exactly there rather than at
    # a jittered point; this matters a lot for the unconstrained-vector normal
    # parameterization (u ~ N(u_hat, 0.2)), where a jittered start was
    # observed to occasionally land near a degenerate sign-flipped normal,
    # causing divergences and a stuck chain.
    #
    # nutpie (if installed) would otherwise be preferred here -- it adapts a
    # richer mass matrix than PyMC's diagonal adapt_diag, which matters
    # because several sub-models couple parameters nonlinearly (e.g. the
    # log-velocity spline feeding a cumulative sum that must hit tight
    # per-boundary phase targets), producing a curved, correlated posterior
    # that a diagonal mass matrix struggles with. But PyMC does not forward
    # `initvals` to nutpie (only a raw `init_mean` array in nutpie's own
    # flattened/transformed parameter order, an undocumented private-API
    # detail not worth depending on). Without controlled initialization,
    # nutpie's own default init reintroduced the same sign-flip failure mode
    # (observed as a wild spline excursion between two knots of opposite
    # sign). Plain PyMC NUTS with explicit initvals is slower per model but
    # was verified to reliably avoid this.
    kwargs = dict(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=1,  # avoid Windows multiprocessing re-import issues under pytest
        progressbar=False,
        random_seed=random_seed,
        nuts_sampler="pymc",
        target_accept=target_accept,
        init="adapt_diag",
        initvals=initvals,
    )
    if use_numba:
        kwargs["compile_kwargs"] = {"mode": "NUMBA"}
    return kwargs


def _fit_layer1(
    X,
    fs,
    tau_idx,
    T0,
    R_X,
    xbar,
    draws,
    tune,
    chains,
    target_accept,
    random_seed,
    use_numba,
):
    pm = _import_pymc()
    pt = _import_pytensor_tensor()
    az = _import_arviz()

    n_time = X.shape[0]
    t_grid = np.arange(n_time) / fs
    K = len(tau_idx)

    tau_hat = tau_idx / fs
    c_hat = seed_cycle_centers(X, tau_idx)
    n_hat = seed_cycle_normals(X, tau_idx)
    u_hat = n_hat.copy()  # unconstrained-vector seed = unit normal seed

    t_grid_const = pt.constant(t_grid, name="t_grid")
    X_const = pt.constant(X, name="X_grid")

    with pm.Model():
        tau = pm.Normal("tau", mu=tau_hat, sigma=_BOUNDARY_TIMING_SD_FRAC * T0, shape=K)

        T = pm.Deterministic("T", tau[1:] - tau[:-1])
        T_safe = pt.maximum(T, 1e-3 * T0)
        pm.Potential(
            "log_duration_prior",
            pm.logp(
                pm.Normal.dist(mu=np.log(T0), sigma=_LOG_DURATION_SD), pt.log(T_safe)
            ).sum(),
        )

        c = pm.Normal("c", mu=c_hat, sigma=_CENTER_PRIOR_SD_FRAC * R_X, shape=c_hat.shape)
        if K - 1 > 1:
            dc = c[1:] - c[:-1]
            pm.Potential(
                "center_smoothness",
                pm.logp(
                    pm.Normal.dist(mu=0.0, sigma=_CENTER_CHANGE_SD_FRAC * R_X), dc
                ).sum(),
            )

        u = pm.Normal("u", mu=u_hat, sigma=_NORMAL_VECTOR_SD, shape=u_hat.shape)
        u_norm = pt.sqrt(pt.sum(u**2, axis=-1, keepdims=True))
        n = pm.Deterministic("n", u / u_norm)
        if K - 1 > 1:
            # Spec: cos^-1(|n_k . n_{k-1}|) ~ HalfNormal(0.10) ("|.| handles
            # sign ambiguity"). Implemented via a smooth small-angle-equivalent
            # proxy rather than literal arccos: HalfNormal(sigma).logpdf(x) is
            # -x^2/(2 sigma^2) + const, and for small angles x^2 = arccos(c)^2
            # ~= 2*(1-c). Substituting gives -(1-|cos_angle|)/sigma^2, which
            # matches the spec's prior in the small-angle regime it targets
            # (sigma = 0.10 rad) but stays C-infinity in cos_angle everywhere
            # -- unlike arccos, whose gradient diverges as |cos_angle| -> 1,
            # i.e. exactly at the alignment the prior rewards most. Literal
            # arccos caused visible sampler pathology (one chain converging to
            # a sign-flipped normal ~10 prior-SDs from its own u prior).
            cos_angle = pt.abs(pt.sum(n[1:] * n[:-1], axis=-1))
            pm.Potential(
                "normal_smoothness",
                (-(1.0 - cos_angle) / _NORMAL_ANGLE_SD**2).sum(),
            )

        X_tau = _pt_interp_at(t_grid_const, X_const, tau, n_time, pt)
        pm.Deterministic("a", X_tau[:-1] - c)

        mu_tau = pm.Normal("mu_tau", mu=xbar, sigma=R_X, shape=3)
        rho_tau = pm.Lognormal(
            "rho_tau", mu=_BOUNDARY_SCATTER_LOGNORMAL_MU, sigma=_BOUNDARY_SCATTER_LOGNORMAL_SD
        )
        sigma_tau_x = R_X * rho_tau
        pm.Potential(
            "boundary_cluster",
            pm.logp(pm.Normal.dist(mu=mu_tau, sigma=sigma_tau_x), X_tau).sum(),
        )

        initvals = {
            "tau": tau_hat,
            "c": c_hat,
            "u": u_hat,
            "mu_tau": xbar,
            "rho_tau": 0.10,
        }
        idata1 = pm.sample(
            **_sample_kwargs(
                draws, tune, chains, target_accept, random_seed, use_numba, initvals
            )
        )

    post = idata1.posterior
    tau_mean = post["tau"].mean(("chain", "draw")).values
    tau_sd = post["tau"].std(("chain", "draw")).values
    T_mean = post["T"].mean(("chain", "draw")).values
    T_sd = post["T"].std(("chain", "draw")).values
    c_mean = post["c"].mean(("chain", "draw")).values
    c_sd = post["c"].std(("chain", "draw")).values
    u_mean = post["u"].mean(("chain", "draw")).values
    u_sd = post["u"].std(("chain", "draw")).values
    n_mean = normalize(post["n"].mean(("chain", "draw")).values)
    a_mean = post["a"].mean(("chain", "draw")).values
    a_sd = post["a"].std(("chain", "draw")).values
    rho_tau_mean = float(post["rho_tau"].mean(("chain", "draw")).values)

    # Angular SD of the posterior normal-vector samples around their mean
    # direction, per cycle (used for the "normal dominated by prior"
    # diagnostic and reported uncertainty).
    n_samples = post["n"].values.reshape(-1, K - 1, 3)
    normal_angular_sd = np.empty(K - 1)
    for k in range(K - 1):
        cos_to_mean = np.clip(n_samples[:, k, :] @ n_mean[k], -1.0, 1.0)
        normal_angular_sd[k] = np.std(np.arccos(np.abs(cos_to_mean)))

    return _Layer1Summary(
        tau_mean=tau_mean,
        tau_sd=tau_sd,
        period_mean=T_mean,
        period_sd=T_sd,
        center_mean=c_mean,
        center_sd=c_sd,
        u_hat=u_hat,
        u_mean=u_mean,
        u_sd=u_sd,
        normal_mean=n_mean,
        normal_angular_sd=normal_angular_sd,
        boundary_direction_mean=a_mean,
        boundary_direction_sd=a_sd,
        rho_tau_mean=rho_tau_mean,
        idata=idata1,
    )


def _pt_cross(a, b, pt):
    """Cross product for the last axis of two ``(..., 3)`` PyTensor tensors."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return pt.stack(
        [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1
    )


def _default_n_velocity_knots(n_cycles):
    return int(np.clip(2 * n_cycles, 4, 20))


@dataclass
class _Layer2Summary:
    """Internal posterior summary of the Layer 2 instantaneous model."""

    time: np.ndarray
    phase_mean: np.ndarray
    phase_sd: np.ndarray
    phase_velocity_mean: np.ndarray
    phase_velocity_sd: np.ndarray
    center_mean: np.ndarray
    center_sd: np.ndarray
    normal_mean: np.ndarray           # normalized posterior mean direction (n_time, 3)
    normal_raw_mean: np.ndarray       # unnormalized posterior mean (n_time, 3) for resultant length
    normal_resultant_length: np.ndarray  # ||E[n(t)|X]|| (n_time,), < 1 flags bimodal normal
    normal_angular_sd: np.ndarray
    e1_mean: np.ndarray
    e2_mean: np.ndarray
    boundary_direction_mean: np.ndarray
    projection_norm_mean: np.ndarray
    radius_mean: np.ndarray
    radius_sd: np.ndarray
    perp_deviation_mean: np.ndarray
    perp_deviation_sd: np.ndarray
    predicted_trajectory_mean: np.ndarray
    sigma_x_mean: float
    idata: Any


def _fit_layer2(
    X,
    fs,
    layer1,
    T0,
    R_X,
    n_velocity_knots,
    draws,
    tune,
    chains,
    target_accept,
    random_seed,
    use_numba,
    _sigma_x_override=None,
    _init_override=None,
):
    """
    Instantaneous model (spec: "Layer 2: instantaneous model"). Uses the
    Layer 1 posterior summaries as priors for smoothly varying phase, center,
    normal, radius, and perpendicular deviation.

    Reparameterization (vs. original implementation):

    1. Normal: tangent-plane deviations (delta_n) around Layer 1 posterior
       mean normals, not raw unconstrained 3D vector spline (u2).  Prevents
       the near-zero spline interpolant that produced localized normal artifacts.

    2. Phase: boundary-normalized positive speed -- phi(t) satisfies
       phi(tau_k) = 2*pi*k exactly by construction.  Removes the tight
       phase_boundary soft-Potential that was causing divergences.

    3. sigma_a2 now has a floor of 0.02*R_X (previously had no floor).

    Fixed deviation from literal spec: spline knot *locations* are fixed at
    tau_mean rather than being re-drawn Layer 2 free parameters, keeping the
    spline linear in its knot values (a plain matrix multiply inside PyMC).
    """
    pm = _import_pymc()
    pt = _import_pytensor_tensor()

    tau_mean = layer1.tau_mean
    K = len(tau_mean)

    i0 = max(0, int(np.ceil(tau_mean[0] * fs)))
    i1 = min(X.shape[0] - 1, int(np.floor(tau_mean[-1] * fs)))
    if i1 - i0 < 10:
        raise ValueError(
            "Not enough time samples spanned by the detected cycle boundaries "
            "to fit the instantaneous (Layer 2) model."
        )
    X_fit = X[i0 : i1 + 1]
    t_fit = np.arange(i0, i1 + 1) / fs
    n_time = X_fit.shape[0]
    dt = 1.0 / fs

    def pad(v):
        return np.vstack([v, v[-1:]])

    c_mean_p = pad(layer1.center_mean)
    c_sd_p = pad(layer1.center_sd)
    a_mean_p = pad(layer1.boundary_direction_mean)
    a_sd_p = pad(layer1.boundary_direction_sd)

    sigma_c2 = np.maximum(_LAYER2_PADDING * c_sd_p, _LAYER2_CENTER_FLOOR_FRAC * R_X)
    # Floor added for boundary-direction SD (deviation #4 resolved)
    sigma_a2 = np.maximum(_LAYER2_PADDING * a_sd_p, _LAYER2_BOUNDARY_DIR_FLOOR_FRAC * R_X)

    # --- Sign-aligned Layer 1 normals and tangent bases for Layer 2 ---
    # Pad angular SD from (K-1,) to (K,) then apply 1.5x padding + floor
    n_angular_sd_raw = layer1.normal_angular_sd                        # (K-1,)
    n_angular_sd_p = np.append(n_angular_sd_raw, n_angular_sd_raw[-1])  # (K,)
    sigma_theta2 = np.maximum(_LAYER2_PADDING * n_angular_sd_p, _LAYER2_NORMAL_FLOOR)  # (K,)
    n_mean_p = normalize(pad(layer1.normal_mean))  # (K, 3) padded, re-normalised
    m_p = align_normal_signs(n_mean_p)             # (K, 3) sign-aligned adjacent pairs
    Q_p = orthonormal_tangent_basis(m_p)           # (K, 3, 2)

    # --- Spline matrices ---
    B = cubic_spline_matrix(tau_mean, t_fit)  # (n_time, K) from boundary knots to t_fit
    B_const = pt.constant(B)

    if n_velocity_knots is None:
        n_velocity_knots = _default_n_velocity_knots(K - 1)
    vel_knot_t = np.linspace(t_fit[0], t_fit[-1], n_velocity_knots)
    Bg = cubic_spline_matrix(vel_knot_t, t_fit)
    Bg_const = pt.constant(Bg)

    # --- Per-cycle membership masks for boundary-normalized phase ---
    # cycle_idx_arr[i] = k  iff  tau_mean[k] <= t_fit[i] < tau_mean[k+1]
    cycle_idx_arr = np.searchsorted(tau_mean, t_fit, side="right") - 1
    cycle_idx_arr = np.clip(cycle_idx_arr, 0, K - 2)
    cycle_phase_offset = (2 * np.pi * cycle_idx_arr).astype(float)    # (n_time,)
    # cumul_mask[i, j] = 1 iff j is in same cycle as i AND j < i
    same_cycle_mat = (cycle_idx_arr[:, None] == cycle_idx_arr[None, :]).astype(float)
    before_i_mat = (np.arange(n_time)[:, None] > np.arange(n_time)[None, :]).astype(float)
    cumul_mask = same_cycle_mat * before_i_mat                         # (n_time, n_time)

    # PyTensor constants
    Q_const = pt.constant(Q_p)
    m_const = pt.constant(m_p)
    cumul_mask_const = pt.constant(cumul_mask)
    same_cycle_const = pt.constant(same_cycle_mat)
    cycle_phase_offset_const = pt.constant(cycle_phase_offset)

    r0_hat = float(np.median(np.linalg.norm(layer1.boundary_direction_mean, axis=-1)))
    r0_hat = max(r0_hat, 1e-3 * R_X)

    # Better a2 initval: direction from estimated center to the ACTUAL DATA at each
    # boundary time, not Layer 1's per-cycle mean boundary direction.  Layer 1's
    # a_mean_p is the posterior mean of the cycle-level boundary direction, which
    # can deviate >20° from the true data at tau_mean (because it's an average over
    # the whole cycle, not the point estimate at tau_k).  Starting from a2=a_mean_p
    # with sigma_x=0.02 produces ~19-sigma residuals at boundary points, corrupting
    # the MAP and NUTS warm-up. Using the actual data at tau_k reduces initval
    # residuals to ~noise level.
    a2_init = np.zeros_like(a_mean_p)
    for k in range(K):
        tau_k = tau_mean[k]
        idx_k = int(round((tau_k - t_fit[0]) * fs))
        idx_k = max(0, min(n_time - 1, idx_k))
        a2_init[k] = X_fit[idx_k] - c_mean_p[k]
    # Last row: copy from second-to-last (matches how a_mean_p is padded)
    a2_init[-1] = a2_init[-2]

    with pm.Model():
        # --- Center ---
        c2 = pm.Normal("c2", mu=c_mean_p, sigma=sigma_c2, shape=c_mean_p.shape)
        c_t = pm.Deterministic("center", pt.dot(B_const, c2))

        # --- Normal: tangent-plane deviations ---
        # delta_n[j] ~ N(0, sigma_theta2[j]^2 * I_2); both components same sigma
        delta_n = pm.Normal(
            "delta_n", mu=0.0, sigma=sigma_theta2[:, None], shape=(K, 2)
        )
        # Tang displacement: sum_d Q_p[j, :, d] * delta_n[j, d]  -> (K, 3)
        tang_disp = pt.sum(Q_const * delta_n[:, None, :], axis=-1)
        n_tilde = m_const + tang_disp                                  # (K, 3)
        n_tilde_norm = pt.sqrt(pt.sum(n_tilde**2, axis=-1, keepdims=True) + 1e-12)
        n_knots_pt = n_tilde / n_tilde_norm                            # (K, 3) normalised knots

        # Smoothness across adjacent normal knots (no absolute value; signs aligned)
        if K > 1:
            cos_adj = pt.sum(n_knots_pt[1:] * n_knots_pt[:-1], axis=-1)
            pm.Potential(
                "normal2_smoothness",
                (-(1.0 - cos_adj) / _LAYER2_NORMAL2_SMOOTHNESS_SD**2).sum(),
            )

        # Spline knots through time, renormalise
        n_bar_t = pt.dot(B_const, n_knots_pt)                         # (n_time, 3)
        n_bar_norm = pt.sqrt(pt.sum(n_bar_t**2, axis=-1, keepdims=True) + 1e-12)
        n_t = pm.Deterministic("normal", n_bar_t / n_bar_norm)

        # --- Boundary direction ---
        # Use a2_init (data at tau_k minus estimated center) as the prior mean,
        # not Layer 1's a_mean_p. Layer 1's a_mean_p is a posterior average of
        # the cycle-level boundary direction; for the boundary-anchored frame we
        # need the instantaneous direction from the estimated center to the
        # actual data point at tau_k. Using a_mean_p as the prior mean was shown
        # to create ~19-sigma boundary residuals that corrupt NUTS warmup.
        a2 = pm.Normal("a2", mu=a2_init, sigma=sigma_a2, shape=a_mean_p.shape)
        a_t = pm.Deterministic("a", pt.dot(B_const, a2))

        # Boundary-anchored in-plane frame
        proj = a_t - n_t * pt.sum(n_t * a_t, axis=-1, keepdims=True)
        proj_norm = pt.sqrt(pt.sum(proj**2, axis=-1, keepdims=True) + 1e-12)
        e1_t = pm.Deterministic("e1", proj / proj_norm)
        e2_t = pm.Deterministic("e2", _pt_cross(n_t, e1_t, pt))
        pm.Deterministic("projection_norm", proj_norm[:, 0])

        # --- Phase: boundary-normalized positive speed ---
        # q(t) spline; prior centred at 0 because per-cycle normalisation
        # removes the absolute scale of exp(q).
        q_knots = pm.Normal(
            "q_knots", mu=0.0, sigma=_PHASE_VELOCITY_LOGKNOT_SD, shape=n_velocity_knots
        )
        if n_velocity_knots > 1:
            dq = q_knots[1:] - q_knots[:-1]
            pm.Potential(
                "velocity_smoothness",
                pm.logp(pm.Normal.dist(0.0, _PHASE_VELOCITY_SMOOTHNESS_SD), dq).sum(),
            )
        q_t = pt.dot(Bg_const, q_knots)     # (n_time,) log speed
        w_t = pt.exp(q_t)                   # (n_time,) unnormalised speed > 0

        # S_{k,i} = sum_{j in k, j<i} w_j * dt  (cumulative, not including i)
        # S_{k,total} = sum_{j in k} w_j * dt
        S_ki = pt.dot(cumul_mask_const, w_t) * dt         # (n_time,)
        S_k_total = pt.dot(same_cycle_const, w_t) * dt    # (n_time,)
        phi_t = pm.Deterministic(
            "phase",
            cycle_phase_offset_const + 2 * np.pi * S_ki / (S_k_total + 1e-12),
        )
        # Instantaneous phase velocity: d phi/dt = 2*pi * w_i / S_{k,total}
        pm.Deterministic(
            "phase_velocity",
            2 * np.pi * w_t / (S_k_total + 1e-12),
        )

        # --- Radius ---
        h_r_knots = pm.Normal(
            "h_r_knots", mu=np.log(r0_hat), sigma=0.3, shape=n_velocity_knots
        )
        if n_velocity_knots > 1:
            dhr = h_r_knots[1:] - h_r_knots[:-1]
            pm.Potential(
                "radius_smoothness",
                pm.logp(pm.Normal.dist(0.0, _PHASE_VELOCITY_SMOOTHNESS_SD), dhr).sum(),
            )
        r_t = pm.Deterministic("radius", pt.exp(pt.dot(Bg_const, h_r_knots)))

        # --- Perpendicular deviation ---
        h_z_knots = pm.Normal("h_z_knots", mu=0.0, sigma=0.2 * R_X, shape=n_velocity_knots)
        if n_velocity_knots > 1:
            dhz = h_z_knots[1:] - h_z_knots[:-1]
            pm.Potential(
                "perp_smoothness",
                pm.logp(
                    pm.Normal.dist(0.0, _PHASE_VELOCITY_SMOOTHNESS_SD * R_X), dhz
                ).sum(),
            )
        z_t = pm.Deterministic("perp_deviation", pt.dot(Bg_const, h_z_knots))

        # --- Observation model ---
        pred = (
            c_t
            + e1_t * (r_t * pt.cos(phi_t))[:, None]
            + e2_t * (r_t * pt.sin(phi_t))[:, None]
            + n_t * z_t[:, None]
        )
        pm.Deterministic("predicted_trajectory", pred)

        if _sigma_x_override is not None:
            sigma_x_val = float(_sigma_x_override)
            pm.Normal("X_obs", mu=pred, sigma=sigma_x_val, observed=X_fit)
            initvals = {
                "c2": c_mean_p,
                "delta_n": np.zeros((K, 2)),
                "a2": a2_init,
                "q_knots": np.zeros(n_velocity_knots),
                "h_r_knots": np.full(n_velocity_knots, np.log(r0_hat)),
                "h_z_knots": np.zeros(n_velocity_knots),
            }
        else:
            rho_x = pm.Lognormal("rho_x", mu=_OBS_NOISE_LOGNORMAL_MU, sigma=_OBS_NOISE_LOGNORMAL_SD)
            sigma_x = pm.Deterministic("sigma_x", R_X * rho_x)
            pm.Normal("X_obs", mu=pred, sigma=sigma_x, observed=X_fit)
            initvals = {
                "c2": c_mean_p,
                "delta_n": np.zeros((K, 2)),
                "a2": a2_init,
                "q_knots": np.zeros(n_velocity_knots),
                "h_r_knots": np.full(n_velocity_knots, np.log(r0_hat)),
                "h_z_knots": np.zeros(n_velocity_knots),
                "rho_x": 0.03,
            }
        sample_kw = _sample_kwargs(
            draws, tune, chains, target_accept, random_seed, use_numba, initvals
        )
        if _init_override is not None:
            sample_kw["init"] = _init_override
        idata2 = pm.sample(**sample_kw)

    post = idata2.posterior

    def pmean(name):
        return post[name].mean(("chain", "draw")).values

    def psd(name):
        return post[name].std(("chain", "draw")).values

    phase_mean = pmean("phase")
    phase_sd = psd("phase")

    # Compute raw (unnormalized) mean for resultant-length diagnostic
    normal_raw_mean = post["normal"].mean(("chain", "draw")).values  # (n_time, 3)
    normal_resultant_length = np.linalg.norm(normal_raw_mean, axis=-1)  # (n_time,)
    normal_mean = normalize(normal_raw_mean)

    normal_samples = post["normal"].values.reshape(-1, n_time, 3)
    cos_to_mean = np.clip(
        np.einsum("dtj,tj->dt", normal_samples, normal_mean), -1.0, 1.0
    )
    normal_angular_sd = np.std(np.arccos(np.abs(cos_to_mean)), axis=0)

    # Reconstruct e1/e2 from posterior-mean n and a so the frame is exactly orthonormal
    a_mean_fit = pmean("a")
    e1_mean, e2_mean, projection_norm_mean = construct_frame(normal_mean, a_mean_fit)

    return _Layer2Summary(
        time=t_fit,
        phase_mean=phase_mean,
        phase_sd=phase_sd,
        phase_velocity_mean=pmean("phase_velocity"),
        phase_velocity_sd=psd("phase_velocity"),
        center_mean=pmean("center"),
        center_sd=psd("center"),
        normal_mean=normal_mean,
        normal_raw_mean=normal_raw_mean,
        normal_resultant_length=normal_resultant_length,
        normal_angular_sd=normal_angular_sd,
        e1_mean=e1_mean,
        e2_mean=e2_mean,
        boundary_direction_mean=a_mean_fit,
        projection_norm_mean=projection_norm_mean,
        radius_mean=pmean("radius"),
        radius_sd=psd("radius"),
        perp_deviation_mean=pmean("perp_deviation"),
        perp_deviation_sd=psd("perp_deviation"),
        predicted_trajectory_mean=pmean("predicted_trajectory"),
        sigma_x_mean=_sigma_x_override if _sigma_x_override is not None else float(pmean("sigma_x")),
        idata=idata2,
    )


# ---------------------------------------------------------------------------
# Diagnostics (spec: "Diagnostics")
# ---------------------------------------------------------------------------

_EPS = 1e-9


def _angle_between(a, b):
    """Angle (radians) between two vectors, sign-ambiguity-insensitive."""
    a = normalize(np.asarray(a, dtype=float))
    b = normalize(np.asarray(b, dtype=float))
    cos = np.clip(np.abs(np.dot(a, b)), -1.0, 1.0)
    return float(np.arccos(cos))


def _check_boundary_multimodality(idata1, K, T_for_floor):
    """
    Spec: fail if two posterior modes of a boundary tau_k each have mass
    > 0.20 and are separated by more than 0.20 * T_k.
    """
    from scipy.stats import gaussian_kde

    samples = idata1.posterior["tau"].values.reshape(-1, K)
    for k in range(K):
        s = samples[:, k]
        lo, hi = s.min(), s.max()
        if hi - lo < 1e-9:
            continue
        try:
            kde = gaussian_kde(s)
        except (np.linalg.LinAlgError, ValueError):
            continue
        grid = np.linspace(lo, hi, 512)
        density = kde(grid)
        peaks, _ = find_peaks(density)
        if len(peaks) < 2:
            continue
        top2 = sorted(peaks[np.argsort(-density[peaks])[:2]])
        p1, p2 = top2
        split = p1 + int(np.argmin(density[p1 : p2 + 1]))
        total = np.trapz(density, grid)
        if total <= 0:
            continue
        mass1 = np.trapz(density[: split + 1], grid[: split + 1]) / total
        mass2 = np.trapz(density[split:], grid[split:]) / total
        separation = float(grid[p2] - grid[p1])
        if mass1 > 0.20 and mass2 > 0.20 and separation > 0.20 * T_for_floor[k]:
            return True
    return False


def _compute_diagnostics(layer1, layer2, R_X):
    """Build a :class:`BayesianPhaseDiagnostics` from the two layers' summaries."""
    failures = []
    warns = []
    K = len(layer1.tau_mean)
    T_for_floor = np.append(layer1.period_mean, layer1.period_mean[-1])

    boundary_multimodal = _check_boundary_multimodality(layer1.idata, K, T_for_floor)
    if boundary_multimodal:
        failures.append(
            "Boundary posterior is multimodal: two modes each with mass > 0.20 "
            "separated by more than 0.20*T_k."
        )

    rho_tau = float(np.sqrt(3.0) * layer1.rho_tau_mean)
    if rho_tau > 0.40:
        failures.append(f"Boundary cloud too large: rho_tau = {rho_tau:.3f} > 0.40.")
    elif rho_tau > 0.25:
        warns.append(f"Boundary cloud spread elevated: rho_tau = {rho_tau:.3f} > 0.25.")

    a_norm = np.linalg.norm(layer2.boundary_direction_mean, axis=-1)
    projection_ratio = layer2.projection_norm_mean / np.maximum(a_norm, _EPS)
    med_projection_ratio = float(np.median(projection_ratio))
    if med_projection_ratio < 0.10:
        failures.append(
            "Projection of a(t) into the instantaneous plane is near zero: "
            f"median p_t/||a(t)|| = {med_projection_ratio:.3f} < 0.10."
        )
    elif med_projection_ratio < 0.20:
        warns.append(
            "Projection of a(t) into the instantaneous plane is small: "
            f"median p_t/||a(t)|| = {med_projection_ratio:.3f} < 0.20."
        )

    normal_prior_dominated = []
    for k in range(K - 1):
        angular_shift = _angle_between(layer1.u_hat[k], layer1.normal_mean[k])
        post_sd = layer1.normal_angular_sd[k]
        if angular_shift < 0.25 * _NORMAL_VECTOR_SD and post_sd > 0.75 * _NORMAL_VECTOR_SD:
            normal_prior_dominated.append(k)
    if normal_prior_dominated:
        warns.append(
            f"Cycle normal(s) dominated by the prior (little posterior update): "
            f"cycles {normal_prior_dominated}."
        )

    rho_z = np.abs(layer2.perp_deviation_mean) / (layer2.radius_mean + _EPS)
    rho_z_median = float(np.median(rho_z))
    frac_large_rho_z = float(np.mean(rho_z > 0.50))
    if rho_z_median > 0.50:
        warns.append(f"STRONG: median perpendicular-deviation ratio {rho_z_median:.3f} > 0.50.")
    elif rho_z_median > 0.25:
        warns.append(f"Perpendicular-deviation ratio elevated: median {rho_z_median:.3f} > 0.25.")
    if frac_large_rho_z > 0.10:
        warns.append(
            f"{100 * frac_large_rho_z:.1f}% of time samples have perpendicular-"
            "deviation ratio > 0.50."
        )

    c_bar = layer2.center_mean.mean(axis=0)
    center_drift = np.linalg.norm(layer2.center_mean - c_bar, axis=-1)
    center_drift_ratio = float((center_drift.max() - center_drift.min()) / R_X)
    if center_drift_ratio > 0.50:
        warns.append(f"STRONG: center drift D_c/R_X = {center_drift_ratio:.3f} > 0.50.")
    elif center_drift_ratio > 0.25:
        warns.append(f"Center drift elevated: D_c/R_X = {center_drift_ratio:.3f} > 0.25.")

    omega_95 = np.percentile(layer2.phase_velocity_mean, 95)
    omega_5 = np.percentile(layer2.phase_velocity_mean, 5)
    omega_ratio = float(omega_95 / max(omega_5, _EPS))
    if omega_ratio > 5:
        warns.append(f"STRONG: phase-velocity degeneracy omega_95/omega_5 = {omega_ratio:.3f} > 5.")
    elif omega_ratio > 3:
        warns.append(f"Phase-velocity degeneracy: omega_95/omega_5 = {omega_ratio:.3f} > 3.")

    sigma_x_over_RX = layer2.sigma_x_mean / R_X
    if sigma_x_over_RX > 0.25:
        warns.append(f"STRONG: observation noise sigma_x/R_X = {sigma_x_over_RX:.3f} > 0.25.")
    elif sigma_x_over_RX > 0.10:
        warns.append(f"Observation noise elevated: sigma_x/R_X = {sigma_x_over_RX:.3f} > 0.10.")

    # Low normal mean-resultant-length diagnostic: warns when normalising the
    # posterior mean direction is misleading (spec: "Low normal mean resultant
    # length").  This can catch cases where most individual draws agree on a
    # stable-but-wrong direction without the posterior SD being elevated.
    normal_resultant_length = layer2.normal_resultant_length
    low_rl_mask = normal_resultant_length < 0.80
    low_rl_frac = float(np.mean(low_rl_mask))
    low_rl_min = float(np.min(normal_resultant_length))
    if low_rl_frac > 0:
        warns.append(
            f"Normal mean resultant length < 0.80 at {100 * low_rl_frac:.1f}% of time samples "
            f"(min {low_rl_min:.3f}). Normalised mean direction may be misleading at those points."
        )

    dphi = np.diff(layer2.phase_mean)
    phase_monotonic = bool(np.all(dphi >= -1e-6))
    if not phase_monotonic:
        failures.append("Phase is not monotonically increasing (sanity check failed).")

    return BayesianPhaseDiagnostics(
        failures=failures,
        warnings=warns,
        boundary_multimodal=boundary_multimodal,
        rho_tau=rho_tau,
        projection_ratio=projection_ratio,
        normal_prior_dominated=normal_prior_dominated,
        normal_resultant_length=normal_resultant_length,
        rho_z_median=rho_z_median,
        center_drift_ratio=center_drift_ratio,
        omega_ratio=omega_ratio,
        sigma_x_over_RX=sigma_x_over_RX,
        phase_monotonic=phase_monotonic,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_bayesian_phase_coordinates(
    X,
    sampling_rate_hz,
    columns=None,
    n_velocity_knots=None,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.9,
    random_seed=None,
    return_report=False,
):
    """
    Fit the Bayesian two-layer phase-coordinate model (spec:
    ``docs/bayesian_two_layer_spec.md``).

    Parameters
    ----------
    X : array-like or pandas.DataFrame, shape (n_time, 3)
        3-D movement trajectory. Exactly 3 features are required (the model
        uses a genuine :math:`\\mathbb{R}^3` cross product for the in-plane
        frame).
    sampling_rate_hz : float
        Sampling rate in Hz.
    columns : list of str, optional
        Subset of columns to use when ``X`` is a :class:`pandas.DataFrame`.
    n_velocity_knots : int, optional
        Number of knots for the log phase-velocity / log-radius / perpendicular
        deviation splines. Defaults to ``clip(2 * n_cycles, 4, 20)``.
    draws, tune, chains : int
        MCMC sampling settings, applied to both Layer 1 and Layer 2.
    target_accept : float
        NUTS target acceptance rate.
    random_seed : int, optional
        Random seed for reproducibility.
    return_report : bool
        If ``True``, retain the full posterior (Layer 1 and Layer 2 ArviZ
        ``InferenceData`` objects) on ``result.bayesian_report``. If
        ``False`` (default), discard the raw posterior draws after computing
        summaries.

    Returns
    -------
    BayesianPhaseResult
    """
    import pandas as pd

    if isinstance(X, pd.DataFrame):
        X_arr = X[columns].to_numpy(dtype=float) if columns else X.to_numpy(dtype=float)
    else:
        X_arr = np.asarray(X, dtype=float)

    if X_arr.ndim != 2 or X_arr.shape[1] != 3:
        raise ValueError(
            f"fit_bayesian_phase_coordinates requires 3-D data, shape "
            f"(n_time, 3); got shape {X_arr.shape}."
        )
    if not np.all(np.isfinite(X_arr)):
        raise ValueError("X contains non-finite values (NaN or Inf).")

    fs = float(sampling_rate_hz)
    if fs <= 0:
        raise ValueError(f"sampling_rate_hz must be positive, got {fs}.")

    # Force an early, clear error (before any expensive sampling) if PyMC /
    # ArviZ are not installed.
    _import_pymc()
    _import_pytensor_tensor()
    _import_arviz()
    use_numba = _numba_available()

    R_X, xbar = robust_movement_scale(X_arr)
    ref = dominant_reference_signal(X_arr)
    T0 = estimate_dominant_period(ref, fs)
    tau_idx = seed_boundary_indices(ref, fs, T0)

    layer1 = _fit_layer1(
        X_arr, fs, tau_idx, T0, R_X, xbar,
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=random_seed, use_numba=use_numba,
    )
    layer2 = _fit_layer2(
        X_arr, fs, layer1, T0, R_X, n_velocity_knots=n_velocity_knots,
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=random_seed, use_numba=use_numba,
    )

    estimates = BayesianPhaseEstimates(
        tau=layer1.tau_mean,
        period=layer1.period_mean,
        cycle_center=layer1.center_mean,
        cycle_normal=layer1.normal_mean,
        boundary_direction=layer1.boundary_direction_mean,
        time=layer2.time,
        phase=layer2.phase_mean,
        phase_velocity=layer2.phase_velocity_mean,
        center=layer2.center_mean,
        normal=layer2.normal_mean,
        e1=layer2.e1_mean,
        e2=layer2.e2_mean,
        radius=layer2.radius_mean,
        perp_deviation=layer2.perp_deviation_mean,
        predicted_trajectory=layer2.predicted_trajectory_mean,
    )

    uncertainty = BayesianPhaseUncertainty(
        tau_sd=layer1.tau_sd,
        period_sd=layer1.period_sd,
        cycle_center_sd=layer1.center_sd,
        cycle_normal_angular_sd=layer1.normal_angular_sd,
        boundary_direction_sd=layer1.boundary_direction_sd,
        phase_sd=layer2.phase_sd,
        phase_velocity_sd=layer2.phase_velocity_sd,
        center_sd=layer2.center_sd,
        normal_angular_sd=layer2.normal_angular_sd,
        radius_sd=layer2.radius_sd,
        perp_deviation_sd=layer2.perp_deviation_sd,
        observation_noise_sd=layer2.sigma_x_mean,
    )

    diagnostics = _compute_diagnostics(layer1, layer2, R_X)

    bayesian_report = None
    if return_report:
        bayesian_report = {"layer1": layer1.idata, "layer2": layer2.idata}

    return BayesianPhaseResult(
        estimates=estimates,
        uncertainty=uncertainty,
        diagnostics=diagnostics,
        bayesian_report=bayesian_report,
    )
