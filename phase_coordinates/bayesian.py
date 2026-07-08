"""
Bayesian two-layer phase-coordinate estimator.

Implements the model described in ``docs/bayesian_two_layer_spec.md``: a
coarse cycle-level model (Layer 1) estimating boundary times, cycle centers,
and cycle normals with posterior uncertainty, followed by an instantaneous
model (Layer 2) that uses the Layer 1 posterior summaries as priors for
smoothly varying phase, center, normal, radius, and perpendicular deviation.

This module is independent of :mod:`phase_coordinates.core` and does not
replace :func:`phase_coordinates.core.hilbert_phase` or
:func:`phase_coordinates.core.fit_pca_phase_coordinates`.

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

from .epochs import CycleEpochs, epochs_from_boundary_indices
from .geometry import (
    interp_X_at_times,
    oriented_frame_from_anchors,
)

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




# ---------------------------------------------------------------------------
# Vector utilities
# ---------------------------------------------------------------------------

def normalize(v, axis=-1, eps=1e-12):
    """Normalize vectors along ``axis`` to unit length."""
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.clip(norm, eps, None)


def _oriented_frame_from_anchors_with_diag(x0, x90, c, eps=1e-12):
    """
    Layer-1/2 wrapper around :func:`phase_coordinates.geometry.oriented_frame_from_anchors`
    that additionally returns the anchor vectors ``a0``/``a90`` and the norm
    of the a90-orthogonal component (used for pre-sampling diagnostics).
    """
    x0 = np.asarray(x0, dtype=float)
    x90 = np.asarray(x90, dtype=float)
    c = np.asarray(c, dtype=float)
    a0 = x0 - c
    a90 = x90 - c
    e1, e2, n = oriented_frame_from_anchors(x0, x90, c, eps=eps)
    dot_a90_e1 = np.sum(a90 * e1, axis=1, keepdims=True)
    a90_orth = a90 - e1 * dot_a90_e1
    a90_orth_norm = np.linalg.norm(a90_orth, axis=1)
    return a0, a90, e1, e2, n, a90_orth_norm




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

_LAYER2_CENTER_FLOOR_FRAC = 0.02          # * R_X

_OBS_NOISE_LOGNORMAL_MU = np.log(0.03)
_OBS_NOISE_LOGNORMAL_SD = 0.5     # reverted: 0.3 caused divergences; correct-mode initvals now prevent wrong-mode escape

_PHASE_VELOCITY_SMOOTHNESS_SD = 0.15


@dataclass
class _Layer1Summary:
    """Internal posterior summary handed from Layer 1 to Layer 2."""

    tau_mean: np.ndarray
    tau_sd: np.ndarray
    period_mean: np.ndarray
    period_sd: np.ndarray
    center_mean: np.ndarray
    center_sd: np.ndarray
    a0_mean: np.ndarray             # (K_cyc, 3) phase-zero anchor vectors
    a90_mean: np.ndarray            # (K_cyc, 3) quarter-phase anchor vectors
    e1_mean: np.ndarray             # (K_cyc, 3) oriented e1 per cycle
    e2_mean: np.ndarray             # (K_cyc, 3) oriented e2 per cycle
    normal_mean: np.ndarray         # (K_cyc, 3) n = cross(e1, e2) per cycle
    rho_tau_mean: float
    idata: Any


def _sample_kwargs(draws, tune, chains, target_accept, random_seed, use_numba, initvals=None):
    # Seeds (tau_hat, c_hat, ...) are informative and already near the mode.
    # Explicit initvals start every chain exactly there rather than at a
    # jittered point. This matters for c_k and log_R_k starting near the
    # correct mode.
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
    seed_epochs,
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
    """
    Fit the Layer 1 boundary/center model, seeded by ``seed_epochs``.

    ``seed_epochs.tau`` supplies the initial mean of the boundary-time prior;
    per-cycle seed centers are the sample-mean of ``X`` between successive
    seed boundaries (recomputed here so the seeds match ``seed_epochs.tau``
    even if the epochs come from a non-index source).
    """
    pm = _import_pymc()
    pt = _import_pytensor_tensor()

    n_time = X.shape[0]
    t_grid = np.arange(n_time) / fs

    tau_hat = np.asarray(seed_epochs.tau, dtype=float)
    K = len(tau_hat)
    # Convert real-valued boundary times to integer sample indices for
    # ``seed_cycle_centers`` (which slices X directly).  Values are clamped
    # into [0, n_time-1] so that a boundary that lies at the exact end of the
    # signal doesn't overflow.
    tau_idx_seed = np.clip(np.round(tau_hat * fs).astype(int), 0, n_time - 1)
    # Make sure they are strictly increasing so slices in seed_cycle_centers
    # are non-empty.
    for i in range(1, len(tau_idx_seed)):
        if tau_idx_seed[i] <= tau_idx_seed[i - 1]:
            tau_idx_seed[i] = tau_idx_seed[i - 1] + 1
    tau_idx_seed = np.clip(tau_idx_seed, 0, n_time - 1)
    c_hat = seed_cycle_centers(X, tau_idx_seed)

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
    rho_tau_mean = float(post["rho_tau"].mean(("chain", "draw")).values)

    # Compute oriented frame deterministically from posterior-mean tau and c.
    # The frame is derived from real-valued interpolation at tau_mean, not
    # sampled, so there is no sign ambiguity.
    T_k = tau_mean[1:] - tau_mean[:-1]
    x0 = interp_X_at_times(X, fs, tau_mean[:-1])
    x90 = interp_X_at_times(X, fs, tau_mean[:-1] + 0.25 * T_k)
    a0_mean, a90_mean, e1_mean, e2_mean, n_mean, _ = \
        _oriented_frame_from_anchors_with_diag(x0, x90, c_mean)

    return _Layer1Summary(
        tau_mean=tau_mean,
        tau_sd=tau_sd,
        period_mean=T_mean,
        period_sd=T_sd,
        center_mean=c_mean,
        center_sd=c_sd,
        a0_mean=a0_mean,
        a90_mean=a90_mean,
        e1_mean=e1_mean,
        e2_mean=e2_mean,
        normal_mean=n_mean,
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
    cycle_center_mean: np.ndarray      # (K_cyc, 3) posterior-mean c_k from Layer 2
    cycle_e1_mean: np.ndarray          # (K_cyc, 3) frame from posterior-mean c_k + anchors
    cycle_e2_mean: np.ndarray          # (K_cyc, 3)
    cycle_normal_mean: np.ndarray      # (K_cyc, 3)
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
    n_velocity_knots,   # ignored; kept for API compatibility
    draws,
    tune,
    chains,
    target_accept,
    random_seed,
    use_numba,
):
    """
    Cycle-fixed geometry Layer 2 model.

    The oriented frame is derived deterministically from the sampled center c_k
    and fixed interpolated anchor points x0_arr / x90_arr (held as PyTensor
    constants). This removes a0_k and a90_k as independent Bayesian variables;
    they become deterministic functions of c_k alone.

    Frame convention:
        a0_k  = x0_const[k] - c_k        (phase-zero anchor)
        a90_k = x90_const[k] - c_k       (quarter-phase anchor)
        e1_k  = normalize(a0_k)
        e2_k  = normalize(a90_k - e1_k * dot(a90_k, e1_k))
        n_k   = normalize(cross(e1_k, e2_k))
    """
    pm = _import_pymc()
    pt = _import_pytensor_tensor()

    tau_mean = layer1.tau_mean      # (K,) boundary times
    K = len(tau_mean)
    K_cyc = K - 1

    i0 = max(0, int(np.ceil(tau_mean[0] * fs)))
    i1 = min(X.shape[0] - 1, int(np.floor(tau_mean[-1] * fs)))
    if i1 - i0 < 10:
        raise ValueError(
            "Not enough time samples spanned by the detected cycle boundaries "
            "to fit the Layer 2 model."
        )
    X_fit = X[i0 : i1 + 1]
    t_fit = np.arange(i0, i1 + 1) / fs
    n_time = X_fit.shape[0]

    # --- Cycle membership ---
    cycle_idx_arr = np.searchsorted(tau_mean, t_fit, side="right") - 1
    cycle_idx_arr = np.clip(cycle_idx_arr, 0, K_cyc - 1).astype(int)
    cycle_idx_const = pt.constant(cycle_idx_arr)

    # --- Layer 1 per-cycle summaries ---
    c1_mean = layer1.center_mean   # (K_cyc, 3)

    # --- Fixed anchor points from interpolation at tau_mean ---
    T_k = tau_mean[1:] - tau_mean[:-1]               # (K_cyc,)
    x0_arr = interp_X_at_times(X, fs, tau_mean[:-1])                   # (K_cyc, 3)
    x90_arr = interp_X_at_times(X, fs, tau_mean[:-1] + 0.25 * T_k)    # (K_cyc, 3)
    x0_const = pt.constant(x0_arr)
    x90_const = pt.constant(x90_arr)

    # --- Frame at Layer 1 center means (for R1_mean and diagnostics) ---
    _, _, e1_prior, e2_prior, n_prior, a90_orth_norms = \
        _oriented_frame_from_anchors_with_diag(x0_arr, x90_arr, c1_mean)

    # --- Pre-sampling frame diagnostics ---
    a90_mean_arr = x90_arr - c1_mean
    a90_normed = a90_mean_arr / np.maximum(
        np.linalg.norm(a90_mean_arr, axis=1, keepdims=True), 1e-12
    )
    orient_scores = np.sum(e2_prior * a90_normed, axis=1)  # (K_cyc,)
    print("\nPre-sampling oriented frame (Layer 2 prior means):")
    print(f"  {'cyc':>4}  {'dot(e1,e2)':>11}  {'|e1|':>6}  {'|e2|':>6}  "
          f"{'|n|':>6}  {'orient_score':>12}  {'a90_orth_norm':>14}")
    for k in range(K_cyc):
        print(
            f"  {k:>4}  "
            f"{float(np.dot(e1_prior[k], e2_prior[k])):>11.4f}  "
            f"{float(np.linalg.norm(e1_prior[k])):>6.4f}  "
            f"{float(np.linalg.norm(e2_prior[k])):>6.4f}  "
            f"{float(np.linalg.norm(n_prior[k])):>6.4f}  "
            f"{float(orient_scores[k]):>12.4f}  "
            f"{float(a90_orth_norms[k]):>14.4f}"
        )
    print()

    # --- R1_mean: median in-plane radius using oriented normal ---
    R1_mean = np.zeros(K_cyc)
    for k in range(K_cyc):
        mask_k = cycle_idx_arr == k
        if mask_k.sum() < 2:
            R1_mean[k] = max(float(np.linalg.norm(x0_arr[k] - c1_mean[k])), 0.1 * R_X)
            continue
        diff = X_fit[mask_k] - c1_mean[k]
        perp = diff - (diff @ n_prior[k])[:, None] * n_prior[k]
        R1_mean[k] = float(np.median(np.linalg.norm(perp, axis=1)))
    R1_mean = np.maximum(R1_mean, 0.1 * R_X)
    log_R1_mean = np.log(R1_mean)

    # --- Hierarchical prior scales ---
    sigma_c_scale = np.maximum(
        np.std(c1_mean, axis=0), _LAYER2_CENTER_FLOOR_FRAC * R_X
    )
    sigma_logR_scale = float(max(float(np.std(log_R1_mean)), 0.03))

    # --- Linear phase: fully deterministic ---
    tau_k_arr = tau_mean[cycle_idx_arr]
    tau_kp1_arr = tau_mean[cycle_idx_arr + 1]
    phi_t_np = (
        2 * np.pi * cycle_idx_arr
        + 2 * np.pi * (t_fit - tau_k_arr) / (tau_kp1_arr - tau_k_arr)
    )
    phi_t_const = pt.constant(phi_t_np)
    phase_vel_np = (2 * np.pi / T_k)[cycle_idx_arr]

    # --- Spline matrix for z (K boundary-time knots) ---
    B = cubic_spline_matrix(tau_mean, t_fit)
    B_const = pt.constant(B)

    with pm.Model():
        # --- Hierarchical center ---
        sigma_c = pm.HalfNormal("sigma_c", sigma=sigma_c_scale, shape=(3,))
        c_k = pm.Normal("c_k", mu=c1_mean, sigma=sigma_c, shape=(K_cyc, 3))
        c_t = pm.Deterministic("center", c_k[cycle_idx_const])

        # --- Deterministic anchors from sampled center and fixed interpolated points ---
        a0_k = x0_const - c_k    # (K_cyc, 3) — moves with c_k
        a90_k = x90_const - c_k  # (K_cyc, 3) — moves with c_k

        # --- Oriented frame: e1 from a0, e2 from a90 ortho, n = cross(e1, e2) ---
        a0_norm = pt.sqrt(pt.sum(a0_k**2, axis=-1, keepdims=True) + 1e-12)
        e1_k = pm.Deterministic("e1_k", a0_k / a0_norm)

        dot_a90_e1 = pt.sum(a90_k * e1_k, axis=-1, keepdims=True)
        a90_orth = a90_k - e1_k * dot_a90_e1
        a90_orth_n = pt.sqrt(pt.sum(a90_orth**2, axis=-1, keepdims=True) + 1e-12)
        e2_k = pm.Deterministic("e2_k", a90_orth / a90_orth_n)

        n_cross = _pt_cross(e1_k, e2_k, pt)
        n_cross_n = pt.sqrt(pt.sum(n_cross**2, axis=-1, keepdims=True) + 1e-12)
        n_k = pm.Deterministic("n_k", n_cross / n_cross_n)

        # Per-time expansions
        e1_t = pm.Deterministic("e1", e1_k[cycle_idx_const])
        e2_t = pm.Deterministic("e2", e2_k[cycle_idx_const])
        n_t = pm.Deterministic("normal", n_k[cycle_idx_const])

        # --- Hierarchical cycle mean radius ---
        sigma_log_R = pm.HalfNormal("sigma_log_R", sigma=sigma_logR_scale)
        log_R_k = pm.Normal("log_R_k", mu=log_R1_mean, sigma=sigma_log_R, shape=(K_cyc,))
        R_k = pm.Deterministic("R_k", pt.exp(log_R_k))
        r_t = pm.Deterministic("radius", R_k[cycle_idx_const])

        # --- Perpendicular deviation: K-knot spline ---
        h_z_knots = pm.Normal("h_z_knots", mu=0.0, sigma=0.2 * R_X, shape=K)
        if K > 1:
            dhz = h_z_knots[1:] - h_z_knots[:-1]
            pm.Potential(
                "perp_smoothness",
                pm.logp(pm.Normal.dist(0.0, _PHASE_VELOCITY_SMOOTHNESS_SD * R_X), dhz).sum(),
            )
        z_t = pm.Deterministic("perp_deviation", pt.dot(B_const, h_z_knots))

        # --- Observation model ---
        pred = (
            c_t
            + e1_t * (r_t * pt.cos(phi_t_const))[:, None]
            + e2_t * (r_t * pt.sin(phi_t_const))[:, None]
            + n_t * z_t[:, None]
        )
        pm.Deterministic("predicted_trajectory", pred)

        rho_x = pm.Lognormal("rho_x", mu=_OBS_NOISE_LOGNORMAL_MU, sigma=_OBS_NOISE_LOGNORMAL_SD)
        sigma_x = pm.Deterministic("sigma_x", R_X * rho_x)
        pm.Normal("X_obs", mu=pred, sigma=sigma_x, observed=X_fit)

        initvals = {
            "sigma_c": sigma_c_scale,
            "c_k": c1_mean,
            "sigma_log_R": np.array(sigma_logR_scale),
            "log_R_k": log_R1_mean,
            "h_z_knots": np.zeros(K),
            "rho_x": 0.03,
        }
        idata2 = pm.sample(**_sample_kwargs(
            draws, tune, chains, target_accept, random_seed, use_numba, initvals
        ))

    post = idata2.posterior

    def pmean(name):
        return post[name].mean(("chain", "draw")).values

    def psd(name):
        return post[name].std(("chain", "draw")).values

    # Reconstruct consistent (e1, e2, n) frame from posterior mean c_k and
    # fixed anchor points. Averaging per-draw unit vectors (pmean("e1") etc.)
    # produces vectors shorter than 1 when draws spread in direction, and the
    # averaged e1/e2/n are not exactly consistent. Reconstruction from
    # c_mean_cyc gives exact orthonormality by construction.
    c_mean_cyc = pmean("c_k")        # (K_cyc, 3)
    a0_pm = x0_arr - c_mean_cyc      # (K_cyc, 3)
    a90_pm = x90_arr - c_mean_cyc    # (K_cyc, 3)
    eps = 1e-12
    e1_cyc = a0_pm / np.maximum(np.linalg.norm(a0_pm, axis=1, keepdims=True), eps)
    dot_a90_e1c = np.sum(a90_pm * e1_cyc, axis=1, keepdims=True)
    a90_orth_cyc = a90_pm - e1_cyc * dot_a90_e1c
    e2_cyc = a90_orth_cyc / np.maximum(np.linalg.norm(a90_orth_cyc, axis=1, keepdims=True), eps)
    n_cross_cyc = np.cross(e1_cyc, e2_cyc)
    n_cyc = n_cross_cyc / np.maximum(np.linalg.norm(n_cross_cyc, axis=1, keepdims=True), eps)
    # Expand to per-time (exact unit norm and mutual orthogonality)
    e1_mean = e1_cyc[cycle_idx_arr]
    e2_mean = e2_cyc[cycle_idx_arr]
    normal_mean = n_cyc[cycle_idx_arr]

    # Resultant length diagnostic uses the raw Deterministic posterior mean
    # (not the reconstruction) so it can detect bimodal frame distributions.
    normal_raw_mean = pmean("normal")                       # (n_time, 3)
    normal_resultant_length = np.linalg.norm(normal_raw_mean, axis=-1)

    normal_samples = post["normal"].values.reshape(-1, n_time, 3)
    cos_to_mean = np.clip(
        np.einsum("dtj,tj->dt", normal_samples, normal_mean), -1.0, 1.0
    )
    normal_angular_sd = np.std(np.arccos(np.abs(cos_to_mean)), axis=0)

    # boundary_direction_mean = per-time a0 from posterior-mean c_k
    boundary_dir_mean = a0_pm[cycle_idx_arr]             # (n_time, 3)
    projection_norm_mean = np.linalg.norm(boundary_dir_mean, axis=1)  # (n_time,)

    return _Layer2Summary(
        time=t_fit,
        phase_mean=phi_t_np,
        phase_sd=np.zeros(n_time),
        phase_velocity_mean=phase_vel_np,
        phase_velocity_sd=np.zeros(n_time),
        center_mean=pmean("center"),
        center_sd=psd("center"),
        normal_mean=normal_mean,
        normal_raw_mean=normal_raw_mean,
        normal_resultant_length=normal_resultant_length,
        normal_angular_sd=normal_angular_sd,
        e1_mean=e1_mean,
        e2_mean=e2_mean,
        cycle_center_mean=c_mean_cyc,
        cycle_e1_mean=e1_cyc,
        cycle_e2_mean=e2_cyc,
        cycle_normal_mean=n_cyc,
        boundary_direction_mean=boundary_dir_mean,
        projection_norm_mean=projection_norm_mean,
        radius_mean=pmean("radius"),
        radius_sd=psd("radius"),
        perp_deviation_mean=pmean("perp_deviation"),
        perp_deviation_sd=psd("perp_deviation"),
        predicted_trajectory_mean=pmean("predicted_trajectory"),
        sigma_x_mean=float(pmean("sigma_x")),
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
        total = np.trapezoid(density, grid)
        if total <= 0:
            continue
        mass1 = np.trapezoid(density[: split + 1], grid[: split + 1]) / total
        mass2 = np.trapezoid(density[split:], grid[split:]) / total
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

    # Layer 1 normal is derived deterministically from anchor interpolation,
    # not independently sampled, so there is no prior-dominated check.
    normal_prior_dominated = []

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
    *,
    sampling_rate_hz,
    seed_epochs: Optional[CycleEpochs] = None,
    columns=None,
    draws=1000,
    tune=1000,
    chains=4,
    target_accept=0.9,
    random_seed=None,
    return_report=False,
):
    """
    Fit the Bayesian two-layer phase-coordinate model.

    Parameters
    ----------
    X : array-like or pandas.DataFrame, shape (n_time, 3)
        3-D movement trajectory.
    sampling_rate_hz : float
        Sampling rate in Hz.
    seed_epochs : CycleEpochs, optional
        Seed cycle boundaries.  If omitted, seeds are built internally by
        running the explicit pipeline::

            ref = dominant_reference_signal(X)
            T0 = estimate_dominant_period(ref, fs)
            tau_idx = seed_boundary_indices(ref, fs, T0)
            seed_epochs = epochs_from_boundary_indices(tau_idx, ...)

        Pass this argument to inspect or override the seed path.
    columns : list of str, optional
        Subset of columns to use when ``X`` is a :class:`pandas.DataFrame`.
    draws, tune, chains : int
        MCMC sampling settings, applied to both Layer 1 and Layer 2.
    target_accept : float
        NUTS target acceptance rate.
    random_seed : int, optional
        Random seed for reproducibility.
    return_report : bool
        If ``True``, add ``"report"`` key to details with the full posterior
        (Layer 1 and Layer 2 ArviZ ``InferenceData`` objects).

    Returns
    -------
    samples : pandas.DataFrame
        One row per input time sample. Columns = SAMPLE_COLUMNS.
    cycles : pandas.DataFrame
        One row per detected cycle. Columns = CYCLE_COLUMNS.
    details : dict
        Algorithm-specific diagnostics and uncertainty information.
    """
    import pandas as pd
    from .core import SAMPLE_COLUMNS, CYCLE_COLUMNS

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

    n_input_time = X_arr.shape[0]

    R_X, xbar = robust_movement_scale(X_arr)
    ref = dominant_reference_signal(X_arr)
    T0 = estimate_dominant_period(ref, fs)

    if seed_epochs is None:
        tau_idx = seed_boundary_indices(ref, fs, T0)
        seed_epochs = epochs_from_boundary_indices(
            tau_idx,
            sampling_rate_hz=fs,
            n_time=n_input_time,
            source="periodogram_peaks",
            metadata={"T0": T0},
        )
    elif not isinstance(seed_epochs, CycleEpochs):
        raise TypeError(
            f"seed_epochs must be a CycleEpochs, got {type(seed_epochs).__name__}."
        )

    layer1 = _fit_layer1(
        X_arr, fs, seed_epochs, T0, R_X, xbar,
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=random_seed, use_numba=use_numba,
    )
    layer2 = _fit_layer2(
        X_arr, fs, layer1, T0, R_X, n_velocity_knots=None,
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=random_seed, use_numba=use_numba,
    )

    diagnostics = _compute_diagnostics(layer1, layer2, R_X)

    K = len(layer1.tau_mean)
    K_cyc = K - 1

    # ---- Determine fitted window indices ----
    i0 = max(0, int(np.ceil(layer1.tau_mean[0] * fs)))
    i1 = min(n_input_time - 1, int(np.floor(layer1.tau_mean[-1] * fs)))

    # ---- Build samples DataFrame ----
    # Start with NaN-filled frame for all input samples
    all_sample_index = np.arange(n_input_time)
    all_time = all_sample_index / fs

    # Fitted-window values
    phase_in_cycle_fit = np.mod(layer2.phase_mean, 2 * np.pi)
    u_fit = layer2.radius_mean * np.cos(phase_in_cycle_fit)
    v_fit = layer2.radius_mean * np.sin(phase_in_cycle_fit)
    theta_wrapped_fit = np.angle(np.exp(1j * phase_in_cycle_fit))

    # Cycle membership for fitted window
    cycle_idx_arr = np.searchsorted(layer1.tau_mean, layer2.time, side="right") - 1
    cycle_idx_arr = np.clip(cycle_idx_arr, 0, K_cyc - 1).astype(int)

    # Build full arrays (NaN outside window)
    cycle_full = np.full(n_input_time, np.nan)
    phase_full = np.full(n_input_time, np.nan)
    phase_in_cycle_full = np.full(n_input_time, np.nan)
    u_full = np.full(n_input_time, np.nan)
    v_full = np.full(n_input_time, np.nan)
    radius_full = np.full(n_input_time, np.nan)
    theta_full = np.full(n_input_time, np.nan)
    theta_wrapped_full = np.full(n_input_time, np.nan)
    perp_full = np.full(n_input_time, np.nan)

    fit_slice = slice(i0, i1 + 1)
    n_fit = i1 - i0 + 1

    # layer2.time may not cover exactly i0..i1 in edge cases; use min length
    n_use = min(n_fit, len(layer2.time))
    fit_indices = np.arange(i0, i0 + n_use)

    cycle_full[fit_indices] = cycle_idx_arr[:n_use]
    phase_full[fit_indices] = layer2.phase_mean[:n_use]
    phase_in_cycle_full[fit_indices] = phase_in_cycle_fit[:n_use]
    u_full[fit_indices] = u_fit[:n_use]
    v_full[fit_indices] = v_fit[:n_use]
    radius_full[fit_indices] = layer2.radius_mean[:n_use]
    theta_full[fit_indices] = phase_in_cycle_fit[:n_use]
    theta_wrapped_full[fit_indices] = theta_wrapped_fit[:n_use]
    perp_full[fit_indices] = layer2.perp_deviation_mean[:n_use]

    samples = pd.DataFrame({
        "sample_index": all_sample_index,
        "time": all_time,
        "cycle": cycle_full,
        "phase": phase_full,
        "phase_in_cycle": phase_in_cycle_full,
        "u": u_full,
        "v": v_full,
        "radius": radius_full,
        "theta": theta_full,
        "theta_wrapped": theta_wrapped_full,
        "perp": perp_full,
    })

    # ---- Get R_k posterior stats from layer2 idata ----
    try:
        R_k_post = layer2.idata.posterior["R_k"]
        R_k_mean = R_k_post.mean(("chain", "draw")).values
        R_k_sd = R_k_post.std(("chain", "draw")).values
    except Exception:
        R_k_mean = layer2.radius_mean[0:K_cyc] if len(layer2.radius_mean) >= K_cyc else np.full(K_cyc, np.nan)
        R_k_sd = np.zeros(K_cyc)

    # ---- Build cycles DataFrame ----
    cycle_rows = []
    for k in range(K_cyc):
        t_start_k = float(layer1.tau_mean[k])
        t_stop_k = float(layer1.tau_mean[k + 1])
        s_start_k = max(0, int(np.ceil(t_start_k * fs)))
        s_stop_k = min(n_input_time, int(np.floor(t_stop_k * fs)) + 1)
        duration_k = t_stop_k - t_start_k
        t_quarter_k = t_start_k + 0.25 * duration_k
        n_samples_k = s_stop_k - s_start_k

        # Compute perp stats within this cycle from fitted window
        fit_mask_k = (cycle_idx_arr[:n_use] == k)
        if fit_mask_k.sum() > 0:
            perp_k = layer2.perp_deviation_mean[:n_use][fit_mask_k]
            perp_mean_k = float(np.mean(perp_k))
            perp_sd_k = float(np.std(perp_k))
        else:
            perp_mean_k = np.nan
            perp_sd_k = np.nan

        R_k_mean_k = float(R_k_mean[k]) if k < len(R_k_mean) else np.nan
        R_k_sd_k = float(R_k_sd[k]) if k < len(R_k_sd) else np.nan

        cycle_rows.append({
            "cycle": k,
            "sample_start": s_start_k,
            "sample_stop": s_stop_k,
            "time_start": t_start_k,
            "time_stop": t_stop_k,
            "time_quarter": t_quarter_k,
            "duration": duration_k,
            "center_x": float(layer2.cycle_center_mean[k, 0]),
            "center_y": float(layer2.cycle_center_mean[k, 1]),
            "center_z": float(layer2.cycle_center_mean[k, 2]),
            "e1_x": float(layer2.cycle_e1_mean[k, 0]),
            "e1_y": float(layer2.cycle_e1_mean[k, 1]),
            "e1_z": float(layer2.cycle_e1_mean[k, 2]),
            "e2_x": float(layer2.cycle_e2_mean[k, 0]),
            "e2_y": float(layer2.cycle_e2_mean[k, 1]),
            "e2_z": float(layer2.cycle_e2_mean[k, 2]),
            "normal_x": float(layer2.cycle_normal_mean[k, 0]),
            "normal_y": float(layer2.cycle_normal_mean[k, 1]),
            "normal_z": float(layer2.cycle_normal_mean[k, 2]),
            "radius_mean": R_k_mean_k,
            "radius_sd": R_k_sd_k,
            "perp_mean": perp_mean_k,
            "perp_sd": perp_sd_k,
            "n_samples": n_samples_k,
            "fit_ok": True,
        })

    cycles = pd.DataFrame(cycle_rows, columns=CYCLE_COLUMNS)

    # ---- Build details dict ----
    details = {
        "algorithm": "bayesian",
        "diagnostics": {
            "failures": diagnostics.failures,
            "warnings": diagnostics.warnings,
            "boundary_multimodal": diagnostics.boundary_multimodal,
            "rho_tau": diagnostics.rho_tau,
            "rho_z_median": diagnostics.rho_z_median,
            "center_drift_ratio": diagnostics.center_drift_ratio,
            "omega_ratio": diagnostics.omega_ratio,
            "sigma_x_over_RX": diagnostics.sigma_x_over_RX,
            "phase_monotonic": diagnostics.phase_monotonic,
        },
        "uncertainty": {
            "tau_sd": layer1.tau_sd,
            "period_sd": layer1.period_sd,
            "center_sd": layer1.center_sd,
            "sigma_x_mean": layer2.sigma_x_mean,
            "radius_sd": R_k_sd,
        },
        "sampling_metadata": {
            "draws": draws,
            "tune": tune,
            "chains": chains,
            "target_accept": target_accept,
            "random_seed": random_seed,
        },
    }

    if return_report:
        details["report"] = {"layer1": layer1.idata, "layer2": layer2.idata}

    return samples, cycles, details


def _fit_bayesian_phase_coordinates_legacy(
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
    """Legacy wrapper returning BayesianPhaseResult (for internal use)."""
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

    _import_pymc()
    _import_pytensor_tensor()
    _import_arviz()
    use_numba = _numba_available()

    R_X, xbar = robust_movement_scale(X_arr)
    ref = dominant_reference_signal(X_arr)
    T0 = estimate_dominant_period(ref, fs)
    tau_idx = seed_boundary_indices(ref, fs, T0)
    seed_epochs = epochs_from_boundary_indices(
        tau_idx, sampling_rate_hz=fs, n_time=X_arr.shape[0],
        source="periodogram_peaks", metadata={"T0": T0},
    )

    layer1 = _fit_layer1(
        X_arr, fs, seed_epochs, T0, R_X, xbar,
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=random_seed, use_numba=use_numba,
    )
    layer2 = _fit_layer2(
        X_arr, fs, layer1, T0, R_X, n_velocity_knots=n_velocity_knots,
        draws=draws, tune=tune, chains=chains, target_accept=target_accept,
        random_seed=random_seed, use_numba=use_numba,
    )

    K = len(layer1.tau_mean)
    estimates = BayesianPhaseEstimates(
        tau=layer1.tau_mean,
        period=layer1.period_mean,
        cycle_center=layer1.center_mean,
        cycle_normal=layer1.normal_mean,
        boundary_direction=layer1.a0_mean,
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
        cycle_normal_angular_sd=np.zeros(K - 1),
        boundary_direction_sd=np.zeros_like(layer1.a0_mean),
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
