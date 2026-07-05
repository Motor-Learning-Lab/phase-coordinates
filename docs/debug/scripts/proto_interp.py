import time
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def pt_linear_interp(t_grid_c, X_grid_c, tau, n_grid):
    idx = pt.extra_ops.searchsorted(t_grid_c, tau)
    idx = pt.clip(idx, 1, n_grid - 1)
    t0 = t_grid_c[idx - 1]
    t1 = t_grid_c[idx]
    w = (tau - t0) / (t1 - t0)
    w = pt.clip(w, 0.0, 1.0)
    X0 = X_grid_c[idx - 1]
    X1 = X_grid_c[idx]
    return X0 + w[:, None] * (X1 - X0)


def main():
    rng = np.random.default_rng(0)

    fs = 50.0
    T0 = 1.0
    n_time = 400
    t_grid = np.arange(n_time) / fs

    true_tau = np.array([0.9, 1.95, 3.05, 3.98])
    X_grid = np.zeros((n_time, 3))
    X_grid[:, 0] = np.cos(2 * np.pi * t_grid / T0)
    X_grid[:, 1] = np.sin(2 * np.pi * t_grid / T0)
    X_grid[:, 2] = 0.05 * np.sin(2 * np.pi * t_grid / T0 * 3)
    X_grid += rng.normal(scale=0.01, size=X_grid.shape)

    t_grid_pt = pt.constant(t_grid, name="t_grid")
    X_grid_pt = pt.constant(X_grid, name="X_grid")

    hat_tau = np.array([0.88, 1.9, 3.0, 4.0])

    print("building model...", flush=True)
    with pm.Model() as m:
        tau = pm.Normal("tau", mu=hat_tau, sigma=0.075 * T0, shape=4)
        X_tau = pt_linear_interp(t_grid_pt, X_grid_pt, tau, n_time)
        pm.Deterministic("X_tau", X_tau)
        mu_tau = pm.Normal("mu_tau", mu=0.0, sigma=1.0, shape=3)
        rho_tau = pm.Lognormal("rho_tau", mu=np.log(0.10), sigma=0.5)
        sigma_tau_x = rho_tau * 1.0
        pm.Potential(
            "boundary_cluster",
            pm.logp(pm.Normal.dist(mu=mu_tau, sigma=sigma_tau_x), X_tau).sum(),
        )
        print("model built, checking logp...", flush=True)
        print(m.point_logps(), flush=True)
        print("sampling...", flush=True)
        t0 = time.time()
        idata = pm.sample(
            draws=200, tune=200, chains=2, cores=1, progressbar=True, random_seed=1
        )
        print("sampling done in", time.time() - t0, "s", flush=True)

    import arviz as az
    print(az.summary(idata, var_names=["tau", "mu_tau", "rho_tau"]))
    print("true tau:", true_tau)
    print("n_divergences:", int(idata["sample_stats"]["diverging"].sum()))


if __name__ == "__main__":
    main()
