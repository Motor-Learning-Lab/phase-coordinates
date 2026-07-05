import time
import numpy as np
import pymc as pm
import pymc_extras as pmx


def main():
    rng = np.random.default_rng(0)
    x = rng.normal(size=50)
    y = 2.0 * x + 1.0 + rng.normal(scale=0.5, size=50)

    with pm.Model() as m:
        a = pm.Normal("a", 0, 5)
        b = pm.Normal("b", 0, 5)
        sigma = pm.HalfNormal("sigma", 2)
        mu = a * x + b
        pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        t0 = time.time()
        idata = pmx.fit_laplace(draws=500, progressbar=False, random_seed=0)
        print("fit_laplace took", time.time() - t0, "s")

    print(type(idata))
    post = idata.posterior
    print("dims:", post.dims)
    print("a mean:", float(post["a"].mean()), "expected ~2.0")
    print("b mean:", float(post["b"].mean()), "expected ~1.0")
    print("a sd:", float(post["a"].std()))


if __name__ == "__main__":
    main()
