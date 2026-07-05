# Layer 2 sampling debug materials

This directory holds raw scratch scripts and run logs collected while
diagnosing an unresolved NUTS convergence problem in Layer 2 of the Bayesian
two-layer estimator (`phase_coordinates/bayesian.py`). See
`docs/PROGRESS.md` for the full narrative; this directory is the
supporting evidence, kept so a different agent/session can inspect raw
numbers and rerun things without re-deriving them.

All scripts can now be run with:

```bash
pixi run python docs/debug/scripts/<script>.py
```

from the repo root (the pixi environment at the repo root has all required
packages). The scripts auto-detect the repo root via `__file__`, so the
hard-coded Windows path (`D:\Repositories\phase-coordinates`) is no longer
needed (the updated scripts handle both Linux and Windows via `os.path`).

Legacy note: the original miniforge environment (`/c/Users/User/miniforge3/python.exe`
on Windows) is superseded by the `pixi.toml` environment added on 2026-07-05.

## scripts/

| File | Purpose |
|---|---|
| `test_utils.py` | Sanity-checks the pure-numpy/scipy utility functions (robust scale, seeding, frame construction, spline matrices). All passed; not implicated in the Layer 2 issue. |
| `proto_interp.py`, `proto_interp_numba.py` | Earliest feasibility prototypes for differentiable linear interpolation of a fixed data grid at continuous-time latent query points (used by Layer 1's boundary-clustering likelihood). Confirmed the technique works; `_numba.py` variant confirmed the NUMBA compile backend fixes a severe slowdown from the missing C++ compiler. Superseded by the real implementation; kept for reference only. |
| `test_layer1.py` | End-to-end Layer 1 check on synthetic fixed-plane data. **Passes reliably** (see logs). |
| `test_layer1_changing.py` | End-to-end Layer 1 check on synthetic changing-plane-per-cycle data. **Passes** at a gentle (8 deg/cycle) rotation rate; a steeper 30 deg/cycle rate (matching the existing deterministic-estimator test fixture) is in real conflict with the model's own `HalfNormal(0.10 rad)` cycle-to-cycle smoothness prior and isn't a fair test of this model — see PROGRESS.md deviation #5. |
| `debug_layer1.py` | Per-chain diagnostic dump (posterior mean/SD of each cycle's normal, per chain) — used to catch the Layer 1 sign-flip bug (now fixed). |
| `test_layer2.py` | **The main repro script.** End-to-end Layer 1 + Layer 2 fit on synthetic fixed-plane data, with assertions on normal recovery, phase monotonicity, radius, perpendicular deviation, and e1/e2/n frame orthonormality. All checks pass after reparameterization (log 08) — primary artifact gone, amplitude params still slightly off. |
| `test_layer2_tune.py` | Variant of the above used to try `n_velocity_knots=6` and `target_accept=0.95/0.99` instead of the defaults. Made things worse, not better (log 03). |
| `test_layer2_tune1000.py` | Variant of `test_layer2.py` with `tune=1000` (vs 400) and two added assertions: radius median > 0.85 and sigma_x < 0.10. Motivated by Chain 1 max_treedepth and low ESS in log 08 — more warmup should let mass matrix adapt to amplitude parameters. |
| `debug_layer2_normal.py` | Finds the time index/indices with the worst (lowest) cos-similarity to the known true normal in a Layer 2 fit, and prints the recovered normal vectors and `normal_angular_sd` (posterior uncertainty) around that region. This is what revealed the artifact is localized to a narrow window between two spline knots (log 04). |
| `test_laplace_proto.py` | Prototype of `pymc_extras.fit_laplace()` as an alternative inference backend (MAP + Gaussian/Laplace approximation instead of full MCMC) on a trivial unrelated regression model, just to confirm the API and output shape work in this environment. **Not yet wired into the real Layer 2 model** — this was queued as a fallback option but deprioritized in favor of first trying to fix the NUTS parameterization directly. `pymc_extras` (0.12.1) is installed in the miniforge env if this route is picked back up; note the environment had a stale/broken editable install pointing at a nonexistent `C:\Repositories\pymc-extras` that had to be uninstalled and reinstalled properly first (`pip uninstall pymc-extras -y && pip install pymc-extras --no-cache-dir`) — if `import pymc_extras` fails with `ModuleNotFoundError` despite `pip show` claiming it's installed, check for this. |

## logs/

Raw stdout/stderr captured from background runs of the scripts above, in
roughly chronological order (numbered prefix). PyMC's own convergence
warnings (divergences, max-treedepth, r_hat, ESS) are the important content;
the trailing `print()` output from each test script (cos_sim stats, timing,
etc.) is the interpreted result.

| File | Script | Config | Result |
|---|---|---|---|
| `01_proto_interp_pure_python_slow.log` | `proto_interp.py` | 8 params, no numba | 35s for 400 draws total — motivated the numba backend. |
| `02_layer2_first_full_run_17div_423s.log` | `test_layer2.py` | 12 vel knots, target_accept=0.9, plain PyMC, no explicit initvals (relying on `init="adapt_diag"` only) | Layer2: 423s, **17 divergences**, max-treedepth both chains, r_hat/ESS warnings. Despite the warnings, **normal recovery was actually good this run** (min cos_sim 0.997, median 0.9998) — the only assertion failure was an overly-strict e1⊥n tolerance in the test itself (since fixed by reconstructing e1/e2 from posterior-mean n/a instead of averaging per-draw e1/e2). |
| `03_layer2_tuning_experiment_6knots_worse.log` | `test_layer2_tune.py` | 6 vel knots, target_accept=0.95 then 0.99 | Both configs worse: 610s / 47 divergences for the first (accept=0.95); second config (accept=0.99) was killed before finishing because the first result already showed the direction was wrong. Fewer knots + higher target_accept made divergences worse, not better. |
| `04_layer2_worst_cossim_location_debug.log` | `debug_layer2_normal.py` | (analyzing a run with the sign-flip artifact) | Worst cos_sim (~0.40) is localized to t=0.58-0.67s, sitting between spline knots at tau_mean[0]=0.295 and tau_mean[1]=1.288. **Important**: `normal_angular_sd` in that bad window (0.060-0.073 rad) is *not* elevated relative to typical values seen elsewhere in a healthy fit (~0.06-0.07 rad in Layer 1) — i.e. the model's own posterior-SD-based uncertainty does not flag this region as uncertain, even though the point estimate there is badly wrong. This suggests the bad draws are not a case of genuine, detectable bimodality/high-variance in the posterior; more likely most draws agree on a stable-but-wrong compromise there. See "leads" below. |
| `05_layer2_nutpie_signflip_221s.log` | `test_layer2.py` (with `nuts_sampler="nutpie"`, later reverted) | 12 vel knots, nutpie backend, no working initvals (nutpie doesn't honor PyMC's `initvals=`) | 221s (much faster than plain PyMC), but **min cos_sim 0.40** — the sign/spline-excursion artifact, attributed at the time to nutpie's own default (jittered) initialization not being controllable. |
| `06_layer2_initvals_run_interrupted_by_crash.log` | `test_layer2.py` (plain PyMC + explicit `initvals=` fix applied) | 12 vel knots, target_accept=0.9 | Session crashed mid-run; Layer 1 had completed cleanly (34s, 0 divergences, no rhat warnings). Layer 2 had just started sampling. Superseded by log 07 (rerun after crash). |
| `07_layer2_initvals_rerun_84div_999s_WORST.log` | `test_layer2.py` (same as 06, rerun after crash) | 12 vel knots, target_accept=0.9, plain PyMC, explicit initvals for every free var | **999s, 84 divergences** (worse than log 02's 17) — explicit initvals did not fix the problem and made sampling harder, not easier. min cos_sim 0.51 (same artifact, still present). This is the strongest evidence that **initialization is not the (main) root cause** — Layer 1's identical fix (explicit initvals + `init="adapt_diag"`) fully resolved its own, mechanistically similar sign-flip bug, but the same fix did not transfer to Layer 2. |
| `08_layer2_reparam_0div_418s_CHECKS_PASSED.log` | `test_layer2.py` (reparameterized model — tangent-plane normals + boundary-normalized phase) | 12 vel knots, tune=400, draws=400, chains=2, target_accept=0.9 | **0 divergences, 418s, ALL CHECKS PASSED.** Normal artifact: GONE (min cos_sim 0.9915 vs 0.51 before). Phase monotone by construction. Radius median 0.709 (barely above 0.7 floor), sigma_x_mean 0.467 (expected ~0.02). Chain 1 max_treedepth, rhat/ESS warnings — amplitude parameters haven't converged yet but normal and phase fully recovered. |
| `09_layer2_reparam_debug_normal_worst_cossim_0.9915.log` | `debug_layer2_normal.py` (on reparameterized model) | Same as 08 | Worst cos_sim across all time: **0.9915** (t≈1.10-1.17s, between spline knots 0 and 1). True normal cos_sim ≥ 0.9915 everywhere — the old localized artifact (cos_sim 0.40) is completely gone. normal_angular_sd at worst points ≈ 0.050 rad (model slightly underestimates uncertainty at these points but no longer confidently wrong). |

## Reparameterization implemented (2026-07-05)

All changes from `docs/claude_layer2_reparameterization_prompt.md` have been
applied to `phase_coordinates/bayesian.py`. See `docs/PROGRESS.md` for the
full list. Key changes:

1. **Normal**: `u2 ~ N(u_mean, sigma_u2)` replaced by tangent-plane deviations
   `delta_n ~ N(0, sigma_theta2 * I_2)` with Layer 1 angular SD as scale.
   Normalized normal knots spliced into a spline, then renormalized at each t.
   Adjacent-knot smoothness Potential added (`sigma_Delta_n = 0.10`).

2. **Phase**: `phi0`, `g_knots`, and `phase_boundary` Potential removed entirely.
   Replaced by `q_knots ~ N(0, 0.20^2)` with boundary-normalized positive speed:
   `phi(t) = 2*pi*k + 2*pi * S_{k,t} / S_{k,total}`. Boundaries satisfied by
   construction.

3. **`sigma_a2` floor**: 0.02*R_X floor added (was missing before).

4. **Resultant-length diagnostic**: `_compute_diagnostics` now warns if
   `||E[n(t)|X]|| < 0.80`.

**Test run result (log 08)**: 0 divergences, 418s, ALL CHECKS PASSED. Normal
artifact gone (min cos_sim 0.9915). Amplitude parameters (radius, sigma_x) still
converging — Chain 1 max_treedepth, low ESS. See log 08 for full output.

**Follow-up (log 09)**: `debug_layer2_normal.py` confirmed the localized artifact
is gone — worst cos_sim across all 600 time points is 0.9915 (vs 0.40 before).

**Ongoing**: `test_layer2_tune1000.py` (tune=1000) running to investigate amplitude
convergence.

## Post-reparameterization status (2026-07-06)

The phase-velocity banana hypothesis was **CONFIRMED AND FIXED**. Replacing the
`g_knots + phase_boundary Potential` model with boundary-normalized positive phase
(`q_knots ~ N(0, 0.20^2)`, boundaries satisfied by construction) eliminated all
84 divergences (log 08: 0 divergences). The localized normal artifact is also gone
(log 09: worst cos_sim 0.9915 everywhere, vs 0.40 before).

**Remaining issue — amplitude convergence.** Chain 1 hits max_treedepth and ESS
< 100 for some parameters. Radius posterior median = 0.709 (expected ~1.0);
sigma_x_mean = 0.467 (expected ~0.02). These indicate the amplitude subspace
(h_r_knots / rho_x) hasn't converged yet. Most likely cause: `adapt_diag` hasn't
had enough tune steps to learn the right per-parameter scales for the amplitude
parameters. Experiment running: tune=1000 vs previous tune=400 (see
`test_layer2_tune1000.py`).

## Ideas remaining

1. **Amplitude convergence (in progress)**: try tune=1000. If radius and sigma_x
   still don't converge, try a stronger prior on rho_x (currently
   `Lognormal(log(0.03), 0.5)`) or a prior directly on sigma_x/R_X from a
   small-noise assumption.
2. **Run with 4 chains** for better r_hat diagnostics (2 chains gives marginal
   statistical power to distinguish a stuck chain from sampling variance).
3. **`pymc_extras.fit_laplace()` fallback** remains available if MCMC proves too
   slow for the pytest suite (a full fit takes 418s at the base setting). See
   `test_laplace_proto.py` and the spec's own allowance for this ("If full PyMC
   sampling is too heavy, implement deterministic seeds plus uncertainty
   approximations first, but keep the structure ready for sampling").
