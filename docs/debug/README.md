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
| `test_layer2.py` | **The main repro script.** End-to-end Layer 1 + Layer 2 fit on synthetic fixed-plane data, with assertions on normal recovery, phase monotonicity, radius, perpendicular deviation, and e1/e2/n frame orthonormality. This is what keeps intermittently failing on the `normal cos_sim` min-value assertion (or would, if that assertion weren't currently loose) — see logs 02, 05, 07. |
| `test_layer2_tune.py` | Variant of the above used to try `n_velocity_knots=6` and `target_accept=0.95/0.99` instead of the defaults. Made things worse, not better (log 03). |
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

**Test run launched**: `pixi run python docs/debug/scripts/test_layer2.py`.
Will add results log and update this README when it completes.

## Current best hypothesis (not confirmed)

The phase-velocity sub-model has a tight, nonlinear constraint: a handful of
`g_knots` (log phase-velocity spline coefficients, default count =
`clip(2*n_cycles, 4, 20)`, 12 in these tests) get exponentiated, cumulatively
summed over ~600 time steps (`phi_t = phi0 + cumsum(exp(spline(g_knots)) *
dt)`), and must land within `sigma=0.15 rad` of `2*pi*k` at *every one* of
`K=7` boundary times simultaneously (the `phase_boundary` `pm.Potential` in
`_fit_layer2`, `phase_coordinates/bayesian.py`). This is a classic
"banana"/funnel-shaped posterior that PyMC's default diagonal mass matrix
(`adapt_diag`) handles poorly (confirmed indirectly: nutpie's richer,
non-diagonal mass-matrix adaptation ran ~2x faster per log 05, even though it
had its own separate initialization problem).

Because HMC/NUTS proposes all parameters jointly within one leapfrog
trajectory, a badly-behaved trajectory driven by the phase sub-model's
difficulty could be dragging otherwise well-behaved parameters (like a `u2`
spline knot for the normal vector) along with it, transiently, in a way that
biases the posterior mean without necessarily showing up as elevated
posterior SD for those other parameters (see the `04_...` log finding above).
This would explain why the *localized artifact* (not a full, sustained sign
flip) tracks with the *severity* of the divergence problem across attempts
(worse divergences -> worse artifact: log 02's 17 divergences produced a
*good* fit; log 07's 84 divergences produced the worst artifact seen).

This has NOT been confirmed with a targeted experiment (e.g., isolating the
phase/velocity sub-model alone, without the normal/center/radius splines, to
see if it reproduces divergences on its own).

## Ideas not yet tried, for whoever picks this up

1. **Isolate the phase-velocity sub-model.** Build a minimal PyMC model with
   *only* `g_knots`, `phi0`, and the `phase_boundary` potential (no
   center/normal/radius/observation likelihood at all) and see if it alone
   produces divergences. This would confirm or rule out the hypothesis above
   cheaply (much smaller model, faster to sample) before spending more time
   reparameterizing the full thing.
2. **Reparameterize the phase-boundary constraint to be less like a
   near-equality constraint on a highly nonlinear function.** E.g., instead
   of free `g_knots` with an independent prior around `log(omega0)` PLUS a
   separate smoothness potential PLUS a separate tight boundary-hitting
   potential (three separate soft constraints all fighting to shape the same
   handful of parameters), consider directly parameterizing velocity in a way
   that satisfies the boundary constraint *by construction*, with only
   smoothness left as a free/random property. (E.g., anchor the average
   velocity per inter-boundary segment to exactly `2*pi / T_k` deterministically,
   and let `g_knots` express only the *within-segment shape* deviation around
   that anchored mean, subject to a mean-zero-per-segment constraint. This
   would make the tight `phase_boundary` potential largely unnecessary since
   correctness would be structural rather than fitted.)
3. **Try a non-centered parameterization for `g_knots`** even though this
   isn't a classical hierarchical-funnel case — worth ruling out quickly.
4. **Try natural cubic spline alternatives for the `u(t)`/`n(t)` spline**
   (`bc_type="not-a-knot"` or `"clamped"` instead of `"natural"` in
   `cubic_spline_matrix()`) to see if the localized artifact is partly a
   spline-overshoot artifact independent of the phase-velocity coupling
   hypothesis — cheap to test in isolation with `test_utils.py`-style checks
   (no PyMC needed) by checking spline overshoot on cyclic constant-vector
   data.
5. **Run with 4 chains instead of 2** and check r_hat carefully — with only 2
   chains a single bad chain can't be statistically distinguished from a
   sampling fluke as easily.
6. **Try raising `target_accept` alone (0.95, 0.97, 0.99) on the *original*
   12-knot configuration** — this specific combination (original knot count +
   higher target_accept) was never actually tested; log 03 confounded the
   knot-count change with the target_accept change.
7. If none of the above converges cleanly in reasonable time, the
   `pymc_extras.fit_laplace()` fallback (see `test_laplace_proto.py`) reuses
   the *exact same* PyMC model definition (same priors/likelihood, "structure
   ready for sampling" per the spec's own allowance for this) and replaces
   full MCMC with MAP + a Gaussian (Laplace) approximation at the mode — much
   cheaper and avoids the NUTS geometry problem entirely, at the cost of a
   less faithful (Gaussian-approximate rather than exact) posterior. This
   was queued as the next thing to try before the debugging focus shifted to
   reparameterization; it remains available as a documented spec-sanctioned
   fallback (see `docs/bayesian_two_layer_spec.md` / the Claude prompt's
   "Implementation strategy" step 3: "If full PyMC sampling is too heavy,
   implement deterministic seeds plus uncertainty approximations first, but
   keep the structure ready for sampling").
