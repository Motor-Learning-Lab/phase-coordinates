# Bayesian two-layer estimator — implementation progress log

Working notes for `phase_coordinates/bayesian.py`, updated as work proceeds.
This machine has had a couple of crashes mid-session, so this file exists to
let work resume without re-deriving context. Update it whenever something
meaningfully changes (a fix lands, a test passes/fails, a new problem is
found).

Branch: `bayesian-two-layer-estimator` (based on `bayesian-estimation-spec-main`,
which already has the two spec docs; NOT based on `main`, which lacks them).

Environment notes (see "Environment" section below) — **use the miniforge
python for everything**, not the `python` on PATH.

## Status as of last update (2026-07-06)

**Reparameterization complete; a2_init fixes applied; amplitude-noise ridge
diagnosed in detail. Root cause: adapt_diag warmup drifts to wrong mode and
the adapted mass matrix locks it in. Fix identified: initialize from MAP or
from a fixed-sigma fit (seed=0).**

The full reparameterization (tangent-plane normals + boundary-normalized phase)
eliminated divergences and the localized normal artifact. Amplitude parameters
(radius, sigma_x) still converge to wrong values due to an amplitude-noise ridge
in the posterior: `r_t` and `sigma_x` can trade off (`r × sigma_x ≈ const`) along
a nearly-flat likelihood direction. NUTS warmup drifts to the ridge and the adapted
mass matrix then locks the chain in — both chains stuck from draw 1 (finding 12).

**Fixes applied (findings 8, 11):**
- initval: `a2_init[k] = X_fit[idx_k] - c_mean_p[k]` (reduced ~19-sigma boundary
  residuals to noise level; finding 8)
- prior mean: changed `a2 ~ Normal(mu=a_mean_p, ...)` to `a2 ~ Normal(mu=a2_init, ...)`
  (theoretically correct; marginal improvement only — finding 11)
- Reverted `_OBS_NOISE_LOGNORMAL_SD` from 0.3 back to 0.5 (finding 7)

**Current test result** (`test_layer2.py`, finding 11): test assertions PASS, 40
divergences, radius median 1.123, sigma_x 0.362. sigma_x is still ~18x ground truth.

**Diagnosed (finding 12):** chains start near truth (r≈1, sigma_x≈0.03) but warmup
carries them to the ridge (corr = -0.993, stuck from draw 1, zero draws near true
mode). The amplitude is run-to-run unstable — different wrong ridge points each run
(r=0.709/0.88/1.12/1.25 across logs 08–11). True mode is 4763 nats better than the
posterior mean but never found. Fixed sigma_x (seed=0) reaches r=0.961 in 8 div
(finding 9), confirming the correct mode exists; the obstacle is warmup, not geometry.

**Next step:** initialize the free-sigma fit from a fixed-sigma (seed=0) run or from
`pm.find_MAP()`. See finding 12 for full evidence and recommendation.

**See "Layer 2 findings" items 4-12 for the detailed evidence trail.**

**Environment**: a new pixi environment is now set up for this project
(`pixi.toml` at the repo root). Use `pixi run python <script>` or
`pixi run pytest` instead of the miniforge python from the old Windows notes.
The pixi env has PyMC 6.0.1, ArviZ 1.2.0, numba 0.65.1.

### Reparameterization changes applied (2026-07-05)

1. **New utilities**: `align_normal_signs(normals)` and
   `orthonormal_tangent_basis(normals)` added to `bayesian.py`. All tests
   pass (unit norm, orthogonal to normal, mutually orthogonal, edge cases
   near coordinate axes, sign alignment).

2. **Layer 2 normal**: replaced the raw-vector `u2 ~ N(u_mean_p, sigma_u2)`
   spline with tangent-plane deviations `delta_n ~ N(0, sigma_theta2 * I_2)`.
   The prior scale `sigma_theta2` now uses Layer 1 angular posterior SD (not
   componentwise vector SD), with floor 0.03 rad. Spline still goes through
   the normalized normal knots, followed by renormalization at each time step.

3. **Layer 2 normal smoothness**: added a `normal2_smoothness` Potential
   across adjacent normalized normal knots, with `sigma_Delta_n = 0.10`.
   No absolute value (signs explicitly aligned).

4. **Layer 2 boundary-direction floor**: `sigma_a2` now has a `0.02*R_X`
   floor (previously had no floor; this was deviation #4 from the spec).

5. **Phase reparameterization**: removed `phi0`, `g_knots`, and the
   `phase_boundary` Potential entirely. Replaced with `q_knots ~ N(0, 0.20^2)`
   (mean-zero prior), smoothness Potential, and a boundary-normalized positive
   speed model. Phase satisfies `phi(tau_k) = 2*pi*k` exactly by construction.
   Per-cycle cumulative weight matrices are precomputed as constant PyTensor
   tensors. Phase velocity computed analytically from `2*pi*w / S_{k,total}`.

6. **Normal mean resultant length diagnostic**: added `normal_raw_mean` and
   `normal_resultant_length` to `_Layer2Summary` and `BayesianPhaseDiagnostics`.
   `_compute_diagnostics` warns if `||E[n(t)|X]|| < 0.80` at any time point.

**All pure-numpy / non-MCMC tests pass:**
- `align_normal_signs` and `orthonormal_tangent_basis`: unit norm, tangent-plane
  orthogonality, mutual orthogonality, edge cases — ALL PASS.
- Boundary-normalized phase construction: phi(tau_k) = 2*pi*k exactly, monotone
  with non-uniform speed, invariant to constant offset in q within a cycle — ALL PASS.
- Frame construction (e1/e2/n): unchanged from previous validated version.

**MCMC test results (log 08):** 0 divergences / 418s / min cos_sim 0.9915 /
ALL LAYER2 CHECKS PASSED. Amplitude parameters still converging (max_treedepth,
low ESS). See "Layer 2 findings" item 4 below for full detail.

**All raw diagnostic scripts and run logs are in `docs/debug/` (see
`docs/debug/README.md` for an index).**

## What's done

- [x] Branch created, `pyproject.toml` has `[project.optional-dependencies] bayes = ["pymc", "arviz"]`.
- [x] Utility functions in `bayesian.py`: `robust_movement_scale`, `dominant_reference_signal`,
      `estimate_dominant_period`, `seed_boundary_indices`, `seed_cycle_centers`,
      `seed_cycle_normals`, `seed_boundary_vectors`, `normalize`, `construct_frame`,
      `cubic_spline_matrix`, `spline_eval`, `_linear_interp_matrix`, `_pt_interp_at`.
      All validated against direct scipy/numpy equivalents in `docs/debug/scripts/test_utils.py`. PASS.
- [x] Result dataclasses: `BayesianPhaseEstimates`, `BayesianPhaseUncertainty`,
      `BayesianPhaseDiagnostics`, `BayesianPhaseResult` (+ internal `_Layer1Summary`,
      `_Layer2Summary`).
- [x] Layer 1 coarse PyMC model (`_fit_layer1`). Validated on fixed-plane and
      changing-plane (8 deg/cycle) synthetic data — normal recovery cos_sim > 0.97
      for all cycles, ~7-30s to fit. PASS (see "Layer 1 findings" below for two bugs
      found and fixed along the way).
- [x] Layer 2 instantaneous PyMC model (`_fit_layer2`): fixed-knot-location cubic
      splines for center/normal/boundary-direction, positive phase-velocity spline
      with cumulative-sum phase, radius/perp-deviation splines, observation
      likelihood. Structurally implemented and correct — frame construction,
      spline math, and observation model all verified right when sampling
      behaves — but NUTS sampling itself has an **unresolved convergence
      problem** (high divergence rate + a localized sign/spline-excursion
      artifact in the recovered normal). See "Layer 2 findings" below and
      `docs/debug/README.md` for the full diagnosis; this is the open issue
      blocking everything downstream (tests, README, deliverables).
- [x] Diagnostics (`_compute_diagnostics`): boundary multimodality (KDE-based),
      rho_tau, projection ratio, normal-dominated-by-prior, rho_z, center drift,
      omega ratio, sigma_x/R_X, phase monotonicity sanity check. Not yet tested.
- [x] Public API `fit_bayesian_phase_coordinates()` wired up in `bayesian.py`,
      exported from `phase_coordinates/__init__.py` alongside the dataclasses.
      Not yet tested end-to-end via the public entry point (only via internal
      `_fit_layer1`/`_fit_layer2` calls in scratch scripts).

## What's NOT done yet

- [ ] **Amplitude convergence** — radius and sigma_x still off ground truth
      (current: radius 1.148 vs expected ~1.0; sigma_x 0.341 vs expected ~0.02).
      Root cause: amplitude-noise ridge (`r × sigma_x ≈ const`) — NUTS exploits
      this during warmup and gets stuck off the true mode. a2_init fix improved
      radius from 0.709 → 1.148 but didn't resolve the ridge. Best result to date:
      fixed sigma_x gives radius=0.961 (finding 9). Current approach: two-phase
      sampling (finding 10), not yet completed. Test assertions pass with loose
      bounds (0.7 < median radius < 1.3), but amplitude accuracy needs improvement.
- [ ] pytest test suite (`tests/test_bayesian_phase_coordinates.py`) covering the
      7 required scenarios from the prompt (see "Required tests" below).
- [ ] Run full suite (existing + new) on both the no-pymc default env and the
      miniforge env with pymc.
- [ ] README updates.
- [ ] Final deliverables write-up (files changed, API, tests run, limitations,
      deviations from spec).

## OPEN ISSUE — status and options

Layer 2's NUTS sampling has a real, unresolved convergence difficulty: a high
divergence rate (10-20% of post-tuning draws) and a correlated, localized
artifact in the recovered instantaneous normal (median cos_sim to ground
truth is excellent at 0.999, but a narrow time window has cos_sim as low as
0.40-0.51 across two different attempts). Two rounds of targeted fixes
(smooth angular potential, then explicit initvals) did not resolve it — see
"Layer 2 findings" items 2-3 below for full diagnosis and what's been ruled
out, and **`docs/debug/README.md` for the complete evidence trail (raw logs,
scripts, a root-cause hypothesis, and prioritized untried ideas)**.

**Decision (current instruction): try, in this order —**
1. ~~Fall back to the spec's own explicitly-sanctioned lighter path for Layer 2
   (deterministic point estimates + approximate uncertainty via
   `pymc_extras.fit_laplace()`, reusing the identical PyMC model, keeping
   Layer 1 fully Bayesian)~~ — **deprioritized before being wired in**; a
   prototype confirmed `pymc_extras.fit_laplace()` works in this environment
   (`docs/debug/scripts/test_laplace_proto.py`, ~1s per fit on a trivial
   model) and remains available as a fallback (see `docs/debug/README.md`
   idea #7), but the user redirected effort to reparameterization first.
2. **Fix nutpie's initvals properly** — dig into `nutpie.compile_pymc_model`'s
   internal variable ordering (or find upstream docs/support) to construct a
   correct `init_mean` array; nutpie's richer mass-matrix adaptation is the
   textbook fix for this exact "banana" geometry and already showed ~2x
   better raw speed despite its own separate init problem (see log 05 in
   `docs/debug/logs/`).
3. **Tune/reparameterize the plain-PyMC-NUTS model directly** — see
   `docs/debug/README.md`'s "ideas not yet tried" list (isolating the
   phase-velocity sub-model to confirm the root-cause hypothesis cheaply is
   the recommended first step before broad parameter sweeps).

If none of the above resolves it, stop and report back rather than
improvising further (e.g. don't silently fall back to a documented-limitation
or reduced-scope approach without checking in first) — per standing
instruction to flag rather than route around genuine workability problems.

Other options that were on the table and remain available if 1-3 don't pan
out (not being pursued right now):
- **Ship with a documented limitation**: keep full MCMC for Layer 2, note in
  diagnostics/README that convergence can be marginal for some datasets, and
  rely on the existing diagnostics to warn users when this happens.
- **Try a different environment** (a machine with a C++ compiler and multiple
  cores) where iteration is fast enough to tune this properly in minutes
  rather than 15-20-minute increments.

## Environment

**Current (Linux, 2026-07-05):** A pixi environment has been set up at the
repo root (`pixi.toml`). Run scripts with `pixi run python <script>` or
activate with `pixi shell`. Has Python 3.12, PyMC 6.0.1, ArviZ 1.2.0,
numba 0.65.1, numpy, scipy, scikit-learn, pandas, pytest all installed.
The package is editable-installed via the `pypi-dependencies` table in
`pixi.toml`. This replaces the Windows miniforge environment.

**Note on PyMC 6 vs PyMC 5:** The original code was written for PyMC 5.28
(`idata.posterior` returned an InferenceData). PyMC 6 returns an xarray
DataTree, but `idata.posterior` still works as an attribute (it's a property
returning the "posterior" child DataTree). All existing posterior access
patterns (`post["var"].mean(("chain","draw")).values`) still work unchanged.

**Old environment (Windows, for historical reference):**
- Default `python` on PATH (3.14) has **no** numpy/pandas/sklearn/scipy installed.
  Do not use it.
- **`/c/Users/User/miniforge3/python.exe`** (Python 3.12.12, conda env) had
  pymc 5.28.0, arviz 0.23.0, numba 0.62.1. Now superseded by pixi environment.
- No C++ compiler (`g++`) is available, so PyTensor falls back to pure-Python
  graph execution by default — very slow (a trivial 8-parameter/400-draw model
  took 35s). Fix: pass `compile_kwargs={"mode": "NUMBA"}` to `pm.sample()` when
  `numba` is importable (see `_numba_available()` / `_sample_kwargs()` in
  `bayesian.py`). This brought the same toy model down to ~1s of actual sampling.
- `nutpie` is installed (0.16.11) but **deliberately not used from `bayesian.py`**
  — see "Layer 2 findings" below for why. It remains installed in the env in
  case option 2 above (fixing its `init_mean` handling) is pursued.
- `pymc_extras` is installed (0.12.1) — needed if the `fit_laplace()` fallback
  is picked back up. **Gotcha**: this environment initially had a *stale
  editable install* pointing at a nonexistent path
  (`C:\Repositories\pymc-extras`), left over from unrelated prior work on this
  machine — `pip show` claimed it was installed but `import pymc_extras`
  raised `ModuleNotFoundError`. Fixed with
  `pip uninstall pymc-extras -y && pip install pymc-extras --no-cache-dir`.
  If you hit the same symptom, check `pip show pymc-extras` for an
  `Editable project location` pointing somewhere that doesn't exist.
- Scratch/prototype scripts were developed in `/d/tmp/` (NOT `/tmp/`, which
  resolves to somewhere unhelpful for the Windows python interpreters in this
  Bash tool) and are now **committed under `docs/debug/scripts/`** (with raw
  run output under `docs/debug/logs/`) so they survive across sessions/agents.
  See `docs/debug/README.md` for a full index and what each one showed. Quick
  summary:
  - `test_utils.py` — utility function checks. PASS.
  - `test_layer1.py` — Layer 1 fixed-plane end-to-end check. PASS.
  - `test_layer1_changing.py` — Layer 1 changing-plane check (needs gentle
    rotation rate, see deviation #5 below). PASS at 8 deg/cycle.
  - `debug_layer1.py` — per-chain diagnostic dump for Layer 1 (used to find
    the sign-flip bug).
  - `test_layer2.py` — **the main Layer 2 repro script**, still intermittently
    failing (see "Layer 2 findings" and `docs/debug/README.md`).
  - `test_layer2_tune.py` — variant used to try fewer knots / higher
    target_accept; made things worse.
  - `debug_layer2_normal.py` — finds the worst-cos_sim time index in a Layer 2
    fit (used to diagnose the sign-flip/spline-excursion artifact).
  - `proto_interp.py`, `proto_interp_numba.py` — original
    differentiable-interpolation feasibility prototypes (superseded, kept for
    reference).
  - `test_laplace_proto.py` — confirms `pymc_extras.fit_laplace()` works in
    this environment, on a trivial unrelated model (not yet wired into the
    real Layer 2 model).
- **Always run PyMC scripts with `cores=1`** and a `if __name__ == "__main__":`
  guard. Windows multiprocessing (`cores>1`) re-imports the launching script in
  each worker process; without the guard this recursively re-executes the whole
  script (observed as an apparent hang — actually infinite recursive spawning).
  `_sample_kwargs()` always sets `cores=1` for this reason, independent of the
  guard, since library code can't force callers to add the guard.
- Sampling with `progressbar=True` from a backgrounded Bash tool call can hang
  the *tool's* output capture (box-drawing progress bar characters seem to
  confuse it) — use `progressbar=False` in any script whose output needs to be
  read back by the agent. `_sample_kwargs()` always sets this.

## Layer 1 findings (both fixed, both confirmed)

1. **`pt.clip(x, lo, None)` fails.** PyTensor's `clip` doesn't accept `None`
   like numpy does ("Cannot convert None to a tensor variable"). Fix: use
   `pt.maximum(x, lo)` instead for a one-sided clip.

2. **Sign-flip / divergence bug in the cycle-normal smoothness prior.** The
   literal spec formula `cos^-1(|n_k . n_{k-1}|) ~ HalfNormal(0.10)` has a
   gradient singularity in `arccos` exactly at `|cos| -> 1` — i.e. exactly
   where the prior concentrates mass for slowly-rotating data. Observed
   effect: with 2 chains, one chain converged to a sign-flipped normal for one
   cycle (~10 prior-SDs from its own `u ~ N(u_hat, 0.2)` prior — should be
   astronomically unlikely) with 0-3 divergences, while the other chain
   converged correctly. **Fix applied**: replaced the arccos-based potential
   with a smooth small-angle-equivalent proxy that matches the same
   `HalfNormal(0.10)` prior in the regime the spec targets (small angles) but
   has no singularity: `HalfNormal(sigma).logpdf(x) = -x^2/(2 sigma^2)`, and
   for small angles `x^2 = arccos(cos)^2 ~= 2*(1-cos)`, so the potential becomes
   `-(1 - |cos_angle|) / sigma^2` — smooth (C-infinity) in `cos_angle`
   everywhere. This is documented inline in `bayesian.py` at the
   `normal_smoothness` Potential.

   This alone did NOT fully fix it, though — divergences persisted. Root cause
   turned out to be **initialization**, not the potential's shape:

3. **Root cause of the sign flip: PyMC's default `init="jitter+adapt_diag"`
   perturbs the informative seeds by too much.** The seeds (`tau_hat`, `c_hat`,
   `u_hat`) are already excellent (typically cos_sim > 0.999 to ground truth
   before any sampling). Default jitter adds up to +/-1 in unconstrained space,
   which relative to `u ~ N(u_hat, 0.2)` is enormous — large enough to
   occasionally jitter a starting point into a degenerate sign-flipped basin,
   from which the chain got stuck (divergences prevented it from correctly
   relaxing back). **Fix applied**: `init="adapt_diag"` (no jitter) plus,
   belt-and-suspenders, explicit `initvals=` pinning every free variable to its
   seed value. Confirmed fix: re-ran with same random seed, 0 divergences, both
   chains agree to 3 decimal places, cos_sim > 0.999 for all 6 cycles.

## Layer 2 findings

1. **Same class of sign-flip bug reappeared, worse, when nutpie was tried.**
   Layer 1's fixes (arccos->smooth potential, `init="adapt_diag"`) don't
   directly carry over to Layer 2, because Layer 2 has NO cross-knot
   smoothness potential on `u2` at all (each spline knot's `u2` value has only
   an independent Gaussian prior from its Layer-1-derived mean/SD; nothing
   couples adjacent knots the way Layer 1 couples adjacent cycles). This is
   fine IF each knot's own prior mean is correctly signed (it should be, since
   it's derived from Layer 1's now-reliable posterior) AND initialization
   doesn't jitter it away.

   First full Layer 2 run (before trying nutpie) actually PASSED completely
   (all checks green) using plain PyMC NUTS with `init="adapt_diag"` (no
   explicit initvals yet) — but took ~423s with 17 divergences + max-treedepth
   warnings in both chains. That run's numbers were scientifically correct
   despite the warnings (normal cos_sim median 0.9998, phase monotonic, radius
   ~1.0, perp deviation small) — the warnings indicate sampling *inefficiency*,
   not incorrect output, in that particular run.

2. **Tried nutpie to fix the slowness** (pymc-modeling skill recommends it;
   installed via `pip install nutpie`, works — 0.16.11). Nutpie's mass-matrix
   adaptation (richer than PyMC's diagonal `adapt_diag`) is the standard fix
   for exactly the kind of curved/correlated posterior this model has (the
   log-velocity spline feeds a cumulative sum that must hit tight per-boundary
   phase targets — a classic "banana" geometry). It compiles through numba, so
   no C++ compiler needed either.

   BUT: switching to nutpie reintroduced the sign-flip bug, this time as a
   **wild spline excursion**: one `u2` knot ended up with flipped sign relative
   to its neighbor, and since natural-cubic-spline interpolation *between* a
   +v knot and a -v knot must pass through near-zero magnitude at some
   intermediate point, normalizing that (`n_t = u_t / ||u_t||`) produced an
   essentially arbitrary, wildly-off direction in a localized time window
   between those two knots (observed: cos_sim to true normal dropped to ~0.40
   for samples 28-37 out of ~600, sitting between spline knots 0 and 1, while
   the rest of the trajectory had cos_sim > 0.999). Diagnosed by finding the
   worst-cos_sim indices and checking which knots straddle that time range
   (`docs/debug/scripts/debug_layer2_normal.py`).

   Cause: nutpie does NOT honor PyMC's `initvals=` (PyMC emits a
   `UserWarning: initvals are currently not passed to nutpie sampler. Use
   init_mean kwarg following nutpie specification instead`). `init_mean` wants
   a raw flattened, transformed-space numpy array in nutpie's own internal
   variable ordering — reconstructing that correctly would mean depending on
   private/undocumented details of how `nutpie.compile_pymc_model` orders
   `CompiledPyMCModel`'s underlying value vars (inspected: no public method
   found that maps a `{var_name: value}` dict to that ordering). Decided this
   was too fragile to depend on for a maintainable library.

   **Decision: dropped nutpie, reverted to plain PyMC NUTS.** Kept
   `init="adapt_diag"` AND added explicit `initvals=` for every free variable
   in both `_fit_layer1` and `_fit_layer2` (belt-and-suspenders — `initvals`
   alone should already be sufficient and is honored by plain PyMC regardless
   of the `init=` strategy).

3. **RE-TEST RESULT (post-crash, post-initvals-fix): still broken, and worse.**
   Fresh run of `docs/debug/scripts/test_layer2.py` with explicit `initvals` + `init=
   "adapt_diag"`, plain PyMC NUTS (no nutpie): Layer 1 was clean (34s, 0
   divergences, no rhat warnings — the Layer 1 fix is solid). Layer 2 took
   **999s** (vs 220-423s in earlier runs) with **84 divergences** (vs 17
   before) and hit max treedepth in both chains. Result: normal recovery
   median cos_sim = 0.999 (excellent) but **min cos_sim = 0.51** — the same
   kind of narrow, localized bad-region artifact as the nutpie run, just from
   plain PyMC this time. So explicit initvals, which fixed Layer 1 completely,
   did *not* fix Layer 2, and the sampler's overall difficulty (divergence
   count, runtime) got worse rather than better.

   **Reassessment of root cause**: initialization is probably not actually the
   (main) problem for Layer 2. More likely explanation: HMC/NUTS updates *all*
   parameters jointly within one leapfrog trajectory. The phase-velocity
   sub-model has a genuinely difficult geometry independent of initialization
   — `g_knots` (a handful of log-velocity spline coefficients) get pushed
   through `exp()`, then a cumulative sum over ~600 time steps, then must land
   within `sigma=0.15 rad` of `2*pi*k` at *every one* of the `K=7` boundary
   times simultaneously (the `phase_boundary` Potential). This is a tight,
   nonlinear, essentially near-equality constraint on a small number of
   parameters — a classic "banana"/funnel shape that a diagonal mass matrix
   (PyMC's `adapt_diag`) handles poorly. When NUTS struggles badly on *that*
   sub-space, divergent/max-treedepth trajectories affect the *entire* joint
   sample for that draw, which can drag otherwise-easy parameters (like a
   `u2` spline knot) into transiently bad values too, and if those
   contaminate the post-warmup samples enough, they bias the posterior mean
   used as the point estimate — producing exactly the kind of localized,
   between-knots artifact observed (near-zero-magnitude spline interpolant at
   one location once you average in some bad draws), even though *most* of
   the trajectory across *most* of the posterior looks fine (hence median
   cos_sim staying excellent).

   Things tried/considered and their status:
   - More/fewer velocity knots: fewer (6 instead of 12) made divergences worse
     (47 vs 17), not better — ruled out as a quick fix.
   - Higher `target_accept` (0.95, 0.99) with fewer knots: also worse (610s,
     47 divergences) — confounded with the knot-count change, not cleanly
     tested in isolation with the *original* 12-knot config, but combined with
     the knot-count result this doesn't look like a promising direction to
     pursue blindly.
   - nutpie (richer, non-diagonal mass matrix — the textbook fix for exactly
     this "banana" geometry): works, but its `initvals` incompatibility
     reintroduces sign instability (see finding above) and fixing that
     properly would mean depending on nutpie's undocumented internal variable
     ordering for `init_mean` — not something to do without more research or
     nutpie-side documentation/support.
   - Explicit initvals under plain PyMC (this section): tried, did not help,
     made it slightly worse.
   - NOT yet tried: raising `target_accept` alone (e.g. 0.95-0.99) on the
     *original* 12-knot config without changing knot count; explicitly raising
     `nuts={"max_treedepth": ...}`; a non-centered-style reparameterization of
     the phase-velocity sub-model; hard-constraining phase at the boundaries
     analytically instead of via a soft Potential; running with more chains
     (4+) so bad chains are visible/discardable via r_hat; testing on a
     machine with a C++ compiler and multiple cores (each iteration here costs
     15-20 minutes, which makes this kind of iterative sampler-tuning very
     expensive to do blind).

4. **Reparameterization result (2026-07-06, log 08): primary issues resolved.**
   Applied all 6 changes described in `docs/claude_layer2_reparameterization_prompt.md`:
   tangent-plane normal deviations, smoothness potential, sigma_a2 floor, resultant-length
   diagnostic, and boundary-normalized positive phase. Run with draws=400, tune=400,
   chains=2, target_accept=0.9, random_seed=0.

   **Primary issues FIXED:**
   - Divergences: **0** (was 84 in the worst previous run)
   - Normal artifact: **gone** (min cos_sim 0.9915, median 0.9978 — was 0.51 / 0.999)
   - Phase monotone by construction; all test assertions passed.
   - Runtime: 418s (vs 999s worst, comparable to 423s best prior case).

   **Remaining issue — amplitude convergence:**
   Chain 1 hit max_treedepth; rhat > 1.01 for some parameters; ESS < 100 for some
   parameters. Radius posterior median = 0.709 (expected ~1.0), sigma_x_mean = 0.467
   (expected ~0.02 given noise scale 0.02, R_X ≈ 1.0). These amplitude parameters
   have not converged yet. The radius barely clears the test's 0.7 floor; the
   sigma_x value being off by 20x suggests the amplitude model is trading radius
   against noise in a way the sampler hasn't sorted out yet.

   Likely cause: the mass matrix hasn't had enough tune steps to adapt to the
   amplitude subspace. Phase is now well-identified (by construction), so the
   remaining difficulty is between radius/perp-deviation/sigma_x parameters, which
   tend to be correlated and need a non-diagonal mass matrix approximation to
   sample efficiently. Tried: tune=1000 → did NOT fix (see finding 5 below).

5. **Amplitude convergence root cause: noise prior too wide (log 10, 2026-07-06).**
   `test_layer2_tune1000.py` with tune=1000 gave: **40 divergences, radius 1.251**
   (opposite direction of error from log 08's 0.709). The error flip between runs
   indicates the chain is NOT sampling from the posterior — it's exploring wildly
   different wrong regions depending on the warmup path.

   Root cause analysis: `_OBS_NOISE_LOGNORMAL_SD = 0.5` gives rho_x a 95% CI of
   [0.011, 0.082] (as rho = sigma_x/R_X). During NUTS warmup, a wrong initial step
   can place rho_x at ~0.47. At that point, with wrong radius (say 0.71), the
   likelihood STRONGLY prefers high sigma_x: absorbing a 0.29-per-step residual
   into sigma_x=0.47 is ~5800 nats better than sigma_x=0.02 over 569 steps. This
   benefit vastly exceeds the prior penalty at rho_x=0.47 (only -15 nats for
   sigma=0.5). So the chain gets stuck in the (wrong r, wrong sigma_x) region and
   adapt_diag calibrates there, trapping subsequent draws.

   **Fix attempted: `_OBS_NOISE_LOGNORMAL_SD = 0.3`** — see finding 7 for results.

6. **Runtime concern**: a full Layer 2 fit at draws=400/tune=400/chains=2 takes
   ~418s (fastest successful run). For the pytest suite this is slow — a separate
   "short" test configuration (fewer cycles, fewer draws) may be needed.

7. **sigma=0.3 caused 400–664 divergences (2026-07-06, logs 09-11).**
   Two experiments with `_OBS_NOISE_LOGNORMAL_SD = 0.3`:
   - ta=0.9, tune=400: **400 divergences**, radius ~0.5-0.7 (various wrong modes)
   - ta=0.95, tune=400: **664 divergences** (worse, not better)
   - ta=0.99, tune=1000, sigma=0.5: **30 divergences, radius 1.739** — a completely
     different wrong mode; tiny steps from high ta allow the chain to slowly crawl
     up the amplitude-noise ridge during 1000 warmup steps.

   Why sigma=0.3 causes divergences: the tighter prior creates steeper gradient
   walls in rho_x space. The chain's adapted step size (calibrated to the wrong
   mode's geometry) generates leapfrog trajectories that crash into these steep
   walls, causing divergences. The prior gradient is 2.75× steeper than sigma=0.5,
   and the chain was already in the wrong region (radius far from 1.0).

   **Decision: reverted `_OBS_NOISE_LOGNORMAL_SD` back to 0.5.** Tightening the
   noise prior is not the right fix — it either causes divergences (from the wrong
   region) or forces the chain to a different wrong mode. The root problem is that
   the chain reaches the wrong region in the first place.

8. **True root cause: a2_init mismatch causes 19-sigma initval residuals (2026-07-06).**
   Diagnosed via MAP analysis and logp evaluation. Layer 1's `a_mean_p` (per-cycle
   mean boundary direction) deviates ~21.6° from the actual data direction at tau_k.
   This is because Layer 1 estimates the CYCLE-AVERAGE boundary direction, not the
   instantaneous direction at the boundary time itself.

   Effect: at initvals (a2=a_mean_p, rho_x=0.03), predicted trajectory at tau_k is
   ~0.376 units away from observed data while sigma_x ≈ 0.02 — a ~19-sigma residual.
   L-BFGS-B MAP optimization from BOTH the correct and wrong starting points found
   the same wrong mode (radius=0.840, rho_x=0.411) with ||grad||=690 (not converged).

   **Fix applied:** compute `a2_init[k] = X_fit[idx_k] - c_mean_p[k]`, the vector
   from estimated center to actual data at the boundary time. This gives near-zero
   residuals at initvals (residuals ≈ noise level). Both `_fit_layer2` initvals
   blocks updated to use `a2_init` instead of `a_mean_p`.

   Also added debug parameters `_sigma_x_override` and `_init_override` to
   `_fit_layer2` signature for diagnostic experiments.

9. **Post-a2_init-fix experiments (2026-07-06).**
   Multiple experiments after implementing the a2_init fix:

   - **sigma=0.5 + a2_init (test_layer2.py)**: 51 divergences, radius 1.148,
     sigma_x 0.341. Test PASSES. Progress from 0.709→1.148 but still far from 1.0.
   - **sigma=0.3 + a2_init**: 0 divergences, but radius 0.570, sigma_x 0.513.
     Different (worse) wrong mode. Tighter prior doesn't help — chain still slides
     along amplitude-noise ridge during warmup, settling at a different bad point.
   - **h_r_knots sigma=0.1 + a2_init**: 390 divergences, radius 1.355. Much tighter
     radius prior creates steep gradient walls → divergences.
   - **Layer 1 radius uncertainty**: sigma_r_k ≈ 0.17-0.21 per cycle (high), giving
     a data-informed sigma_logr ≈ 0.25 — barely tighter than the current 0.3. Layer
     1's boundary direction estimates are noisy at the per-cycle level.
   - **Fixed sigma_x = R_X*0.03**: **0 divergences, radius 0.961** — closest to
     true 1.0 seen yet. Max_treedepth in both chains and somewhat poor normal
     recovery, but PROVES trajectory params converge near-correctly when the
     amplitude-noise ridge is eliminated.
   - **adapt_full + sigma=0.5, tune=400**: 275 divergences. PyMC's adapt_full is
     experimental and needs ~10× tune steps for reliable 93-parameter covariance
     estimation.
   - **advi+adapt_diag**: 470 divergences. ADVI's mean-field approximation doesn't
     capture parameter correlations; the resulting initial mass matrix is worse than
     identity for this geometry.
   - **Fixed sigma_x + adapt_full, tune=1000**: 48 divergences, radius 0.902.
     adapt_full still struggles.

   **Key insight from fixed-sigma_x result**: the amplitude-noise ridge (`r × sigma_x
   ≈ const`) is the sole obstacle. Once sigma_x is held fixed (ridge broken), radius
   converges to 0.961 with 0 divergences. The problem is purely about sampling sigma_x
   jointly with r.

10. **Current approach: two-phase warmup (2026-07-06, in progress).**
    Phase 1: run `_fit_layer2` with `_sigma_x_override = R_X * 0.03` (breaks the
    amplitude-noise ridge). Phase 2: extract trajectory posterior means from Phase 1,
    use as initvals for full model with free sigma_x. Theory: Phase 1 finds the correct
    trajectory neighborhood; Phase 2's warmup starts from a good trajectory point so
    the mass matrix adapts to the true posterior geometry rather than the ridge.

    A `test_layer2_twophase.py` script was written and launched but was killed (OOM,
    exit 137) before completing. Needs re-run or memory investigation.

11. **a2 prior-mean hypothesis tested: marginal improvement, ridge persists (2026-07-06).**
    Hypothesis: using `a_mean_p` as the Layer 2 prior mean for `a2` (not just the
    initval) creates a persistent pull on the frame direction throughout sampling, not
    just at initialization. Because `a(t)` defines `e1(t)` (the direction in which
    radius acts), a bad `a2` prior mean could mis-orient the cyclic term and cause the
    model to compensate with a wrong radius + inflated sigma_x — i.e., produce an
    apparent amplitude-noise ridge even if the deeper problem is a bad frame prior.

    **Change applied:** `a2 ~ Normal(mu=a2_init, sigma=sigma_a2)` instead of
    `a2 ~ Normal(mu=a_mean_p, sigma=sigma_a2)`. Both initval and prior mean now use
    the data-at-tau_k direction.

    **Result** (`test_layer2.py`, 2026-07-06, 417s):
    - Divergences: **40** (was 51 — slight improvement)
    - Radius median: **1.123** (was 1.148 — slight improvement)
    - sigma_x_mean: **0.362** (was 0.341 — essentially unchanged)
    - normal cos_sim: min 0.989, median 0.997 (comparable)
    - Phase monotonic: yes
    - Perp deviation median abs: 0.043 (was 0.062 — improvement)
    - All assertions: PASS

    **Conclusion:** the a2 prior-mean hypothesis is not the main driver. The amplitude-
    noise ridge remains essentially intact. sigma_x is still ~18x the true value and
    radius is still ~12% above ground truth. The change is kept (it is theoretically
    correct — a2_init is a better prior mean than a_mean_p), but it does not resolve
    the convergence problem.

    Recommended next diagnostic: four-condition comparison to isolate contributions:
    - A: free sigma_x, prior mean a_mean_p (baseline before all recent fixes)
    - B: fixed sigma_x, prior mean a_mean_p
    - C: free sigma_x, prior mean a2_init (current state)
    - D: fixed sigma_x, prior mean a2_init
    Or: a log-likelihood slice over (radius, sigma_x) with all geometry fixed, to
    confirm the ridge shape and magnitude directly.

12. **Amplitude diagnostics confirm warmup failure; trajectory direction is
    also wrong (2026-07-06, log 11).**
    `docs/debug/scripts/diagnose_layer2_amplitude.py` ran 7 diagnostics on the
    current model (draws=400/tune=400/chains=2/seed=0). Key numbers:

    **Free-sigma fit** (442s, 40 divergences — same as previous):
    - sigma_x_mean = 0.362 (18×true); r_median = 1.123; r×sigma = 0.406 (vs true 0.020)
    - Pearson corr(sigma_x, r_median) = **-0.993** — ridge confirmed, extremely tight
    - r×sigma range [0.249, 0.363], SD/mean = 0.078 — the two dimensions trade off
      almost perfectly while the product stays constant

    **Chains stuck from draw 1 (Diagnostic 5):**
    - Chain 0: sigma_x ∈ [0.503, 0.542], r ∈ [0.505, 0.654] in first AND last 50 draws
    - Chain 1: sigma_x ∈ [0.202, 0.218], r ∈ [1.496, 1.618] in first AND last 50 draws
    - **Zero** draws from either chain have sigma_x < 4×true = 0.08
    - The two chains landed on completely different points on the same ridge during
      warmup and never moved. No exploration across the ridge or toward the true mode.

    **True mode is far better but never found (Diagnostic 7):**
    - log-lik at true params: +4258 nats
    - log-lik at posterior mean: -505 nats — gap of **4763 nats**
    - The wrong mode is not a nearby local optimum; it is qualitatively worse
    - Ridge scan (all points at r×sigma ≈ 0.406): log-lik ranges from -2224 to -75000
      — the entire ridge is far below the true mode. NUTS is not in a competing local
      posterior mode; it is simply far from the posterior.

    **Trajectory direction also wrong (Diagnostics 2, 6):**
    - MLE sigma for posterior-mean trajectory = 0.321 (true 0.02): the
      posterior-mean trajectory does not fit the data at the true noise level —
      confirming the trajectory itself (not just scale) is off
    - OLS radius from posterior trajectory direction = 1.22 (true 1.0), SD = 0.73
    - RMS cyclic residual = 0.42, RMS tangential = 0.36 (should both be ~0.02)

    **Fixed-sigma fit (seed=42): 458 divergences, cos_sim median 0.58 — seed-specific
    failure, but informative.** This contradicts finding 9 (seed=0, fixed sigma_x →
    8 div, radius 0.961, all checks passed), confirming the fixed-sigma model is
    initialization-sensitive. Two things this run tells us beyond "seed=42 is bad":
    (a) Even with 458 divergences and cos_sim 0.58, the fixed-sigma posterior mean
    achieves log-lik +2462 vs the free-sigma run's -505 — the correct amplitude
    matters more than the normal being somewhat off. (b) The 458-divergence pattern
    (tight sigma + bad warmup → steep gradient walls → cascade of divergences) is
    exactly the mechanism identified in finding 7 to explain why sigma_lognormal_SD=0.3
    caused 400–664 divergences. The fixed-sigma run is the limiting case of "tight
    sigma prior", confirming that interpretation.

    **Cross-run amplitude instability (logs 08, 10, 11 compared):**
    Across all free-sigma runs with similar configuration, the amplitude result is
    different every time — landing on different points on the ridge depending on the
    warmup path:

    | Log | tune | div | r_median | sigma_x | r×sigma |
    |-----|------|-----|----------|---------|---------|
    | 08  | 400  | 0   | 0.709    | 0.467   | 0.331   |
    | 10  | 1000 | 40  | 1.251    | ~0.28 est. | ~0.35 |
    | 11  | 400  | 40  | 1.123    | 0.362   | 0.406   |

    The r×sigma product varies (0.33–0.41) — these are not the same wrong mode. The
    chains reach different ridge points on each run. Notably, log 08 (0 divergences —
    the "best" run by NUTS health metrics) gives the worst amplitude (r=0.709). Zero
    divergences means the sampler adapted cleanly to the geometry of whatever mode it
    found, not that it found the right mode. More warmup (log 10, tune=1000) moved the
    amplitude further from truth (r=1.251), not closer — more adapt_diag iterations
    deepen the commitment to whatever wrong ridge point was found early in warmup.

    **Conclusion:** adapt_diag warmup drives chains to the ridge and the adapted mass
    matrix locks them in. The chains start near truth (r≈1, sigma_x≈0.03) but warmup
    carries them to the ridge; subsequent draws can't escape. The amplitude failure is
    run-to-run unstable, independent of divergence count, and worsens with more tune
    steps — all signatures of mass-matrix-reinforced warmup drift. The fixed-sigma
    result (finding 9, seed=0, r=0.961, 8 div) proves the correct mode is reachable
    when the ridge is removed. The fix must target warmup initialization.

    **Recommended next fix:** run a fixed-sigma fit (seed=0, which reliably converges)
    and use its trajectory posterior means as initvals for the full free-sigma model.
    Alternatively, use `pm.find_MAP()` to locate the MAP before sampling. Either
    approach prevents warmup drift by starting at a point already near the true mode,
    so the mass matrix adapts to the correct geometry. The fixed-sigma phase must use
    seed=0 (or check for convergence and retry) since fixed-sigma is itself
    initialization-sensitive.

## Design decisions / deviations from the literal spec (for the final report)

These are deliberate, reasoned engineering choices, not workarounds forced by
something being "unworkable" — flagging them here so they all make it into the
final deliverables write-up's "deviations" section.

1. **Layer 2 spline knot locations are fixed at the Layer 1 posterior-mean
   boundary times**, rather than re-drawn as Layer 2 free parameters with
   their own (padded) prior, as the literal spec implies (`tau_k^{(2)} ~
   N(...)` feeding into `CubicSpline(tau_k^{(2)}, ...)`. Reason: with fixed
   knot locations, a natural cubic spline is a *linear* function of its knot
   *values*, so it can be implemented as a plain (differentiable) matrix
   multiply by precomputing the interpolation matrix once with scipy. With
   resampled knot *locations*, the spline coefficients would need to be
   re-derived at every NUTS step via a differentiable tridiagonal solve —
   substantially more complex, and standard PyMC spline practice (see the
   pymc-extras skill's `references/splines.md`) uses fixed-knot basis
   matrices for exactly this reason. Layer 1's own boundary-time uncertainty
   (`tau_sd`) is still computed and reported; it just isn't re-propagated into
   the Layer 2 spline geometry. This is the single most significant modeling
   simplification in the implementation.

2. **Padded knot arrays**: Layer 1 produces per-*cycle* summaries (`K-1` of
   them: `c_k`, `u_k`, `a_k`) but the spec's spline formula pairs them with
   `K` boundary times `tau_k`. Resolved by padding: cycle `k`'s value is used
   as the knot value at boundary `k` (its start), and the final boundary
   reuses the last cycle's value (`np.vstack([v, v[-1:]])`). Reasonable
   resolution of an indexing ambiguity, not a modeling change.

3. **`arccos`-based angular smoothness potentials replaced with a smooth
   small-angle-equivalent proxy** (see Layer 1 finding #2 above) for numerical
   stability. Mathematically equivalent to the literal spec in the small-angle
   regime the spec's own `sigma=0.10 rad` targets; diverges from the literal
   formula only in behavior extremely far from that regime (which the model
   should essentially never visit anyway, by construction of the same prior).

4. **`sigma_a2` floor added** — spec gives explicit uncertainty floors for `tau`,
   `c`, `u` in Layer 2's `1.5x`-padding rule but not for `a`. Originally
   implemented as plain `1.5 * sd` with no floor; a `0.02 * R_X` floor has since
   been added (this was the missing deviation flagged in the reparameterization
   plan — now fixed).

5. **"Changing planes" test data uses a gentler rotation rate (~8 deg/cycle)
   than the existing deterministic estimator's stress-test fixture** (which
   uses 30 deg/cycle). Reason: the coarse model's cycle-normal smoothness
   prior is `HalfNormal(0.10 rad)` (~5.7 deg SD) — a real, spec-mandated prior
   that assumes gradually-changing planes. 30 deg/cycle is a ~5-sigma conflict
   with that prior and is a reasonable stress test for the *deterministic*,
   assumption-free estimator, but is not a fair test of a model whose entire
   point is Bayesian pooling across cycles under a smoothness assumption. At
   30 deg/cycle the Bayesian model's coarse normals get pulled toward a
   compromise/blend across cycles (as they should, given the prior), which
   isn't a bug. At 8 deg/cycle (within ~1.4 prior SDs) recovery is good
   (cos_sim > 0.97 for all cycles) — confirmed in `docs/debug/scripts/test_layer1_changing.py`.

## Required tests (from the prompt) — status

1. Existing API still imports and current deterministic tests still pass — not
   yet re-verified after all bayesian.py changes (should still be fine, core.py
   untouched, but must confirm).
2. New Bayesian API imports or gives a clear optional-dependency error — not
   yet written as a pytest test (informally confirmed bayesian.py imports fine
   without pymc since imports are lazy — see `_import_pymc()` etc. — but no
   formal test yet, and the ImportError message itself not yet exercised).
3. Frame construction test (e1/e2/n unit vectors, mutually perpendicular,
   e2 = cross(n,e1)) — validated informally in `docs/debug/scripts/test_utils.py` and
   `docs/debug/scripts/test_layer2.py`, not yet a formal pytest test.
4. Synthetic fixed-plane trajectory (normal recovery, phase monotone, radius
   positive, perp deviation small) — exercised repeatedly via
   `docs/debug/scripts/test_layer2.py`, but **blocked on the open Layer 2
   convergence issue** (median recovery is good, but the min-cos_sim
   assertion is not reliably satisfied — see "OPEN ISSUE" above). Not yet a
   formal pytest test; writing one should wait until the convergence issue is
   resolved (or a decision is made on how to handle it in tests, e.g. loose
   tolerances plus a documented flaky-region caveat).
5. Synthetic cycle-level changing planes (coarse normals recover known
   per-cycle normals up to sign) — validated informally in
   `docs/debug/scripts/test_layer1_changing.py` at a gentler rotation rate (see deviation
   #5 above), not yet a formal pytest test.
6. Boundary convention test (phase-zero points cluster near known synthetic
   boundary event) — NOT yet checked at all, even informally.
7. Diagnostic test (a(t) nearly parallel to n(t) triggers warning/failure) —
   NOT yet checked at all, even informally. Should be straightforward: the
   `_compute_diagnostics` projection-ratio check should catch this given a
   constructed `layer2`-like summary, or could be tested by feeding synthetic
   data where the boundary event sits very close to the plane's normal
   direction from the center.

None of the 7 are yet formal pytest tests in `tests/`. That's the next major
chunk of work once the Layer 2 convergence issue (see "OPEN ISSUE" above) is
resolved.
