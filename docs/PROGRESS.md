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

## Status as of last update

**Handed off for reparameterization debugging.** All of Layer 1, Layer 2,
diagnostics, and the public API are implemented in
`phase_coordinates/bayesian.py` (~1130 lines). Utility functions and Layer 1
are fully validated with passing standalone checks. Layer 2 is implemented
and structurally correct (frame construction, splines, likelihood all
verified right when sampling behaves), but NUTS sampling for Layer 2 has an
unresolved convergence problem: a high divergence rate (10-20%) and, correlated
with it, a localized sign/spline-excursion artifact in the recovered
instantaneous normal in one narrow time window. Two rounds of fixes (see
"Layer 2 findings" below) did not resolve it — the second attempt (explicit
`initvals`) made it *worse* (84 divergences, ~1000s runtime, same artifact).

**All raw diagnostic scripts and run logs behind these findings are committed
under `docs/debug/` (see `docs/debug/README.md` for an index) — start there
for concrete numbers, a root-cause hypothesis, and a prioritized list of
untried reparameterization ideas.** The user has asked for a fresh
agent/session to focus specifically on debugging the model's
*parameterization* (not the fallback options below, at least not yet). Each
full test iteration costs ~15-20 minutes in this environment (no C++
compiler, single-core sampling — see "Environment" below for real timing
numbers and how to reproduce), so prioritize the cheap isolation experiment
in `docs/debug/README.md` idea #1 before broad parameter sweeps.

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

- [ ] **OPEN ISSUE, being handed off for debugging** — Layer 2 NUTS sampling
      has an unresolved convergence problem (high divergence rate, localized
      sign/spline-excursion artifact). See `docs/debug/README.md` for the
      full evidence, current best hypothesis, and a prioritized list of
      untried ideas. Everything below this is blocked on resolving it (or on
      a decision to fall back to one of the other options if reparameterization
      doesn't pan out — see the numbered options preserved below).
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

- Default `python` on PATH (3.14) has **no** numpy/pandas/sklearn/scipy installed,
  let alone pymc/arviz. It cannot even run the existing deterministic test suite.
  Do not use it for anything in this task except the "no optional deps" import/
  error-message tests, and even then, it would need to at least run
  `phase_coordinates.bayesian` in isolation without importing the rest of the
  package's numpy-dependent bits... in practice this project's real dev env is:
- **`/c/Users/User/miniforge3/python.exe`** (Python 3.12.12, conda env) has numpy,
  pandas, scikit-learn, scipy, pytest, pymc 5.28.0, arviz 0.23.0, numba 0.62.1
  all installed. **Use this interpreter for everything** (tests, scratch scripts).
  Package is editable-installed there (`pip install -e ".[dev]" --no-deps`).
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

4. **Runtime concern (independent of the divergence issue)**: even in the
   "good" run (17 divergences, scientifically correct output), a full Layer 2
   fit at draws=400/tune=400/chains=2 took ~220-423s; the "bad" run took 999s.
   For the pytest suite this is likely too slow to run more than once or
   twice even if the divergence issue is resolved.

5. **The model's own posterior-SD uncertainty does NOT flag the bad region.**
   In the artifact window (t=0.58-0.67s in the nutpie run), `normal_angular_sd`
   — the posterior SD of the normal direction, which is exactly the quantity
   the diagnostics module reports to users as an uncertainty estimate — was
   0.060-0.073 rad, **not elevated** relative to typical healthy values seen
   elsewhere (~0.06-0.07 rad in Layer 1). So this isn't a case where the model
   "knows" it's uncertain there and a user could catch it from the reported
   uncertainty; the point estimate is confidently wrong. This is important
   for whoever debugs the parameterization: it suggests most individual
   posterior draws agree on a stable-but-wrong compromise in that window,
   rather than the window having genuine high posterior variance/bimodality.
   See `docs/debug/README.md`'s "current best hypothesis" section for the
   leading theory (contamination from the phase-velocity sub-model's
   difficult geometry within shared HMC trajectories) and full detail
   (`docs/debug/logs/04_layer2_worst_cossim_location_debug.log` has the raw
   numbers).

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

4. **No explicit floor for `sigma_a2` (Layer 2 boundary-direction prior SD)**:
   spec gives explicit uncertainty floors for `tau`, `c`, `u` in Layer 2's
   `1.5x`-padding rule but not for `a`. Implemented as plain `1.5 * sd` with no
   floor, per literal reading (no floor specified = no floor applied).

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
