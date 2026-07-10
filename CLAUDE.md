# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`phase-coordinates` is an experimental Python package that describes cyclic multivariate movement (e.g. 3-D marker trajectories) in terms of phase, in-plane radius/angle, and perpendicular deviation from a local plane. It exposes **two peer algorithms** behind an identical output contract:

- `fit_pca_phase_coordinates` (`phase_coordinates/core.py`) — fast, fits an independent PCA plane per cycle.
- `fit_bayesian_phase_coordinates` (`phase_coordinates/bayesian.py`) — slow, MCMC-based two-layer model with posterior uncertainty; 3-D only.

Both return `(samples, cycles, details)`. `samples`/`cycles` always have the exact column sets `SAMPLE_COLUMNS`/`CYCLE_COLUMNS` (defined in `core.py`); `details` is algorithm-specific (`details["algorithm"]` is `"pca"` or `"bayesian"`). `reconstruct_phase_coordinates(samples, cycles)` reconstructs an `(n_time, 3)` trajectory from either algorithm's output using the same geometry (`center + u*e1 + v*e2 + perp*normal`). Preserving this shared schema across both algorithms is the core design constraint of the package — when changing either algorithm, keep `SAMPLE_COLUMNS`/`CYCLE_COLUMNS` and the sign/units conventions in sync.

Read `README.md` for the full user-facing API reference, parameter semantics, and known limitations of each algorithm before changing behavior.

## Environment and commands

The dependency environment is managed with **pixi**, resolved into `.pixi/envs/default`. There is no `pixi.toml` committed (only `pixi.lock`, currently untracked) — use the env's binaries directly rather than `pixi run`:

```bash
# Run full test suite (excludes slow/MCMC tests)
.pixi/envs/default/bin/pytest tests/ -q -m "not slow"

# Run everything including the slow Bayesian smoke test (~10-60s MCMC run)
.pixi/envs/default/bin/pytest tests/ -q

# Run a single test
.pixi/envs/default/bin/pytest tests/test_phase_coordinates.py::TestFitPcaPhaseCoordinates::test_name -q

# Quick import sanity check
.pixi/envs/default/bin/python -c "from phase_coordinates import fit_pca_phase_coordinates, fit_bayesian_phase_coordinates, reconstruct_phase_coordinates; print('OK')"
```

The system/bare `python` and `python3` do **not** have numpy/pandas/etc. installed — always use `.pixi/envs/default/bin/python` (or `pytest`) for anything that imports this package.

`pytest.mark.slow` marks tests that run real MCMC sampling (PyMC/ArviZ); default CI-style runs should pass `-m "not slow"`. The Bayesian dependencies (`pymc`, `arviz`) are optional (`pip install -e ".[bayes]"` outside pixi) and imported lazily inside `bayesian.py`, so importing `phase_coordinates` itself must never require them — `TestPublicContract::test_bayesian_import_without_pymc` and `test_bayesian_clear_error_without_deps` guard this.

## Architecture

### `core.py` — PCA algorithm
For each cycle (defined by 2π windows of the input `phase`), fits an independent 3-component PCA to that cycle's data. PC1/PC2 span the in-plane axes (`e1`/`e2`), PC3 is the plane normal; per-sample `u`,`v` are PC1/PC2 scores, `radius`/`theta` are their polar form, `perp` is the PC3 score. Consecutive cycles' PCA axes are sign/orientation-aligned to the previous cycle's axes (dot-product sign flip, then a right-handedness check via cross product) — PCA components are otherwise sign-arbitrary and would flip randomly cycle to cycle. Phase can be supplied directly or estimated via `hilbert_phase` (bandpass + Hilbert transform, with a warning if the unwrapped phase looks unreliable — see `_PHASE_JUMP_THRESHOLD`/`_NON_MONOTONIC_THRESHOLD`).

### `bayesian.py` — Bayesian two-layer algorithm
Independent of `core.py` (does not call it). Two-stage PyMC model:
1. **Layer 1** (`_fit_layer1`) — coarse per-cycle model: boundary times `tau`, cycle centers, cycle normals, with posterior uncertainty. Seeded deterministically first (`dominant_reference_signal` → `estimate_dominant_period` → `seed_boundary_indices` → `seed_cycle_centers`) via the top PCA-score periodogram peak, then refined with MCMC.
2. **Layer 2** (`_fit_layer2`) — instantaneous model using Layer 1 posterior summaries as priors for smoothly varying phase, center, normal, radius, and perpendicular deviation across the whole time series; uses cubic-spline/linear interpolation matrices (`cubic_spline_matrix`, `_linear_interp_matrix`) for smooth per-sample latents.

`_compute_diagnostics` assembles convergence/quality flags (`BayesianPhaseDiagnostics`) such as boundary multimodality, phase monotonicity, and normal-vector prior dominance — surfaced in the public `details["diagnostics"]`. Internal dataclasses (`BayesianPhaseEstimates`, `BayesianPhaseUncertainty`, `BayesianPhaseDiagnostics`, `BayesianPhaseResult`) mirror a design spec (`docs/bayesian_two_layer_spec.md`, referenced in the module docstring but not present in this repo) describing the model math and priors in more detail. `fit_bayesian_phase_coordinates` (the public entry point) wires layer1/layer2 output back onto the shared `SAMPLE_COLUMNS`/`CYCLE_COLUMNS` schema; only samples within the fitted boundary window are populated, everything else is NaN. There is also a `_fit_bayesian_phase_coordinates_legacy` function retained near the end of the file — check whether new work should touch the current or legacy path.

PyMC/PyTensor/ArviZ imports are wrapped in `_import_pymc()`/`_import_pytensor_tensor()`/`_import_arviz()` helpers that raise a clear `ImportError` with install instructions rather than a bare `ModuleNotFoundError`. `_numba_available()` is checked to opt into numba-accelerated sampling when present.

### Shared reconstruction
`reconstruct_phase_coordinates` in `core.py` is algorithm-agnostic: it left-merges `samples` onto `cycles` on the `cycle` key and reconstructs `X_hat = center + u*e1 + v*e2 + perp*normal`, leaving NaN rows where `fit_ok` is false or `u`/`v`/`perp` are missing.

## Notebooks
`notebooks/pca_phase_coordinates_demo.ipynb` and `notebooks/bayesian_phase_coordinates_demo.ipynb` demonstrate each algorithm end to end; keep them in sync with any public API changes.

## Working efficiently in this repo

**Commit and push after each round.** When doing a round of fixes (e.g. resolving review feedback), commit and push to the current branch's remote as soon as the round is done, without waiting for separate confirmation. Still: one commit per round (don't amend), descriptive message, no force-push.

**Invoke pixi directly, never through `pixi run`.** `.pixi/envs/default/bin/python` / `.../bin/pytest` are the real, already-resolved binaries — call them directly. `pixi run pytest ...` and `.pixi/envs/default/bin/pytest ...` run the same thing, but only the direct-path form matches this repo's `Bash(.../pytest *)` allowlist entry (`.claude/settings.json`); the `pixi run` wrapper is a different leading command and will prompt for approval every time even though the underlying command is already trusted.

**Prefer real pytest tests over ad hoc `python -c` snippets.** A one-off `python -c "..."` is unique text each time and can never be pre-approved, even when the check is trivial (shape asserts, import checks, printing a value) — every single one prompts. Write the check as an actual test (or extend an existing one) in `tests/` and run it through the pre-approved pytest binary instead. For genuine one-off numeric exploration that doesn't belong in the test suite, write it to a script under `docs/debug/scripts/` with the Write tool (no prompt) and execute it once via the direct pixi python binary — this turns N prompts into 1 and leaves a reviewable artifact, instead of many small inline snippets. Prefer Read/Grep over spinning up Python at all for structural questions ("does this function exist", "what's this signature").
