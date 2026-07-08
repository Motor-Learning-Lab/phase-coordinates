# Code review: `bayesian-robust-cycle-identification`

**Branch:** `bayesian-robust-cycle-identification`
**Base:** `public-two-algorithm-api` tip (`38eedb2`)
**Date:** 2026-07-08
**Scope:** Architecture review before proceeding to robust Bayesian duration modeling

---

## Summary judgment

The branch is directionally correct and the new module boundaries are clean. It is
close to being ready to build on, but has four issues that should be fixed first:
the legacy wrapper in `bayesian.py` is dead code that will confuse future work;
`CycleEpochs` has no invariant validation; the PCA estimator silently drops cycles
from the cycles table rather than representing them as `fit_ok=False`; and the
Bayesian fitter always computes `T0` internally even when `seed_epochs` is supplied
externally. None of these are correctness bugs in the current test suite, but each
is the kind of thing that causes confusing failures when testing harder cases.

---

## Concern-by-concern evaluation

### 1. Bayesian fitter hides `T0` when `seed_epochs` is provided

**Partially valid. Severity: Important.**

The concern as stated (internal seed path when `seed_epochs=None`) is the expected,
acceptable behavior — the docstring is clear about it and the explicit pipeline is
available. The subtler problem is real: `fit_bayesian_phase_coordinates` always runs
`ref = dominant_reference_signal(X_arr)` and `T0 = estimate_dominant_period(ref, fs)`
regardless of whether `seed_epochs` is provided, and that `T0` is passed to
`_fit_layer1` where it controls the boundary-timing prior sigma
(`_BOUNDARY_TIMING_SD_FRAC * T0`) and the log-duration prior mean. If you pass
geometric-score epochs with a period that differs from the periodogram estimate, the
Layer 1 prior is still keyed to the internally-computed period. This is a hidden
coupling.

**Recommended direction:** Make `T0` an explicit parameter with `None` as default.
When `None`, compute it internally as now. When provided, use it directly. This
allows the full seed path to be external without changing the default experience.

---

### 2. Legacy Bayesian wrapper remains

**Valid. Severity: Minor/Important.**

`_fit_bayesian_phase_coordinates_legacy` (line 1294 of `bayesian.py`) is dead: it is
not called from tests, notebooks, or other modules. `BayesianPhaseResult` and
`BayesianPhaseEstimates` dataclasses exist only to serve it. The function duplicates
essentially all of `fit_bayesian_phase_coordinates`. It should be removed before new
modeling work is added — otherwise those old types will get entangled in new code.

**Recommended direction:** Delete all three (`_fit_bayesian_phase_coordinates_legacy`,
`BayesianPhaseResult`, `BayesianPhaseEstimates`). If any caller surfaces in future,
the answer is `fit_bayesian_phase_coordinates(X, ..., return_report=True)`.

---

### 3. `CycleEpochs` has no invariant validation

**Valid. Severity: Important.**

There is no `__post_init__`. The docstring documents invariants (`tau` strictly
increasing, `duration == diff(tau)`, matching array lengths) but they are not
enforced. Since `CycleEpochs` is the contract between all pipeline stages, silent
invariant violations will cause confusing failures deep downstream rather than at the
construction site. The properties `sample_start` and `sample_stop` already assume
these invariants and will silently return wrong values if violated.

**Recommended direction:** Add `__post_init__` that validates: `tau` is 1-D with at
least 2 elements; strictly increasing; `duration == diff(tau)` within tolerance;
`time` has the same length as `cycle_index`; if `phase` is not `None` it has the same
length as `time`. Raise `ValueError` with a clear message for each.

---

### 4. Endpoint convention

**Partially valid. Severity: Minor.**

The half-open convention `[tau[k], tau[k+1])` is internally consistent across
`epochs_from_boundary_indices`, `candidate_epochs_from_period_offset`, and the
`cycle_index` assignment throughout. The subtle issue: a sample at exactly `tau[-1]`
gets `cycle_index = -1`. For a recording where the last cycle ends exactly at the
last sample (`tau[-1] = (n_time-1)/fs`), that last sample is excluded. This is a
one-sample edge effect, harmless in practice, but the convention is not stated in the
`CycleEpochs` docstring.

**Recommended direction:** Document the half-open convention explicitly in
`CycleEpochs`. Verify the `+ 1e-12` tolerance in `edge_valid` (diagnostics line 170)
is correct. No code change required beyond clarifying comments.

---

### 5. Interpolation silently clamps out-of-range queries

**Valid. Severity: Important.**

`interp_X_at_times` documents clamping explicitly, but documented is not the same as
safe. This is the behavior that contributed to endpoint confusion in earlier Bayesian
debugging. The scoring and diagnostics functions call `interp_X_at_times` for anchor
positions at `tau[:-1]` and `tau[:-1] + 0.25*duration`. If any of those times fall
outside the signal window, the anchor is silently wrong. In scoring, this produces a
plausibly-valued score for a bad candidate.

**Recommended direction:** Add an optional `bounds_error=True` parameter to
`interp_X_at_times`. When `True` (the default for scoring and diagnostics), raise if
any query time falls outside `[0, (n_time-1)/fs]`. Keep `bounds_error=False` clamping
available for callers that genuinely need extrapolation.

---

### 6. `identify_cycles_from_phase` assumes monotone phase

**Valid. Severity: Important.**

`np.searchsorted(phase0, target, side="left")` (epochs.py line 192) is only correct
when `phase0` is non-decreasing. There is no monotonicity check. If `hilbert_phase`
produces local reversals (common with low SNR or a poor reference signal), cycle
boundaries will be wrong without any warning.

**Recommended direction:** After computing `phase0`, check `np.any(np.diff(phase0) < 0)`.
If true, raise `ValueError` with the count and location of the first reversal. Callers
with a clean phase signal will never see this; callers with a bad phase signal get a
useful diagnostic instead of silent wrong output.

---

### 7. Orientation diagnostics may hide sign flips

**Partially valid. Severity: Minor, but docstring is incorrect.**

After sign-alignment (diagnostics.py lines 127–129), all `n_aligned` vectors are in
the same hemisphere relative to the reference normal. `orientation_score` is then
`dot(n_aligned[k], global_n_mean)`. This means a cycle whose traversal direction is
genuinely reversed will score near `+1` (not `−1`) because the sign-alignment already
corrects it. The docstring on line 75 says "values near −1 mean it is flipped but
otherwise consistent" — this is wrong; that case cannot occur after alignment.

The result is that `orientation_score` cannot detect traversal-direction reversal,
which was a root cause of earlier Bayesian failures.

**Recommended direction:** Add a second column `signed_orientation_score = dot(n_arr[k], global_n_mean)` (without prior sign-alignment). A cycle traversed in the opposite direction will show a negative value here while `orientation_score` stays positive. Fix the docstring.

---

### 8. Geometric scoring does not include coverage metrics

**Valid. Severity: Important for production use, minor for current stage.**

`find_epochs_by_geometric_score` returns `n_cycles` in the candidate table but not
`fraction_samples_assigned`, `min_samples_per_cycle`, or `coverage_duration_fraction`.
A candidate that assigns 30% of the recording to cycles will score identically to one
that assigns 90%, if planarity and anchor metrics are the same.

**Recommended direction:** Add these four columns to the candidate table:
`fraction_samples_assigned`, `min_samples_per_cycle`, `coverage_duration_fraction`
(`(tau[-1] - tau[0]) / total_duration`), and `n_cycles` (already present). Report only
— do not fold into `total_score` yet.

---

### 9. PCA skipped cycles missing from cycles table

**Valid. Severity: Important.**

When a cycle has fewer than `min_samples_per_cycle` valid samples,
`fit_pca_phase_coordinates` uses `continue` and that cycle is silently absent from
`cycle_rows`. Since `CycleEpochs` is now the authoritative cycle contract, the caller
cannot distinguish "this cycle was fitted but reconstruction failed" from "this cycle
was never fitted." `reconstruct_phase_coordinates` silently produces NaN rows for
both cases.

**Recommended direction:** The cycles table should have one row per epoch cycle.
Skipped cycles appear with `fit_ok=False` and NaN geometry columns. This makes the
invariant `len(cycles) == epochs.n_cycles` hold and makes skipped cycles visible to
downstream callers. `fit_bayesian_phase_coordinates` already follows this pattern.

---

### 10. Bayesian seed centers computed from rounded indices

**Valid. Severity: Minor.**

Even when `seed_epochs` carries real-valued `tau` (e.g. from a geometric scorer),
`_fit_layer1` computes seed centers via `seed_cycle_centers` which slices `X` at
integer-rounded boundaries (lines 454–461 of `bayesian.py`). For inter-sample
boundaries this adds up to half-sample error to the seed center. The practical effect
is small because Layer 1 samples its own `c` from a prior and the seed only sets the
prior mean.

**Recommended direction:** Use `interp_X_at_times` to compute anchor midpoints, or
use the sample mask from `seed_epochs.cycle_index` to take a proper mean. Low
priority.

---

## Additional concerns

### A. `sample_start` / `sample_stop` are O(K × N) per call

Both properties loop K times doing a full-array `np.where(self.cycle_index == k)`.
On a 10-minute recording at 200 Hz with 600 cycles that is ~72 M comparisons per
property access, called as a pair inside `compute_cycle_quality`. Replace with a
single `np.unique` + `np.searchsorted` pass.

### B. `epochs.phase` stores raw phase, not zero-referenced phase

`identify_cycles_from_phase` stores `phase=phase` (the raw input) but `phase_in_cycle`
is computed from `phase0 = phase - phase[0]`. A downstream consumer that interprets
`epochs.phase` as the zero-referenced phase will get the wrong values. Either store
`phase0` or document unambiguously that `phase` is the raw unwrapped input.

### C. `period_search.py` not audited

This review did not inspect `period_search.py` in detail. It should be reviewed
before it is used to drive real data analysis.

---

## Recommended cleanup sequence

Items 1–6 should be completed before any new modeling work.
Items 7–9 should be completed before connecting `find_epochs_by_geometric_score`
to the Bayesian seed path or testing on real data.
Items 10–12 can be deferred.

| Priority | Fix |
|----------|-----|
| 1 | Delete `_fit_bayesian_phase_coordinates_legacy`, `BayesianPhaseResult`, `BayesianPhaseEstimates` |
| 2 | Add `__post_init__` validation to `CycleEpochs` |
| 3 | Emit `fit_ok=False` rows in `fit_pca_phase_coordinates` for skipped cycles |
| 4 | Add monotonicity check to `identify_cycles_from_phase` |
| 5 | Add `bounds_error=True` to `interp_X_at_times` |
| 6 | Clarify `epochs.phase` vs `phase0` storage (docstring or field rename) |
| 7 | Add coverage columns to `find_epochs_by_geometric_score` candidate table |
| 8 | Add `signed_orientation_score` to `compute_cycle_quality`; fix docstring |
| 9 | Make `T0` an explicit parameter on `fit_bayesian_phase_coordinates` |
| 10 | Document half-open endpoint convention in `CycleEpochs` |
| 11 | Fix O(K×N) cost in `sample_start` / `sample_stop` |
| 12 | Use `interp_X_at_times` for Bayesian seed centers |

---

## Open design questions

These need a decision before the corresponding code is written.

**Q1 — `T0` exposure.** When `seed_epochs` is provided externally, should the caller
also supply `T0` so the Layer 1 duration prior matches their intended period? Or is it
acceptable for `T0` to always be computed internally from the periodogram, even when
the seed epochs come from a different source?

**Q2 — Endpoint convention.** For a cycle `[tau_k, tau_{k+1})` where `tau_{k+1}`
falls exactly at the last sample time, should that sample be in the cycle or not? The
current code says no. Is that the intended behavior?

**Q3 — `epochs.phase` semantics.** Should `CycleEpochs.phase` store the raw unwrapped
phase as passed in (current behavior) or the zero-referenced `phase - phase[0]`? The
`phase_in_cycle` field is computed from `phase0`, so storing raw `phase` means the two
fields use different references.
