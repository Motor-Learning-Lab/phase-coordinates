# Debug materials for cycle-fixed geometry branch

This branch is no longer pursuing the full instantaneous-geometry Layer 2 diagnostic path from `bayesian-two-layer-estimator`.

The old branch remains available for that history. On this branch, stale amplitude-diagnostic prompt files, review notes, and logs specific to the instantaneous-geometry effort have been removed where they were likely to mislead follow-up work.

## Active debug target

The active debug target is the cycle-fixed geometry Layer 2 model described in:

```text
docs/bayesian_two_layer_spec.md
docs/cycle_fixed_geometry_model.md
```

The first implementation should add:

```text
docs/debug/scripts/test_layer2_cycle_fixed.py
```

using the same synthetic fixed-plane data as the previous Layer 2 tests.

## Required report for the first cycle-fixed test

The first cycle-fixed debug script should report:

```text
divergence count
max treedepth count
sigma_x mean/median/5%/95%
R_k by cycle
center norm by cycle
normal cos_sim by cycle
z_rms
RMSE total
RMSE normal
RMSE cyclic
RMSE tangential
phase monotonicity
frame orthonormality
hierarchical scales: sigma_c, sigma_log_R, sigma_n_angle, sigma_a
```

## Logs

| File | Script | Config | Result |
|---|---|---|---|
| `13_layer2_cycle_fixed_first_run.log` | `test_layer2_cycle_fixed.py` | tune=400, draws=400, chains=2, target_accept=0.9, seed=0 | **3 divergences, 19s.** Normal cos_sim=1.0000 all cycles. Center norms 0.04–0.19 (confound eliminated). z_rms=0.004. BUT sigma_x=0.591 (30×true), R_k=0.217 (5×too small) — amplitude ridge persists in R_k/sigma_x space. Assertion on R_k median fails. See PROGRESS.md finding 1. |
| `14_cycle_fixed_oriented_frame.log` | `test_layer2_cycle_fixed.py` | tune=400, draws=400, chains=2, target_accept=0.9, seed=0 | **3 divergences, 20s.** Two-anchor oriented frame. R_k=0.999 (**fixed**, was 0.217). sigma_x=0.052 (2.6×true, was 30×). Signed normal cos_sim min=0.9954 (all positive). RMSE total=0.044, z_rms=0.036 (cycle-5 boundary artifact). ALL CHECKS PASSED. See PROGRESS.md finding 3. |

## Legacy scripts

Some older utility and Layer 1 debug scripts may still be useful as fixtures or examples, especially for synthetic data generation and Layer 1 sanity checks. But old scripts whose purpose was diagnosing the instantaneous `center(t)` / `normal(t)` / radius / `sigma_x` model should not guide the implementation on this branch.

When in doubt, follow the active spec and design note, not the old instantaneous-geometry debug trail.
