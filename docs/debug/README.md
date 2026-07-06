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

## Legacy scripts

Some older utility and Layer 1 debug scripts may still be useful as fixtures or examples, especially for synthetic data generation and Layer 1 sanity checks. But old scripts whose purpose was diagnosing the instantaneous `center(t)` / `normal(t)` / radius / `sigma_x` model should not guide the implementation on this branch.

When in doubt, follow the active spec and design note, not the old instantaneous-geometry debug trail.
