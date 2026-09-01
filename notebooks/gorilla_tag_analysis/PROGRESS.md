# Gorilla-tag extended analysis — progress

Plan: `/home/opher/.claude/plans/vectorized-pondering-orbit.md` (approved 2026-09-01).
Executing autonomously per that plan; only pausing to ask if execution goes
off-plan or needs a creative/unexpected solution.

## Stage 1 — Shared loader + plotting helpers
Status: DONE

Built `_loading.py` and `_plotting.py`.

**`_loading.py`**: sets `GORILLA_TAG_PROJECT_ROOT=test_data/` before importing
Noam's `config_paths`/`names_format` (works as verified earlier). Adapts her
`repetitive_pos_path`, `load_position_file`, `bridge_boolean_mask_gaps`,
`mask_to_runs`, `count_overlapping_windows`, `build_runs_from_positive_windows`
verbatim (logic unchanged, only path wiring changed). Adds `_load_rep_df()`
(Noam's own rep_df-construction cell wasn't captured by the extraction pass,
so this was built fresh from the raw `window_predictions_{walk,jump,climb}.csv`
files, which already have subject/day/session_part/task as separate columns —
no source_file parsing needed). Public API: `list_available_blocks(subject=)`,
`load_block(subject, day, part, skill, run_index=0|"longest", hand=)`,
`hand_y_relative_to_root(...)`, `get_qc_table()`.

**`_plotting.py`**: `SKILL_COLORS/LABELS`, `HAND_COLORS/LABELS/MARKERS`,
`DAY_LABELS/MARKERS/LINESTYLES`, `PART_LABELS`, `get_subject_color`,
`save_figure` (writes to `notebooks/gorilla_tag_analysis/figures/*.png`,
simplified from Noam's cache/tables/qc/report tree since we don't need it),
`equal_limits` — all adapted from her notebook.

**Verified:**
- `pixi run -e dev python -c "from _loading import ..."` — imports cleanly,
  lists all available blocks for subject 002, loads full (non-truncated)
  walk/jump/climb signals. Example: walk day1-s longest run = 1559 frames /
  25.98s; jump day3-e longest run = 1859 frames / 30.98s; climb day1-s
  longest run = 1379 frames / 22.98s.
- `_plotting.save_figure` writes a real PNG; test file created and removed.
- `pixi run -e dev pytest -q` → 62 passed, 1 skipped (unchanged; nothing in
  `phase_coordinates/` touched).

**IMPORTANT DEVIATION FROM THE PLAN'S ASSUMPTIONS — "6 blocks" was wrong:**
The plan assumed subject 002 × {day1-s, day3-e} × {walk,jump,climb} = 6
single continuous blocks. In reality, `build_runs_from_positive_windows`
finds **58 disjoint runs** across those 6 (day,part,skill) conditions —
positive-labeled movement is fragmented into many bouts of ~4-31s each, not
one long continuous recording per file. Per-condition run counts:

| day,part | climb | jump | walk |
|---|---|---|---|
| 1,s | 11 | 13 | 11 |
| 3,e | 6 | 7 | 10 |

`load_block(..., run_index="longest")` picks the single longest run per
condition (closest to what the original smoke-test notebook did, just
systematized across all 6 conditions) — this is what Stage 2 will use as
each condition's primary block. `list_available_blocks()` exposes the full
inventory of all 58 runs, which Stage 5 (within-condition variability) should
use directly instead of treating each condition as a single block — repeated
bouts of the same skill on the same day/part are exactly the kind of
within-condition variability the user asked to see, and we now have real
replicate data for that rather than needing to invent it. Flagging this
prominently for the coordinating thread/user rather than silently picking
one interpretation.

## Stage 2 — Full-block PCA pass, all 3 movement types
Status: DONE

Built `01_full_block_pca.ipynb`. Fit `fit_pca_phase_coordinates` on all 58
disjoint runs x 2 hands (r_hand, l_hand) = 116 fits total, full untruncated
signal each time (3-D hand-relative-to-root, hand.y-relative-to-root as the
Hilbert reference signal, matching Noam's own convention).

**f_range:** probed a single shared `(0.5, 3.0)` Hz band against the longest
run of each of the 6 conditions before committing to it for all 116 fits —
resulting median cycle durations (walk ~0.67-1.03s, jump ~0.75-0.93s, climb
~0.75-1.07s) land comfortably inside Noam's own skill-specific expected
ranges (walk 0.5-3.1s, jump 0.4-1.5s, climb 0.4-1.7s), zero phase-monotonicity
warnings. No per-skill differentiation was needed — used `(0.5, 3.0)`
uniformly for walk/jump/climb.

**Results:** all 116 fits succeeded, **zero anomalies** (no zero-cycle runs,
no exceptions). 1,877 cycles total across all fits (climb 638 over 34
run×hand fits, jump 611 over 40, walk 628 over 42 — note not every run has
both hands' data equally represented, hence uneven counts per skill).
Reconstruction error on all 6 primary (longest-per-condition) blocks is
~2e-14 to 5e-14 (essentially exact round-trip), with 1377-1859/1379-1859
samples fitted per block (only a couple of edge samples near cycle
boundaries unfitted, as expected).

**Cache:** `cache/pca_results.pkl` — a pickled dict keyed by
`(subject, day, part, skill, source_file, movement_run_id, hand)` →
`{"samples": DataFrame, "cycles": DataFrame, "meta": dict}`. `meta` carries
run_duration_sec, n_cycles, reconstruction_error_max, warnings, error (for
programmatic filtering). Stages 3-6 should load this directly rather than
refitting.

**For Stage 3/6:** no runs need to be excluded — even the shortest run
(3.98s, climb) produced 3 clean cycles. All 58 runs × both hands are usable
replicate data for Stage 5's within-condition variability work.

Verified: `pixi run -e dev pytest -q` → 62 passed, 1 skipped (unchanged,
`phase_coordinates/` untouched).

## Stage 3 — Compare against Noam's method (Hilbert consistency + side-by-side)
Status: DONE

Built `02_compare_noam.ipynb`. Reused Noam's `extract_cycles_from_signal`
(and `smooth_signal`, `CYCLE_PARAMS`) verbatim from
`_noam_cycle_extraction_reference.py`, run against the *identical* raw
hand-Y-relative-to-root signal used for our cached PCA fits — nothing
refit, all 116 cached (run x hand) entries read from
`cache/pca_results.pkl`.

**3.1 — Hilbert vs. Noam agreement: strong.**
- Boundary-time offset (|our Hilbert cycle-boundary time − nearest Noam
  peak/trough event time|, all 116 runs, ~650-670 boundaries per skill):
  median 0.17-0.20s across walk/jump/climb (IQR roughly 0.13-0.27s). That's
  ~15-25% of a typical cycle duration (0.65-1.1s) — small, and expected,
  since the two methods define "boundary" differently (phase-zero-crossing
  vs. a smoothed peak/trough location within the cycle), not because either
  method is unstable.
- Hilbert phase-monotonicity warnings (from `hilbert_phase`, computed during
  Stage 2): essentially none — 0/34 climb, 1/40 jump (2.5%), 0/42 walk.
- **0 runs skipped** — both methods produced usable cycles for all 116
  cached fits.

**3.2 — Cycle count / duration / shape: strong agreement.**
Per the 6 primary blocks x 2 hands (12 rows), cycle counts differ by only
1-4 cycles between methods (e.g. climb day1-s r_hand: 20 ours vs. 18 Noam's;
jump day3-e: 41 vs. 40 both hands), and median cycle durations match almost
exactly in most conditions (many exact frame-level matches, e.g. walk
day1-s both hands: 1.033s vs. 1.033s). Full table in the notebook's last
cell. Mean-cycle-shape overlay (Noam's raw-signal `y_cycle_norm` vs. our
PCA-reconstruction's hand-Y-relative-to-root, both normalized to 100 points)
plotted in `figures/stage3_mean_cycle_shape_comparison.png` — visually
consistent per skill/hand/day.

**Bottom line: the two independent methods substantially agree** — this is
a real validation result, not just "the code ran." No anomalies worth
escalating; nothing suggests either method is broken.

Verified: `pixi run -e dev jupyter nbconvert --execute --inplace` succeeded
with 0 errors; `pixi run -e dev pytest -q` → 62 passed, 1 skipped, unchanged.

**Note for the coordinating thread:** git commit/push were blocked by a
permission-classifier issue at the time this stage ran (per the plan's
operational-decisions section, commits should happen after every stage) —
this stage's files are uncommitted on disk, same as Stage 2's.

## Stage 4+5 — Plane geometry / radius/perp + drift & variability
Status: NOT STARTED

## Stage 6 — Bayesian analysis for real
Status: NOT STARTED
(Feasibility pre-confirmed 2026-09-01: pymc 5.28.5 / arviz 0.23.4 install and
run correctly in the `bayes` pixi env; smoke MCMC fit succeeded in ~53s on a
tiny synthetic case. Real risk is runtime at full-block scale, not
installability — see plan for scoping approach.)
