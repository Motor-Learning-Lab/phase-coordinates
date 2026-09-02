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

**2026-09-02 fix — phase-anchoring correction.** The original 3.1 compared
our cycle *boundaries* (`fit_pca_phase_coordinates`'s `cycle` column /
`cycles["time_start"/"time_stop"]`, anchored to `phase[0]` — whatever
absolute phase happened to be at each run's first sample) directly against
Noam's peak/trough events, and found a systematic ~15-25%-of-a-cycle offset.
Investigated per the user's request: is this a constant phase shift (they
suspected our boundary sits at a zero-crossing, ~pi/2 from her peak)?

Checked directly, two things:
1. Our raw absolute `phase` at Noam's own events is already extremely
   tightly locked to the waveform — circular resultant length R~0.98 at
   phase=0 for "peaks"-type hands/skills, and R~0.98 at phase=pi for
   "troughs"-type (walk l_hand). No shift in the phase estimate itself.
2. But `phase[0]` (the reference our *cycle boundaries* are anchored to)
   is **not** random across the 116 runs — it clusters bimodally right at
   +-90 degrees (R~0.05 for a single circular mean, because it's genuinely
   bimodal at +90 AND -90, not one direction). That's because these runs
   are extracted from a movement-onset classifier's positive windows, and a
   limb is most likely to be flagged "moving" right as it crosses the
   middle of its range (fastest-changing, easiest to detect), not at the
   top/bottom of a swing where it's momentarily still.

So the user's intuition was correct in substance (our boundary sits at a
positive- or negative-going zero-crossing, hers at the peak) even though the
literal "constant pi/2" framing undersold it slightly — it's not a fixed
pi/2 *rotation* to undo, it's that the *reference point itself* (`phase[0]`)
is uninformatively tied to wherever the recording happens to start rather
than to any feature of the waveform.

**Fix applied** (comparison-only — does not touch `phase_coordinates/` or
any other stage's cycle fits): re-derive "our" cycle boundaries directly
from raw absolute `phase`, anchored at 0 (peaks) or pi (troughs) per the
same `event_type_by_hand` lookup Noam's method uses, instead of at
`phase[0]`. Implemented as `phase_reference()` + `phase_anchored_windows()`
in the notebook, used by both 3.1 and 3.2.

**Result — offset collapses to ~1 frame:**

| skill | median offset (before) | median offset (after) |
|---|---|---|
| climb | 0.183s | 0.017s |
| jump  | 0.200s | 0.000s |
| walk  | 0.167s | 0.017s |

0.017s = 1 frame at 60Hz — i.e. after the correction, our boundaries and
Noam's events land on the same frame essentially every time. This
confirms the fix (the "agreement" claim below is now much stronger than it
was, not just re-confirmed) and confirms the mechanism was a boundary
*reference* artifact, not a phase-estimation problem.

**3.1 — Hilbert vs. Noam agreement: now near-exact** (see table above; was
"strong" pre-fix at ~15-25% offset).
- Hilbert phase-monotonicity warnings (from `hilbert_phase`, computed during
  Stage 2): essentially none — 0/34 climb, 1/40 jump (2.5%), 0/42 walk.
- **0 runs skipped** — both methods produced usable cycles for all 116
  cached fits.

**3.2 — Cycle count / duration / shape: strong agreement** (re-run with
phase-anchored windows on the "ours" side; numbers barely moved from the
pre-fix version since 3.2 was never using the mis-anchored boundaries the
same way 3.1 was — the fix mainly sharpens the mean-cycle-shape overlay's
phase-axis alignment). Per the 6 primary blocks x 2 hands (12 rows), cycle
counts differ by only 1-4 cycles between methods (e.g. climb day1-s r_hand:
21 ours vs. 18 Noam's; jump day3-e: 42 vs. 40 both hands), and median cycle
durations match almost exactly in most conditions (many exact frame-level
matches, e.g. jump day1-s r_hand: 0.925s vs. 0.925s). Full table in the
notebook's last cell. Mean-cycle-shape overlay (Noam's raw-signal
`y_cycle_norm` vs. our PCA-reconstruction's hand-Y-relative-to-root, both
normalized to 100 points, now both starting at the same peak/trough)
plotted in `figures/stage3_mean_cycle_shape_comparison.png` — visually
consistent per skill/hand/day.

**Bottom line: the two independent methods substantially agree, and once
compared on a like-for-like phase reference, agree almost exactly on where
cycle boundaries fall** — this is a real, now sharper validation result, not
just "the code ran." No anomalies worth escalating; nothing suggests either
method is broken.

Verified: `pixi run -e dev jupyter nbconvert --execute --inplace` succeeded
with 0 errors; `pixi run -e dev pytest -q` → 62 passed, 1 skipped, unchanged.

**Note for the coordinating thread:** git commit/push were blocked by a
permission-classifier issue at the time this stage ran (per the plan's
operational-decisions section, commits should happen after every stage) —
this stage's files are uncommitted on disk, same as Stage 2's.

## Stage 4+5 — Plane geometry / radius/perp + drift & variability
Status: DONE

Built `03_geometry_and_drift.ipynb`, reading only Stage 2's cache — nothing
refit. 14 figures saved to `figures/`.

**Subphase 4a (plane geometry):** for each of the 6 primary blocks, first/
middle/last fitted cycle drawn as a translucent PCA-plane patch (spanned by
`e1`/`e2` over that cycle's observed `(u,v)` range) with the cycle's actual
measured 3-D hand-relative-to-root points scattered on it
(`stage4_plane_<skill>_day<day><part>.png`).

**Subphase 4b (radius/perp/phase):** full-block (untruncated) time series of
`radius`, `perp`, `phase_in_cycle` for all 6 primary blocks
(`stage4_signals_*.png`).

**Subphase 5a (within-block drift):** one combined 6-condition x 5-metric
grid (`stage5_within_block_drift_grid.png`) — duration, center-distance-
from-block-mean, normal-angular-deviation-from-block-mean, radius_mean,
**perp_sd** (within-cycle off-plane *width*), all vs. cycle index.

**Subphase 5b (across-run, day 1 vs day 3):** using all 58 runs (not just
the 6 primary blocks) as real bout-to-bout replicate data, per-run means of
the same 5 metrics compared Day-1-start vs Day-3-end per skill
(`stage5_across_run_day1_vs_day3.png`). **Real, consistent finding:** for
every skill, Day 3 shows *lower* median duration, *lower* center-drift,
*lower* normal-drift, and *lower* spread (std) in all of these than Day 1 —
e.g. climb duration 1.09s→0.77s, walk center_dist spread 3.02→0.87, jump
normal_dev_deg median 17.4°→7.3°. Consistent with a practice effect: cycles
get shorter/faster and both individual cycles and bout-to-bout geometry
become more stable with practice. Worth showing the user prominently — this
is the first genuinely interesting substantive result beyond validation.

**2026-09-02 fix:** the original version of this stage tracked `perp_mean`
as its 5th metric, which is trivially ~0 for every cycle by construction
(PCA mean-centering guarantees the 3rd component's mean over the same
points used to fit it is exactly zero — not a bug, just uninformative).
Swapped to `perp_sd` (within-cycle off-plane width) in both subphases per
explicit user request. Re-executed clean (0 errors), `pytest` still 62
passed/1 skipped.

**perp_sd Day-1-vs-Day-3 result — mostly consistent with the practice
effect above, but not uniformly:** median per-run perp_sd (r_hand):

| skill | day 1 | day 3 |
|---|---|---|
| walk  | 1.54 | 0.48 |
| jump  | 2.07 | 0.61 |
| climb | 1.38 | 1.75 |

Walk and jump both show off-plane width shrinking ~3x with practice,
matching the duration/center/normal-drift story. **Climb goes the other
way** — width *increases* from day 1 to day 3, and still does even after
normalizing by `radius_mean` (0.089→0.106) to rule out it just being a
side-effect of climb's own radius changing. Worth flagging as a genuine
exception rather than smoothing it into the "practice makes it more stable"
narrative — climb's cycles may be getting less planar, not more, with
practice, or this metric may behave differently for climbing specifically.

Verified: `pixi run -e dev jupyter nbconvert --execute --inplace` succeeded
(0 errors); `pixi run -e dev pytest -q` → 62 passed, 1 skipped, unchanged.
`phase_coordinates/` untouched.

**Note for the coordinating thread:** git commit/push not attempted here per
directive — leaving Stage 4+5's files uncommitted on disk for the
coordinator to handle, consistent with Stage 2/3.

## Stage 6 — Bayesian analysis for real
Status: DONE

**Process note:** the subagent assigned to this stage fit all 6 blocks
successfully (see below) but then stalled for ~1hr building the notebook/
writeup — its own background MCMC-watcher outlived its turn twice, and after
a third check-in with no response the coordinating thread took over directly
rather than waiting further. Actual Bayesian compute was never the
bottleneck (all 6 fits together took well under 10 minutes); this was a
subagent-coordination issue, not a modeling-cost one. Built
`04_bayesian.ipynb` directly, executed clean (0 errors), `pytest` still 62
passed/1 skipped.

**What was fit:** all 6 primary (longest-per-condition) blocks, reduced MCMC
settings (`draws=300, tune=300, chains=2`), cached to `cache/bayes_*.pkl`
and combined in `cache/bayes_results.pkl` (same `samples`/`cycles` schema as
`pca_results.pkl` — plain DataFrames, no pymc/arviz objects, so the notebook
loads it under `-e dev` with no pymc import needed).

| skill | day/part | n_cycles | elapsed_sec | diagnostic failures |
|---|---|---|---|---|
| climb | 1,s | 20 | 30 | rho_tau=0.724 > 0.40 |
| climb | 3,e | 37 | 37 | boundary posterior multimodal |
| jump  | 1,s | 9  | 25 | none |
| jump  | 3,e | 40 | 30 | rho_tau=0.408 > 0.40 |
| walk  | 1,s | 24 | 175| none |
| walk  | 3,e | 22 | 23 | rho_tau=0.518 > 0.40 |

**Headline finding — reliability, not just agreement:** on the cleanest fit
(walk, day 1: no diagnostic failures), PCA and Bayesian estimates track
closely for radius/perp/phase and produce visually similar plane geometry,
extending Stage 3's PCA-vs-Noam validation to a third independent estimator.
But **4 of 6 blocks trip a hard diagnostic failure** the package itself
defines (`rho_tau` boundary-cloud spread too large, or multimodal boundary
posterior) at these reduced settings. This is an honest scoping result, not
a finished comparison — open question for a follow-up: does scaling to
default settings (`draws=1000, tune=1000, chains=4`) resolve it, or does it
reflect a real modeling difficulty for short/low-amplitude jump/climb
cycles specifically?

Not attempted: scaling any block to full default MCMC settings (out of
scope for this stage's reduced-settings scoping pass; a natural next step).
