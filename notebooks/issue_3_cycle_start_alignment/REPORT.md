# Issue #3: inconsistent cycle-start alignment

Workup of [issue #3](https://github.com/Motor-Learning-Lab/phase-coordinates/issues/3), against
`main` at `d83b30d` (the commit named in the issue).

Every number and figure here is produced by `issue_3_cycle_start_alignment.ipynb` in this folder.
Nothing in `phase_coordinates/` was changed; the proposed behaviour is prototyped in `_issue3.py`.

## Summary

Noam's observation is correct, and most of it comes from the package, not from her pipeline or
from the movement.

With the current package, **19% of subject 006's cycles start more than an eighth of a cycle away
from where the other cycles of the same condition start** (13% for subject 002). With the changes
proposed here that falls to **2%** (1% for subject 002), and her cycle-variability measure for
subject 006 falls by half (0.18 to 0.09). So about half of the variability she would have measured
was segmentation error.

There are three causes, in order of size:

1. **Phase zero is set once per run, from one unreliable sample.** If that sample is off, every
   cycle in the run is shifted by the same amount. About 7% of runs are shifted by more than an
   eighth of a cycle, some by half a cycle.
2. **The leftover pieces at the two ends of each run are returned as cycles.** They are 14% of all
   cycles for subject 006, and 72% of them are misaligned.
3. **The dominant frequency is estimated separately for each run, with a rule that slow drift can
   capture.** In 24 of 794 fits the run is cut into fewer than two complete cycles.

The fix for 1 and 2 is small and gives most of the benefit. Until it is in the package, Noam can
get the same result with a short wrapper around the current public functions
([Workaround](#workaround-with-the-current-package)).

## What was done

- **Data.** Subject 006, all three days and session parts (the files attached to the issue), and
  subject 002, day 1 start and day 3 end (the files already in `test_data/`). That is 397 movement
  runs and 794 run × hand fits. Runs were built with the rules in `labeled_analysis.py`.
- **Reproduction.** `fit_pca_phase_coordinates(xyz, sampling_rate_hz=60)` per run and hand, with
  boundaries from `sample_start` / `sample_stop`, as in the issue. The prototype in `_issue3.py`,
  with all its switches at the current setting, returns exactly the package's cycles in all 794
  fits, so the changes below are measured against the real behaviour.
- **The figure from the issue.** The common-plane QC figure and the subject 002 files were
  described in the issue but not attached. Figure 1 rebuilds the figure from
  `plot_common_plane_cycle_qc()` for the same condition (subject 006, day 3, start).

![Figure 1](figures/fig01_issue_reproduction.png)

*Figure 1. Cycles from the current package in a shared PCA plane, built as in the issue. Most
starts cluster; some are scattered around the loop.*

## How misalignment is measured

The measure must not depend on the phase the package computed. Each time-normalised 3-D cycle is
treated as a closed loop and slid along itself until it best matches the mean loop of its condition
(subject × skill × day × part × hand). The slide needed is the cycle's **start lag**, in degrees of
a cycle. A cycle is called **misaligned** when its start lag exceeds 45°, an eighth of a cycle.
Only the shapes of the cycles are compared.

![Figure 2](figures/fig02_start_lags_current.png)

*Figure 2. Start lags under the current package, subject 006. Partial cycles (pink) are spread over
the whole circle. Complete cycles (grey) have a main peak and a second group near ±180°.*

## Cause 1: one sample sets phase zero for the whole run

For the automatic reference, `fit_pca_phase_coordinates` finds the first local peak of the
band-passed reference and subtracts the phase at that sample from the whole run
(`_first_positive_peak_index`, `core.py:180` and `core.py:351-360`). Every later boundary is a
multiple of 2π counted from that sample.

That sample is the least reliable one in the run. It lies within about one cycle of the start of
the record, where the band-pass filter and the Hilbert transform are still settling, and
`find_peaks` accepts any local maximum, however small.

![Figure 3](figures/fig05_anchor_example.png)

*Figure 3. One jumping run. Top: the anchor lands on a small bump in a trough, so every boundary
falls in a trough. Bottom: boundaries at the peak of the band-passed reference in every cycle.*

The Hilbert phase at the anchor sample should be 0. Its actual value is the run's **anchor error**.

| | Subject 006 | Subject 002 |
|---|---|---|
| Runs with anchor error above 45° | 6.8% | 6.0% |
| Spread of start lags between runs (complete cycles) | 32° | 25° |
| Spread of start lags within a run | 14° | 9° |
| Spread between runs after subtracting each run's anchor error | 19° | 8° |

Cycles within a run agree with each other. Runs disagree with one another, and the anchor error
predicts by how much.

![Figure 4](figures/fig04_anchor_statistics.png)

*Figure 4. Left: anchor error over subject 006's runs. Right: the mean start lag of each run's
complete cycles against its anchor error.*

This also explains why Noam's first attempt, with hand Y as an explicit `ref_signal`, was
inconsistent. For an explicit reference or a supplied phase, phase zero is the phase of the run's
first sample (`core.py:381`), which is wherever the run happened to begin.

## Cause 2: partial cycles at the ends of each run

A run starts and stops where the movement classifier switched on and off. The pieces before the
first boundary and after the last one are kept as cycles whenever they have at least 10 samples,
and their `sample_start` is the first sample of the run.

| | Subject 006 | Subject 002 |
|---|---|---|
| Partial cycles, share of all cycles | 14.4% | 10.8% |
| Misaligned, among partial cycles | 72% | 76% |
| Misaligned, among complete cycles | 10% | 6% |

For subject 006, partial cycles are 54% of all misaligned cycles. They also explain the "unusual
trajectories" in the issue: a partial cycle is a fraction of a loop stretched to 100 points.

![Figure 5](figures/fig03_partial_cycles.png)

*Figure 5. Left: one walking run with its two partial cycles shaded. Right: partial cycles are
mostly much shorter than the complete cycles of their run.*

## Cause 3: the per-run dominant frequency

The band-pass is centred on the largest periodogram bin of that run's reference, at any positive
frequency (`estimate_dominant_period`). Runs are 3 to 31 s long. In a short run a slow drift can
hold more power than the movement, and the band follows the drift.

![Figure 6](figures/fig07_frequency_example.png)

*Figure 6. A 10 s walking run. The current rule picks 0.20 Hz and returns one complete cycle. A
rule that detrends and requires at least three cycles in the run picks 1.35 Hz and returns twelve.*

With the current rule, 24 of 794 fits give fewer than two complete cycles. With the proposed rule,
one does, and 318 more complete cycles are recovered (8290 to 8608). It has little effect on the
alignment of the cycles that were already found.

![Figure 7](figures/fig06_frequency_per_run.png)

*Figure 7. Per-run frequency against a frequency pooled over the run's condition, subject 006.
The low outliers on the left are drift. The remaining disagreements are all climbing; see
[Open questions](#open-questions).*

## Two smaller effects

- **Cycles that are not one block of samples.** A cycle is every sample whose phase lies in
  [2πk, 2π(k+1)). When the phase briefly runs backwards across a boundary, samples are assigned
  back to the previous cycle, and `sample_start`..`sample_stop` overlaps the neighbouring cycle.
  This affects 105 of 7662 cycles for subject 006 and 9 of 1912 for subject 002.
- **The reference axis differs between runs.** Each run gets its own dominant axis. The median
  angle to the condition's pooled axis is 6° to 13°, but 44 of 674 fits for subject 006 are more
  than 30° away, and in 2 the Y-based sign rule flips the axis.

## The proposed fix, one change at a time

| | Change |
|---|---|
| A | The current package. |
| B | A, keeping only complete cycles. |
| C | B, with phase zero at the peak of the band-passed reference in every cycle, and every cycle one block of samples. |
| D | C, with the more robust per-run frequency. |
| E | D, with the reference axis estimated once per condition and shared by its runs. |

Subject 006 (674 fits):

| | Cycles | Misaligned | Start-lag spread | Hand-Y variability |
|---|---|---|---|---|
| A | 7662 | 19.2% | 40° | 0.182 |
| B | 6555 | 10.1% | 30° | 0.138 |
| C | 6582 | 3.4% | 16° | 0.102 |
| D | 6865 | 3.2% | 15° | 0.103 |
| E | 6872 | 2.1% | 13° | 0.091 |

Subject 002 (120 fits):

| | Cycles | Misaligned | Start-lag spread | Hand-Y variability |
|---|---|---|---|---|
| A | 1912 | 13.3% | 30° | 0.134 |
| B | 1706 | 5.9% | 16° | 0.107 |
| C | 1708 | 0.7% | 11° | 0.098 |
| D | 1743 | 0.8% | 10° | 0.100 |
| E | 1744 | 0.9% | 9° | 0.093 |

"Misaligned" is over all cycles pooled. "Start-lag spread" (circular SD) and "hand-Y variability"
are medians over conditions. Hand-Y variability is the measure from Noam's section 8: RMS deviation
of each time-normalised hand-Y cycle from the condition mean, divided by cycle amplitude.

![Figure 8](figures/fig08_fix_steps.png)

*Figure 8. The same three measures by skill. B and C carry almost all of the improvement.*

![Figure 9](figures/fig09_before_after_common_plane.png)

*Figure 9. Figure 1 again, with the proposal (E) beside the current package. The mean cycle changes
too: averaging misaligned cycles shrinks and distorts it.*

![Figure 10](figures/fig10_before_after_hand_y.png)

*Figure 10. The same cycles as hand Y against cycle time.*

**Sharing the frequency across runs was tried and is not recommended.** It changes nothing where
the movement is stable and makes subject 006's day-1 climbing worse (12.7% misaligned against 4.1%
with a per-run frequency).

## Changes to make in the package

In order of value. Changes 1 to 3 belong in one pull request.

1. **Use the Hilbert phase as it is.** Remove the single-sample anchor. Phase zero is then the peak
   of the band-passed reference in every cycle. Do the same for an explicit `ref_signal`, which
   today is zeroed at the run's first sample.
2. **Make every cycle one block of samples.** Take the running maximum of the unwrapped phase
   before `floor(phase / 2π)`.
3. **Mark partial cycles and report per-cycle quality.** Add to `cycles`:
   - `complete`: the cycle has a boundary inside the record at both ends;
   - `duration_ratio`: duration over the median duration of the run's complete cycles;
   - `amplitude_ratio`: mean Hilbert amplitude over the run median;
   - `backward_steps`: samples where the phase ran backwards.
4. **Make `estimate_dominant_period` robust.** Detrend, taper, zero-pad, and consider only
   frequencies with at least three cycles in the record. `bayesian.py` uses the same function for
   its seeds, so the Bayesian path changes too and needs its own check.
5. **Allow a shared reference axis.** Add a `ref_axis` argument to `fit_pca_phase_coordinates` and
   a helper that estimates one axis from several runs. Also return the band actually used in
   `details`; today it has to be recomputed.

Existing behaviour that changes: cycle numbering and `samples["phase"]` on the automatic path, and
`details["phase_zero"]`. `test_automatic_reference_is_y_oriented_and_peak_anchored` asserts the
single-anchor behaviour and has to be rewritten. New tests should cover a run that starts
mid-cycle, a run whose first local peak is a small bump, a run with slow drift larger than the
movement, and a phase that steps backwards across a boundary.

## Workaround with the current package

`fit_with_absolute_phase_zero` in `_issue3.py` gives changes B and C using public functions only:

1. Compute the Hilbert phase of the package's own automatic reference with `hilbert_phase`.
2. Trim the run to start and end on an upward crossing of a multiple of 2π.
3. Pass the trimmed phase with `phase=`.

It returns exactly the cycles of variant C in all 783 fits where it applies, with the usual
per-cycle PCA coordinates. The other 11 runs have fewer than one complete cycle because of the
frequency problem, and it raises an error for those.

## Answers to the six questions in the issue

1. **Does the automatic reference aim for consistent starts across runs?** It aims to, and fails in
   the two ways above: the single anchor, and partial cycles.
2. **Is the dominant-PC reference suitable for elongated, back-and-forth trajectories?** Yes. The
   first component holds on average 70% (jumping) to 84% (climbing) of the variance in these data, and once the anchor is fixed the
   starts are tight for all three skills. Flatter loops are the easy case; it is the rounder
   jumping loops where the axis direction is less stable between runs.
3. **Can independently estimated references or bands move phase zero between runs?** Yes. The
   anchor is the large effect. Run-to-run differences in the axis are the next, and sharing the
   axis removes them. The band matters mainly when it is badly wrong.
4. **Do partial or irregular cycles stay in `cycles`? Is there a criterion?** Yes, both stay, and
   there is no criterion today. Drop cycles that touch either end of the run. For the rest, flag a
   cycle when its duration ratio is outside 0.6 to 1.5, its amplitude ratio is below 0.5, or its
   phase ran backwards. Under the proposal this flags 5.5% of subject 006's cycles: 33% of the
   flagged cycles are misaligned against 0.3% of the unflagged, and 85% of the misaligned cycles
   are caught. Subject 002 has too few misaligned cycles left for this to be informative.
5. **Which outputs to examine?** Today: whether a cycle touches sample 0 or the last sample;
   `duration` against the run median; `n_samples` against `sample_stop - sample_start` (unequal
   means a non-contiguous cycle); `details["amp_hilbert"]`; and the number of cycles against the
   run length, which exposes a wrong frequency.
6. **Recommended approach?** Keep the automatic reference. Use the workaround now. When the package
   changes, also share the axis across the runs of a condition, and keep the frequency per run.

## Open questions

- **Day-1 climbing for subject 006.** Per-run frequencies there are 2.4 to 3.7 Hz, while the pooled
  spectrum peaks near 1.2 to 1.4 Hz, about half. Later days are stable near 1.7 Hz. I could not
  establish which of the two is the movement cycle on day 1. The per-run cycles are the more
  self-consistent, but that does not settle it. Noam is better placed to say what the hands are
  doing in those runs.
- **Two subjects.** Subject 002 has only two of nine conditions. The causes are mechanical, so I
  expect them to hold for the other subjects, but the sizes may differ.
- **Phase zero is the peak of the band-passed reference, not of the raw signal.** For a waveform
  that is not symmetric these differ by a constant. That does not affect consistency, and I did not
  measure the offset.
- **The 45° threshold is a choice.** The start-lag spread does not depend on it and tells the same
  story.
- **The Bayesian estimator was not examined.**

## Files

| File | Contents |
|---|---|
| `REPORT.md` | This report. |
| `issue_3_cycle_start_alignment.ipynb` | Executed notebook; every number and figure above. |
| `_issue3.py` | Loading, the switchable segmentation prototype, the workaround, the measures. |
| `_issue3_plots.py` | Figure code. |
| `data/hand_root_xyz.npz` | Root and hand positions for the 35 source files, stored as float32. |
| `figures/` | The eleven figures the notebook writes. |
