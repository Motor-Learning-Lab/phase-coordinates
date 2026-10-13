"""Figures for the issue #3 workup.  Every function returns the figure and saves a PNG."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

import _issue3 as I

CURRENT = "#D55E00"
PROPOSED = "#0072B2"
PARTIAL = "#CC79A7"
NEUTRAL = "#8C8C8C"
INK = "#222222"
SKILL_COLORS = {"walk": "#1b9e77", "jump": "#d95f02", "climb": "#7570b3"}
SKILL_MARKERS = {"walk": "o", "jump": "s", "climb": "^"}
SKILL_LABELS = {"walk": "Walking", "jump": "Jumping", "climb": "Climbing"}
HAND_LABELS = {"r_hand": "Right hand", "l_hand": "Left hand"}
PART_LABELS = {"s": "Start", "m": "Middle", "e": "End"}


def use_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })


def save(fig, stem):
    I.FIGURES_DIR.mkdir(exist_ok=True)
    fig.savefig(I.FIGURES_DIR / f"{stem}.png", bbox_inches="tight")
    return fig


def _condition(table, subject, skill, day, part, hand):
    return table[
        table["subject"].eq(subject) & table["skill"].eq(skill) & table["day"].eq(day)
        & table["part"].eq(part) & table["hand"].eq(hand)
    ]


# ---------------------------------------------------------------------------
# Noam's common-plane QC view
# ---------------------------------------------------------------------------

def _common_plane_panel(ax, positions, runs, cycles, color):
    """Cycle-centred trajectories in one shared PCA plane, as in Noam's QC figure."""
    traj = I.cycle_trajectories(positions, runs, cycles, centre="cycle")
    if len(traj) < 2:
        ax.text(0.5, 0.5, "Not enough cycles", ha="center", va="center", transform=ax.transAxes)
        return
    basis = PCA(n_components=2).fit(traj.reshape(-1, 3)).components_.copy()
    if basis[0, 1] < 0:
        basis[0] *= -1
    if basis[1, np.argmax(np.abs(basis[1]))] < 0:
        basis[1] *= -1
    flat = traj @ basis.T
    for curve in flat:
        ax.plot(curve[:, 0], curve[:, 1], color=NEUTRAL, alpha=0.18, lw=0.6)
    ax.scatter(flat[:, 0, 0], flat[:, 0, 1], s=9, color=color, alpha=0.7, zorder=3, linewidths=0)
    mean = flat.mean(axis=0)
    ax.plot(mean[:, 0], mean[:, 1], color=INK, lw=1.8, zorder=4)
    ax.scatter(*mean[0], marker="*", s=110, color="white", edgecolor=INK, linewidths=1.0, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def common_plane(positions, runs, tables, subject, day, part, stem, colors=None):
    """
    Rows = skills.  Columns = hand x variant, where ``tables`` maps a variant
    label to its cycle table.
    """
    labels = list(tables)
    colors = colors or [CURRENT, PROPOSED][:len(labels)]
    n_cols = len(I.HANDS) * len(labels)
    fig, axes = plt.subplots(
        len(I.SKILLS), n_cols, figsize=(3.1 * n_cols, 3.0 * len(I.SKILLS)),
        constrained_layout=True, squeeze=False,
    )
    for i, skill in enumerate(I.SKILLS):
        for j, hand in enumerate(I.HANDS):
            for k, label in enumerate(labels):
                ax = axes[i, j * len(labels) + k]
                cycles = _condition(tables[label], subject, skill, day, part, hand)
                _common_plane_panel(ax, positions, runs, cycles, colors[k])
                ax.set_title(f"{SKILL_LABELS[skill]}, {HAND_LABELS[hand].lower()}\n{label} ({len(cycles)} cycles)")
    handles = [
        Line2D([], [], color=NEUTRAL, lw=0.8, label="One cycle"),
        Line2D([], [], color=INK, lw=1.8, label="Mean cycle"),
        Line2D([], [], marker="*", ls="", markersize=10, markerfacecolor="white", markeredgecolor=INK, label="Start of mean cycle"),
    ] + [
        Line2D([], [], marker="o", ls="", markersize=5, color=c, label=f"Cycle starts: {lab}")
        for lab, c in zip(labels, colors)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(
        f"Cycles in a shared PCA plane: subject {subject}, day {day}, {PART_LABELS[part].lower()} of session"
    )
    return save(fig, stem)


def hand_y_cycles(positions, runs, tables, subject, day, part, stem, colors=None):
    """Time-normalised hand-Y cycles (run-centred), one panel per skill x hand x variant."""
    labels = list(tables)
    colors = colors or [CURRENT, PROPOSED][:len(labels)]
    n_cols = len(I.HANDS) * len(labels)
    fig, axes = plt.subplots(
        len(I.SKILLS), n_cols, figsize=(3.1 * n_cols, 2.3 * len(I.SKILLS)),
        constrained_layout=True, squeeze=False, sharex=True,
    )
    t = np.linspace(0, 100, I.NORMALIZED_CYCLE_POINTS)
    for i, skill in enumerate(I.SKILLS):
        for j, hand in enumerate(I.HANDS):
            panels = axes[i, j * len(labels):(j + 1) * len(labels)]
            for k, label in enumerate(labels):
                ax = panels[k]
                cycles = _condition(tables[label], subject, skill, day, part, hand)
                y = I.cycle_trajectories(positions, runs, cycles)[:, :, 1]
                for curve in y:
                    ax.plot(t, curve, color=colors[k], alpha=0.12, lw=0.6)
                if len(y):
                    ax.plot(t, y.mean(axis=0), color=INK, lw=1.8)
                ax.set_title(f"{SKILL_LABELS[skill]}, {HAND_LABELS[hand].lower()}\n{label}")
                if k:
                    ax.sharey(panels[0])
                    ax.tick_params(labelleft=False)
            panels[0].set_ylabel("Hand Y − root Y\n(run mean removed)")
    for ax in axes[-1]:
        ax.set_xlabel("Cycle time (%)")
    handles = [Line2D([], [], color=c, lw=0.8, label=f"One cycle: {lab}") for lab, c in zip(labels, colors)]
    handles.append(Line2D([], [], color=INK, lw=1.8, label="Mean cycle"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))
    fig.suptitle(f"Hand-Y cycles: subject {subject}, day {day}, {PART_LABELS[part].lower()} of session")
    return save(fig, stem)


# ---------------------------------------------------------------------------
# Start-lag distributions
# ---------------------------------------------------------------------------

def start_lag_histograms(table, stem, title):
    """Stacked histogram of start lags per skill, split into partial and complete cycles."""
    bins = np.linspace(-180, 180, 37)
    fig, axes = plt.subplots(1, len(I.SKILLS), figsize=(11, 3.0), constrained_layout=True, sharey=False)
    for ax, skill in zip(axes, I.SKILLS):
        sub = table[table["skill"].eq(skill) & table["lag"].notna()]
        lag = np.degrees(sub["lag"].to_numpy())
        complete = sub["complete"].to_numpy()
        ax.hist([lag[complete], lag[~complete]], bins=bins, stacked=True,
                color=[NEUTRAL, PARTIAL], edgecolor="white", linewidth=0.4)
        ax.axvspan(-45, 45, color="#000000", alpha=0.05, lw=0)
        off = np.abs(lag) > 45
        ax.set_title(f"{SKILL_LABELS[skill]}: {100 * off.mean():.0f}% of {len(lag)} cycles start\nmore than 45° from the common start")
        ax.set_xlabel("Start lag (degrees of a cycle)")
        ax.set_xticks(np.arange(-180, 181, 90))
    axes[0].set_ylabel("Cycles")
    fig.legend(handles=[
        Patch(color=NEUTRAL, label="Complete cycle"),
        Patch(color=PARTIAL, label="Partial cycle at a run edge"),
        Patch(color="#000000", alpha=0.08, label="Within ±45°"),
    ], loc="outside lower center", ncol=3)
    fig.suptitle(title)
    return save(fig, stem)


# ---------------------------------------------------------------------------
# Single-run timelines
# ---------------------------------------------------------------------------

def _timeline(ax, fit, color, *, show_filtered=True, mark_partial=True):
    t = np.arange(len(fit["ref"])) / I.FS_HZ
    ax.plot(t, fit["ref"], color=NEUTRAL, lw=1.0, label="Reference (score on dominant axis)")
    if show_filtered:
        ax.plot(t, fit["filtered"], color=INK, lw=1.0, label="Band-passed reference")
    cycles = fit["cycles"]
    for row in cycles.itertuples():
        if mark_partial and not row.complete:
            ax.axvspan(row.sample_start / I.FS_HZ, row.sample_stop / I.FS_HZ, color=PARTIAL, alpha=0.25, lw=0)
    starts = cycles["sample_start"].to_numpy()
    starts = starts[starts > 0]
    for s in starts:
        ax.axvline(s / I.FS_HZ, color=color, lw=1.0, alpha=0.9)
    ax.set_xlim(t[0], t[-1])
    ax.margins(y=0.15)


def partial_cycles_example(fit, table, stem, title):
    """One run with its partial edge cycles shaded, plus durations of partial vs complete cycles."""
    fig, (ax, ax_hist) = plt.subplots(
        1, 2, figsize=(11, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [2.2, 1]}
    )
    _timeline(ax, fit, CURRENT, show_filtered=False)
    ax.set_xlabel("Time in run (s)")
    ax.set_ylabel("Reference")
    ax.set_title(title)
    ax.legend(handles=[
        Line2D([], [], color=NEUTRAL, lw=1.0, label="Reference (score on dominant axis)"),
        Line2D([], [], color=CURRENT, lw=1.0, label="Cycle start (before the fix)"),
        Patch(color=PARTIAL, alpha=0.35, label="Partial cycle"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

    bins = np.linspace(0, 2.5, 26)
    ratio = table["duration_ratio"].to_numpy()
    complete = table["complete"].to_numpy()
    ax_hist.hist(ratio[complete], bins=bins, color=NEUTRAL, alpha=0.9, density=True, label="Complete")
    ax_hist.hist(ratio[~complete], bins=bins, color=PARTIAL, alpha=0.75, density=True, label="Partial")
    ax_hist.set_xlabel("Cycle duration / median duration in its run")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title(f"{100 * (~complete).mean():.0f}% of all cycles are partial")
    ax_hist.legend()
    return save(fig, stem)


def anchor_example(fit_current, fit_proposed, stem, title):
    """Cycle starts before and after the fix on one run whose anchor landed off the peak."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.6), constrained_layout=True, sharex=True, sharey=True)
    _timeline(axes[0], fit_current, CURRENT, mark_partial=False)
    _timeline(axes[1], fit_proposed, PROPOSED, mark_partial=False)
    from scipy.signal import find_peaks
    anchor = find_peaks(fit_current["filtered"])[0][0]
    axes[0].scatter([anchor / I.FS_HZ], [fit_current["filtered"][anchor]], s=70, facecolor="white",
                    edgecolor=CURRENT, linewidths=1.8, zorder=5)
    axes[0].annotate(
        "anchor: first local peak\nof the band-passed reference",
        (anchor / I.FS_HZ, fit_current["filtered"][anchor]), xytext=(18, 22), textcoords="offset points",
        arrowprops=dict(arrowstyle="-", color=INK, lw=0.8), fontsize=8,
    )
    axes[0].set_title(
        f"Before: phase zero fixed once, at the anchor "
        f"(anchor is {np.degrees(fit_current['anchor_error']):+.0f}° from a true peak)"
    )
    axes[1].set_title("After: phase zero at the peak of the band-passed reference in every cycle")
    axes[1].set_xlabel("Time in run (s)")
    for ax in axes:
        ax.set_ylabel("Reference")
    fig.legend(handles=[
        Line2D([], [], color=NEUTRAL, lw=1.0, label="Reference (score on dominant axis)"),
        Line2D([], [], color=INK, lw=1.0, label="Band-passed reference"),
        Line2D([], [], color=CURRENT, lw=1.0, label="Cycle start, before"),
        Line2D([], [], color=PROPOSED, lw=1.0, label="Cycle start, after"),
    ], loc="outside lower center", ncol=4)
    fig.suptitle(title)
    return save(fig, stem)


def anchor_statistics(run_table, stem):
    """Anchor error across runs, and how well it predicts each run's mean start lag."""
    fig, (ax_hist, ax_scatter) = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    error = run_table["anchor_error_deg"].to_numpy()
    ax_hist.hist(error, bins=np.linspace(-180, 180, 37), color=CURRENT, edgecolor="white", linewidth=0.4)
    ax_hist.set_xlabel("Anchor error (degrees): phase of the anchor sample\nrelative to a true peak of the band-passed reference")
    ax_hist.set_ylabel("Runs")
    ax_hist.set_xticks(np.arange(-180, 181, 90))
    ax_hist.set_title(
        f"{100 * (np.abs(error) > 45).mean():.0f}% of {len(error)} runs are anchored\nmore than 45° away from a peak"
    )
    for skill in I.SKILLS:
        sub = run_table[run_table["skill"].eq(skill)]
        ax_scatter.scatter(sub["anchor_error_deg"], sub["run_lag_deg"], s=12, alpha=0.7,
                           color=SKILL_COLORS[skill], marker=SKILL_MARKERS[skill],
                           linewidths=0, label=SKILL_LABELS[skill])
    ax_scatter.plot([-180, 180], [-180, 180], color=INK, lw=0.8, ls="--", label="Equal")
    ax_scatter.set_xlabel("Anchor error (degrees)")
    ax_scatter.set_ylabel("Mean start lag of the run's\ncomplete cycles (degrees)")
    ax_scatter.set_xticks(np.arange(-180, 181, 90))
    ax_scatter.set_yticks(np.arange(-180, 181, 90))
    ax_scatter.set_title("A run's cycles are all shifted by its anchor error")
    ax_scatter.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    return save(fig, stem)


# ---------------------------------------------------------------------------
# Frequency
# ---------------------------------------------------------------------------

def frequency_scatter(freq_table, stem):
    """Per-run dominant frequency against the frequency shared by the run's condition."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), constrained_layout=True, sharex=True, sharey=True)
    for ax, column, label in zip(
        axes, ["f_periodogram", "f_robust"],
        ["Before: largest periodogram bin", "After: detrended, ≥ 3 cycles in the run"],
    ):
        for skill in I.SKILLS:
            sub = freq_table[freq_table["skill"].eq(skill)]
            ax.scatter(sub["f_shared"], sub[column], s=12, alpha=0.6, color=SKILL_COLORS[skill],
                       marker=SKILL_MARKERS[skill], linewidths=0, label=SKILL_LABELS[skill])
        line = np.array([0.5, 3.0])
        ax.plot(line, line, color=INK, lw=0.8, ls="--", label="Equal")
        ax.fill_between(line, 0.6 * line, 1.7 * line, color="#000000", alpha=0.05, lw=0,
                        label="Within 0.6× to 1.7× of pooled")
        ratio = freq_table[column] / freq_table["f_shared"]
        outside = (ratio < 0.6) | (ratio > 1.7)
        ax.set_title(f"{label}\n{outside.sum()} of {len(ratio)} run fits outside the band")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency pooled over the condition's runs (Hz)")
        ax.set_xticks([0.5, 1, 2, 3], ["0.5", "1", "2", "3"])
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 4], ["0.1", "0.2", "0.5", "1", "2", "4"])
        ax.minorticks_off()
    axes[0].set_ylabel("Frequency estimated from one run (Hz)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=len(labels))
    return save(fig, stem)


def frequency_example(fit_current, fit_proposed, stem, title):
    """A run where the periodogram picked a slow drift instead of the movement."""
    fig, axes = plt.subplots(2, 1, figsize=(11, 4.4), constrained_layout=True, sharex=True, sharey=True)
    _timeline(axes[0], fit_current, CURRENT)
    _timeline(axes[1], fit_proposed, PROPOSED)
    axes[0].set_title(
        f"Before: dominant frequency {fit_current['f0']:.2f} Hz, "
        f"complete cycles: {int(fit_current['cycles']['complete'].sum())}"
    )
    axes[1].set_title(
        f"After: dominant frequency {fit_proposed['f0']:.2f} Hz, "
        f"complete cycles: {int(fit_proposed['cycles']['complete'].sum())}"
    )
    axes[1].set_xlabel("Time in run (s)")
    for ax in axes:
        ax.set_ylabel("Reference")
    fig.legend(handles=[
        Line2D([], [], color=NEUTRAL, lw=1.0, label="Reference (score on dominant axis)"),
        Line2D([], [], color=INK, lw=1.0, label="Band-passed reference"),
        Line2D([], [], color=CURRENT, lw=1.0, label="Cycle start, before"),
        Line2D([], [], color=PROPOSED, lw=1.0, label="Cycle start, after"),
        Patch(color=PARTIAL, alpha=0.35, label="Partial cycle"),
    ], loc="outside lower center", ncol=5)
    fig.suptitle(title)
    return save(fig, stem)


# ---------------------------------------------------------------------------
# Step-by-step summary
# ---------------------------------------------------------------------------

def fix_steps(metrics, stem):
    """Three metrics against the five variants; one line per skill, one row per subject."""
    panels = [
        ("misaligned", "Cycle starts more than 45°\nfrom the common start (%)", 100.0),
        ("lag_sd_deg", "Spread of cycle starts\n(circular SD, degrees)", 1.0),
        ("shape_rms", "Hand-Y cycle variability\n(RMS deviation / amplitude)", 1.0),
    ]
    subjects = sorted(metrics["subject"].unique())
    variants = list(I.VARIANTS)
    fig, axes = plt.subplots(len(subjects), len(panels), figsize=(11, 2.9 * len(subjects)),
                             constrained_layout=True, squeeze=False, sharex=True)
    x = np.arange(len(variants))
    for i, subject in enumerate(subjects):
        for j, (column, label, scale) in enumerate(panels):
            ax = axes[i, j]
            for skill in I.SKILLS:
                sub = metrics[metrics["subject"].eq(subject) & metrics["skill"].eq(skill)]
                values = sub.groupby("variant")[column].median().reindex(variants) * scale
                ax.plot(x, values, color=SKILL_COLORS[skill], marker=SKILL_MARKERS[skill], lw=1.6,
                        markersize=5, label=SKILL_LABELS[skill])
            ax.set_ylim(bottom=0)
            ax.set_xticks(x, [v.split(".")[0] for v in variants])
            ax.set_title(f"Subject {subject}" if j == 1 else "")
            ax.set_ylabel(label)
    axes[0, 0].legend(loc="upper right")
    key = "     ".join(variants)
    fig.supxlabel(key, fontsize=8)
    fig.suptitle("Effect of each change (median over conditions)")
    return save(fig, stem)
