"""
Shared plotting style, adapted from Noam's Labeled_data_analyses.ipynb so
figures in this analysis read as an extension of hers rather than a
different visual language. Simplified: writes plain PNGs to a local
`figures/` dir instead of her full cache/tables/qc/report output tree.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FS_HZ = 60.0

SKILL_LABELS = {"walk": "Walking", "jump": "Jumping", "climb": "Climbing"}
SKILL_COLORS = {"walk": "#1b9e77", "jump": "#d95f02", "climb": "#7570b3"}
HAND_COLORS = {"r_hand": "#D55E00", "l_hand": "#6A5ACD"}
HAND_LABELS = {"r_hand": "Right hand", "l_hand": "Left hand"}
DAY_LABELS = {1: "Day 1", 3: "Day 3"}
PART_LABELS = {"s": "start", "m": "middle", "e": "end"}
DAY_MARKERS = {1: "o", 3: "s"}
HAND_MARKERS = {"r_hand": "o", "l_hand": "^"}
DAY_LINESTYLES = {1: "--", 3: "-"}

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def build_subject_color_table(subjects=("002",)):
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(subjects))]
    return pd.DataFrame({"canonical_subject": list(subjects), "color": colors})


SUBJECT_COLOR_TABLE = build_subject_color_table()
SUBJECT_COLOR_MAP = SUBJECT_COLOR_TABLE.set_index("canonical_subject")["color"].to_dict()


def get_subject_color(subject):
    subject = f"{int(float(subject)):03d}"
    if subject not in SUBJECT_COLOR_MAP:
        SUBJECT_COLOR_MAP[subject] = plt.get_cmap("tab10")(len(SUBJECT_COLOR_MAP) % 10)
    return SUBJECT_COLOR_MAP[subject]


def save_figure(fig, stem, pdf=False):
    """Save fig as figures/<stem>.png (300dpi); optionally also a PDF."""
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    saved = {"png": png_path}
    if pdf:
        pdf_path = FIGURES_DIR / f"{stem}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        saved["pdf"] = pdf_path
    return saved


def equal_limits(*series, padding_fraction=0.08, positive=False, log_scale=False):
    values = np.concatenate([np.asarray(s, dtype=float).ravel() for s in series])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    lo, hi = float(values.min()), float(values.max())
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1.0)
    pad = span * padding_fraction
    lo, hi = lo - pad, hi + pad
    if positive:
        lo = max(lo, 0.0)
    return (lo, hi)
