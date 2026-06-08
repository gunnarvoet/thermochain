"""Shared drawing helpers and palette for the manuscript schematic figures.

Both ``pipeline_schematic.py`` (high-level processing-toolbox flow) and
``drift_procedure_schematic.py`` (detailed CvHG16 drift procedure) draw
rounded, content-sized boxes connected by arrows in an inches-based
coordinate system (1 data unit = 1 inch). This module factors out the box
geometry, the arrow style, the text-measurement helper, and the shared
colour/font constants so the two scripts stay visually consistent without
duplicating code.

The font-size and box-style defaults match ``pipeline_schematic.py`` (the
canonical layout). ``pipeline_schematic_horizontal.py`` keeps its own
slightly different defaults and remains self-contained.

The backend is forced to Agg here so the helpers render headlessly and
reproducibly regardless of which script imports them.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --- palette -----------------------------------------------------------------
TEXT = "#1a1a1a"
C_RAW = "#e8e8e8"
E_RAW = "#7a7a7a"
# blue progression for the processing levels
LEVELS = {
    "L0": ("#dce6f2", "#2f5c8a"),
    "L1": ("#c5d8ec", "#2f5c8a"),
    "gL1": ("#aecae6", "#2f5c8a"),
    "L2": ("#8ab4dd", "#2f5c8a"),
}
C_FINAL, E_FINAL = "#bfe3c6", "#2e7d4f"  # final product (green)
C_DRIFT, E_DRIFT = "#fbe2c0", "#c47f1d"  # drift-fit branch (amber)
C_INPUT, E_INPUT = "#f4f1ea", "#8a8076"  # data tables / inputs
C_CFG, E_CFG = "#f0eef6", "#6f6790"  # configuration rail

A_MAIN = "#444444"  # main-chain arrows
A_CFG = "#9a93b5"  # config -> stage arrows

# --- font sizes (pt) ---------------------------------------------------------
FS_TITLE = 8.0       # level label / box heading (bold)
FS_SUB = 6.5         # box description
FS_INPUT = 7.0       # input-table label (bold)
FS_DRIFT_T, FS_DRIFT_S = 7.5, 6.0
FS_CFG_T, FS_CFG_S = 7.2, 6.0

# --- box style (inches) ------------------------------------------------------
PAD_X = 0.13         # horizontal text padding inside a box
PAD_Y = 0.10         # vertical text padding inside a box
GAP_TS = 0.05        # gap between a title and its subtitle
ROUND = 0.06         # corner rounding radius


def _measure(fig, ax, s, fs, weight="normal", linespacing=1.2):
    """Return (width, height) of a text string in inches (dpi-independent)."""
    t = ax.text(0, 0, s, fontsize=fs, fontweight=weight, ha="center",
                va="center", linespacing=linespacing)
    fig.canvas.draw()
    bb = t.get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi, bb.height / fig.dpi


def draw_box(ax, cx, cy, w, h, title, sub, fc, ec, th, sh, *,
             title_fs=FS_TITLE, sub_fs=FS_SUB, dashed=False, lw=1.1):
    """Draw a rounded box of outer size w x h and place its title/subtitle."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        linestyle="--" if dashed else "-", zorder=2,
    ))
    if sub:
        content_h = th + GAP_TS + sh
        top = cy + content_h / 2
        ax.text(cx, top - th / 2, title, ha="center", va="center",
                fontsize=title_fs, fontweight="bold", color=TEXT, zorder=3)
        ax.text(cx, top - th - GAP_TS - sh / 2, sub, ha="center", va="center",
                fontsize=sub_fs, color=TEXT, zorder=3, linespacing=1.2)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_fs,
                fontweight="bold", color=TEXT, zorder=3)


def arrow(ax, x0, y0, x1, y1, color=A_MAIN, lw=1.3, ls="-", alpha=1.0, mut=9):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=mut,
        color=color, linewidth=lw, linestyle=ls, alpha=alpha,
        shrinkA=0, shrinkB=0, zorder=1,
    ))


__all__ = [
    "plt", "FancyArrowPatch", "FancyBboxPatch",
    "TEXT", "C_RAW", "E_RAW", "LEVELS", "C_FINAL", "E_FINAL",
    "C_DRIFT", "E_DRIFT", "C_INPUT", "E_INPUT", "C_CFG", "E_CFG",
    "A_MAIN", "A_CFG",
    "FS_TITLE", "FS_SUB", "FS_INPUT", "FS_DRIFT_T", "FS_DRIFT_S",
    "FS_CFG_T", "FS_CFG_S",
    "PAD_X", "PAD_Y", "GAP_TS", "ROUND",
    "_measure", "draw_box", "arrow",
]
