#!/usr/bin/env python
"""Generate the CvHG16 drift-procedure schematic.

One generator, two layouts:

* ``build_horizontal()`` -> wide PDF for the manuscript
  (``schematics/pdf/drift_procedure_schematic.pdf``);
* ``build_vertical()`` -> portrait SVG embedded in the docs
  (``src/thermochain/_static/drift_procedure_schematic.svg``).

The vertical layout lays the two refinement passes out as side-by-side
columns (first pass | second pass) joined by a loop-back arrow, making the
two-step structure explicit. Run via ``make schematics`` or::

    uv run python schematics/drift_procedure_schematic.py
"""
from __future__ import annotations

from pathlib import Path

from _drift_procedure import (
    C_PW, C_TS, CHAIN, E_PW, E_TS, LOOP_LABEL, OUTPUT_KEY, PASS1_LABEL,
    PASS2_LABEL, PW_LABEL, TS_LABEL,
)
from _schematic import (
    A_MAIN, FancyArrowPatch, FancyBboxPatch, GAP_TS, PAD_X, ROUND, _measure,
    arrow, draw_box, plt,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PDF_OUT = HERE / "pdf" / "drift_procedure_schematic.pdf"
SVG_OUT = REPO / "src" / "thermochain" / "_static" / "drift_procedure_schematic.svg"


def build_horizontal():
    """Wide drift schematic: per-window block left + two-row pass grid right."""
    # compact labels (title, subtitle) for the tight wide layout
    LABELS = {
        "gL1": ("Gridded L1", "full record"),
        "bg": ("Windowed\nbackground fit", "spline · per window"),
        "off1": ("First-guess\noffsets", "residual from bg"),
        "sig": ("±3σ rejection", "drop time-series\noutliers"),
        "sfc1": ("Shared component", "neighbour triplet"),
        "off2": ("Second-guess\noffsets", "− shared comp."),
        "interim": ("Interim drift fit", "per sensor"),
        "two": ("Recompute\nshared component", "from detrended\noffsets"),
        "clean": ("Cleaned offsets", "orig. − recomputed"),
        "model": ("Final drift model", "lin/exp · R²"),
        "L2": ("Subtract → L2", "drift-corrected"),
    }

    # geometry (inches)
    VPAD = 0.08
    GH = 0.40            # horizontal gap between grid columns
    GV = 0.52            # vertical gap between the two pass rows (room for loop-back)
    MARGIN = 0.16
    GAP_LR = 0.46        # gap between the per-window block and the grid
    PANEL_PAD = 0.10
    TOPLAB = 0.24
    RLAB = 0.30          # right gutter for the rotated pass labels

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })

    colors = {k: (fc, ec) for k, _t, _s, fc, ec in CHAIN}

    # ---- measurement pass --------------------------------------------------
    scratch = plt.figure()
    sax = scratch.add_subplot()
    meas = {}  # key -> (w, h, th, sh)
    for key, (title, sub) in LABELS.items():
        tw, th = _measure(scratch, sax, title, 8.0, "bold")
        sw, sh = _measure(scratch, sax, sub, 6.5)
        meas[key] = (max(tw, sw), th + GAP_TS + sh + 2 * VPAD, th, sh)
    plt.close(scratch)

    Wc = max(m[0] for m in meas.values()) + 2 * PAD_X

    # roles ------------------------------------------------------------------
    left_stack = ["gL1", "bg", "off1"]          # input + per-window, top->down
    row1 = ["sig", "sfc1", "off2", "interim"]   # first pass  (cols g0..g3)
    row2 = ["two", "clean", "model"]            # second pass (cols g1..g3)

    # ---- x layout ----------------------------------------------------------
    left_cx = MARGIN + Wc / 2
    grid_left = MARGIN + Wc + GAP_LR
    gx = [grid_left + Wc / 2 + j * (Wc + GH) for j in range(4)]
    grid_right = gx[-1] + Wc / 2
    W = grid_right + PANEL_PAD + RLAB + MARGIN

    # ---- y layout ----------------------------------------------------------
    hA = max(meas[k][1] for k in row1)
    hB = max(meas[k][1] for k in row2)
    hL2 = meas["L2"][1]
    grid_h = hA + GV + hB + GV + hL2            # rows + L2 below final fit
    left_h = sum(meas[k][1] for k in left_stack) + GV * (len(left_stack) - 1)
    block_h = max(grid_h, left_h)
    H = MARGIN + TOPLAB + block_h + MARGIN

    top = H - MARGIN - TOPLAB
    yA = top - hA / 2
    yB = yA - hA / 2 - GV - hB / 2
    yL2 = yB - hB / 2 - GV - hL2 / 2

    xc, yc, hh = {}, {}, {}
    for j, k in enumerate(row1):
        xc[k], yc[k], hh[k] = gx[j], yA, hA
    for j, k in enumerate(row2):                 # placed under g1, g2, g3
        xc[k], yc[k], hh[k] = gx[j + 1], yB, hB
    xc["L2"], yc["L2"], hh["L2"] = gx[3], yL2, hL2

    # left stack centred vertically against the right block
    cur = top - (block_h - left_h) / 2
    for k in left_stack:
        h = meas[k][1]
        yc[k], hh[k] = cur - h / 2, h
        xc[k] = left_cx
        cur -= h + GV

    def t_(k):
        return yc[k] + hh[k] / 2

    def b_(k):
        return yc[k] - hh[k] / 2

    # ---- draw --------------------------------------------------------------
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # phase panels + labels
    pw_top = t_("bg") + PANEL_PAD
    pw_bot = b_("off1") - PANEL_PAD
    ax.add_patch(FancyBboxPatch(
        (left_cx - Wc / 2 - PANEL_PAD, pw_bot), Wc + 2 * PANEL_PAD,
        pw_top - pw_bot, boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor=C_PW, edgecolor=E_PW, linewidth=1.0, zorder=0,
    ))
    ax.text(left_cx, pw_top + 0.09, PW_LABEL, ha="center", va="bottom",
            fontsize=6.5, color=E_PW, zorder=3)

    ts_left = gx[0] - Wc / 2 - PANEL_PAD
    ts_right = gx[-1] + Wc / 2 + PANEL_PAD
    ts_top = yA + hA / 2 + PANEL_PAD
    ts_bot = yB - hB / 2 - PANEL_PAD
    ax.add_patch(FancyBboxPatch(
        (ts_left, ts_bot), ts_right - ts_left, ts_top - ts_bot,
        boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor=C_TS, edgecolor=E_TS, linewidth=1.0, zorder=0,
    ))
    ax.text((ts_left + ts_right) / 2, ts_top + 0.09, TS_LABEL, ha="center",
            va="bottom", fontsize=6.5, color=E_TS, zorder=3)
    # pass labels in the right gutter, aligned with each grid row
    ax.text(ts_right + 0.13, yA, PASS1_LABEL, rotation=90, ha="center",
            va="center", fontsize=6.5, color=E_TS, zorder=3)
    ax.text(ts_right + 0.13, yB, PASS2_LABEL, rotation=90, ha="center",
            va="center", fontsize=6.5, color=E_TS, zorder=3)

    # boxes
    for key, (title, sub) in LABELS.items():
        _, h, th, sh = meas[key]
        fc, ec = colors[key]
        lw = 1.8 if key in ("two", "L2") else 1.1
        draw_box(ax, xc[key], yc[key], Wc, h, title, sub, fc, ec, th, sh, lw=lw)

    # arrows: left stack (down) + hand-off into the grid
    for a, b in zip(left_stack[:-1], left_stack[1:]):
        arrow(ax, xc[a], b_(a), xc[b], t_(b))
    arrow(ax, xc["off1"] + Wc / 2, yc["off1"], xc["sig"] - Wc / 2, yc["sig"])
    # row 1 (first pass, left -> right)
    for a, b in zip(row1[:-1], row1[1:]):
        arrow(ax, xc[a] + Wc / 2, yc[a], xc[b] - Wc / 2, yc[b])
    # row 2 (second pass, left -> right)
    for a, b in zip(row2[:-1], row2[1:]):
        arrow(ax, xc[a] + Wc / 2, yc[a], xc[b] - Wc / 2, yc[b])
    # final model -> L2 (down)
    arrow(ax, xc["model"], b_("model"), xc["L2"], t_("L2"))

    # loop-back: interim fit -> (detrend) -> recompute, the second pass
    ax.add_patch(FancyArrowPatch(
        (xc["interim"], b_("interim")), (xc["two"], t_("two")),
        connectionstyle=f"arc,angleA=-90,angleB=90,armA=16,armB=16,rad={ROUND}",
        arrowstyle="-|>", mutation_scale=10, color=A_MAIN, linewidth=1.3,
        shrinkA=1, shrinkB=1, zorder=1,
    ))
    ax.text((xc["two"] + xc["interim"]) / 2, (yA + yB) / 2, LOOP_LABEL,
            ha="center", va="center", fontsize=6.0, color=A_MAIN, zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    return fig


def build_vertical():
    """Portrait drift schematic: two side-by-side passes + loop-back arrow."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })

    # geometry (inches)
    VPAD = 0.08
    GV = 0.16            # vertical gap between boxes within a column
    GH = 0.55            # horizontal gap between the two pass columns
    MARGIN = 0.16
    LLAB = 0.34          # left gutter for the rotated phase labels
    PANEL_PAD = 0.10     # phase-panel horizontal padding
    PANEL_PV = 0.06      # phase-panel vertical padding
    G_UP = 0.36          # gap: upper stack -> pass-label band
    TOPLAB = 0.22        # height of the "first/second pass" label band
    G_DOWN = 0.32        # gap: grid -> L2

    # ---- measurement pass --------------------------------------------------
    scratch = plt.figure()
    sax = scratch.add_subplot()
    meas = {}  # key -> (content_w, box_h, th, sh)
    for key, title, sub, fc, ec in CHAIN:
        tw, th = _measure(scratch, sax, title, 8.0, "bold")
        sw, sh = _measure(scratch, sax, sub, 6.5)
        meas[key] = (max(tw, sw), th + GAP_TS + sh + 2 * VPAD, th, sh)
    plt.close(scratch)

    Wc = max(m[0] for m in meas.values()) + 2 * PAD_X   # uniform box width

    upper = ["gL1", "bg", "off1", "sig"]   # single centred column on top
    col1 = ["sfc1", "off2", "interim"]     # first pass (left column)
    col2 = ["two", "clean", "model"]       # second pass (right column)
    row_h = [max(meas[col1[i]][1], meas[col2[i]][1]) for i in range(3)]

    # ---- x layout ----------------------------------------------------------
    col1_cx = MARGIN + LLAB + PANEL_PAD + Wc / 2
    col2_cx = col1_cx + Wc + GH
    center_x = (col1_cx + col2_cx) / 2
    gutter_x = MARGIN + LLAB * 0.45
    W = col2_cx + Wc / 2 + PANEL_PAD + MARGIN

    # ---- y layout (top-down) ----------------------------------------------
    H_upper = sum(meas[k][1] for k in upper) + GV * (len(upper) - 1)
    grid_h = sum(row_h) + GV * 2
    hL2 = meas["L2"][1]
    H = MARGIN + H_upper + G_UP + TOPLAB + grid_h + G_DOWN + hL2 + MARGIN

    yc, hh = {}, {}
    cur = H - MARGIN
    for k in upper:
        h = meas[k][1]
        yc[k], hh[k] = cur - h / 2, h
        cur -= h + GV
    cur += GV
    cur -= G_UP
    band_cy = cur - TOPLAB / 2
    cur -= TOPLAB
    grid_top = cur
    for i in range(3):
        h = row_h[i]
        y = cur - h / 2
        for col in (col1, col2):
            k = col[i]
            yc[k], hh[k] = y, meas[k][1]
        cur -= h + GV
    cur += GV
    grid_bot = cur
    cur -= G_DOWN
    yc["L2"], hh["L2"] = cur - hL2 / 2, hL2

    def t_(k):
        return yc[k] + hh[k] / 2

    def b_(k):
        return yc[k] - hh[k] / 2

    x_of = {k: center_x for k in upper}
    for i in range(3):
        x_of[col1[i]] = col1_cx
        x_of[col2[i]] = col2_cx
    x_of["L2"] = col2_cx

    # ---- draw --------------------------------------------------------------
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    # per-window phase panel (wraps bg + off1 in the upper stack)
    pw_top = t_("bg") + PANEL_PV
    pw_bot = b_("off1") - PANEL_PV
    ax.add_patch(FancyBboxPatch(
        (center_x - Wc / 2 - PANEL_PAD, pw_bot), Wc + 2 * PANEL_PAD,
        pw_top - pw_bot, boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor=C_PW, edgecolor=E_PW, linewidth=1.0, zorder=0,
    ))
    ax.text(gutter_x, (pw_top + pw_bot) / 2, PW_LABEL, rotation=90,
            ha="center", va="center", fontsize=6.5, color=E_PW, zorder=3)

    # offset-time-series phase panel (wraps the two pass columns)
    ts_left = col1_cx - Wc / 2 - PANEL_PAD
    ts_right = col2_cx + Wc / 2 + PANEL_PAD
    ts_top = grid_top + PANEL_PV
    ts_bot = grid_bot - PANEL_PV
    ax.add_patch(FancyBboxPatch(
        (ts_left, ts_bot), ts_right - ts_left, ts_top - ts_bot,
        boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor=C_TS, edgecolor=E_TS, linewidth=1.0, zorder=0,
    ))
    ax.text(gutter_x, (ts_top + ts_bot) / 2, TS_LABEL, rotation=90,
            ha="center", va="center", fontsize=6.5, color=E_TS, zorder=3)

    # pass labels above each column
    ax.text(col1_cx, band_cy, PASS1_LABEL, ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=E_TS, zorder=3)
    ax.text(col2_cx, band_cy, PASS2_LABEL, ha="center", va="center",
            fontsize=7.0, fontweight="bold", color=E_TS, zorder=3)

    # boxes
    for key, title, sub, fc, ec in CHAIN:
        _, _, th, sh = meas[key]
        lw = 1.8 if key in ("two", OUTPUT_KEY) else 1.1
        draw_box(ax, x_of[key], yc[key], Wc, hh[key], title, sub, fc, ec,
                 th, sh, lw=lw)

    # arrows: upper stack (top -> down)
    for a, b in zip(upper[:-1], upper[1:]):
        arrow(ax, center_x, b_(a), center_x, t_(b))
    # sig -> first pass (down-left into col1)
    arrow(ax, center_x, b_("sig"), col1_cx, t_("sfc1"))
    # first-pass column (down)
    for a, b in zip(col1[:-1], col1[1:]):
        arrow(ax, col1_cx, b_(a), col1_cx, t_(b))
    # second-pass column (down)
    for a, b in zip(col2[:-1], col2[1:]):
        arrow(ax, col2_cx, b_(a), col2_cx, t_(b))
    # final model -> L2 (down)
    arrow(ax, col2_cx, b_("model"), col2_cx, t_("L2"))

    # loop-back: interim fit (end of pass 1) -> recompute shared comp (pass 2)
    ax.add_patch(FancyArrowPatch(
        (col1_cx + Wc / 2, yc["interim"]), (col2_cx, t_("two")),
        connectionstyle=f"arc,angleA=0,angleB=90,armA=18,armB=14,rad={ROUND}",
        arrowstyle="-|>", mutation_scale=10, color=A_MAIN, linewidth=1.3,
        shrinkA=1, shrinkB=1, zorder=1,
    ))
    ax.text((col1_cx + col2_cx) / 2, (yc["interim"] + t_("two")) / 2, LOOP_LABEL,
            ha="center", va="center", fontsize=6.0, color=A_MAIN, zorder=3,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    return fig


def main():
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    SVG_OUT.parent.mkdir(parents=True, exist_ok=True)

    fig_h = build_horizontal()
    w, h = fig_h.get_size_inches()
    fig_h.savefig(PDF_OUT, format="pdf")
    plt.close(fig_h)
    print(f"wrote {PDF_OUT}  ({w:.2f} x {h:.2f} in)")

    fig_v = build_vertical()
    w, h = fig_v.get_size_inches()
    fig_v.savefig(SVG_OUT, format="svg")
    plt.close(fig_v)
    print(f"wrote {SVG_OUT}  ({w:.2f} x {h:.2f} in)")


if __name__ == "__main__":
    main()
