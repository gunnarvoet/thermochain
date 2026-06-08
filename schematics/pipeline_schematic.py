#!/usr/bin/env python
"""Generate the thermochain processing-pipeline schematic.

One generator, two layouts of the same content:

* ``build_horizontal()`` -> wide PDF for the manuscript
  (``schematics/pdf/pipeline_schematic.pdf``);
* ``build_vertical()`` -> portrait SVG embedded in the docs
  (``src/thermochain/_static/pipeline_schematic.svg``).

Layout is content-driven (each label is measured and boxes are sized to fit),
geometry in inches (1 data unit = 1 inch). Run via ``make schematics`` or::

    uv run python schematics/pipeline_schematic.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / reproducible rendering
from matplotlib.patches import FancyBboxPatch

from _schematic import (
    A_CFG, C_CFG, C_DRIFT, C_FINAL, C_INPUT, C_RAW, E_CFG, E_DRIFT, E_FINAL,
    E_INPUT, E_RAW, FS_CFG_T, FS_DRIFT_S, FS_DRIFT_T, FS_INPUT, FS_TITLE,
    GAP_TS, LEVELS, PAD_X, PAD_Y, ROUND, TEXT, _measure, arrow, draw_box, plt,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
PDF_OUT = HERE / "pdf" / "pipeline_schematic.pdf"
SVG_OUT = REPO / "src" / "thermochain" / "_static" / "pipeline_schematic.svg"


def build_horizontal():
    """Build the wide (landscape) pipeline schematic for the manuscript PDF."""
    # --- local font/geometry constants (horizontal layout) -------------------
    FS_SUB = 6.3
    FS_CFG_S = 6.3
    GH = 0.40            # horizontal gap between chain boxes
    G_ABOVE = 0.34       # gap chain -> inputs/drift row (above)
    G_BELOW = 0.32       # gap chain -> config band (below)
    MARGIN = 0.15

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,
    })

    # content definitions (subtitles wrapped narrow to keep the row compact) --
    chain = [
        ("raw", "Raw logger files", "RBR Solo · SBE56", C_RAW, E_RAW),
        ("L0", "L0", "Clock calibration\n(per sensor)", *LEVELS["L0"]),
        ("L1", "L1", "Cut to deployment\nwindow + in-situ\nCTD calibration",
         *LEVELS["L1"]),
        ("gL1", "Gridded L1", "Interpolate onto\ncommon depth–time\ngrid",
         *LEVELS["gL1"]),
        ("L2", "L2", "Subtract drift\n(per sensor)", *LEVELS["L2"]),
        ("gL2", "Gridded L2", "Drift-corrected\ntemperature product",
         C_FINAL, E_FINAL),
    ]
    inputs = [("Sensor sheet", "L0"), ("Cal-stops table", "L1"),
              ("CTD cal casts", "L1")]
    cfg_title = "Configuration (per mooring)"
    cfg_items = ("data paths  ·  sensor & mooring layout  ·  segments  ·  "
                 "gridding  ·  calibration  ·  drift parameters")

    # ---- measurement pass --------------------------------------------------
    scratch = plt.figure()
    sax = scratch.add_subplot()

    chain_m = {}
    for key, title, sub, fc, ec in chain:
        tw, th = _measure(scratch, sax, title, FS_TITLE, "bold")
        sw, sh = _measure(scratch, sax, sub, FS_SUB)
        chain_m[key] = (max(tw, sw), th, sh, title, sub, fc, ec)
    Wc = max(m[0] for m in chain_m.values()) + 2 * PAD_X
    Hc = max(m[1] + GAP_TS + m[2] for m in chain_m.values()) + 2 * PAD_Y

    in_m = {}
    for label, _ in inputs:
        in_m[label] = _measure(scratch, sax, label, FS_INPUT, "bold")
    Wi = max(w for w, _ in in_m.values()) + 2 * PAD_X
    Hi = max(h for _, h in in_m.values()) + 2 * PAD_Y

    dtw, dth = _measure(scratch, sax, "Drift fit", FS_DRIFT_T, "bold")
    dsw, dsh = _measure(scratch, sax, "CvHG16\ndense segments", FS_DRIFT_S)
    Wd = max(dtw, dsw) + 2 * PAD_X
    Hd = dth + GAP_TS + dsh + 2 * PAD_Y

    ctw, cth = _measure(scratch, sax, cfg_title, FS_CFG_T, "bold")
    ciw, cih = _measure(scratch, sax, cfg_items, FS_CFG_S)
    Hband = cth + GAP_TS + cih + 2 * PAD_Y
    plt.close(scratch)

    # ---- horizontal layout (chain centres) ---------------------------------
    pitch = Wc + GH
    x0 = MARGIN + Wc / 2
    keys = [k for k, *_ in chain]
    xc = {k: x0 + i * pitch for i, k in enumerate(keys)}
    chain_left = MARGIN
    chain_right = xc[keys[-1]] + Wc / 2
    W = chain_right + MARGIN

    # ---- vertical layout ---------------------------------------------------
    H_above = max(Hi, Hd)
    H = MARGIN + H_above + G_ABOVE + Hc + G_BELOW + Hband + MARGIN
    y_above = H - MARGIN - H_above / 2
    y_chain = H - MARGIN - H_above - G_ABOVE - Hc / 2
    y_band = MARGIN + Hband / 2

    # ---- draw --------------------------------------------------------------
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    top = y_chain + Hc / 2
    bot = y_chain - Hc / 2

    for key in keys:
        cw_, th, sh, title, sub, fc, ec = chain_m[key]
        lw = 1.6 if key == "gL2" else 1.1
        draw_box(ax, xc[key], y_chain, Wc, Hc, title, sub, fc, ec, th, sh, lw=lw)

    # main-chain arrows (left -> right)
    for a, b in zip(keys[:-1], keys[1:]):
        arrow(ax, xc[a] + Wc / 2, y_chain, xc[b] - Wc / 2, y_chain)

    # --- configuration band (below; parameterizes every stage) --------------
    ax.add_patch(FancyBboxPatch(
        (chain_left, y_band - Hband / 2), chain_right - chain_left, Hband,
        boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor=C_CFG, edgecolor=E_CFG, linewidth=1.1, zorder=2,
    ))
    bcx = (chain_left + chain_right) / 2
    ax.text(bcx, y_band + (GAP_TS + cih) / 2, cfg_title, ha="center",
            va="center", fontsize=FS_CFG_T, fontweight="bold", color=TEXT,
            zorder=3)
    ax.text(bcx, y_band - (GAP_TS + cth) / 2, cfg_items, ha="center",
            va="center", fontsize=FS_CFG_S, color=TEXT, zorder=3)
    for k in ["L0", "L1", "gL1", "L2", "gL2"]:
        arrow(ax, xc[k], y_band + Hband / 2, xc[k], bot, color=A_CFG, lw=0.9,
              alpha=0.85, mut=7)

    # --- data inputs (above, feeding the early stages) ----------------------
    x_ss = xc["L0"]
    x_cs = xc["L1"] - 0.46
    x_ctd = xc["L1"] + 0.66
    in_x = {"Sensor sheet": x_ss, "Cal-stops table": x_cs, "CTD cal casts": x_ctd}
    in_aim = {"Sensor sheet": xc["L0"], "Cal-stops table": xc["L1"] - 0.10,
              "CTD cal casts": xc["L1"] + 0.10}
    for label, _ in inputs:
        tw, th = in_m[label]
        draw_box(ax, in_x[label], y_above, Wi, Hi, label, "", C_INPUT, E_INPUT,
                 th, 0.0, title_fs=FS_INPUT, dashed=True)
        arrow(ax, in_aim[label], y_above - Hi / 2, in_aim[label], top,
              color=E_INPUT, lw=1.0, mut=8)

    # --- drift-fit branch (above, between gridded L1 and L2) ----------------
    dcx = (xc["gL1"] + xc["L2"]) / 2
    draw_box(ax, dcx, y_above, Wd, Hd, "Drift fit", "CvHG16\ndense segments",
             C_DRIFT, E_DRIFT, dth, dsh, title_fs=FS_DRIFT_T, sub_fs=FS_DRIFT_S)
    arrow(ax, xc["gL1"], top, dcx - Wd / 2, y_above - Hd / 2 + 0.05,
          color=E_DRIFT, lw=1.2)
    arrow(ax, dcx + Wd / 2, y_above - Hd / 2 + 0.05, xc["L2"], top,
          color=E_DRIFT, lw=1.2)

    return fig


def build_vertical():
    """Build the portrait (tall) pipeline schematic for the docs SVG."""
    # --- local font/geometry constants (vertical layout) ---------------------
    GV = 0.30            # vertical gap between consecutive chain boxes
    G_RAIL = 0.46        # gap rail -> chain (room for config arrows)
    G_RIGHT = 0.46       # gap chain -> right column (room for input arrows)
    MARGIN = 0.14        # outer figure margin
    FS_SUB = 6.5
    FS_CFG_S = 6.0

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "pdf.fonttype": 42,  # embed TrueType so text stays selectable
    })

    # content definitions ----------------------------------------------------
    chain = [
        ("raw", "Raw logger files", "RBR Solo · SBE56", C_RAW, E_RAW),
        ("L0", "L0", "Clock calibration\n(per sensor)", *LEVELS["L0"]),
        ("L1", "L1", "Cut to deployment window\n+ in-situ CTD calibration",
         *LEVELS["L1"]),
        ("gL1", "Gridded L1", "Interpolate onto common\ndepth–time grid",
         *LEVELS["gL1"]),
        ("L2", "L2", "Subtract drift (per sensor)", *LEVELS["L2"]),
        ("gL2", "Gridded L2", "Drift-corrected\ntemperature product",
         C_FINAL, E_FINAL),
    ]
    inputs = [("Sensor sheet", "L0"), ("Cal-stops table", "L1"),
              ("CTD cal casts", "L1")]
    cfg_items = ("data paths\nsensor &\nmooring layout\nsegments\ngridding\n"
                 "calibration\ndrift params")

    # ---- measurement pass (scratch figure) ---------------------------------
    scratch = plt.figure()
    sax = scratch.add_subplot()

    chain_m = {}  # key -> (content_w, content_h, th, sh, title, sub, fc, ec)
    for key, title, sub, fc, ec in chain:
        tw, th = _measure(scratch, sax, title, FS_TITLE, "bold")
        sw, sh = _measure(scratch, sax, sub, FS_SUB)
        chain_m[key] = (max(tw, sw), th + GAP_TS + sh, th, sh, title, sub, fc, ec)
    Wc = max(m[0] for m in chain_m.values()) + 2 * PAD_X       # uniform width
    Hc = max(m[1] for m in chain_m.values()) + 2 * PAD_Y       # uniform height

    in_m = {}
    for label, tgt in inputs:
        tw, th = _measure(scratch, sax, label, FS_INPUT, "bold")
        in_m[label] = (tw, th)
    Wi = max(m[0] for m in in_m.values()) + 2 * PAD_X
    Hi = max(m[1] for m in in_m.values()) + 2 * PAD_Y

    dtw, dth = _measure(scratch, sax, "Drift fit", FS_DRIFT_T, "bold")
    dsw, dsh = _measure(scratch, sax, "CvHG16\ndense segments", FS_DRIFT_S)
    Wd = max(dtw, dsw) + 2 * PAD_X
    Hd = dth + GAP_TS + dsh + 2 * PAD_Y

    ctw, cth = _measure(scratch, sax, "Configuration\n(per mooring)",
                        FS_CFG_T, "bold")
    ciw, cih = _measure(scratch, sax, cfg_items, FS_CFG_S, linespacing=1.45)
    Wcfg = max(ctw, ciw) + 2 * PAD_X
    plt.close(scratch)

    # ---- horizontal layout -------------------------------------------------
    rail_left = MARGIN
    rail_right = rail_left + Wcfg
    chain_left = rail_right + G_RAIL
    cx = chain_left + Wc / 2
    chain_right = chain_left + Wc
    xr = chain_right + G_RIGHT            # right-column left edge
    W = xr + max(Wi, Wd) + MARGIN

    # ---- vertical layout (top-down depth, then flip to data y) -------------
    gap_after = {"raw": GV, "L0": GV, "L1": GV, "gL1": Hd + 0.42, "L2": GV}
    inner = sum(Hc for _ in chain) + sum(gap_after.values())
    H = MARGIN + inner + MARGIN

    yc = {}
    depth = MARGIN
    for key, *_ in chain:
        yc[key] = H - (depth + Hc / 2)
        depth += Hc + gap_after.get(key, 0.0)

    # ---- draw --------------------------------------------------------------
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    for key, *_ in chain:
        cw_, ch_, th, sh, title, sub, fc, ec = chain_m[key]
        lw = 1.6 if key == "gL2" else 1.1
        draw_box(ax, cx, yc[key], Wc, Hc, title, sub, fc, ec, th, sh, lw=lw)

    # main-chain arrows (skip gL1->L2, handled with the drift branch below)
    seq = [k for k, *_ in chain]
    for a, b in zip(seq[:-1], seq[1:]):
        arrow(ax, cx, yc[a] - Hc / 2, cx, yc[b] + Hc / 2)

    # --- drift-fit branch ---------------------------------------------------
    y_drift = (yc["gL1"] + yc["L2"]) / 2
    dcx = xr + Wd / 2
    draw_box(ax, dcx, y_drift, Wd, Hd, "Drift fit", "CvHG16\ndense segments",
             C_DRIFT, E_DRIFT, dth, dsh, title_fs=FS_DRIFT_T, sub_fs=FS_DRIFT_S)
    arrow(ax, chain_right, yc["gL1"] - 0.06, dcx - Wd / 2, y_drift + 0.06,
          color=E_DRIFT, lw=1.2)
    arrow(ax, dcx - Wd / 2, y_drift - 0.06, chain_right, yc["L2"] + 0.06,
          color=E_DRIFT, lw=1.2)

    # --- configuration rail -------------------------------------------------
    rail_top = yc["raw"] + Hc / 2 + 0.10
    rail_bot = yc["gL2"] - Hc / 2 - 0.10
    ax.add_patch(FancyBboxPatch(
        (rail_left, rail_bot), Wcfg, rail_top - rail_bot,
        boxstyle=f"round,pad=0,rounding_size={ROUND}",
        facecolor=C_CFG, edgecolor=E_CFG, linewidth=1.1, zorder=2,
    ))
    rcx = (rail_left + rail_right) / 2
    title_cy = rail_top - 0.12 - cth / 2
    ax.text(rcx, title_cy, "Configuration\n(per mooring)", ha="center",
            va="center", fontsize=FS_CFG_T, fontweight="bold", color=TEXT,
            zorder=3, linespacing=1.15)
    ax.text(rcx, title_cy - cth / 2 - 0.12 - cih / 2, cfg_items, ha="center",
            va="center", fontsize=FS_CFG_S, color=TEXT, zorder=3,
            linespacing=1.45)
    for k in ["L0", "L1", "gL1", "L2", "gL2"]:
        arrow(ax, rail_right, yc[k], chain_left, yc[k], color=A_CFG, lw=0.9,
              alpha=0.85, mut=7)

    # --- data inputs (right of the early stages) ----------------------------
    in_cx = xr + Wi / 2
    span = 0.5 * Hi + 0.06
    y_in = {"Sensor sheet": yc["L0"],
            "Cal-stops table": yc["L1"] + span,
            "CTD cal casts": yc["L1"] - span}
    aim = {"Sensor sheet": yc["L0"],
           "Cal-stops table": yc["L1"] + 0.10,
           "CTD cal casts": yc["L1"] - 0.10}
    for label, tgt in inputs:
        tw, th = in_m[label]
        draw_box(ax, in_cx, y_in[label], Wi, Hi, label, "", C_INPUT, E_INPUT,
                 th, 0.0, title_fs=FS_INPUT, dashed=True)
        arrow(ax, in_cx - Wi / 2, y_in[label], chain_right, aim[label],
              color=E_INPUT, lw=1.0, mut=8)

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
