"""Shared content for the drift-procedure schematics.

Both ``drift_procedure_schematic.py`` (vertical) and
``drift_procedure_schematic_horizontal.py`` (wide) draw the same CvHG16
drift-estimation flow; only the layout differs. The box definitions, palette,
and the per-window / offset-time-series phase grouping live here so the two
figures stay in lock-step.

The procedure has two conceptual phases:

* **Per time window** --- the background fit and first-guess offsets are
  computed independently in each short (≈ daily) window.
* **Offset time series** --- everything downstream (±3σ rejection, the shared
  fluctuating component, the drift fits) operates on how each sensor's offset
  evolves across all windows.

The gridded L1 record (input) and the L2 product (output) are full-record I/O
that bracket the two phases.
"""

from __future__ import annotations

from _schematic import LEVELS

# step-box palette (light blue, matching the processing levels) ---------------
C_STEP, E_STEP = "#eef3f9", "#2f5c8a"
C_HI, E_HI = "#fbe2c0", "#c47f1d"   # highlighted two-step box (amber)
A_FB = "#c47f1d"                    # feedback arrow (amber)

# phase-grouping panels -------------------------------------------------------
C_PW, E_PW = "#f4ecdd", "#c4ab7e"   # per-window phase (warm)
C_TS, E_TS = "#e8eef5", "#9fb4ca"   # offset-time-series phase (cool)

# chain: key -> (title, subtitle, facecolor, edgecolor). ``gL1``/``L2`` reuse
# the processing-level palette; ``two`` is the highlighted two-step box.
CHAIN = [
    ("gL1", "Gridded L1", "full deployment record", *LEVELS["gL1"]),
    ("bg", "Windowed background fit",
     "smoothing spline (or polynomial)\nper window; exclude gross outliers",
     C_STEP, E_STEP),
    ("off1", "First-guess offsets", "residual from the background", C_STEP, E_STEP),
    ("sig", "±3σ rejection", "drop time-series outliers", C_STEP, E_STEP),
    ("sfc1", "First-guess shared component",
     "demeaned neighbour triplet", C_STEP, E_STEP),
    ("off2", "Second-guess offsets", "offsets − shared component", C_STEP, E_STEP),
    ("interim", "Interim drift fit", "per-sensor model", C_STEP, E_STEP),
    ("two", "Detrend → recompute\nshared component",
     "from the detrended offsets\n(two-step refinement)", C_HI, E_HI),
    ("clean", "Cleaned offsets",
     "original offsets − recomputed\nshared component", C_STEP, E_STEP),
    ("model", "Final drift model",
     "linear or exponential (CvHG16)\nR² model selection", C_STEP, E_STEP),
    ("L2", "Subtract → L2", "drift-corrected product", *LEVELS["L2"]),
]

# phase membership (by key); gL1 / L2 are the full-record input / output.
INPUT_KEY = "gL1"
OUTPUT_KEY = "L2"
PHASE_PW = ["bg", "off1"]                                    # per time window
PHASE_TS = ["sig", "sfc1", "off2", "interim", "two", "clean", "model"]  # series
PW_LABEL = "per time window (≈ daily)"
TS_LABEL = "offset time series — per sensor, all windows"

# The shared component, offsets, and drift fit are estimated twice: a first
# pass, then a second pass after detrending with the interim fit (the two-step
# refinement). The wide schematic lays these out as two aligned rows.
PASS1_LABEL = "first pass"
PASS2_LABEL = "second pass"
LOOP_LABEL = "detrend with\ninterim fit"
