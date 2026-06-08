"""Smoke test: every schematic builder renders a non-empty figure.

The generators live in the repo-level ``schematics/`` directory (build tooling,
not an installed package), so it is added to ``sys.path`` here. The suite runs
under ``filterwarnings = ["error", ...]``; a clean render keeps it green.
"""
import sys
from pathlib import Path

import matplotlib
import pytest

SCHEM_DIR = Path(__file__).resolve().parents[1] / "schematics"
if str(SCHEM_DIR) not in sys.path:
    sys.path.insert(0, str(SCHEM_DIR))

import drift_procedure_schematic  # noqa: E402
import pipeline_schematic  # noqa: E402

BUILDERS = [
    pipeline_schematic.build_horizontal,
    pipeline_schematic.build_vertical,
    drift_procedure_schematic.build_horizontal,
    drift_procedure_schematic.build_vertical,
]


@pytest.mark.parametrize(
    "builder", BUILDERS, ids=lambda b: b.__module__ + "." + b.__name__
)
def test_builder_renders_nonempty_figure(builder):
    fig = builder()
    try:
        width, height = fig.get_size_inches()
        assert width > 0 and height > 0
        assert fig.axes, "figure has no axes"
    finally:
        matplotlib.pyplot.close(fig)
