# Documentation assets

Static figures embedded into the pdoc-generated API documentation (pdoc inlines
local images referenced from a module docstring as base64 data URIs).

- `pipeline_schematic.svg` — the configuration-driven processing pipeline
  (raw → L0 → L1 → gridded L1 → L2), embedded in the package overview
  (`thermochain/__init__.py`).
- `drift_procedure_schematic.svg` — the CvHG16 shared-fluctuation drift
  procedure, embedded in the sensor-drift section.

## Provenance / regeneration

These are SVG exports of the horizontal schematics from the accompanying
JTECH manuscript (`motive/research/doc/thermistor-chain-manuscript/figures/`,
scripts `pipeline_schematic_horizontal.py` and
`drift_procedure_schematic_horizontal.py`). To refresh them, regenerate from
those scripts with `format="svg"` and copy the output here, dropping the
`_horizontal` suffix. They are intentionally committed as static assets so the
docs build has no cross-repository dependency.
