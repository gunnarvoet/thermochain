# Documentation assets

Static figures embedded into the pdoc-generated API documentation (pdoc inlines
local images referenced from a module docstring as base64 data URIs).

- `pipeline_schematic.svg` — the configuration-driven processing pipeline
  (raw → L0 → L1 → gridded L1 → L2), embedded in the package overview
  (`thermochain/__init__.py`).
- `drift_procedure_schematic.svg` — the CvHG16 shared-fluctuation drift
  procedure (two-pass layout), embedded in the sensor-drift section.

## Provenance / regeneration

These SVGs are generated, not hand-edited. The generators live in the repo-level
`schematics/` directory and emit a portrait **SVG** here (for the docs) and a
wide **PDF** in `schematics/pdf/` (consumed by the JTECH manuscript). Regenerate
both with:

    make schematics

(or run `uv run python schematics/<name>_schematic.py` directly). They are
committed as static assets so the docs build has no build-time figure step.
