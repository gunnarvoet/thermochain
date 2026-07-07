# thermochain

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21251092-blue)](https://doi.org/10.5281/zenodo.21251092)

In-situ calibration and processing of high-density moored thermistor strings
(RBR Solo, SBE 56).

* Free software: MIT license

## Overview

`thermochain` takes raw logger files from a moored thermistor string and carries
them through a configuration-driven pipeline to a drift-corrected, depth–time
gridded temperature product. For chains with sufficient sensor density it
resolves temperature variability at $\mathcal{O}(10^{-4})$ K — more than an order
of magnitude better than the factory calibration of commercial RBR Solo and
SBE 56 thermistors.

The in-situ calibration at the core of the package follows the iterative
shared-fluctuation method of
[Cimatoribus et al. (2016)](https://journals.ametsoc.org/view/journals/atot/33/7/jtech-d-15-0243_1.xml)
(CvHG16), which separates slow sensor drift from real ocean variability using the
signal shared across neighbouring sensors.

## Features

- **Config-driven pipeline.** A single per-mooring YAML file drives every
  processing level. `thermochain.pipeline.Mooring.run()` orchestrates the
  canonical stage chain:

      process_l0 → compute_ctd_offsets → cut_and_cal
        → grid_l1 → fit_drift → make_l2 → grid_l2

  Stages run in a fixed order, are idempotent (existing outputs are skipped
  unless `overwrite=True`), and accept a `segments=` filter.
- **Clock calibration** anchoring each sensor's internal clock to UTC during
  raw → L0 conversion (RBR via `rbrmoored`, SBE 56 via `sbemoored`).
- **CTD rosette calibration** at L0 → L1 from pre- and post-deployment co-located
  casts, with linear-in-time interpolation between endpoints.
- **Depth and time gridding** of all sensors onto a shared `(depth, time)` grid,
  masking native gaps so the interpolant does not bridge real outages.
- **Sensor drift calibration** via the CvHG16 procedure, with optional iterative
  subtraction of large-drift outliers and gap-aware triplet selection.
- **Status reporting** across L0/L1/drift/L2 via per-sensor `status()` and a
  `status_summary()` table.

## Installation

A few dependencies (`sbemoored`, `rbrmoored`, `mixsea`) live on GitHub rather
than PyPI, so they need to be installed from git.

The most convenient route is [`uv`](https://docs.astral.sh/uv/), which resolves
those sources automatically from the versions pinned in `uv.lock`:

```sh
uv sync
```

`uv` is not required, though. With plain `pip`, install the git dependencies
first, then the package:

```sh
pip install git+https://github.com/gunnarvoet/sbemoored \
            git+https://github.com/gunnarvoet/rbrmoored \
            git+https://github.com/modscripps/mixsea
pip install .
```

To develop against local sibling checkouts, overlay them into the environment
without editing `pyproject.toml`:

```sh
uv pip install -e ../rbrmoored -e ../sbemoored --no-deps
```

## Usage

Processing is driven by a per-mooring YAML config consumed by
`thermochain.pipeline.Mooring` (a subclass of
`thermochain.io.ProcessThermistorMooring`):

```python
from thermochain.pipeline import Mooring

m = Mooring("path/to/mooring_config.yml")
m.run()                 # run the full stage chain
m.status_summary()      # report progress across L0/L1/drift/L2
```

Sensor and mooring metadata are supplied as spreadsheets — see the canonical
column layouts in [`templates/`](templates/) (CSV / XLSX / ODS). Keep the column
names as given, and store any datetime columns as plain text in
`yyyy-mm-dd hh:mm:ss` format to ease parsing. The sensor sheet may span several
moorings but must hold only one row per serial number.

## Configuration

Each mooring is described by one YAML config file. Start from the annotated,
fully-populated template at
[`templates/mooring_template.yml`](templates/mooring_template.yml) — copy it,
rename it, and edit the values. Relative paths resolve against the config file's
grandparent directory (the project root); `$root/` and `$data/` prefixes anchor
explicitly against the project root and an external data root.

Which keys you need depends on which stages you run — required-ness is
*stage-dependent*. The only unconditionally required keys are
`meta.{mooring_name, project}`, `path.{fig, sensors, mooring}`,
`path.data.{raw, proc}`, and `start_time` / `end_time`. The CTD-cal, gridding,
segment, and drift blocks are each consumed by their respective stages. Every
key is annotated inline in the template, and the package documentation
(`thermochain` module docstring, "Configuration") has the full block → stage
table.

## Development

```sh
make test      # or: pytest
make lint      # ruff check src/thermochain
make docs      # build pdoc HTML (needs the .pdoc-theme-gv submodule)
make servedocs # serve docs with live reload
```

`make` targets do not prefix `uv run`, so run them inside an activated `.venv`
or prepend `uv run` yourself. The test suite escalates warnings to errors
(`filterwarnings = ["error", ...]`).

## Documentation

This software comes with [pdoc](https://pdoc.dev/) documentation. Build the HTML
with `make docs`, or serve it with live reload via `make servedocs`. The package
module docstring (`src/thermochain/__init__.py`) is the long-form reference for
each calibration stage.

