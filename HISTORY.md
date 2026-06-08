# History

## unreleased

### Breaking changes
-   Package renamed `thermodrift` → `thermochain` as it grows from a
    drift-correction library into the full moored-thermistor processing
    pipeline. Update imports to `import thermochain`. `import thermodrift` is
    kept as a **deprecated alias** that re-exports `thermochain` and emits a
    `DeprecationWarning`; it will be removed in a future release. The GitHub
    repository was renamed accordingly (with redirects from the old name).

### New Features
-   **Config-driven pipeline (`thermochain.pipeline`).** A major rewrite turns
    the package from a drift-correction toolkit into a full moored-thermistor
    processing pipeline. A new `Mooring` class (subclassing
    `ProcessThermistorMooring`) drives every processing level from a single
    per-mooring YAML config, with `Mooring.run()` orchestrating the canonical
    stage chain:

        process_l0 -> compute_ctd_offsets -> cut_and_cal
          -> grid_l1 -> fit_drift -> make_l2 -> grid_l2

    Stages always execute in the fixed `STAGE_ORDER` (selecting a subset never
    reorders), each stage is idempotent (skips existing outputs unless
    `overwrite=True`), and the segment-aware stages accept a `segments=` filter.
    The algorithms themselves are promoted, largely verbatim, from the former
    processing notebooks into reusable, tested pipeline methods:
    -   `compute_ctd_offsets` — config-driven CTD cal-offset computation over
        `rbr_ctd_cal_find_offset`, honoring `cal_ignore_sns_{source}`, skipping
        sensors with no samples in the cast window, and writing per-sensor
        `source_cast` / `source_cruise` provenance. `detect_cal_stops` promotes
        the pressure-plateau cal-stop detection; `load_cal_offsets` the offset
        loader.
    -   `cut_and_cal` (L0→L1) over the `rbr_cut_and_cal` primitives, with
        `cal_diagnostic_attrs` recording L1 calibration provenance.
    -   `fit_drift` (L1→drift) over the `sensor_drift` primitives, with a typed
        `DriftParameters` schema (rejects unknown keys),
        `drift_provenance_attrs` (NetCDF-safe param flattening), and
        `drift_diag_bundle` diagnostics.
    -   `make_l2` / `grid_l2` apply the fitted drift (`correct_drift`) to build
        per-sensor and gridded L2 products.
    -   `grid_l1` writes gridded L1 under `grid/l1/`.
    -   Per-sensor `status()` and a `status_summary()` table report progress
        across L0/L1/drift/L2.
-   `sensor_drift`: gap-aware triplet selection. The new `max_triplet_gap_m`
    drift parameter (default `None` = off) makes `select_triplet` skip
    neighbour triplets that span a vertical gap wider than the threshold, so
    drift fits don't bridge large holes in the sensor stack.

### Bug fixes
-   `sensor_drift` / CvHG16: detrended offsets are now wired into the
    second-guess shared fluctuating component (previously the raw offsets were
    used), fixing contamination of the shared component by per-sensor trends.

<!-- ### Documentation -->

### Internal Changes
-   New `src/thermochain/pipeline.py` module holds the `Mooring` orchestration
    and pipeline-stage code, built on the existing `io.py` primitives.
-   Dropped the `gvpy` dependency. The handful of helpers used (logger setup
    and the plotting helpers `quickfig`, `axstyle`, `concise_date`,
    `annotate_corner`, and a figure-saver) are now vendored locally: a new
    `src/thermochain/_log.py` holds the loguru logger setup and `plot.py` holds
    slimmed-down plotting helpers (`axstyle` is simplified to a minimal house
    style). `loguru` is now a direct dependency. This removes the mandatory
    `../gvpy` editable sibling checkout and prunes a large set of transitive
    dependencies (cartopy, pyproj, shapely, dask, …).


## 2026.03.0

### New Features
-   `sensor_drift`: optional iterative subtraction of large-drift outliers.
    Set `iterate_subtract=True` (plus `amplitude_threshold_mK` and/or
    `manual_outlier_sns`) in `drift_parameters` to fit one extra pass
    after the standard pipeline: flagged sensors' pass-1 drift is
    subtracted from `offsets`, and the neighbour-stack stages are
    re-run so the neighbours' fits aren't contaminated by the outlier's
    drift leakage. Pass-1 state is preserved as `*_pass1` attributes
    for diagnostics; new `plot_iteration_diagnostic` emits before/after
    figures for flagged sensors. Output `drift` DataArray gains
    `iteration_count` and `flagged_outlier_sns` attrs for provenance.


## 2023.12.0
-   Created package.
