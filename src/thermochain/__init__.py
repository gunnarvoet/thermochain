r"""In-situ sensor calibration and general data processing for high-density moored thermistor strings.

# Overview

This software package applies sensor calibration and general depth-gridding to a set of data files collected with moored thermistors.
The in-situ sensor calibration correcting for sensor offset and sensor drift with time at the core of this package follows the method developed in [Cimatoribus et. al. (2016)](https://journals.ametsoc.org/view/journals/atot/33/7/jtech-d-15-0243_1.xml).
For thermistor strings with sufficient sensor density (depending on ambient stratification between less than one to several meters) the in-situ calibration method allows for measuring temperature variability at $\mathcal{O}(10^{-4})$ K.

Currently the package works with raw data files from RBR Solo and SBE 56 thermistors.

Processing steps are
- conversion
- clock calibration
- CTD rosette calibration
- depth and time gridding
- sensor drift calibration

Individual steps are detailed in the following.

![Configuration-driven thermochain processing pipeline](_static/pipeline_schematic.svg)

*The configuration-driven `thermochain` pipeline. A single per-mooring
configuration file, together with the sensor sheet, the cal-stops table, and the
CTD calibration casts, parameterises and feeds a chain of stages that carry the
raw logger files through the processing levels: clock calibration (L0), cutting
to the deployment window and in-situ CTD calibration (L1), interpolation onto a
common depth–time grid (gridded L1), the CvHG16 drift fit applied to the densely
instrumented segments, and subtraction of the per-sensor drift (L2) to yield the
final drift-corrected, gridded temperature product.*

# Calibration approach

Differential analyses on dense moored thermistor arrays demand **sub-millikelvin relative accuracy** between sensors on the same chain — more than an order of magnitude better than the $\mathcal{O}(2\times 10^{-3})$ K precision of factory-calibrated commercial RBR Solo and SBE 56 thermistors. The package closes that gap with four chained calibration stages:

1. **Clock calibration** during raw → L0 conversion, anchoring each sensor's internal clock to UTC.
2. **CTD rosette calibration** at L0 → L1, providing the absolute-temperature anchor from pre- and post-deployment co-located CTD casts.
3. **Depth and time gridding** of all sensors onto a shared `(depth, time)` array — the regular layout that the drift step needs to compare neighbours.
4. **Sensor drift calibration** via the iterative method of [Cimatoribus et. al. (2016)](https://journals.ametsoc.org/view/journals/atot/33/7/jtech-d-15-0243_1.xml) (**CvHG16**), which separates slow sensor drift from real ocean variability using the shared fluctuating signal across neighbouring sensors.

The clock and CTD stages provide absolute anchors; gridding produces the regular array CvHG16 needs; CvHG16 removes the slow drift that accumulates over months of deployment. The package implements the CvHG16 calibration procedure only — downstream scientific analyses of the calibrated product are out of scope.

# Configuration

The whole pipeline is driven by a single per-mooring YAML config consumed by
`thermochain.pipeline.Mooring` (a subclass of
`thermochain.io.ProcessThermistorMooring`). An annotated, fully-populated
template lives at `templates/mooring_template.yml` — copy it, rename it, and
edit the values for your mooring. Relative paths resolve against the config
file's grandparent directory (the project root); the `$root/` and `$data/`
prefixes anchor explicitly against the project root and an external
`data_root`, respectively.

Not every key is needed for every run — required-ness is *stage-dependent*. The
table below maps each config block to the pipeline stage(s) that read it. The
only **unconditionally** required keys are `meta.mooring_name`, `meta.project`,
`path.{fig, sensors, mooring}`, `path.data.{raw, proc}`, and
`start_time` / `end_time`; everything else is required only once you run the
stage that consumes it.

| Config block / key | Consumed by |
| --- | --- |
| `meta.{mooring_name, project}` · `path.{fig, sensors, mooring}` · `path.data.{raw, proc}` · `start_time` · `end_time` | **all stages** (always required) |
| `path.ctd` · `path.cal_stops` · `calibration.cal_casts_{pre,post}` · `calibration.cal_ignore_sns_{pre,post}` | `compute_ctd_offsets` |
| `calibration.{method, offsets_pre, offsets_post}` | `cut_and_cal` (offsets are the *output* of `compute_ctd_offsets`) |
| `path.data.grid` · `gridding.{dt, max_gap, chunk}` | `grid_l1`, `grid_l2` |
| `segments.<name>.{select, drift, gridding, calibration}` | `cut_and_cal`, `grid_l1`, `fit_drift`, `make_l2`, `grid_l2` |
| `path.aux` · `drift_parameters.*` (incl. `label`) | `fit_drift`, `make_l2` |
| `ignore_sns` | all segment-aware stages (quality drop, unioned with sensor-sheet `exclude == 1`) |

Unknown keys in `gridding` and `drift_parameters` are rejected rather than
silently ignored, so a typo fails fast. The default values for the CvHG16
`drift_parameters` are the fields of `thermochain.pipeline.DriftParameters`.

# Raw file conversion

Initialize a processing object to process data from one mooring with `thermochain.io.ProcessThermistorMooring` using a `.yaml`-configuration file. The processing object will:

- Read the  configuration file (one per mooring) that contains paths to data (both raw input & output) directories, sensor and mooring spreadsheets, and other processing parameters.
- Read sensor information from the sensor spreadsheet. See [templates](#templates) in both `.csv` and `.xlsx` format in the `templates` folder. Either of them works, just make sure to keep the column names as they are. Also, keeping any datetime columns in plain text format and sticking to `yyyy-mm-dd hh:mm:ss` format will help with parsing the spreadsheet. The sensor spreadsheet may contain sensors from several moorings but only one row per serial number.

# Clock calibration

Each thermistor's internal clock is anchored to UTC during the raw → L0 conversion. Two clock-cal CTD-cast times are stored per sensor in the sensor spreadsheet (`time_cal1`, `time_cal2`) and surfaced through `thermochain.io.ProcessThermistorMooring.get_clock_cals`. For RBR Solo these are passed to `rbrmoored.solo.proc` as the `cal_time` argument with `apply_time_offset=True`, which corrects the linear drift of the rsk-internal clock between the two cast times. For SBE 56, `get_clock_reads` additionally surfaces `clock_read_utc` / `clock_read_logger` — the logger and reference clock readouts taken on deck — which are forwarded to `sbemoored.sbe56.proc` to bracket the clock offset. The dispatch happens through `run_proc_single_rbr` / `run_proc_single_sbe`; one L0 NetCDF per sensor is written to the configured `procl0` directory.

# CTD rosette calibration

Each sensor is calibrated against co-located CTD rosette casts at deployment and recovery. Pre- and post-deployment offset arrays (one offset per serial number, computed externally and stored as NetCDF) are consumed by `thermochain.io.rbr_cut_and_cal_interp` during the L0 → L1 step. When both endpoints are available, `thermochain.io.apply_interpolated_ctd_offset` applies a **linear-in-time interpolation** between them; the interpolation factor is clamped to `[0, 1]` so samples outside the cal bracket receive the corresponding endpoint offset. When only one endpoint is available, the function falls back to a scalar offset via `rbr_apply_ctd_offset`. A `cal_method` attribute is written to each L1 file recording which path was taken (`linear_interp`, `scalar`, or `none`).

The same routine also handles end-of-record truncation: the L1 cut window is shortened on the fly by `rbr_find_last_time_stamp` (sensor stopped early), `rbr_find_first_long_gap` (deployment-killing data gap), or an optional per-sensor `end_manually` override supplied through the config.

# Depth and time gridding

`thermochain.io.grid_thermistors` interpolates all L1 sensors on a mooring onto a shared `(depth, time)` grid spanning a configured deployment window. Each sensor's L1 NetCDF is loaded once; native sample gaps wider than `max_gap` are masked as NaN before interpolation so that the linear interpolant does not bridge real outages. The output grid is uniform in time at spacing `dt`; the depth axis is the per-sensor nominal depth from the mooring sheet.

The output is a single `xarray.DataArray` with dimensions `(depth, time)` and auxiliary coordinates `sn` and `sensor_type` on the depth dim. The implementation pre-allocates the `(n_sensors, n_times)` output array and fills it row-by-row rather than accumulating per-sensor DataArrays and concatenating with `xr.concat` — the latter held inputs and merged output coexistent in memory and OOM'd long-deployment deep arrays at fine `dt`. Long deployments are typically processed as fixed-length chunks (e.g. two-day spans) to keep peak memory bounded; one gridded NetCDF per chunk is the input to the drift-calibration step.

# Sensor drift calibration

Commercial thermistors drift on the order of millikelvins per month, which dominates the inter-sensor offset signal after a multi-month deployment. The CvHG16 procedure ([Cimatoribus et. al. 2016](https://journals.ametsoc.org/view/journals/atot/33/7/jtech-d-15-0243_1.xml)) separates the slow drift from real ocean variability by exploiting the fact that, within a short window, neighbouring sensors see a *shared* fluctuating signal as isotherms move past them while their drifts evolve independently.

Per window (default 1 day), each sensor's deviation from a smooth time-mean background profile contains three parts: a slow sensor drift (what we want to estimate and remove), a shared fluctuating component coherent across nearby sensors, and sensor-specific noise. CvHG16 separates them iteratively.

## Procedure

Implemented in `thermochain.io.sensor_drift`. With `run_all=True` the class runs the following stages in sequence; each is also callable individually for debugging.

![Schematic of the CvHG16 shared-fluctuation drift procedure](_static/drift_procedure_schematic.svg)

*The CvHG16 shared-fluctuation drift procedure. The windowed background fit and
first-guess offsets are computed independently in each short (≈ daily) window
(left); all subsequent steps operate on each sensor's offset time series across
the full deployment (right). After ±3σ outlier rejection, the shared fluctuating
component (from each sensor's demeaned neighbour triplet), the offsets, and the
per-sensor drift fit are estimated in two passes: the first pass yields an
interim drift fit used to detrend the offsets so the shared component can be
recomputed (amber), and the second pass forms the cleaned offsets and the final
per-sensor drift model — linear or exponential, selected by the CvHG16 $R^2$
criterion — which is subtracted to yield the drift-corrected L2 product. The
numbered stages below correspond to the boxes in this schematic.*

1. **Background fit** per window — polynomial or smoothing-spline fit in depth to the windowed time-mean profile; each sensor's residual is its first-guess offset for that window. (`windowed_background_fits`, `offsets_from_background_fit`)
2. **Outlier removal** at $\pm 3\sigma$ on each sensor's offset time series. (`remove_outliers`)
3. **First-guess shared fluctuating component** — mean of the triplet (sensor + upper neighbour + lower neighbour) after demeaning each member. (`select_triplet`, `calc_first_guess_shared_fluctuating_component`)
4. **Second-guess offsets** — initial offsets minus the first-guess shared component. (`calc_offsets_second_guess`)
5. **Interim fit** (linear or exponential) to those second-guess offsets. (`fit_second_guess`)
6. **Second-guess shared component** recomputed from the detrended offsets, then subtracted from the *original* offsets to yield **cleaned offsets**. (`calc_second_guess_shared_fluctuating_component`, `calc_cleaned_offsets`)
7. **Final drift fit** on the cleaned offsets. (`fit_cleaned_offsets`)

Step 6 is the CvHG16 two-step refinement and is the default (`two_step_shared=True`). Setting `two_step_shared=False` recovers the legacy single-pass behaviour, in which the recompute reads the (non-detrended) offsets and the second-guess shared component collapses to the first guess — making the interim fit (step 5) a no-op. The flag exists to reproduce pre-fix drift products; new work should leave it at the default.

## Drift model

CvHG16 models the per-sensor drift rate as

$$
\frac{\partial \Delta T}{\partial t} = m + a \exp\!\left[-\left(\frac{t}{\tau}\right)^{\beta}\right]
$$

which integrates to

$$
\Delta T(t) = \Delta T_0 + m\,t + \frac{A}{\beta}\,\gamma\!\left(\frac{1}{\beta},\,\frac{t}{\tau}\right)
$$

with $\gamma$ the lower incomplete gamma function. The five parameters are $\Delta T_0$ (systematic bias), $m$ (asymptotic long-term drift rate), $A = a\tau$ (relaxation amplitude), $\tau$ (relaxation time), and $\beta$ (stretch exponent). The auto-selector (`lin_or_exp`) prefers the exponential over the linear fit when $R_\gamma^{2} > R_l^{2} + 0.3\,(1 - R_l^{2})$ — i.e. when the exponential explains a substantial fraction of the residual variance left by the linear fit.

## Refinement: strongly-drifting sensors

A few sensors in a deployment drift far more than the rest. Because the shared fluctuating component at a given depth is estimated from a sensor's demeaned neighbour triplet, an outlier's large excursions leak into its neighbours' shared-component estimates and bias the drift fit across the whole triplet. CvHG16 guard against this only at the background-profile stage (excluding such sensors from the depth fit); setting `drift_parameters.iterate_subtract=True` additionally removes their influence from the shared-fluctuation step.

After the first full pass through the procedure above, sensors whose pass-1 drift amplitude exceeds `amplitude_threshold_mK` are flagged automatically; additional serials can be forced in with `manual_outlier_sns` (entries not in the deployment are skipped with a warning). The flagged sensors' pass-1 drift is subtracted from `offsets` and the neighbour-stack stages are re-run, so the shared fluctuating component is rebuilt from triplets no longer contaminated by the outliers' drift. Neighbours inherit this cleaned component directly — their fits are unchanged. (This outer flag–subtract–rerun loop is separate from the two-step shared-component refinement above; here "pass 1" and "pass 2" mean one full run of the procedure, before and after the outlier subtraction.)

What happens to a *flagged* sensor's own drift is set by `iterate_mode`:

- `"restore"` *(default)* — the flagged sensor's pass-1 drift is kept. Its pass-2 fit would be the fit of `offsets − pass-1 drift` (the residual leakage), which would zero out the L2 correction at the outlier; restoring pass-1 preserves the drift amplitude, at the cost of the self-bias that the iteration removes from its neighbours.
- `"refit"` — the flagged sensor is re-fit against the *pass-2* (clean) shared component, recovering its drift free of the self-bias that inflated the pass-1 estimate. Prefer this when the residual bias at the outlier itself matters.

Pass-1 state is retained as `*_pass1` attributes for comparison, and `thermochain.io.sensor_drift.plot_iteration_diagnostic` emits before/after figures at a given depth index.

The output is a per-sensor drift trace `drift(sn, time)`; applying it (L2 = L1 − drift on the sensor's native time axis) is performed downstream of `thermochain`.

# Templates

The `templates` directory holds templates for
- Per-mooring processing config (`mooring_template.yml`) — the annotated YAML that drives the pipeline; see [Configuration](#configuration) for the parameter/stage table.
- Sensor spreadsheet (holding all per-sensor info, e.g. clock calibration times, CTD calibration times, specific pre- and post-deployment notes.
- Mooring configuration spreadsheet, laying out the configuration of the sensors on the moorings.

# References

The sensor-drift calibration implemented here follows, and is referred to
throughout as **CvHG16**:

> Cimatoribus, A. A., H. van Haren, and L. Gostiaux (2016): A procedure to
> compensate for the response drift of a large set of thermistors. *Journal of
> Atmospheric and Oceanic Technology*, **33** (7), 1495–1508,
> [doi:10.1175/JTECH-D-15-0243.1](https://doi.org/10.1175/JTECH-D-15-0243.1).

# License
.. include:: ../../LICENSE
"""

__author__ = """Gunnar Voet"""
__email__ = "gvoet@ucsd.edu"
__version__ = "2023.12.0"

__all__ = ["io", "plot", "pipeline", "Mooring"]
from . import io, pipeline, plot
from .pipeline import Mooring
