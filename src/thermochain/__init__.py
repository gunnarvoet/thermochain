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

# Calibration approach

Differential analyses on dense moored thermistor arrays demand **sub-millikelvin relative accuracy** between sensors on the same chain — more than an order of magnitude better than the $\mathcal{O}(2\times 10^{-3})$ K precision of factory-calibrated commercial RBR Solo and SBE 56 thermistors. The package closes that gap with four chained calibration stages:

1. **Clock calibration** during raw → L0 conversion, anchoring each sensor's internal clock to UTC.
2. **CTD rosette calibration** at L0 → L1, providing the absolute-temperature anchor from pre- and post-deployment co-located CTD casts.
3. **Depth and time gridding** of all sensors onto a shared `(depth, time)` array — the regular layout that the drift step needs to compare neighbours.
4. **Sensor drift calibration** via the iterative method of [Cimatoribus et. al. (2016)](https://journals.ametsoc.org/view/journals/atot/33/7/jtech-d-15-0243_1.xml) (**CvHG16**), which separates slow sensor drift from real ocean variability using the shared fluctuating signal across neighbouring sensors.

The clock and CTD stages provide absolute anchors; gridding produces the regular array CvHG16 needs; CvHG16 removes the slow drift that accumulates over months of deployment. The package implements the CvHG16 calibration procedure only — downstream scientific analyses of the calibrated product are out of scope.

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

When `drift_parameters.iterate_subtract=True`, sensors flagged as large-drift outliers in pass 1 (based on `amplitude_threshold_mK`) have their pass-1 drift subtracted from `offsets` and the neighbour-stack stages re-run, so outlier drift does not contaminate neighbours' shared-component estimates. Pass-1 state is retained as `*_pass1` attributes on the class for diagnostic comparison.

The output is a per-sensor drift trace `drift(sn, time)`; applying it (L2 = L1 − drift on the sensor's native time axis) is performed downstream of `thermochain`.

# Templates

The `templates` directory holds templates for
- Sensor spreadsheet (holding all per-sensor info, e.g. clock calibration times, CTD calibration times, specific pre- and post-deployment notes.
- Mooring configuration spreadsheet, laying out the configuration of the sensors on the moorings.

# License
.. include:: ../../LICENSE
"""

__author__ = """Gunnar Voet"""
__email__ = "gvoet@ucsd.edu"
__version__ = "2023.12.0"

__all__ = ["io", "plot", "pipeline", "Mooring"]
from . import io, pipeline, plot
from .pipeline import Mooring
