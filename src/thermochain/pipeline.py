# src/thermochain/pipeline.py
"""Config-driven Mooring pipeline orchestration over thermochain primitives."""

import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
import xarray as xr

from .io import (  # noqa: F401
    ProcessThermistorMooring,
    grid_thermistors,
    logger,
    rbr_ctd_cal_find_offset,
    rbr_cut_and_cal,
    rbr_cut_and_cal_interp,
    sensor_drift,
)

_GRIDDING_KEYS = {"dt", "max_gap", "chunk"}
_CAL_METHODS = {"linear_interp", "scalar", "scalar_pre_only", "none"}

# Canonical pipeline stage order — the single source of truth for Mooring.run().
# Dissipation is intentionally excluded (decided 2026-06-07). ``process_l0`` is
# the public-vocabulary alias of the inherited ``run_proc_all`` (L0 stage).
STAGE_ORDER = [
    "process_l0",
    "compute_ctd_offsets",
    "cut_and_cal",
    "grid_l1",
    "fit_drift",
    "make_l2",
    "grid_l2",
]
# Stages that do not accept a ``segments`` kwarg.
_STAGES_NO_SEGMENTS = {"process_l0", "compute_ctd_offsets"}

_OFFSET_CRUISE_RE = re.compile(r"(cruise\d+)")


def load_cal_offsets(path):
    """Load a CTD cal-offsets NetCDF into one DataArray indexed by ``sn``.

    Promoted from MOTIVE notebook 02's ``load_offsets``. The aggregated
    offsets file (written by the CTD-cal notebook) carries a per-sensor
    ``cast`` coord; it is propagated as ``source_cast`` and a
    ``source_cruise`` coord is parsed from the filename so per-sensor
    provenance survives into the L1 attrs. Returns ``None`` when ``path``
    is absent, so a missing post-deployment side does not raise.

    Parameters
    ----------
    path : pathlib.Path or str
        Offsets NetCDF with an ``offset`` data variable and ``sn`` /
        ``cast`` coordinates.

    Returns
    -------
    xr.DataArray or None
        Per-sensor offsets indexed by ``sn`` (sorted), with
        ``source_cruise`` / ``source_cast`` coords; ``None`` if absent.
    """
    path = Path(path)
    if not path.exists():
        return None
    ds = xr.load_dataset(path)
    m = _OFFSET_CRUISE_RE.search(path.name)
    cruise = m.group(1) if m else "unknown"
    a = ds["offset"].copy()
    a = a.assign_coords(
        source_cruise=("sn", [cruise] * a.sizes["sn"]),
        source_cast=("sn", ds["cast"].values.astype(int)),
    )
    return a.sortby("sn")


def cal_diagnostic_attrs(
    sn, ctdcal_pre, ctdcal_post, *, cal_method, pre_applied, post_applied,
    t_pre=None, t_post=None,
):
    """Build the flat L1 calibration-provenance attrs for one sensor.

    Mirrors MOTIVE notebook 02's ``attach_diagnostic_offsets``: records
    each side's offset / cruise / cast / cast-time when present in the
    cal file (whether or not it was applied), and NaN + sentinels when
    the sensor is absent from that side. ``cal_method`` is recorded
    verbatim.

    Parameters
    ----------
    sn : int
        Sensor serial number.
    ctdcal_pre, ctdcal_post : xr.DataArray or None
        Per-sensor offsets indexed by ``sn`` with ``source_cruise`` /
        ``source_cast`` coords (output of :func:`load_cal_offsets`).
    cal_method : str
        Effective calibration method recorded on the L1 file.
    pre_applied, post_applied : bool
        Whether each side's offset was actually applied to the series.
    t_pre, t_post : datetime-like or None, optional
        Cast times for the pre / post sides.

    Returns
    -------
    dict
        Flat attrs: ``cal_method`` plus ``{pre,post}_cal_{offset,applied,
        cruise,cast,time}``.
    """
    def _one(prefix, ctdcal, applied, t_cal):
        if ctdcal is not None and sn in ctdcal.sn:
            sel = ctdcal.sel(sn=sn)
            return {
                f"{prefix}_offset": float(sel.data),
                f"{prefix}_applied": int(applied),
                f"{prefix}_cruise": str(sel.source_cruise.item()),
                f"{prefix}_cast": int(sel.source_cast.item()),
                f"{prefix}_time": (
                    pd.Timestamp(t_cal).isoformat()
                    if t_cal is not None and not pd.isna(t_cal)
                    else ""
                ),
            }
        return {
            f"{prefix}_offset": float("nan"),
            f"{prefix}_applied": 0,
            f"{prefix}_cruise": "",
            f"{prefix}_cast": -1,
            f"{prefix}_time": "",
        }

    attrs = {"cal_method": cal_method}
    attrs.update(_one("pre_cal", ctdcal_pre, pre_applied, t_pre))
    attrs.update(_one("post_cal", ctdcal_post, post_applied, t_post))
    return attrs


_DRIFT_FIT_MODES = ("linear", "auto", "exp")
_DRIFT_ITERATE_MODES = ("restore", "refit")


@dataclass
class DriftParameters:
    """Typed, validated CvHG16 drift-fit parameters.

    Field names and defaults mirror
    :meth:`thermochain.io.sensor_drift.parse_drift_parameters` exactly, so a
    ``DriftParameters().as_dict()`` reproduces today's defaults. Unlike the
    loose ``setattr`` loop in the primitive, :meth:`from_dict` **rejects
    unknown keys** (a YAML typo like ``spline_smoothh`` raises instead of
    silently falling back to the default - big-picture stretch-cleanup
    Item A, enforced at the pipeline boundary).

    The pipeline-level ``label`` (drift-product filename key) is handled
    separately by :meth:`Mooring.fit_drift` and is *not* a field here.
    """

    exclude: object = field(default_factory=lambda: [1e-2, 5e-3])
    polydeg: int = 8
    outliers_polydeg: int = 8
    use_spline: bool = False
    spline_smooth: float = 2e-4
    exclude_sn: object = None
    tau0: float = 20.0
    tau_bounds: object = field(default_factory=lambda: (5.0, 180.0))
    beta_bounds: object = field(default_factory=lambda: (1.0 / 3.0, 3.0))
    fit_mode: str = "auto"
    iterate_subtract: bool = False
    iterate_mode: str = "restore"
    amplitude_threshold_mK: float = 1.5
    manual_outlier_sns: list = field(default_factory=list)
    two_step_shared: bool = True
    max_triplet_gap_m: object = None

    def __post_init__(self):
        if self.fit_mode not in _DRIFT_FIT_MODES:
            raise ValueError(
                f"fit_mode must be one of {_DRIFT_FIT_MODES}; got {self.fit_mode!r}"
            )
        if self.iterate_mode not in _DRIFT_ITERATE_MODES:
            raise ValueError(
                f"iterate_mode must be one of {_DRIFT_ITERATE_MODES}; got {self.iterate_mode!r}"
            )

    @classmethod
    def from_dict(cls, params):
        """Build from a dict, rejecting any key not a recognised field.

        Parameters
        ----------
        params : dict or None
            Raw ``drift_parameters`` (config block UNION caller override),
            with the pipeline-level ``label`` already removed.

        Returns
        -------
        DriftParameters

        Raises
        ------
        ValueError
            If ``params`` contains a key outside the dataclass fields, or
            ``fit_mode`` / ``iterate_mode`` is invalid.
        """
        params = dict(params or {})
        known = {f.name for f in fields(cls)}
        unknown = set(params) - known
        if unknown:
            raise ValueError(
                f"unknown drift_parameters keys: {sorted(unknown)} "
                f"(allowed: {sorted(known)})"
            )
        return cls(**params)

    def as_dict(self):
        """Return a plain dict suitable for ``sensor_drift(drift_parameters=…)``."""
        return asdict(self)


def drift_provenance_attrs(params, label):
    """Flatten DriftParameters + label into NetCDF-safe attrs.

    NetCDF attrs accept scalars, strings, and 1-D numeric arrays — not
    tuples, bools, or ``None``. Two-element bounds become ``_lo`` / ``_hi``
    floats; bools become 0/1 ints; lists become int64 arrays; ``None``
    becomes an empty string. All keys are prefixed ``drift_param_`` (the
    ``label`` lands in ``drift_label``).

    Parameters
    ----------
    params : DriftParameters
        Validated fit parameters.
    label : str
        Drift-product label (filename key).

    Returns
    -------
    dict
        NetCDF-safe attrs for the drift and diagnostic products.
    """
    attrs = {"drift_label": str(label)}
    for key, value in params.as_dict().items():
        name = f"drift_param_{key}"
        if value is None:
            attrs[name] = ""
        elif isinstance(value, bool):
            attrs[name] = int(value)
        elif isinstance(value, (tuple, list)) and len(value) == 2 and key.endswith("_bounds"):
            attrs[f"{name}_lo"] = float(value[0])
            attrs[f"{name}_hi"] = float(value[1])
        elif isinstance(value, (tuple, list)):
            attrs[name] = np.asarray(value, dtype="int64" if key == "manual_outlier_sns" else float)
        else:
            attrs[name] = value
    return attrs


def drift_diag_bundle(sd, params, label):
    """Assemble the drift diagnostic Dataset from a fitted ``sensor_drift``.

    Port of MOTIVE notebook 05a's ``_run_config`` bundle block: gathers the
    arrays the drift-comparison diagnostics need (``offsets_clean``,
    ``drift_linfit``, ``drift_fit``, ``fit_type``, and — for auto/exp runs —
    ``drift_expfit`` / ``drift_exp_params``, plus the iterate-subtract
    ``*_pass1`` arrays when present), and writes the sd-derived diagnostic
    attrs (matching the baseline) plus :func:`drift_provenance_attrs`.

    Parameters
    ----------
    sd : thermochain.io.sensor_drift
        A fitted instance (``run_all=True`` already executed).
    params : DriftParameters
        Validated parameters for the provenance attrs.
    label : str
        Drift-product label.

    Returns
    -------
    xr.Dataset
    """
    bundle = xr.Dataset(
        {
            "offsets_clean": sd.offsets_clean,
            "drift_linfit": sd.drift_linfit,
            "drift_fit": sd.drift_fit,
            "fit_type": sd.fit_type,
        }
    )
    for opt in ("drift_expfit", "drift_exp_params", "drift_fit_pass1", "offsets_clean_pass1"):
        if hasattr(sd, opt):
            bundle[opt] = getattr(sd, opt)
    # sd-derived diagnostic attrs (match the baseline diag bundle)
    bundle.attrs["fit_mode"] = sd.fit_mode
    bundle.attrs["tau_bounds_lo"] = float(sd.tau_bounds[0])
    bundle.attrs["tau_bounds_hi"] = float(sd.tau_bounds[1])
    bundle.attrs["iteration_count"] = int(getattr(sd, "iteration_count", 0))
    bundle.attrs["flagged_outlier_sns"] = np.asarray(
        getattr(sd, "flagged_outlier_sns", []), dtype="int64"
    )
    bundle.attrs.update(drift_provenance_attrs(params, label))
    return bundle


def correct_drift(sensor, sn, drift):
    """Subtract a sensor's interpolated drift from its L1 series (L1 -> L2).

    Promoted from MOTIVE notebook 06's notebook-local helper. ``drift`` is
    the drift product already re-dimensioned to ``(sn, time)`` (window
    centres on the ``time`` dim). The per-sensor drift is interpolated onto
    the sensor's sample times and subtracted; ``fill_value="extrapolate"``
    linearly extends the fit outside its window (e.g. the few hours at
    deployment start before the first window centre) so the corrected
    series has no NaN edge.

    Parameters
    ----------
    sensor : xr.DataArray
        Per-sensor L1 temperature series with a ``time`` coord.
    sn : int
        Serial number to select from ``drift``.
    drift : xr.DataArray
        Drift product with dims ``(sn, time)`` (i.e. already
        ``swap_dims({"depth": "sn", "window": "time"})``-ed).

    Returns
    -------
    xr.DataArray
        ``sensor`` minus the interpolated per-sensor drift (the L2 series).
    """
    drifts = drift.sel(sn=sn)
    driftsi = drifts.interp_like(sensor.time, kwargs=dict(fill_value="extrapolate"))
    return sensor - driftsi


_CAL_STOP_COLUMNS = ["stop_start", "stop_end", "mean_p", "mean_t", "duration_s"]
_CAL_STOPS_REQUIRED_COLS = ["source", "cast", "stop_start", "stop_end", "ctd_file"]


def detect_cal_stops(
    ctd,
    *,
    p_std_thresh=0.5,
    min_duration="60s",
    smooth_window="30s",
    bridge_gap="20s",
):
    """Propose CTD cal-stop windows from a cast's pressure trace.

    Promoted from MOTIVE notebook 01b's ``find_cal_stops``. A cal stop is
    a contiguous run where the rolling standard deviation of pressure
    stays below ``p_std_thresh`` (dbar) within ``smooth_window``. Adjacent
    plateaus separated by less than ``bridge_gap`` are merged; plateaus
    shorter than ``min_duration`` are dropped. Pure and deterministic —
    it *proposes* candidate windows; the human confirms/nudges them and
    writes the chosen ones to ``cal_stops.csv`` (which
    :meth:`Mooring.compute_ctd_offsets` then consumes).

    Parameters
    ----------
    ctd : xr.Dataset
        Cast dataset with a time-indexed ``p`` (dbar) and ``t1`` (degC).
    p_std_thresh : float, optional
        Pressure rolling-std threshold in dbar. Default 0.5.
    min_duration : str, optional
        Pandas offset; minimum plateau length to keep. Default ``"60s"``.
    smooth_window : str, optional
        Pandas offset for the rolling-std window. Default ``"30s"``.
    bridge_gap : str, optional
        Pandas offset; merge plateaus separated by less than this.
        Default ``"20s"``.

    Returns
    -------
    pd.DataFrame
        One row per candidate stop with columns ``stop_start``,
        ``stop_end`` (``np.datetime64``), ``mean_p`` / ``mean_t`` (mean
        pressure / CTD ``t1`` over the window), and ``duration_s``.
    """
    p = ctd.p.dropna(dim="time")
    times = pd.to_datetime(p.time.values)
    if times.size < 2:
        return pd.DataFrame(columns=_CAL_STOP_COLUMNS)
    dt_s = (times[1] - times[0]).total_seconds()
    nwin = max(1, int(pd.Timedelta(smooth_window).total_seconds() / dt_s))
    rstd = p.rolling(time=nwin, center=True).std().values
    flag = np.isfinite(rstd) & (rstd < p_std_thresh)
    if not flag.any():
        return pd.DataFrame(columns=_CAL_STOP_COLUMNS)

    idx = np.flatnonzero(flag)
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    intervals = [(times[s], times[e]) for s, e in zip(starts, ends)]

    bridge = pd.Timedelta(bridge_gap)
    merged = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= bridge:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    min_dur = pd.Timedelta(min_duration)
    rows = []
    for s, e in merged:
        if (e - s) < min_dur:
            continue
        sl = slice(np.datetime64(s), np.datetime64(e))
        rows.append({
            "stop_start": np.datetime64(s),
            "stop_end": np.datetime64(e),
            "mean_p": float(ctd.p.sel(time=sl).mean()),
            "mean_t": float(ctd.t1.sel(time=sl).mean()),
            "duration_s": float((e - s).total_seconds()),
        })
    return pd.DataFrame(rows, columns=_CAL_STOP_COLUMNS)


def parse_gridding(block, defaults=None):
    """Validate a gridding block and convert dt/max_gap/chunk to np.timedelta64.

    Parameters
    ----------
    block : dict or None
        Per-segment gridding overrides (pandas-style duration strings).
    defaults : dict or None, optional
        Top-level gridding defaults; ``block`` keys override these.

    Returns
    -------
    dict
        All three required keys {dt, max_gap, chunk} as np.timedelta64.

    Raises
    ------
    ValueError
        If ``block`` contains a key outside {dt, max_gap, chunk}.
    ValueError
        If the merged result is missing any of the three required keys.
    """
    block = dict(block or {})
    unknown = set(block) - _GRIDDING_KEYS
    if unknown:
        raise ValueError(f"unknown gridding keys: {sorted(unknown)}")
    merged = dict(defaults or {})
    merged.update(block)
    out = {k: pd.Timedelta(v).to_timedelta64() for k, v in merged.items() if k in _GRIDDING_KEYS}
    missing = _GRIDDING_KEYS - set(out)
    if missing:
        raise ValueError(f"gridding is missing required keys: {sorted(missing)}")
    return out


def resolve_segment_sns(mooring_info, select, root):
    """Return a pandas.Index of SNs for one segment's ``select`` spec.

    ``select`` accepts exactly one of:
      - ``{"segment": name}`` — match the ``segment`` column (PRIMARY).
      - ``{"sns": [..]}``      — explicit serial-number list (fallback).
      - ``{"layout": file}``   — CSV with an ``SN`` column (fallback/migration),
        resolved relative to ``root``.

    Selection is structural only; quality drops (``ignore_sns``) are applied
    separately by the consuming stage.

    Parameters
    ----------
    mooring_info : pd.DataFrame
        Mooring sensor sheet with SN as the index.
    select : dict
        Selection spec with exactly one of the keys above.
    root : str or Path
        Root directory for resolving relative ``layout`` CSV paths.

    Returns
    -------
    pd.Index
        Index of integer serial numbers matching the selection.

    Raises
    ------
    ValueError
        If ``select`` uses ``segment`` but the mooring sheet has no ``segment`` column.
    ValueError
        If ``select`` contains none of the recognised keys.
    """
    select = dict(select or {})
    if "segment" in select:
        if "segment" not in mooring_info.columns:
            raise ValueError("select.segment requires a 'segment' column in the mooring sheet")
        return mooring_info.index[mooring_info["segment"] == select["segment"]]
    if "sns" in select:
        return pd.Index([int(s) for s in select["sns"]], name="SN")
    if "layout" in select:
        layout_path = Path(root).joinpath(select["layout"])
        return pd.Index([int(s) for s in pd.read_csv(layout_path)["SN"]], name="SN")
    raise ValueError(f"segment select needs one of segment/sns/layout: {select!r}")


class Mooring(ProcessThermistorMooring):
    """Config-driven moored-thermistor pipeline for one mooring.

    Extends the L0 processing base with segment resolution and the
    gridding stage. Built from a single YAML config (see the schema in
    plans/generalize-thermistor-pipeline.md).
    """

    def __init__(self, config_file, project_root=None, data_root=None):
        """Initialize the Mooring pipeline.

        Parameters
        ----------
        config_file : pathlib.Path or str
            YAML config file for this mooring.
        project_root : pathlib.Path or None, optional
            Forwarded to ``load_config_box``; defaults to
            ``configfile.parent.parent``.
        data_root : pathlib.Path or None, optional
            Forwarded to ``load_config_box``.
        """
        super().__init__(config_file, project_root=project_root, data_root=data_root)
        self.segments_cfg = {
            name: dict(seg)
            for name, seg in (self.cfg.get("segments", {}) or {}).items()
        }
        default_grid = dict(self.cfg.get("gridding", {}) or {})
        self.gridding = {
            name: parse_gridding(seg.get("gridding"), defaults=default_grid)
            for name, seg in self.segments_cfg.items()
        }

    def process_l0(self):
        """Run the L0 stage (alias of the inherited :meth:`run_proc_all`).

        Provided so the public stage surface reads consistently with the
        other pipeline stages (``process_l0``, ``cut_and_cal``, ``grid_l1``,
        …). Delegates verbatim to :meth:`run_proc_all`: it converts every
        raw sensor file to a clock-calibrated per-sensor L0 NetCDF in
        ``path.procl0``.

        Returns
        -------
        None
        """
        return self.run_proc_all()

    def run(self, stages=None, segments=None, overwrite=False):
        """Run the full pipeline (or a named sub-chain) honoring the config.

        Pure orchestration over the validated stage methods — no new
        numerical code. The canonical chain (dissipation intentionally
        excluded) is::

            process_l0 -> compute_ctd_offsets -> cut_and_cal
              -> grid_l1 -> fit_drift -> make_l2 -> grid_l2

        Stages are always executed in :data:`STAGE_ORDER`; ``stages=`` only
        *selects* a subset, it does not reorder. The drift stages
        (``fit_drift`` / ``make_l2`` / ``grid_l2``) already no-op on
        segments without ``drift: true``, so ``run`` just calls them. Each
        stage is idempotent (skips existing outputs); ``overwrite=True``
        forwards to every stage.

        Parameters
        ----------
        stages : list of str or None, optional
            Subset of :data:`STAGE_ORDER` to run (e.g.
            ``["cut_and_cal", "grid_l1"]`` to skip the expensive L0 step).
            ``None`` runs the full chain.
        segments : str, list of str, or None, optional
            Segment(s) forwarded to the segment-aware stages. ``None``
            runs every defined (or drift-eligible) segment.
        overwrite : bool, optional
            Forwarded to every stage. Default ``False``.

        Returns
        -------
        dict
            ``{stage_name: <that stage's return value>}`` for each stage run.

        Raises
        ------
        ValueError
            If ``stages`` names anything outside :data:`STAGE_ORDER`.
        """
        if stages is None:
            selected = list(STAGE_ORDER)
        else:
            unknown = [s for s in stages if s not in STAGE_ORDER]
            if unknown:
                raise ValueError(
                    f"unknown stage(s): {unknown}; allowed: {STAGE_ORDER}"
                )
            wanted = set(stages)
            selected = [s for s in STAGE_ORDER if s in wanted]

        results = {}
        for name in selected:
            method = getattr(self, name)
            if name == "process_l0":
                results[name] = method()          # run_proc_all takes no args
            elif name in _STAGES_NO_SEGMENTS:
                results[name] = method(overwrite=overwrite)
            else:
                results[name] = method(segments=segments, overwrite=overwrite)
        return results

    def _segment_names(self, segments):
        """Validate and return a list of segment names.

        Parameters
        ----------
        segments : str, list of str, or None
            Segment(s) to validate. ``None`` returns all defined segments.

        Returns
        -------
        list of str
        """
        if segments is None:
            return list(self.segments_cfg)
        if isinstance(segments, str):
            segments = [segments]
        for s in segments:
            if s not in self.segments_cfg:
                raise KeyError(f"unknown segment {s!r}; defined: {list(self.segments_cfg)}")
        return list(segments)

    def _ignore_sns(self):
        """SNs to drop everywhere: config ``ignore_sns`` UNION sensor-sheet ``exclude == 1``.

        Structure (segment membership) and quality (these drops) are
        separate axes; this is the quality axis, applied on top of
        segment selection by every consuming stage.

        Returns
        -------
        list of int
            Sorted list of serial numbers to exclude from all processing stages.
        """
        ignore = {int(s) for s in (self.cfg.get("ignore_sns", []) or [])}
        if "exclude" in self.sensor_info.columns:
            flagged = pd.to_numeric(self.sensor_info["exclude"], errors="coerce") == 1
            ignore |= {int(s) for s in self.sensor_info.index[flagged]}
        return sorted(ignore)

    def segment_sensors(self, segment):
        """Return mooring_info rows for one segment (structural selection only).

        Parameters
        ----------
        segment : str
            Segment name as defined in the config ``segments`` block.

        Returns
        -------
        pd.DataFrame
            Subset of ``mooring_info`` whose SNs belong to ``segment``.
        """
        sns = resolve_segment_sns(
            self.mooring_info,
            self.segments_cfg[segment]["select"],
            self.cfg.path.root,
        )
        return self.mooring_info.loc[self.mooring_info.index.intersection(sns)]

    def _procl0_dir(self):
        return Path(self.cfg.path.data.procl0)

    def _procl1_dir(self):
        return Path(self.cfg.path.data.procl1)

    def _procl2_dir(self):
        return Path(self.cfg.path.data.procl2)

    def _segment_cal_method(self, segment):
        """Resolve a segment's calibration method (segment override, else top-level default)."""
        seg = self.segments_cfg[segment]
        method = (seg.get("calibration") or {}).get("method")
        if method is None:
            method = (self.cfg.get("calibration", {}) or {}).get("method", "linear_interp")
        if method not in _CAL_METHODS:
            raise ValueError(
                f"unknown calibration method {method!r}; allowed: {sorted(_CAL_METHODS)}"
            )
        return method

    def _resolve_offsets(self, ctdcal_pre, ctdcal_post):
        """Return (pre, post) offsets, loading from config paths when args are None."""
        cal = self.cfg.get("calibration", {}) or {}

        def _load(key):
            p = cal.get(key)
            if p is None:
                return None
            p = Path(p)
            if not p.is_absolute():
                p = Path(self.cfg.path.root) / p
            return load_cal_offsets(p)

        if ctdcal_pre is None:
            ctdcal_pre = _load("offsets_pre")
        if ctdcal_post is None:
            ctdcal_post = _load("offsets_post")
        return ctdcal_pre, ctdcal_post

    def _ctd_dir(self):
        """Base directory for CTD cal-cast files (``path.ctd``)."""
        if "ctd" not in self.cfg.path:
            raise KeyError("path.ctd not set in config")
        d = Path(self.cfg.path.ctd)
        if not d.is_absolute():
            d = Path(self.cfg.path.root) / d
        return d

    def _load_cal_stops(self):
        """Read the declarative per-cast cal-stop windows (``path.cal_stops``).

        Columns: ``source`` (pre/post), ``cast`` (int), ``stop_start`` /
        ``stop_end`` (ISO datetimes), ``ctd_file`` (relative to
        :meth:`_ctd_dir`), and advisory ``ref_temp``.
        """
        if "cal_stops" not in self.cfg.path:
            raise KeyError("path.cal_stops not set in config")
        p = Path(self.cfg.path.cal_stops)
        if not p.is_absolute():
            p = Path(self.cfg.path.root) / p
        df = pd.read_csv(p)
        missing = [c for c in _CAL_STOPS_REQUIRED_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"cal_stops file {p} missing required column(s): {missing}")
        return df

    def _cal_cast_pool_dir(self, source):
        """Pool dir of per-sensor cal-cast series (``calibration.cal_casts_{source}``)."""
        cal = self.cfg.get("calibration", {}) or {}
        key = f"cal_casts_{source}"
        if key not in cal:
            raise KeyError(f"calibration.{key} not set in config")
        d = Path(cal[key])
        if not d.is_absolute():
            d = Path(self.cfg.path.root) / d
        return d

    def _offsets_out_path(self, source):
        """Output offsets NetCDF for a source (``calibration.offsets_{source}``)."""
        cal = self.cfg.get("calibration", {}) or {}
        key = f"offsets_{source}"
        if key not in cal:
            raise KeyError(f"calibration.{key} not set in config")
        p = Path(cal[key])
        if not p.is_absolute():
            p = Path(self.cfg.path.root) / p
        return p

    def compute_ctd_offsets(self, sources=None, overwrite=False):
        """Compute per-sensor CTD-cal offsets and write the offsets NetCDF(s).

        The deterministic CTD-offset stage. Consumes the declarative
        ``cal_stops.csv`` (:meth:`_load_cal_stops`), the per-source cal-cast
        pools (:meth:`_cal_cast_pool_dir`), and the sensor-sheet cast
        assignments (``pre_ctd_cal_cast`` / ``post_ctd_cal_cast``). For each
        chosen stop it concatenates every assigned sensor's cal-cast series
        onto a common time grid (``interp_like`` the first sensor), calls the
        unchanged :func:`rbr_ctd_cal_find_offset` kernel, takes the
        both-CTD-sensors mean difference (``m=0``), and assigns per-sensor
        ``cast`` / ``cal_temp`` (= ``mean_temp.mean('sn')``) coords. Offsets
        are concatenated across casts, ``sortby('sn')``-ed, and written to
        ``calibration.offsets_{source}`` — the file Phase-1b
        :meth:`cut_and_cal` reads via :func:`load_cal_offsets`.

        Sensor selection is ``(assignment == cast)`` intersected with
        pool-file presence, in sensor-sheet order. The stage operates over the
        full project sensor sheet, so it produces the **project-wide** offsets
        files. Idempotent: an existing output is skipped unless
        ``overwrite=True``.

        An optional per-source ignore list ``calibration.cal_ignore_sns_{source}``
        (e.g. ``cal_ignore_sns_pre``) excludes listed SNs even when a valid pool
        file exists — parallel to the notebooks' ``pre_cal_ignore_sns`` /
        ``post_cal_ignore_sns``. Pool files that are unreadable (e.g. no data
        variables), empty (zero-length time), or have no samples within the
        cast period are skipped with a warning rather than raising, so a
        degenerate or non-overlapping file does not abort the full source.

        Parameters
        ----------
        sources : str, list of str, or None, optional
            ``"pre"`` / ``"post"`` source(s). ``None`` runs every source
            present in ``cal_stops.csv``.
        overwrite : bool, optional
            Rewrite existing offsets files. Default ``False``.

        Returns
        -------
        dict
            ``{source: {"written": int, "skipped": int, "path": str}}``
            (``written`` is the sensor count; ``skipped`` is 1 when the
            output already existed and was kept).
        """
        cal_stops = self._load_cal_stops()
        if sources is None:
            sources = sorted(cal_stops["source"].unique())
        elif isinstance(sources, str):
            sources = [sources]

        assign_col = {"pre": "pre_ctd_cal_cast", "post": "post_ctd_cal_cast"}
        summary = {}
        for source in sources:
            if source not in assign_col:
                raise ValueError(f"source {source!r} not supported; expected 'pre' or 'post'")
            out_path = self._offsets_out_path(source)
            if out_path.exists() and not overwrite:
                logger.info(f"{source}: offsets {out_path.name} exists, skipping")
                summary[source] = {"written": 0, "skipped": 1, "path": str(out_path)}
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pool = self._cal_cast_pool_dir(source)
            col = assign_col[source]
            rows = cal_stops[cal_stops["source"] == source]

            cal = self.cfg.get("calibration", {}) or {}
            ignore = set(int(s) for s in (cal.get(f"cal_ignore_sns_{source}", []) or []))

            parts = []
            for _, r in rows.iterrows():
                cast_no = int(r["cast"])
                ctd = xr.open_dataset(self._ctd_dir() / str(r["ctd_file"]))
                cast_period = slice(ctd.time.values[0], ctd.time.values[-1])

                assign = pd.to_numeric(self.sensor_info[col], errors="coerce")
                assigned = [int(sn) for sn in self.sensor_info.index[assign == cast_no]]
                assigned = [sn for sn in assigned if sn not in ignore]
                cals, kept = [], []
                for sn in assigned:
                    files = list(pool.glob(f"*{sn:06d}*.nc"))
                    if not files:
                        continue                      # absent from pool -> cal-ignored
                    if len(files) > 1:
                        raise OSError(
                            f"ambiguous pool files for SN{sn:06d} in {pool}: {[f.name for f in files]}"
                        )
                    try:
                        da = xr.open_dataarray(files[0])
                    except ValueError as exc:
                        logger.warning(
                            f"{source} SN{sn:06d}: unreadable pool file {files[0].name} ({exc}); skipping"
                        )
                        continue
                    if da.sizes.get("time", 0) == 0:
                        da.close()
                        logger.warning(
                            f"{source} SN{sn:06d}: empty pool file {files[0].name}; skipping"
                        )
                        continue
                    sliced = da.sel(time=cast_period)
                    if sliced.sizes.get("time", 0) == 0:
                        da.close()
                        logger.warning(
                            f"{source} SN{sn:06d}: pool file {files[0].name} has no samples in cast period; skipping"
                        )
                        continue
                    cals.append(sliced)
                    kept.append(sn)
                if not cals:
                    logger.warning(f"{source} cast {cast_no}: no sensors in pool, skipping")
                    ctd.close()
                    continue

                time = cals[0].time.copy()
                c = xr.concat([ci.interp_like(time) for ci in cals], dim="n")
                for ci in cals:
                    ci.close()
                c["sn"] = (("n"), kept)
                ts = slice(np.datetime64(r["stop_start"]), np.datetime64(r["stop_end"]))
                res = rbr_ctd_cal_find_offset(ts, c, ctd)

                offsets = res.isel(m=0).mean_diff.drop_vars("sensor", errors="ignore")
                temp = float(res.mean_temp.mean(dim="sn").data)
                n_sn = offsets.sizes["sn"]
                part = offsets.assign_coords(
                    cast=("sn", np.full(n_sn, cast_no, dtype=int)),
                    cal_temp=("sn", np.full(n_sn, temp)),
                    temp=("sn", np.full(n_sn, temp)),
                )
                parts.append(part)
                ctd.close()

            if not parts:
                logger.warning(f"{source}: no offsets computed, nothing written")
                summary[source] = {"written": 0, "skipped": 0, "path": str(out_path)}
                continue
            out = xr.concat(parts, dim="sn", coords="minimal").sortby("sn")
            out.name = "offset"
            out.to_netcdf(out_path, mode="w")
            summary[source] = {
                "written": int(out.sizes["sn"]), "skipped": 0, "path": str(out_path),
            }
        return summary

    def _cast_time(self, sn, col):
        """Per-sensor CTD cast time from the sensor sheet, or NaT if absent.

        Guards against a duplicate-SN sensor sheet (``.loc`` would return a
        Series): take the first matching row.
        """
        if sn in self.sensor_info.index and col in self.sensor_info.columns:
            val = self.sensor_info.loc[sn][col]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            return val
        return pd.NaT

    def cut_and_cal(
        self, segments=None, ctdcal_pre=None, ctdcal_post=None,
        end_manually=None, overwrite=False,
    ):
        """Trim L0 to the deployment window and apply per-segment CTD calibration.

        Dispatches each segment's ``calibration.method`` onto the unchanged
        primitives:

        - ``scalar_pre_only`` -> :func:`rbr_cut_and_cal` with the pre-cast
          offset only (the deep-array recipe; drift handled downstream).
        - ``linear_interp`` -> :func:`rbr_cut_and_cal_interp` (two-point
          pre->post, with the primitive's own per-sensor scalar/none
          fallback; recorded ``cal_method`` is the *effective* value).
        - ``scalar`` -> :func:`rbr_cut_and_cal` with whichever single side
          the sensor has (pre preferred).
        - ``none`` -> cut only, no offset applied.

        Offsets default to the ``calibration.offsets_pre`` /
        ``offsets_post`` config paths (load via :func:`load_cal_offsets`)
        unless passed in-memory. ``ignore_sns`` (config UNION sensor-sheet
        ``exclude==1``) is applied on top of segment selection. Idempotent:
        existing L1 files are skipped unless ``overwrite=True``.

        Parameters
        ----------
        segments : str, list of str, or None, optional
            Segment(s) to process. ``None`` processes all defined segments.
        ctdcal_pre, ctdcal_post : xr.DataArray or None, optional
            In-memory offsets; override the config paths when given.
        end_manually : dict or None, optional
            ``{sn: np.datetime64}`` hard cut-window end overrides.
        overwrite : bool, optional
            Rewrite existing L1 files. Default ``False``.

        Returns
        -------
        dict
            ``{segment: {"written": int, "skipped": int, "methods": {m: n}}}``.
        """
        end_manually = end_manually or {}
        ctdcal_pre, ctdcal_post = self._resolve_offsets(ctdcal_pre, ctdcal_post)
        l0dir = self._procl0_dir()
        l1dir = self._procl1_dir()
        l1dir.mkdir(parents=True, exist_ok=True)
        ignore = set(self._ignore_sns())
        summary = {}
        for seg in self._segment_names(segments):
            method = self._segment_cal_method(seg)
            written = skipped = 0
            methods = {}
            for sn in self.segment_sensors(seg).index:
                sn = int(sn)
                if sn in ignore:
                    continue
                sensor_type = str(self.mooring_info.loc[sn]["type"])
                savename = (
                    f"{self.meta.mooring_name.lower().replace(' ', '_')}"
                    f"__{sensor_type.lower()}__{sn:06}_L1.nc"
                )
                l1_path = l1dir / savename
                if l1_path.exists():
                    if not overwrite:
                        skipped += 1
                        continue
                    l1_path.unlink()

                t_pre = self._cast_time(sn, "pre_ctd_cal_time")
                t_post = self._cast_time(sn, "post_ctd_cal_time")
                try:
                    eff_method, pre_applied, post_applied = self._run_one_cut_cal(
                        sn, method, l0dir, l1dir, sensor_type,
                        ctdcal_pre, ctdcal_post, t_pre, t_post, end_manually,
                    )
                except FileNotFoundError as exc:
                    logger.warning(f"SN{sn}: L0 missing ({exc}), skipping")
                    continue
                if eff_method == "empty":
                    continue  # no L1 written (cut window had no samples)

                da = xr.load_dataarray(l1_path)
                da.attrs.update(cal_diagnostic_attrs(
                    sn, ctdcal_pre, ctdcal_post, cal_method=eff_method,
                    pre_applied=pre_applied, post_applied=post_applied,
                    t_pre=t_pre, t_post=t_post,
                ))
                da.to_netcdf(l1_path, mode="w")
                written += 1
                methods[eff_method] = methods.get(eff_method, 0) + 1
            summary[seg] = {"written": written, "skipped": skipped, "methods": methods}
        return summary

    def _run_one_cut_cal(
        self, sn, method, l0dir, l1dir, sensor_type,
        ctdcal_pre, ctdcal_post, t_pre, t_post, end_manually,
    ):
        """Call the right primitive for one sensor; return (eff_method, pre_applied, post_applied)."""
        pre_has = ctdcal_pre is not None and sn in ctdcal_pre.sn
        post_has = ctdcal_post is not None and sn in ctdcal_post.sn
        common = dict(
            sn=sn, l0dir=l0dir, l1dir=l1dir,
            cut_beg=self.cfg.start_time, cut_end=self.cfg.end_time,
            end_manually=end_manually, mooring_name=self.meta.mooring_name,
            sensor_type=sensor_type,
        )
        if method == "scalar_pre_only":
            rbr_cut_and_cal(ctdcal=ctdcal_pre, **common)
            return "scalar_pre_only", pre_has, False
        if method == "scalar":
            ctdcal = ctdcal_pre if pre_has else ctdcal_post
            rbr_cut_and_cal(ctdcal=ctdcal, **common)
            return "scalar", pre_has, (post_has and not pre_has)
        if method == "none":
            _, eff = rbr_cut_and_cal_interp(
                ctdcal_pre=None, ctdcal_post=None, t_pre=t_pre, t_post=t_post, **common
            )
            return eff, False, False
        # linear_interp (default): primitive resolves per-sensor effective method
        _, eff = rbr_cut_and_cal_interp(
            ctdcal_pre=ctdcal_pre, ctdcal_post=ctdcal_post,
            t_pre=t_pre, t_post=t_post, **common
        )
        if eff == "linear_interp":
            return eff, True, True
        if eff == "scalar":
            return eff, pre_has, (post_has and not pre_has)
        return eff, False, False  # "none" or "empty"

    def _aux_dir(self):
        d = Path(self.cfg.path.aux)
        if not d.is_absolute():
            d = Path(self.cfg.path.root) / d
        return d

    def _gridl1_dir(self):
        """Directory holding gridded-L1 chunks (``grid/l1/``; the drift-fit input).

        Both :meth:`grid_l1` (writer) and :meth:`fit_drift` (reader) resolve
        here, matching the on-disk layout from notebook 03 / 05a.
        """
        return self._grid_dir() / "l1"

    def _gridl2_dir(self):
        """Directory holding gridded-L2 chunks (``grid/l2/``).

        Mirrors :meth:`_gridl1_dir`; :meth:`grid_l2` writes here.
        """
        return self._grid_dir() / "l2"

    def _drift_segments(self):
        """Segment names flagged ``drift: true`` in the config."""
        return [name for name, seg in self.segments_cfg.items() if seg.get("drift")]

    def _drift_segment_names(self, segments):
        """Validate ``segments`` against the drift-eligible set."""
        drift = self._drift_segments()
        if segments is None:
            return drift
        names = [segments] if isinstance(segments, str) else list(segments)
        for s in names:
            if s not in self.segments_cfg:
                raise KeyError(f"unknown segment {s!r}; defined: {list(self.segments_cfg)}")
            if s not in drift:
                raise ValueError(
                    f"segment {s!r} is not a drift segment (set drift: true to enable)"
                )
        return names

    def fit_drift(
        self, segments=None, drift_parameters=None, label=None,
        gridl1_dir=None, first_n_chunks=None,
        window_length=np.timedelta64(1, "D"), overwrite=False,
    ):
        """Run the CvHG16 drift fit on the ``drift: true`` segments.

        Wraps the unchanged :class:`thermochain.io.sensor_drift`
        (``run_all=True``) + ``drift_to_netcdf`` primitives. For each drift
        segment it fits the gridded-L1 deep array, writes the drift product
        ``drift_{project}_{mooring}_{label}.nc`` and the diagnostic bundle
        ``diag_{project}_{mooring}_{label}.nc`` (both in ``path.aux``), and
        stamps the full validated ``drift_parameters`` as provenance attrs.

        The base ``drift_parameters`` come from the config block; a ``label``
        there (or ``label=``) names the product. Pass ``drift_parameters=``
        and ``label=`` to write side-by-side sweeps (e.g. ``lin`` vs
        ``slowtau``) without editing the YAML. Unknown parameter keys are
        rejected via :class:`DriftParameters` (typo guard).

        Idempotent: a segment whose drift + diag products already exist is
        skipped (its return value is ``None``) unless ``overwrite=True``.

        .. note::
           Holding two fitted ``sensor_drift`` instances live at once can
           segfault xarray's groupby C-state (documented in notebook 05a).
           When sweeping labels, drop the returned object (``del``; ``gc``)
           before the next ``fit_drift`` call.

        Parameters
        ----------
        segments : str, list of str, or None, optional
            Drift segment(s). ``None`` runs all ``drift: true`` segments.
        drift_parameters : dict or None, optional
            Overrides merged over the config block (may include ``label``).
        label : str or None, optional
            Product label; overrides the config / ``drift_parameters`` label.
        gridl1_dir : pathlib.Path or None, optional
            Gridded-L1 input dir; defaults to :meth:`_gridl1_dir`.
        first_n_chunks : int or None, optional
            Cap on gridded-L1 chunks read (``None`` = all).
        window_length : np.timedelta64, optional
            Background-fit window length. Default 1 day.
        overwrite : bool, optional
            Refit even if the products exist. Default ``False``.

        Returns
        -------
        dict
            ``{segment: sensor_drift | None}`` (``None`` where skipped).
        """
        base = dict(self.cfg.get("drift_parameters", {}) or {})
        config_label = base.pop("label", None)
        override_label = None
        if drift_parameters:
            override = dict(drift_parameters)
            override_label = override.pop("label", None)
            base.update(override)
        # priority: explicit label= arg > drift_parameters override label > config label
        label = label or override_label or config_label
        if label is None:
            raise ValueError(
                "fit_drift needs a label (config drift_parameters.label or label=)"
            )
        params = DriftParameters.from_dict(base)   # validates + rejects unknown keys

        aux = self._aux_dir()
        aux.mkdir(parents=True, exist_ok=True)
        gridl1_dir = Path(gridl1_dir) if gridl1_dir is not None else self._gridl1_dir()
        mooring_id = f"{self.meta.project.lower()}_{self.meta.mooring_name.lower()}"

        results = {}
        for seg in self._drift_segment_names(segments):
            drift_path = aux / f"drift_{mooring_id}_{label}.nc"
            diag_path = aux / f"diag_{mooring_id}_{label}.nc"
            if drift_path.exists() and diag_path.exists() and not overwrite:
                logger.info(f"{seg}: drift products for {label!r} exist, skipping")
                results[seg] = None
                continue

            file_pattern = f"{mooring_id}_{seg}_L1_*.nc"
            sd = sensor_drift(
                mooring_name=mooring_id,
                l1_grid_dir=gridl1_dir,
                file_pattern=file_pattern,
                window_length=window_length,
                drift_parameters=params.as_dict(),
                first_n_chunks=first_n_chunks,
                run_all=True,
            )
            sd.drift_to_netcdf(path=aux, suffix=label)
            self._stamp_drift_provenance(drift_path, params, label)
            drift_diag_bundle(sd, params, label).to_netcdf(diag_path, mode="w")
            results[seg] = sd
        return results

    def _stamp_drift_provenance(self, drift_path, params, label):
        """Re-open the drift product and add the provenance attrs in place."""
        da = xr.load_dataarray(drift_path)
        da.attrs.update(drift_provenance_attrs(params, label))
        da.to_netcdf(drift_path, mode="w")

    def _l2_filename(self, sn, sensor_type):
        """Per-sensor L2 filename (project-prefixed, matching notebook 06).

        ``{project}_{mooring}__{type}__{sn:06}_L2.nc`` (e.g.
        ``motive_a__rbr__236109_L2.nc``). Differs from the per-sensor L1
        name (no project prefix) — flagged for the reorg phase.
        """
        return (
            f"{self.meta.project.lower()}_{self.meta.mooring_name.lower()}"
            f"__{sensor_type.lower()}__{sn:06}_L2.nc"
        )

    def make_l2(self, segments=None, drift_label=None, overwrite=False):
        """Subtract the drift product from per-sensor L1 to make per-sensor L2.

        Runs only on the ``drift: true`` segments (drift handled downstream
        of cut/cal). For each segment it loads the drift product
        ``drift_{project}_{mooring}_{label}.nc`` from ``path.aux``
        (``label`` defaults to the config ``drift_parameters.label``;
        override with ``drift_label=``), re-dimensions it to ``(sn, time)``,
        and for each sensor interpolates the drift onto that sensor's L1
        sample times and subtracts (via :func:`correct_drift`). The L2
        series carries the L1 attrs plus an ``sn`` attr (matching notebook
        06). ``ignore_sns`` (config UNION sensor-sheet ``exclude==1``) is
        applied on top of segment selection. Idempotent: existing L2 files
        are skipped unless ``overwrite=True``.

        Per-sensor L1 is read from :meth:`_procl1_dir`; L2 is written to
        :meth:`_procl2_dir`. Kept separate from :meth:`grid_l2` so the
        per-sensor L2 is inspectable before gridding.

        Parameters
        ----------
        segments : str, list of str, or None, optional
            Drift segment(s). ``None`` runs all ``drift: true`` segments.
        drift_label : str or None, optional
            Drift-product label to consume. Defaults to the config
            ``drift_parameters.label``.
        overwrite : bool, optional
            Rewrite existing L2 files. Default ``False``.

        Returns
        -------
        dict
            ``{segment: {"written": int, "skipped": int}}``.
        """
        label = drift_label or (self.cfg.get("drift_parameters", {}) or {}).get("label")
        if label is None:
            raise ValueError(
                "make_l2 needs a drift label (config drift_parameters.label or drift_label=)"
            )
        aux = self._aux_dir()
        procl1 = self._procl1_dir()
        procl2 = self._procl2_dir()
        procl2.mkdir(parents=True, exist_ok=True)
        ignore = set(self._ignore_sns())
        mooring_id = f"{self.meta.project.lower()}_{self.meta.mooring_name.lower()}"

        summary = {}
        for seg in self._drift_segment_names(segments):
            drift_path = aux / f"drift_{mooring_id}_{label}.nc"
            if not drift_path.exists():
                raise FileNotFoundError(f"drift product not found: {drift_path}")
            drift = xr.open_dataarray(drift_path).swap_dims(
                {"depth": "sn", "window": "time"}
            )
            written = skipped = 0
            for sn in self.segment_sensors(seg).index:
                sn = int(sn)
                if sn in ignore:
                    continue
                sensor_type = str(self.mooring_info.loc[sn]["type"])
                l2_path = procl2 / self._l2_filename(sn, sensor_type)
                if l2_path.exists():
                    if not overwrite:
                        skipped += 1
                        continue
                    l2_path.unlink()

                l1_files = list(procl1.glob(f"*__{sn:06d}_L1.nc"))
                if not l1_files:
                    logger.warning(f"SN{sn}: no L1 file in {procl1}, skipping")
                    continue
                tmp = xr.open_dataarray(l1_files[0])
                tc = correct_drift(tmp, sn, drift)
                tc.attrs = tmp.attrs.copy()
                tc.attrs["sn"] = sn
                tc.to_netcdf(l2_path, mode="w")
                tc.close()
                tmp.close()
                written += 1
            drift.close()
            summary[seg] = {"written": written, "skipped": skipped}
        return summary

    def _grid_dir(self):
        d = Path(self.cfg.path.data.grid)
        if not d.is_absolute():
            d = Path(self.cfg.path.root) / d
        return d

    def _grid_filename(self, segment, level, ti):
        stamp = np.datetime_as_string(np.datetime64(ti, "s")).replace("-", "").replace(":", "")
        return (
            f"{self.meta.project.lower()}_{self.meta.mooring_name.lower()}"
            f"_{segment}_L{level}_{stamp}.nc"
        )

    def grid_l1(self, segments=None, overwrite=False):
        """Grid per-sensor L1 to (depth, time) chunks, one set per segment.

        Mirrors notebook 03: build the full deployment once via
        ``grid_thermistors`` (peak RAM bounded by one (depth x time)
        array), then slice and write fixed-length chunks. Idempotent —
        existing chunk files are skipped unless ``overwrite=True``.
        ``ignore_sns`` is applied via ``exclude_sn`` (structure vs.
        quality kept separate).

        Parameters
        ----------
        segments : str, list of str, or None, optional
            Segment(s) to process. ``None`` processes all defined segments.
        overwrite : bool, optional
            If ``True``, rewrite existing chunk files. Default ``False``.

        Returns
        -------
        dict
            ``{segment: {"written": int, "skipped": int, "chunks": int}}``.
        """
        grid_dir = self._gridl1_dir()
        grid_dir.mkdir(parents=True, exist_ok=True)
        procl1 = Path(self.cfg.path.data.procl1)
        summary = {}
        for seg in self._segment_names(segments):
            gp = self.gridding[seg]
            info_sub = self.segment_sensors(seg)
            start_times = np.arange(
                self.cfg.start_time, self.cfg.end_time, gp["chunk"], dtype="datetime64[s]"
            )
            written = skipped = 0
            full = None
            for ti in start_times:
                fpath = grid_dir / self._grid_filename(seg, 1, ti)
                if fpath.exists() and not overwrite:
                    skipped += 1
                    continue
                if full is None:
                    full = grid_thermistors(
                        info_sub,
                        procl1,
                        start=self.cfg.start_time,
                        end=self.cfg.end_time,
                        dt=gp["dt"],
                        max_gap=gp["max_gap"],
                        exclude_sn=self._ignore_sns(),
                    )
                full.sel(time=slice(ti, ti + gp["chunk"])).to_netcdf(fpath)
                written += 1
            if full is not None:
                del full
            summary[seg] = {"written": written, "skipped": skipped, "chunks": len(start_times)}
        return summary

    def grid_l2(self, segments=None, overwrite=False):
        """Grid per-sensor L2 to (depth, time) chunks, for drift segments.

        Same gridding as :meth:`grid_l1` (per-segment ``dt``/``max_gap``/
        ``chunk``, peak RAM bounded by one (depth x time) array via the
        unchanged :func:`grid_thermistors`), but over the per-sensor L2
        from :meth:`_procl2_dir` and written to :meth:`_gridl2_dir`
        (``grid/l2/``). Runs only on the ``drift: true`` segments (the only
        ones with an L2 product). Idempotent — existing chunk files are
        skipped unless ``overwrite=True``. ``ignore_sns`` is applied via
        ``exclude_sn``.

        Parameters
        ----------
        segments : str, list of str, or None, optional
            Drift segment(s). ``None`` runs all ``drift: true`` segments.
        overwrite : bool, optional
            Rewrite existing chunk files. Default ``False``.

        Returns
        -------
        dict
            ``{segment: {"written": int, "skipped": int, "chunks": int}}``.
        """
        gridl2_dir = self._gridl2_dir()
        gridl2_dir.mkdir(parents=True, exist_ok=True)
        procl2 = self._procl2_dir()
        summary = {}
        for seg in self._drift_segment_names(segments):
            gp = self.gridding[seg]
            info_sub = self.segment_sensors(seg)
            start_times = np.arange(
                self.cfg.start_time, self.cfg.end_time, gp["chunk"], dtype="datetime64[s]"
            )
            written = skipped = 0
            full = None
            for ti in start_times:
                fpath = gridl2_dir / self._grid_filename(seg, 2, ti)
                if fpath.exists() and not overwrite:
                    skipped += 1
                    continue
                if full is None:
                    full = grid_thermistors(
                        info_sub,
                        procl2,
                        start=self.cfg.start_time,
                        end=self.cfg.end_time,
                        dt=gp["dt"],
                        max_gap=gp["max_gap"],
                        exclude_sn=self._ignore_sns(),
                    )
                full.sel(time=slice(ti, ti + gp["chunk"])).to_netcdf(fpath)
                written += 1
            if full is not None:
                del full
            summary[seg] = {"written": written, "skipped": skipped, "chunks": len(start_times)}
        return summary

    def status(self):
        """Per-sensor progress: L0 / L1 / L2 presence + effective cal method.

        Lazy scan of the L0, L1, and L2 dirs on each call. Index is SN;
        columns: ``type``, ``segment``, ``ignored``, ``l0``, ``l1``, ``l2``,
        ``cal_method`` (read from the L1 ``cal_method`` attr).

        Returns
        -------
        pd.DataFrame
            Indexed by SN; one row per sensor in the mooring sheet.
        """
        l0dir = self._procl0_dir()
        l1dir = self._procl1_dir()
        l2dir = self._procl2_dir()
        ignore = set(self._ignore_sns())
        seg_of = {
            int(sn): seg
            for seg in self.segments_cfg
            for sn in self.segment_sensors(seg).index
        }
        rows = []
        for sn in self.mooring_info.index:
            sn = int(sn)
            l1_files = list(l1dir.glob(f"*{sn:06d}*_L1.nc")) if l1dir.exists() else []
            cal_method = ""
            if l1_files:
                with xr.open_dataarray(l1_files[0]) as da:
                    cal_method = da.attrs.get("cal_method", "")
            rows.append({
                "sn": sn,
                "type": str(self.mooring_info.loc[sn]["type"]),
                "segment": seg_of.get(sn, ""),
                "ignored": sn in ignore,
                "l0": bool(list(l0dir.glob(f"*{sn:06d}*.nc"))) if l0dir.exists() else False,
                "l1": bool(l1_files),
                "l2": (
                    bool(list(l2dir.glob(f"*{sn:06d}*_L2.nc"))) if l2dir.exists() else False
                ),
                "cal_method": cal_method,
            })
        return pd.DataFrame(rows).set_index("sn")

    def status_summary(self):
        """Per-segment progress: sensor count, L1, gridded-L1, drift, L2, gridded-L2.

        Lazy scan of the L1 / L2 / grid / aux dirs on each call. Columns:
        ``n`` (sensors after ignore_sns), ``l1`` (``present/n`` per-sensor
        L1), ``gridL1`` (``present/expected`` chunks), ``drift`` (comma-
        joined labels present, or ``-``), ``l2`` (``present/n`` per-sensor
        L2, or ``-`` for non-drift segments), ``gridL2`` (``present/
        expected`` chunks, or ``-`` for non-drift segments).

        Returns
        -------
        pd.DataFrame
            Index is segment name.
        """
        grid_dir = self._gridl1_dir()
        gridl2_dir = self._gridl2_dir()
        l1dir = self._procl1_dir()
        l2dir = self._procl2_dir()
        aux = self._aux_dir()
        ignore = set(self._ignore_sns())
        drift_segments = set(self._drift_segments())
        mooring_id = f"{self.meta.project.lower()}_{self.meta.mooring_name.lower()}"
        rows = []
        for seg in self.segments_cfg:
            gp = self.gridding[seg]
            sns = [int(s) for s in self.segment_sensors(seg).index if int(s) not in ignore]
            l1_present = sum(
                1 for s in sns
                if l1dir.exists() and list(l1dir.glob(f"*{s:06d}*_L1.nc"))
            )
            n_chunks = len(
                np.arange(self.cfg.start_time, self.cfg.end_time, gp["chunk"], dtype="datetime64[s]")
            )
            pattern = f"{mooring_id}_{seg}_L1_*.nc"
            present = len(list(grid_dir.glob(pattern))) if grid_dir.exists() else 0
            if seg in drift_segments:
                labels = sorted(
                    p.name[len(f"drift_{mooring_id}_"):-3]
                    for p in aux.glob(f"drift_{mooring_id}_*.nc")
                ) if aux.exists() else []
                drift = ", ".join(labels) if labels else "-"
                l2_present = sum(
                    1 for s in sns
                    if l2dir.exists() and list(l2dir.glob(f"*{s:06d}*_L2.nc"))
                )
                l2 = f"{l2_present}/{len(sns)}"
                gl2_present = (
                    len(list(gridl2_dir.glob(f"{mooring_id}_{seg}_L2_*.nc")))
                    if gridl2_dir.exists() else 0
                )
                gridL2 = f"{gl2_present}/{n_chunks}"
            else:
                drift = l2 = gridL2 = "-"
            rows.append({
                "segment": seg,
                "n": len(sns),
                "l1": f"{l1_present}/{len(sns)}",
                "gridL1": f"{present}/{n_chunks}",
                "drift": drift,
                "l2": l2,
                "gridL2": gridL2,
            })
        return pd.DataFrame(rows).set_index("segment")
