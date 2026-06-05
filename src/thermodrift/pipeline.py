# src/thermodrift/pipeline.py
"""Config-driven Mooring pipeline orchestration over thermodrift primitives."""

import re
from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd
import xarray as xr

from .io import ProcessThermistorMooring, grid_thermistors  # noqa: F401

_GRIDDING_KEYS = {"dt", "max_gap", "chunk"}

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
        grid_dir = self._grid_dir()
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

    def status_summary(self):
        """Per-segment progress: expected sensor count and gridded-L1 chunk coverage.

        Lazy scan of the grid dir on each call. Columns: ``n`` (sensors
        after ignore_sns), ``gridL1`` (``present/expected`` chunks).

        Returns
        -------
        pd.DataFrame
            Index is segment name; columns are ``n`` (int) and
            ``gridL1`` (str ``"present/expected"``).
        """
        grid_dir = self._grid_dir()
        ignore = set(self._ignore_sns())
        rows = []
        for seg in self.segments_cfg:
            gp = self.gridding[seg]
            sns = [s for s in self.segment_sensors(seg).index if int(s) not in ignore]
            n_chunks = len(
                np.arange(self.cfg.start_time, self.cfg.end_time, gp["chunk"], dtype="datetime64[s]")
            )
            pattern = f"{self.meta.project.lower()}_{self.meta.mooring_name.lower()}_{seg}_L1_*.nc"
            present = len(list(grid_dir.glob(pattern))) if grid_dir.exists() else 0
            rows.append({"segment": seg, "n": len(sns), "gridL1": f"{present}/{n_chunks}"})
        return pd.DataFrame(rows).set_index("segment")
