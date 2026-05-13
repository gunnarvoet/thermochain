#!/usr/bin/env python
# coding: utf-8
"""
I/O functions.

Generate a processing object with `ProcessThermistorMooring`.
"""

import functools
from pathlib import Path
import tqdm
import yaml
from box import Box
import matplotlib.pyplot as plt
import numpy as np
import scipy
import xarray as xr
import pandas as pd
import rbrmoored
import sbemoored
import mixsea as mx
import gvpy as gv


logger = gv.misc.log()


def load_config(configfile):
    """Load configuration (.yaml) file with
    - paths to sensor and mooring sheet
    - processing parameters

    Returns
    -------
    cfg : dict
        Config parameters dictionary.

    See also
    --------
    `load_config_box`
    """
    with open(configfile, "r") as ymlfile:
        return yaml.safe_load(ymlfile)


def _resolve_path_vars(obj, data_root, project_root):
    """Recursively replace ``$data/...`` and ``$root/...`` strings with absolute Paths.

    Parameters
    ----------
    obj : Any
        Raw value from the parsed YAML (str, dict, list, or any leaf type).
    data_root : pathlib.Path or None
        Anchor for ``$data/`` prefixes. May be ``None`` only when no
        ``$data/`` prefix is present in the tree.
    project_root : pathlib.Path
        Anchor for ``$root/`` prefixes.

    Raises
    ------
    ValueError
        If a ``$data/`` prefix is encountered but ``data_root`` is ``None``.
    """
    if isinstance(obj, str):
        if obj.startswith("$data/"):
            if data_root is None:
                raise ValueError(
                    "config uses '$data/' prefix but data_root was not provided "
                    "to load_config_box; pass data_root=<Path> explicitly."
                )
            return data_root / obj[6:]
        if obj.startswith("$root/"):
            return project_root / obj[6:]
        return obj
    if isinstance(obj, dict):
        return {
            k: _resolve_path_vars(v, data_root, project_root)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_resolve_path_vars(item, data_root, project_root) for item in obj]
    return obj


def load_config_box(configfile, project_root=None, data_root=None) -> Box:
    """Load the yaml config file as Box object which is basically a dict with
    dot access.

    Resolves ``$data/`` and ``$root/`` placeholders before wrapping the
    parsed YAML in a Box. ``$root/`` resolves against ``project_root``
    (default: ``configfile.parent.parent``); ``$data/`` resolves against
    ``data_root`` and is required only when the YAML actually contains a
    ``$data/`` prefix.

    Parameters
    ----------
    configfile : pathlib.Path or str
        Path to the YAML config file.
    project_root : pathlib.Path or None, optional
        Override for the project root. Defaults to
        ``configfile.parent.parent.resolve()``.
    data_root : pathlib.Path or None, optional
        Anchor for ``$data/`` prefixes. Required when the YAML contains
        ``$data/``; otherwise ignored.

    Returns
    -------
    cfg : Box
        Config parameters dictionary with dot access.

    Raises
    ------
    ValueError
        If the YAML uses ``$data/`` but ``data_root`` is not supplied.

    See also
    --------
    `load_config`
    """
    configfile = Path(configfile)
    if project_root is None:
        project_root = configfile.parent.parent.resolve()

    with open(configfile, "r") as ymlfile:
        raw = yaml.safe_load(ymlfile)
    raw = _resolve_path_vars(raw, data_root=data_root, project_root=project_root)
    cfg = Box(raw)
    cfg.path.root = project_root
    cfg.path.fig = cfg.path.root.joinpath(cfg.path.fig)
    cfg.path.data.proc = cfg.path.root.joinpath(cfg.path.data.proc)
    # `data.raw` may be a single path or a per-instrument mapping
    # (e.g. {rbr: …, sbe: …}). Resolve each leaf relative to root.
    if isinstance(cfg.path.data.raw, Box):
        for key, val in cfg.path.data.raw.items():
            cfg.path.data.raw[key] = cfg.path.root.joinpath(val)
    else:
        cfg.path.data.raw = cfg.path.root.joinpath(cfg.path.data.raw)
    for level in range(3):
        cfg.path.data[f"procl{level}"] = cfg.path.data.proc.joinpath(f"l{level}")
        # create directories but check if they may not be there because data/ is
        # symlinked to an external drive
        if not cfg.path.root.joinpath("data/").is_symlink():
            cfg.path.data[f"procl{level}"].mkdir(exist_ok=True, parents=True)
    # for level in range(3):
    #     cfg.path.data[f"procl{level}"] = cfg.path.root.joinpath(cfg.path.data[f"procl{level}"])
    #     cfg.path.data[f"procl{level}"].mkdir(exist_ok=True, parents=True)

    # Create figure directory if needed
    cfg.path.fig.mkdir(exist_ok=True, parents=True)

    # parse datetimes
    for time in ["start_time", "end_time"]:
        cfg[time] = np.datetime64(cfg[time])

    return cfg


def mooring_sheet_load(path):
    mooring_info = pd.read_csv(path, sep=",", header=0, index_col="SN")
    unify_sensor_type(mooring_info)
    return mooring_info


def sensor_sheet_load(path):
    """Load sensor spreadsheet in .csv or .xlsx format. Parse and return a
    `pandas.Dataframe` with sensor info.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to sensor spreadsheet

    Returns
    -------
    df : pandas.DataFrame
        Sensor information.
    """
    path = _parse_path(path)
    if path.suffix == ".csv":
        df = sensor_sheet_csv_load(path)
    elif path.suffix == ".xlsx" or path.suffix == ".ods":
        df = sensor_sheet_xlsx_load(path)
    df = sensor_sheet_rename_columns(df)
    df = sensor_sheet_columns_to_dt64(df)
    unify_sensor_type(df)
    return df


def sensor_sheet_csv_load(path):
    """Load sensor spreadsheet in .csv format.
    See template for expected headers etc.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to sensor spreadsheet

    Returns
    -------
    df : pandas.DataFrame
        Sensor information.
    """

    df = pd.read_csv(
        path,
        header=0,
        index_col=0,
    )
    return df


def sensor_sheet_xlsx_load(path):
    """Load sensor spreadsheet in .xlsx format.
    See template for expected headers etc.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to sensor spreadsheet

    Returns
    -------
    df : pandas.DataFrame
        Sensor information.
    """
    df = pd.read_excel(
        path,
        header=0,
        index_col=0,
    )
    return df


def sensor_sheet_rename_columns(df):
    """Change column names in sensor DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor information.

    Returns
    -------
    df : pandas.DataFrame
        Sensor information.
    """
    try:
        df.drop(["Depth rating"], axis=1, inplace=True)
    except:
        pass
    df = df.rename(
        columns={
            "Pre-Deployment Time Calibration": "time_cal1",
            "Post-Deployment Time Calibration": "time_cal2",
            "Post-Deployment UTC Time": "clock_read_utc",
            "Post-Deployment Logger Time": "clock_read_logger",
            "Type": "type",
            "SN": "sn",
            "Pre-Deployment CTD Calibration Time": "pre_ctd_cal_time",
            "Pre-Deployment CTD Calibration Cast": "pre_ctd_cal_cast",
            "Post-Deployment CTD Calibration Time": "post_ctd_cal_time",
            "Post-Deployment CTD Calibration Cast": "post_ctd_cal_cast",
            "Pre-Deployment Notes": "pre_notes",
            "Post-Deployment Notes": "post_notes",
        }
    )
    return df


def unify_sensor_type(df):
    """Parse sensor type in Dataframe. Will change various sensor
    descriptions to either `rbr` or `sbe`.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor information.
    """
    # for i, v in df.type.items():
    #     if "56" in v or "Seabird" in v or "SBE" in v:
    #         df.type.at[i] = "sbe"
    #     elif "rbr" in v or "RBR" in v or "Solo" in v or "solo" in v:
    #         df.type.at[i] = "rbr"
    #     else:
    #         raise ValueError(f'Could not parse instrument type "{v}"')

    # Create a boolean mask for the 'sbe' instruments
    # We use .str.contains and combine multiple checks using the OR operator (|)
    sbe_mask = (
        df["type"].str.contains("56", na=False)
        | df["type"].str.contains("Seabird", na=False)
        | df["type"].str.contains("SBE", na=False)
    )

    # Use .loc to assign the value "sbe" to all rows that match the mask
    df.loc[sbe_mask, "type"] = "sbe"

    # Create a boolean mask for the 'rbr' instruments
    rbr_mask = (
        df["type"].str.contains("rbr", na=False)
        | df["type"].str.contains("RBR", na=False)
        | df["type"].str.contains("Solo", na=False)
        | df["type"].str.contains("solo", na=False)
    )

    df.loc[rbr_mask, "type"] = "rbr"

    # We combine both masks to find the rows that are NOT in sbe_mask AND NOT in rbr_mask
    unparsed_mask = ~sbe_mask & ~rbr_mask
    unmatched_rows = df[unparsed_mask]
    if not unmatched_rows.empty:
        # Get the original values that failed to parse
        failed_values = unmatched_rows["type"].unique()
        # Raise a clear error listing the failed types
        raise ValueError(
            f"Could not parse instrument type(s): {', '.join(failed_values)}"
        )


def sensor_sheet_columns_to_dt64(df):
    """Convert datetimes to np.datetime64.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor information.

    Returns
    -------
    df : pandas.DataFrame
        Sensor information.
    """
    for col in [
        "pre_ctd_cal_time",
        "post_ctd_cal_time",
        "time_cal1",
        "time_cal2",
        "clock_read_utc",
        "clock_read_logger",
    ]:
        df[col] = df[col].astype("datetime64[ns]")
    return df


def proc_db_generate(sensor_sheet):
    """Generate a processing database in form of a pandas.DataFrame.
    Here we can track which files have been processed etc.

    Parameters
    ----------
    sensor_sheet : pandas.DataFrame
        Sensor information generated with `sensor_sheet_load`.

    Returns
    -------
    proc_db : pandas.DataFrame
        Processing database.
    """

    cols = ["type", "time_cal1", "time_cal2"]
    for opt in ("clock_read_utc", "clock_read_logger"):
        if opt in sensor_sheet.columns:
            cols.append(opt)
    proc_db = sensor_sheet[cols].copy()
    n = proc_db.index.shape[0]

    proc_db = proc_db.assign(processed=np.tile(False, n))
    proc_db = proc_db.assign(raw_data_exists=proc_db.processed.copy())
    proc_db = proc_db.assign(figure_exists=proc_db.processed.copy())
    proc_db = proc_db.assign(time_offset_applied=np.full(n, np.nan))
    proc_db = proc_db.assign(comment=np.tile("ok", n))
    proc_db = proc_db.assign(mooring=np.tile(None, n))

    proc_db = proc_db.sort_index()

    return proc_db


def proc_db_assign_mooring(proc_db, mooring_id, sn_list):
    sn_mask = [sni in sn_list for sni in proc_db.index]
    proc_db.loc[sn_mask, "mooring"] = mooring_id


def get_file_name(sn, data_dir, type):
    if type == "rbr":
        extension = "rsk"
        files = list(data_dir.glob(f"*{sn:06}*.{extension}"))
        files = _ignore_hidden_files(files)
        if len(files) == 0:
            files = list(data_dir.glob(f"*{sn:5}*.{extension}"))
            files = _ignore_hidden_files(files)
    elif type == "sbe":
        extension = "csv"
        files = list(data_dir.glob(f"SBE056{sn:05}*.{extension}"))
        files = _ignore_hidden_files(files)
    elif type == "nc" or type == "png":
        extension = type
        files = list(data_dir.glob(f"*{sn:06}*.{extension}"))
        if len(files) == 0:
            files = list(data_dir.glob(f"SBE056{sn:05}*.{extension}"))
        # RBR file could have 5-digit SN in file name
        if len(files) == 0:
            files = list(data_dir.glob(f"*{sn:5}*.{extension}"))
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return None
    else:
        files = _ignore_hidden_files(files)
    if len(files) == 1:
        return files[0]
    else:
        raise OSError(f"more than one file for SN{sn} in {data_dir}")


def proc_db_update_files(proc_db, data_raw, data_out, figure_out):
    has_offset_col = "time_offset_applied" in proc_db.columns
    for g, v in proc_db.groupby("SN"):
        f = get_file_name(g, data_dir=data_raw, type=v.type.item())
        if f is not None:
            proc_db.loc[g, "raw_data_exists"] = True
        f = get_file_name(g, data_dir=data_out, type="nc")
        if f is not None:
            proc_db.loc[g, "processed"] = True
            if has_offset_col:
                proc_db.loc[g, "time_offset_applied"] = _read_time_offset_applied(f)
        f = get_file_name(g, data_dir=figure_out, type="png")
        if f is not None:
            proc_db.loc[g, "figure_exists"] = True


def _read_time_offset_applied(l0_file):
    """Return the ``time offset applied`` attr from an L0 file as float (NaN if absent).

    The attr is stored on the ``t`` data variable by ``rbrmoored`` / ``sbemoored``.
    A value of ``1`` means a clock correction was applied during L0 generation;
    ``0`` or missing means the L0 was written without one. Returning NaN on any
    open failure keeps the proc_db update non-fatal.
    """
    try:
        with xr.open_dataset(l0_file) as ds:
            var = ds["t"] if "t" in ds.data_vars else ds[list(ds.data_vars)[0]]
            val = var.attrs.get("time offset applied", np.nan)
    except (OSError, KeyError, IndexError):
        return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def rbr_save_ctd_cal_time_series(l0_data, ctd_time, save_dir):
    """Save thermistor time series for time of CTD calibration cast."""
    files = sorted(l0_data.glob("*.nc"))

    def save_ctd_cal(file, ctd_time):
        datestr = ctd_time.start[:10].replace("-", "")
        outname = f"{file.stem[: file.stem.find('_')]}_ctd_cal_cast_{datestr}.nc"
        outpath = save_dir.joinpath(outname)
        if outpath.exists() is False:
            r = xr.open_dataarray(file)
            if r.time[0] < np.datetime64(ctd_time.stop):
                r.sel(time=ctd_time).to_netcdf(outpath)
                r.close()
            else:
                logger.info(f"no data during ctd cal for {outname}")

    for file in files:
        save_ctd_cal(file, ctd_time)


def rbr_load_cals(cal_dir):
    calfiles = sorted(cal_dir.glob("*.nc"))
    cals = [xr.open_dataarray(calfile) for calfile in calfiles]
    sns = [ci.attrs["SN"] for ci in cals]

    # pick one time for all of them and interpolate
    time = cals[0].time.copy()
    calsi = [ci.interp_like(time) for ci in cals]
    c = xr.concat(calsi, dim="n")
    c["sn"] = (("n"), sns)
    return c


def plot_zoom(ax, ts, rbr, ctd, add_legend=False):
    t = rbr.sel(time=ts)
    t.plot(hue="n", add_legend=False, linewidth=0.75, color="k", alpha=0.3, ax=ax)
    ctd.t1.sel(time=ts).plot(color="C0", linewidth=1, label="CTD 1", ax=ax)
    ctd.t2.sel(time=ts).plot(color="C4", linewidth=1, label="CTD 2", ax=ax)
    ax.set(xlabel="", ylabel="in-situ temperature [°C]")
    gv.plot.concise_date(ax)
    gv.plot.axstyle(ax, fontsize=9)
    if add_legend:
        ax.legend()


def plot_cal_stop(ts, rbr, ctd, dt=2):
    """Plot CTD calibration stop with zoom."""
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), constrained_layout=True)
    plot_zoom(ax[0], ts, rbr, ctd, add_legend=True)
    tsn = slice(
        np.datetime64(ts.start) - np.timedelta64(dt, "m"),
        np.datetime64(ts.stop) + np.timedelta64(dt, "m"),
    )
    plot_zoom(ax[1], tsn, rbr, ctd)
    mctd = ctd.t1.sel(time=ts).mean()
    ax[1].plot(
        [np.datetime64(ti) for ti in [ts.start, ts.stop]],
        np.tile(mctd - 0.02, 2),
        "r",
    )


def plot_multiple_cal_stop(ts, rbr, ctd, dt=2):
    """Plot multiple CTD calibration stop with zoom."""
    n = len(ts)
    fig, ax = plt.subplots(
        nrows=n,
        ncols=2,
        figsize=(10, 3 * n),
        constrained_layout=True,
    )
    leg = True
    ii = 1
    for tsi, axi in zip(ts, ax):
        plot_zoom(axi[0], tsi, rbr, ctd, add_legend=leg)
        tsn = slice(
            np.datetime64(tsi.start) - np.timedelta64(dt, "m"),
            np.datetime64(tsi.stop) + np.timedelta64(dt, "m"),
        )
        gv.plot.annotate_corner(ii, axi[0], background_circle="0.1", col="w")
        plot_zoom(axi[1], tsn, rbr, ctd)
        mctd = ctd.t1.sel(time=tsi).mean()
        axi[1].plot(
            [np.datetime64(ti) for ti in [tsi.start, tsi.stop]],
            np.tile(mctd - 0.02, 2),
            "r",
        )
        leg = False
        ii += 1
    return fig, ax


class ProcessThermistorMooring:
    """Process thermistors on one mooring."""

    def __init__(self, config_file, project_root=None, data_root=None):
        """Initialize processing object for one mooring.

        Parameters
        ----------
        config_file : pathlib.Path or str
            .yaml configuration file
        project_root : pathlib.Path or None, optional
            Forwarded to ``load_config_box``. See its docstring.
        data_root : pathlib.Path or None, optional
            Forwarded to ``load_config_box``. See its docstring.
        """
        self.cfg = load_config_box(
            config_file, project_root=project_root, data_root=data_root
        )
        # add L0/L1/L2 sub-directories to the figure path
        for i in range(3):
            self.cfg.path[f"figl{i}"] = self.cfg.path.fig.joinpath(f"l{i}")
        """Processing configuration"""
        self.meta = self.cfg.meta
        """Meta data"""
        self.load_sensor_info()
        self.load_mooring_info()
        self.generate_processing_database()

    def load_sensor_info(self):
        """Load sensor configuration sheet from path provided in config."""
        sensor_sheet_path = self.cfg.path.root.joinpath(self.cfg.path.sensors)
        self.sensor_info = sensor_sheet_load(sensor_sheet_path)

    def load_mooring_info(self):
        """Load mooring configuration sheet from path provided in config."""
        mooring_sheet_path = self.cfg.path.root.joinpath(self.cfg.path.mooring)
        self.mooring_info = mooring_sheet_load(mooring_sheet_path)

    def generate_processing_database(self):
        """Generate a processing database in form of a `pandas.DataFrame` to
        track which files have been processed etc.
        """
        mooring_id = self.meta.mooring_name
        self.proc_db = proc_db_generate(self.sensor_info)
        proc_db_assign_mooring(self.proc_db, mooring_id, self.mooring_info.index.values)
        self.proc_db_all_sensors = self.proc_db.copy()
        self.proc_db = self.proc_db.query("mooring == @mooring_id")
        """test"""

    def proc_db_update_files(self):
        """Update processing database.

        Looks for raw & L0 processed data; L0 figure files.
        """
        proc_db_update_files(
            self.proc_db,
            self.cfg.path.data.raw,
            self.cfg.path.data.procl0,
            self.cfg.path.figl0,
        )

    def get_clock_cals(self, sn):
        cal1 = self.proc_db.loc[sn]["time_cal1"].to_datetime64()
        cal2 = self.proc_db.loc[sn]["time_cal2"].to_datetime64()
        cals = []
        for c in [cal1, cal2]:
            if ~np.isnat(c):
                cals.append(c)
        if len(cals) > 1:
            cals = tuple(cals)
        elif len(cals) == 1:
            cals = cals[0]
        else:
            cals = None
        return cals

    def get_clock_reads(self, sn):
        utc = self.sensor_info.loc[sn]["clock_read_utc"].to_datetime64()
        inst = self.sensor_info.loc[sn]["clock_read_logger"].to_datetime64()
        return inst, utc

    def run_proc_single_rbr(self, sn, show_plot=False):
        """Process RBR sensors on the mooring."""
        type = "rbr"
        info = self.proc_db.loc[sn]
        cals = self.get_clock_cals(sn)
        file = get_file_name(sn=sn, data_dir=self.cfg.path.data.raw, type=type)
        print(file.name)
        t = rbrmoored.solo.proc(
            file,
            data_out=self.cfg.path.data.procl0,
            apply_time_offset=True,
            figure_out=self.cfg.path.fig,
            cal_time=cals,
            show_plot=show_plot,
        )
        return t

    def run_proc_single_sbe(self, sn, show_plot=False):
        """Process SBE sensors on the mooring."""
        type = "sbe"
        info = self.proc_db.loc[sn]
        cals = self.get_clock_cals(sn)
        insttime, utctime = self.get_clock_reads(sn)
        file = get_file_name(sn=sn, data_dir=self.cfg.path.data.raw, type=type)
        print(file.name)
        t = sbemoored.sbe56.proc(
            file,
            time_instrument=insttime,
            time_utc=utctime,
            data_out=self.cfg.path.data.procl0,
            figure_out=self.cfg.path.fig,
            cal_time=cals,
            show_plot=show_plot,
        )
        return t

    def run_proc(self, sn, show_plot=False):
        ttype = self.sensor_info.loc[sn].type
        match ttype:
            case "rbr":
                return self.run_proc_single_rbr(sn, show_plot)
            case "sbe":
                return self.run_proc_single_sbe(sn, show_plot)

    def run_proc_all(self):
        """Process all sensors on the mooring."""
        self.proc_db_update_files()
        sn_not_processed_yet = self.proc_db.query("processed == False").index.values
        for sn in tqdm.tqdm(sn_not_processed_yet):
            try:
                t = self.run_proc(sn)
            except:
                print(f"{sn} failed")
        self.proc_db_update_files()


def rbr_ctd_cal_find_offset(ts, rbr, ctd):
    ctdm1 = ctd.t1.sel(time=ts).mean().data
    ctdm2 = ctd.t2.sel(time=ts).mean().data
    print(f"Difference between CTD sensor 1 & 2 mean values: {ctdm1 - ctdm2:.4e}°C")
    ctdm = np.mean([ctdm1, ctdm2])
    ab = rbr.sel(time=ts)
    diffmean_1 = (ctdm1 - ab).mean(dim="time")
    diffmean_1.name = "mean1"
    diffstd_1 = (ctdm1 - ab).std(dim="time")
    diffstd_1.name = "std1"
    diffmean_2 = (ctdm2 - ab).mean(dim="time")
    diffmean_2.name = "mean2"
    diffstd_2 = (ctdm2 - ab).std(dim="time")
    diffstd_2.name = "std2"
    diffmean_both = (ctdm - ab).mean(dim="time")
    diffmean_both.name = "meanboth"
    diffstd_both = (ctdm - ab).std(dim="time")
    diffstd_both.name = "stdboth"
    mean = xr.concat([diffmean_both, diffmean_1, diffmean_2], dim="m")
    mean = mean.swap_dims({"n": "sn"})
    mean.name = "mean_diff"
    std = xr.concat([diffstd_both, diffstd_1, diffstd_2], dim="m")
    std = std.swap_dims({"n": "sn"})
    std.name = "std_diff"

    mean_temp = ab.mean(dim="time")
    mean_temp.name = "mean_temp"
    mean_temp = mean_temp.swap_dims({"n": "sn"})

    out = xr.merge([mean, std, mean_temp])
    out.coords["sensor"] = (("m"), ["both", "1", "2"])

    return out


def rbr_load_proc_level0(sn, l0dir):
    files = list(l0dir.glob(f"*{sn:06}*.nc"))
    files = _ignore_hidden_files(files)
    if len(files) == 1:
        return xr.open_dataarray(files[0])
    elif len(files) == 0:
        return None
    else:
        raise OSError(f"more than one file for SN{sn} in {l0dir}")


def rbr_load_proc_level1(sn, l0dir):
    files = list(l0dir.glob(f"*{sn:06}*.nc"))
    files = _ignore_hidden_files(files)
    if len(files) == 1:
        return xr.open_dataarray(files[0])
    elif len(files) == 0:
        return None
    else:
        raise OSError(f"more than one file for SN{sn} in {l0dir}")


def rbr_find_last_time_stamp(thermistor):
    return thermistor.time.isel(time=-1).data


def rbr_find_gaps(thermistor):
    return find_gaps(thermistor)


def find_gaps(thermistor):
    td = thermistor.time.diff(dim="time")
    dt = np.timedelta64(int(thermistor.attrs["sampling period in s"] * 1000), "ms")
    tdi = td.where(
        (td > dt + np.timedelta64(500, "ms")) | (td < dt - np.timedelta64(500, "ms")),
        drop=True,
    )
    return tdi


def rbr_find_first_long_gap(tdi):
    t0 = find_first_long_gap(tdi)
    if ~np.isnat(t0):
        t0["time"] = (t0.time - t0).data
    return t0


def find_first_long_gap(tdi):
    ti = tdi > np.timedelta64(1, "h")
    if np.any(ti):
        t = tdi.where(ti, drop=True)
        t0 = t.isel(time=0)
    else:
        t0 = np.datetime64("nat")
    return t0


def rbr_apply_ctd_offset(thermistor, sn, ctdcal):
    if sn in ctdcal.sn:
        cal = ctdcal.sel(sn=sn).data
        return thermistor + cal
    else:
        print(f"no cal for {sn}")
        return thermistor


def rbr_cut_and_cal(
    sn, l0dir, l1dir, ctdcal, cut_beg, cut_end, end_manually, mooring_name, sensor_type
):
    savename = f"{mooring_name.lower().replace(' ', '_')}__{sensor_type.lower()}__{sn:06}_L1.nc"
    savepath = l1dir.joinpath(savename)
    if savepath.exists():
        logger.info(f"loading existing SN{sn} L1 data")
        tmpcal = xr.open_dataarray(savepath)
    else:
        logger.info(f"loading SN{sn} L0 data")
        tmp = rbr_load_proc_level0(sn, l0dir)
        attrs = tmp.attrs
        last_time = rbr_find_last_time_stamp(tmp)
        t1 = cut_end
        if last_time < cut_end:
            t1 = last_time
        tdi = rbr_find_gaps(tmp)
        if len(tdi) > 0:
            t = rbr_find_first_long_gap(tdi)
            if ~np.isnat(t):
                if t.time.data < t1:
                    t1 = t.time.data
                if sn in end_manually:
                    if end_manually[sn] < t1:
                        t1 = end_manually[sn]

        tmpcut = tmp.where((tmp.time > cut_beg) & (tmp.time < t1), drop=True)

        tmpcal = rbr_apply_ctd_offset(thermistor=tmpcut, sn=sn, ctdcal=ctdcal)
        tmpcal.attrs = attrs
        tmpcal.attrs["sample size"] = len(tmpcal)
        logger.info(f"saving SN{sn} L1 data")
        tmpcal.to_netcdf(savepath, mode="w")

    return tmpcal


def load_thermistors(proc_dir, time_span, sensor_info, exclude_sn=None):
    if exclude_sn is not None:
        if type(exclude_sn) is not list:
            exclude_sn = [exclude_sn]
        for sni in exclude_sn:
            if sni in sensor_info.index:
                sensor_info = sensor_info.drop(sni)
    out = []
    logger.info("reading sensors")
    for sn, soloi in tqdm.tqdm_notebook(sensor_info.groupby("SN")):
        # file = proc_dir.glob(f"*{sn:06}.nc")
        # tmp = xr.open_dataarray(file.__next__())
        tmp = rbr_load_proc_level1(sn, proc_dir)
        t = tmp.sel(time=time_span).copy()
        t.attrs["depth"] = soloi.depth.values[0]
        t.attrs["height"] = soloi.height.values[0]
        t.attrs["sn"] = sn
        if len(t) > 0:
            out.append(t)
        tmp.close()
    # # read RBR sensors
    # solo = sensor_info.query('Type=="Solo" or Type=="Solo Ti" or Type=="MAVS Solo"')
    # print("reading RBRs")
    # for sn, soloi in tqdm(solo.groupby("SN")):
    #     if proc_level == 0:
    #         file = rbr_proc.glob(f"{sn:06}*.nc")
    #     if proc_level > 0:
    #         file = rbr_proc.glob(f"*{sn:06}.nc")
    #     tmp = xr.open_dataarray(file.__next__())
    #     t = tmp.sel(time=time_span).copy()
    #     t.attrs["depth"] = soloi.depth.values[0]
    #     t.attrs["height"] = soloi.height.values[0]
    #     t.attrs["sn"] = sn
    #     if len(t) > 0:
    #         out.append(t)
    #     tmp.close()
    # # read SBE sensors
    # sbe = sensor_info.query('Type=="SBE56"')
    # print("reading SBEs")
    # for sn, sbei in tqdm(sbe.groupby("SN")):
    #     file = sbe_proc.glob(f"*{sn:04}*.nc")
    #     tmp = xr.open_dataarray(file.__next__())
    #     t = tmp.sel(time=time_span).copy()
    #     t.attrs["depth"] = sbei.depth.values[0]
    #     t.attrs["height"] = sbei.height.values[0]
    #     t.attrs["sn"] = sn
    #     if len(t) > 0:
    #         out.append(t)
    #     tmp.close()
    return out


def grid_thermistors(
    sensor_info,
    proc_dir,
    start="2023-08-01",
    end=None,
    days=2,
    extra_meta_data=None,
    exclude_sn=None,
):
    # construct time slice
    start = np.datetime64(start)
    if end is not None:
        end = np.datetime(end)
    elif days is not None:
        end = start + np.timedelta64(days, "D")
    time_span = slice(start, end)
    out = load_thermistors(proc_dir, time_span, sensor_info, exclude_sn)
    # def load_thermistors(proc_dir, time_span, sensor_info, exclude_sn=None):

    depth = [ti.depth for ti in out]
    height = [ti.height for ti in out]
    sn = [ti.sn for ti in out]

    # 1s time vector
    tstart = np.datetime64(time_span.start)
    tstop = np.datetime64(time_span.stop)
    time = np.arange(tstart, tstop, dtype="datetime64[s]")
    time = time.astype("datetime64[ns]")
    time = xr.DataArray(time, coords=[time], dims=["time"])

    # interpolate to time vector
    out2 = [ti.interp_like(time) for ti in out]

    t = xr.concat(out2, dim="depth")
    t.coords["depth"] = depth
    t.coords["sn"] = (("depth"), sn)
    t = t.sortby("depth")

    # clean up attributes
    t.attrs = {k: v for k, v in t.attrs.items() if k in ["units", "long_name"]}
    t.time.attrs["long_name"] = " "

    # Add more attributes if provided
    if extra_meta_data is not None:
        for k, v in extra_meta_data.items():
            t.attrs[k] = v

    # Add sensor type info to the data structure
    sn = t.sn.data
    sensor = [sensor_info.loc[sni].type for sni in sn]
    t["sensor_type"] = (("depth"), sensor)

    return t


def find_outliers(t, exclusion_criteria, polyfit_order=8, plot=True):
    """Find outliers in the time-mean stratification of a moored thermistor dataset.

    Parameters
    ----------
    t : xr.DataArray
        Thermistor dataset. The background time-mean will be calculated over
        the full length of the time series provided. Dimensions need to be
        `time` and `depth`.
    exclusion_criteria : float or list
        Exclude outliers that deviate more than this value from a polynomial
        fit to the time-mean. Can either be a single float value or a two
        element list. If a list with values is provided then the second
        exclusion criterion is applied to a second fit based on a subset of
        thermistors with those removed not passing the first exclusion
        parameter.
    polyfit_order : int, optional
        Order of the polynomial background fit. Defaults to 8.
    plot : bool, optional
        Plot background fit and outliers.

    Returns
    -------
    xn : array-like
        Boolean indexing array into the depth vector of `t`.
    fig, subfigs, axall
        If plot==True then these plotting objects are returned as well.
    """
    # Determine whether we run an iterative procedure.
    if type(exclusion_criteria) is not list:
        exclusion_criteria = [exclusion_criteria]
        second_fit = False
    else:
        second_fit = True
    # Calculate the time-mean stratification.
    z = t.depth
    mt = t.mean(dim="time")
    # We need to deal with nans in the mean profile - they come from
    # thermistors that dropped out early. The nanratio gets rid of any time
    # series that has more than a few nan's.
    mt_good = ~np.isnan(mt)
    nanratio = np.isnan(t).sum(dim="time") / t.time.shape
    exclude_nan = nanratio < 0.001
    xn = mt_good & exclude_nan
    # Do the first background fit and determine outliers.
    pf = np.polynomial.polynomial.polyfit(z[xn], mt[xn], deg=polyfit_order)
    py = np.polynomial.polynomial.polyval(z, pf)
    offset = mt - py
    xn2 = np.absolute(offset) < exclusion_criteria[0]
    # If requested do the second background fit.
    if second_fit:
        pf2 = np.polynomial.polynomial.polyfit(z[xn2], mt[xn2], deg=polyfit_order)
        py2 = np.polynomial.polynomial.polyval(z, pf2)
        offset2 = mt - py2
        # The sensor closest to the bottom may deviate a bit from the others. We
        # don't want to exclude it as an outlier no matter what. Not ideal, but
        # also not good not to include it in the fit... We could apply a different
        # offset criterion here, say two times the offset? That way we still get
        # rid of the sensor if it is a gross outlier. We divide the offset for the
        # bottom-most sensor by two to account for this.
        zmaxi = z.argmax().data
        offset2[zmaxi] = offset2[zmaxi] / 1.5
        xn3 = np.absolute(offset2) < exclusion_criteria[1]
    else:
        py2 = py
        xn3 = xn2

    def plot_fits(ax, z, mt, py, py2, xn, xn2, xn3, second_fit):
        ax.plot(mt[~xn], z[~xn], linestyle="", marker="x", color="0.3")
        # ax.plot(mt[xn], z[xn], linestyle='', marker='.')

        ax.plot(py, z, color="k", alpha=0.5)
        ax.plot(mt[~xn2], z[~xn2], linestyle="", marker=".", color="r")

        if second_fit:
            ax.plot(py2, z, color="k", alpha=0.5)
            ax.plot(mt[~xn3], z[~xn3], linestyle="", marker=".", color="r")

        ax.plot(mt[xn3], z[xn3], linestyle="", marker=".", color="0.2")

        ax.invert_yaxis()

    if plot:
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
        fig.suptitle("Offsets from background fits", fontsize=14)

        # Create 2x1 subfigs. Working with subfigures here to be able to do titles per row.
        subfigs = fig.subfigures(nrows=2, ncols=1)
        subfigs[0].suptitle(
            f"Outliers (red) excluded in background fit (exclude {exclusion_criteria})"
        )
        # Create 1x3 subplots per subfig.
        [axbot, ax, axtop] = subfigs[0].subplots(nrows=1, ncols=3)
        ax2 = subfigs[1].subplots(nrows=1, ncols=3)
        axall = [[axbot, ax, axtop], ax2]

        plot_fits(ax, z, mt, py, py2, xn, xn2, xn3, second_fit)
        depth_range = z.max() - z.min()
        itop = z < z.min() + depth_range / 5
        plot_fits(
            axtop,
            z[itop],
            mt[itop],
            py[itop],
            py2[itop],
            xn[itop],
            xn2[itop],
            xn3[itop],
            second_fit,
        )
        ibot = z > z.max() - depth_range / 5
        plot_fits(
            axbot,
            z[ibot],
            mt[ibot],
            py[ibot],
            py2[ibot],
            xn[ibot],
            xn2[ibot],
            xn3[ibot],
            second_fit,
        )

        ax.set(title="all", xlabel="temperature [°C]")
        axbot.set(title="zoom bottom 20%", ylabel="depth [m]")
        axtop.set(title="zoom top 20%")

        # We included axes for showing offsets but hide them for now.
        subfigs[1].set_visible(False)
        for axi in ax2:
            axi.set_visible(False)

    if plot:
        return xn3, fig, subfigs, axall
    else:
        return xn3


def offsets_from_background_fit(
    t,
    exclude=[1e-2, 5e-3],
    plot=True,
    polydeg=8,
    outliers_polydeg=8,
    spline=False,
    spline_smooth=2e-4,
    exclude_sn=None,
):
    """Determine sensor offset from a stable time-mean background temperature profile.

    Parameters
    ----------
    t : xr.DataArray
        Thermistor dataset. The background time-mean will be calculated over
        the full length of the time series provided. Dimensions need to be
        `time` and `depth`.
    exclude : float or list
        Exclude outliers that deviate more than this value from a polynomial
        fit to the time-mean. Can either be a single float value or a two
        element list. If a list with values is provided then the second
        exclusion criterion is applied to a second fit based on a subset of
        thermistors with those removed not passing the first exclusion
        parameter.
    plot : bool, optional
        Plot background fit and outliers.
    polydeg : int, optional
        Order of the polynomial background fit used to determine offsets. Defaults to 8.
    outliers_polydeg : int, optional
        Order of the polynomial background fit used to detect outliers. Defaults to 8.
    spline : bool, optional
        Use smooth spline fit instead of polynomial fit for determining offsets.
    spline_smooth : float, optional
        Smoothing factor, only applied if `spline` is True.
    exclude_sn : list, optional
        Exclude specific sensors (for example if no CTD cal available) with a
        list of serial numbers.

    Returns
    -------
    offs : xr.DataArray
        Data array with offset for each sensor.
    """

    depth = t.depth
    mt = t.mean(dim="time")

    if plot:
        # Figure & axes are initiated in find_outliers()
        xn, fig, subfigs, axall = find_outliers(
            t, exclude, plot=plot, polyfit_order=outliers_polydeg
        )
    else:
        xn = find_outliers(t, exclude, plot=plot, polyfit_order=outliers_polydeg)
    print("before excluding:", np.logical_not(xn).sum().data)
    if exclude_sn is not None:
        xn2 = [sni not in exclude_sn for sni in xn.sn]
        xn2 = xn.copy(data=xn2)
        xn = xn * xn2
    print("after excluding:", np.logical_not(xn).sum().data)
    mt_sel = mt[xn]
    depth_sel = depth[xn]

    spl = scipy.interpolate.UnivariateSpline(depth[xn], mt[xn])
    spl.set_smoothing_factor(spline_smooth)
    spline_fit = spl(depth)
    spline_offsets = mt - spline_fit

    pf2 = np.polynomial.polynomial.polyfit(depth[xn], mt[xn], deg=polydeg)
    poly_fit = np.polynomial.polynomial.polyval(depth, pf2)
    poly_offsets = mt - poly_fit

    py = spline_fit if spline else poly_fit
    offset = mt - py

    offs = xr.DataArray(offset, coords=[t.depth.data], dims=["depth"])

    if plot:
        # Title for bottom row
        if spline:
            subfigs[1].suptitle(
                (
                    f"Spline ({spline_smooth} smoothing factor) background fit "
                    f"and offsets, {polydeg}th order polynomial for comparison"
                )
            )
        else:
            subfigs[1].suptitle(
                (
                    f"{polydeg}th order polynomial background fit and offsets, "
                    f"spline ({spline_smooth} smoothing factor) for comparison"
                )
            )
        # Vertical derivative of the fits
        ax = axall[1][0]
        poly_fit.differentiate(coord="depth").plot(
            ax=ax, y="depth", color="C0", alpha=0.5, label="polynomial"
        )
        spline_fit_da = poly_fit.copy()
        spline_fit_da.data = spline_fit
        spline_fit_da.differentiate(coord="depth").plot(
            ax=ax, y="depth", color="C3", alpha=0.5, label="spline"
        )
        ax.set(xlabel=r"dT$_{\mathrm{fit}}$/dz [deg/m]")
        ax.legend()
        ax.set_title("dfit/dz")
        # Zoom bottom 20%
        # ax = axall[1][0]
        # depth_range = depth.max() - depth.min()
        # ibot = depth_sel > depth.max() - depth_range / 15
        # ibot2 = depth > depth.max() - depth_range / 15
        # ax.plot(
        #     mt_sel[ibot], depth_sel[ibot], marker=".", linestyle="", color="k"
        # )
        # ax.plot(
        #     poly_fit[ibot2],
        #     depth[ibot2],
        #     color="C0",
        #     alpha=0.5,
        #     label="polynomial",
        # )
        # ax.plot(
        #     spline_fit[ibot2],
        #     depth[ibot2],
        #     color="C3",
        #     alpha=0.5,
        #     label="spline",
        # )
        # ax.set(xlabel="temperature [°C]", ylabel="depth [m]")
        # ax.legend()
        # Full depth
        ax = axall[1][1]
        ax.plot(mt_sel, depth_sel, marker=".", linestyle="", color="k")
        ax.plot(py, depth, alpha=0.5)
        ax.set(xlabel="temperature [°C]", ylabel="depth [m]")
        # Offsets
        ax = axall[1][2]

        ax.plot(
            poly_offsets,
            depth,
            marker="o",
            linestyle="",
            color="C0",
            alpha=0.5,
            label="polynomial",
        )
        ax.plot(
            spline_offsets,
            depth,
            marker="o",
            linestyle="",
            color="C3",
            alpha=0.5,
            label="spline",
        )
        # ax.plot(offset, depth, "ko")
        ax.set(xlabel=r"Delta T [°C]")
        ax.legend()
        ax.set_title("offsets")

        subfigs[1].set_visible(True)
        for axi in axall[1][:]:
            axi.set_visible(True)
            axi.invert_yaxis()

    return offs


def correct_offset(
    t,
    exclude=[1e-2, 5e-3],
    plot=True,
    polydeg=4,
    outliers_polydeg=8,
    return_offsets=False,
    spline=False,
    spline_smooth=2e-4,
    exclude_sn=None,
):
    """Correct sensor offset based on a stable time-mean background temperature profile.

    Mostly wraps `find_outliers` and `offsets_from_background_fit`.

    Parameters
    ----------
    t : xr.DataArray
        Thermistor dataset. The background time-mean will be calculated over
        the full length of the time series provided. Dimensions need to be
        `time` and `depth`.
    exclude : float or list
        Exclude outliers that deviate more than this value from a polynomial
        fit to the time-mean. Can either be a single float value or a two
        element list. If a list with values is provided then the second
        exclusion criterion is applied to a second fit based on a subset of
        thermistors with those removed not passing the first exclusion
        parameter.
    plot : bool, optional
        Plot background fit and outliers.
    polydeg : int, optional
        Order of the polynomial background fit used to determine offsets. Defaults to 8.
    outliers_polydeg : int, optional
        Order of the polynomial background fit used to detect outliers. Defaults to 8.
    return_offsets: bool, optional
        Return offsets in addition to corrected data.
    spline : bool, optional
        Use smooth spline fit instead of polynomial fit for determining offsets.
    spline_smooth : float, optional
        Smoothing factor, only applied if `spline` is True.
    exclude_sn : list, optional
        Exclude specific sensors (for example if no CTD cal available) with a
        list of serial numbers.

    Returns
    -------
    offs : xr.DataArray
        Data array with offset for each sensor.
    """
    offset = offsets_from_background_fit(
        t,
        exclude=exclude,
        plot=plot,
        polydeg=polydeg,
        outliers_polydeg=outliers_polydeg,
        spline=spline,
        spline_smooth=spline_smooth,
        exclude_sn=exclude_sn,
    )

    offs = xr.DataArray(offset, coords=[t.depth.data], dims=["depth"])

    if return_offsets:
        return t - offs, offs
    else:
        return t - offs


class sensor_drift:
    """Determine sensor drift. Following the method in Cimatoribus et al. (2016)."""

    def __init__(
        self,
        mooring_name,
        l1_grid_dir,
        last_window=None,
        run_all=False,
        drift_parameters=None,
        first_n_chunks=None,
    ):
        # self.mooring_id = mooring_id
        self.l1_grid_dir = l1_grid_dir
        self.mooring_name = mooring_name
        self.last_window = last_window
        self.parse_drift_parameters(drift_parameters)
        self.print_drift_parameters()
        # generate a list of all level 1 processed files
        self.list_gridded_level1_files(first_n_chunks)
        self.windowed_background_fits()  # adds variable "offsets_initial" to instance
        self.remove_outliers(last_window)  # adds variable "offsets" to instance
        if run_all:
            self.calc_first_guess_shared_fluctuating_component()
            self.calc_offsets_second_guess()
            self.fit_second_guess()
            self.calc_second_guess_shared_fluctuating_component()
            self.calc_cleaned_offsets()
            self.fit_cleaned_offsets()

    def parse_drift_parameters(self, drift_parameters):
        drift_defaults = dict(
            exclude=[1e-2, 5e-3],
            polydeg=8,
            outliers_polydeg=8,
            use_spline=False,
            spline_smooth=2e-4,
            exclude_sn=None,
            tau0=20.0,
            tau_bounds=(5.0, 180.0),
            beta_bounds=(1.0 / 3.0, 3.0),
            fit_mode="auto",
        )
        for key, value in drift_defaults.items():
            setattr(self, key, value)
        if drift_parameters is not None:
            assert type(drift_parameters) == dict
            for key, value in drift_parameters.items():
                setattr(self, key, value)
        if self.fit_mode not in ("linear", "auto", "exp"):
            raise ValueError(
                f"fit_mode must be 'linear', 'auto', or 'exp'; got {self.fit_mode!r}"
            )

    def print_drift_parameters(self):
        parameter_list = [
            "l1_grid_dir",
            "mooring_name",
            "last_window",
            "exclude",
            "polydeg",
            "outliers_polydeg",
            "use_spline",
            "spline_smooth",
            "exclude_sn",
            "fit_mode",
            "tau0",
            "tau_bounds",
            "beta_bounds",
        ]
        for key in parameter_list:
            print(key, ":", getattr(self, key))
        print("\n")

    def list_gridded_level1_files(self, first_n_chunks):
        # config = io.load_config()
        # gridded_path_l1 = config.data.gridded.thermistors.level1[
        #     f"mavs{self.mooring_id}"
        # ]
        all_gridded_files_l1 = sorted(list(self.l1_grid_dir.glob("*.nc")))
        all_gridded_files_l1 = _ignore_hidden_files(all_gridded_files_l1)
        if first_n_chunks is not None:
            all_gridded_files_l1 = all_gridded_files_l1[:first_n_chunks]
        self.files_gridded_level1 = all_gridded_files_l1

    def windowed_background_fits(self):
        """This shows the offsets determined from background fits for all
        windows. Outliers not removed yet."""
        all_offs = []
        all_time = []
        time_window = []
        # Iterate over all 2-day-gridded data files. Split up into one-day
        # windows. We treat first and last file differently as we drop times
        # without any data and treat the rest as one window.
        for file in tqdm.tqdm_notebook(self.files_gridded_level1):
            data_gridded = xr.open_dataarray(file)
            data_gridded.close()
            # First or last file.
            if (
                file == self.files_gridded_level1[0]
                or file == self.files_gridded_level1[-1]
            ):
                tmp = data_gridded.dropna(dim="time", how="all")
                offs = offsets_from_background_fit(
                    tmp,
                    exclude=self.exclude,
                    plot=False,
                    polydeg=self.polydeg,
                    outliers_polydeg=self.outliers_polydeg,
                    spline=self.use_spline,
                    spline_smooth=self.spline_smooth,
                    exclude_sn=self.exclude_sn,
                )
                all_offs.append(offs)
                time_window.append(tmp.time.mean(dim="time"))
            # All other files.
            else:
                days = data_gridded.time.dt.day
                ud, ind = np.unique(days, return_index=True)
                ud = days[
                    np.sort(ind)
                ]  # need this trick because unique() returns sorted...
                for udi in ud:
                    tmp = data_gridded.where(days == udi, drop=True)
                    offs = offsets_from_background_fit(
                        tmp,
                        exclude=self.exclude,
                        plot=False,
                        polydeg=self.polydeg,
                        outliers_polydeg=self.outliers_polydeg,
                        spline=self.use_spline,
                        spline_smooth=self.spline_smooth,
                        exclude_sn=self.exclude_sn,
                    )
                    time_window.append(tmp.time.mean(dim="time"))
                    all_offs.append(offs)
            all_time.append(data_gridded.time)

        time = xr.concat(all_time, dim="time")
        off1 = xr.concat(all_offs, dim="window")
        off1.coords["window"] = (("window"), np.arange(len(time_window)))
        tmp = xr.open_dataarray(self.files_gridded_level1[0])
        off1.coords["sn"] = tmp.sn
        off1.name = "offsets"
        tmp.close()
        self.offsets_initial = off1
        self.time_window = [ti.data for ti in time_window]

    def remove_outliers(self, last_window=None):
        """Outliers in the offsets (anything outside +/-3 times standard
        deviation) are removed on initialization. This leads to a few nan's in
        the offset time series that we will have to consider in the
        calculations."""
        self.offsets = self.offsets_initial.copy()
        if last_window is not None:
            self.offsets = self.offsets.isel(window=range(last_window))
        self.n_offset_outliers = 0
        self.offsets.groupby("depth").apply(self.remove_outliers_one_sensor)
        print(
            f"Removed a total of {self.n_offset_outliers} outliers via 3*std criterion"
        )

    def remove_outliers_one_sensor(self, tt):
        tt = tt.squeeze()
        ttm = tt.median(dim="window")
        tts = tt.std(dim="window")
        upper_bound = ttm + 3 * tts
        lower_bound = ttm - 3 * tts
        ind = (tt > upper_bound) | (tt < lower_bound)
        tt[ind] = np.nan
        if np.any(ind):
            self.n_offset_outliers += ind.sum().data
        return tt

    def select_triplet(self, ni):
        n = len(self.offsets.sn)
        if ni == 0:
            tt = self.offsets.isel(depth=[ni, ni + 1])
        elif ni == n - 1:
            tt = self.offsets.isel(depth=[ni - 1, ni])
        else:
            tt = self.offsets.isel(depth=[ni - 1, ni, ni + 1])
        return tt

    def calc_first_guess_shared_fluctuating_component(self):
        """The first step is to calculate the first guess for a shared
        fluctuating component for each sensor from the sensor and its three
        neighbors."""
        n = len(self.offsets.sn)
        # pre-allocate the output DataArray
        self.first_guess_shared_fluct_comp = self.offsets.copy()
        self.first_guess_shared_fluct_comp.data = (
            np.ones_like(self.first_guess_shared_fluct_comp) * np.nan
        )
        for ni in range(n):
            tt = self.select_triplet(ni)
            demeaned_first_guess = tt - tt.groupby("depth").mean(dim="window")
            first_guess = demeaned_first_guess.mean(dim="depth")
            self.first_guess_shared_fluct_comp[:, ni] = first_guess

    def calc_offsets_second_guess(self):
        """Calculate second guess of drift as first guess minus the shared
        fluctuating component. This is a time series for each of the
        neighbors."""
        self.offsets_second_guess = self.offsets - self.first_guess_shared_fluct_comp

    def fit_second_guess(self):
        # Linear fit is always computed — cheap and needed as fallback.
        self.second_guess_linfit = xr.apply_ufunc(
            linfit_ufunc,
            self.offsets_second_guess,
            input_core_dims=[["window"]],
            output_core_dims=[["window"]],
            vectorize=True,
        )

        if self.fit_mode == "linear":
            self.fit = self.second_guess_linfit
        else:
            self.second_guess_expfit = xr.apply_ufunc(
                functools.partial(
                    expfit_ufunc,
                    tau0=self.tau0,
                    tau_bounds=self.tau_bounds,
                    beta_bounds=self.beta_bounds,
                ),
                self.offsets_second_guess,
                input_core_dims=[["window"]],
                output_core_dims=[["window"]],
                vectorize=True,
            )
            if self.fit_mode == "exp":
                self.fit = self.second_guess_expfit
            else:  # "auto"
                self.fit, self.second_guess_fit_type = self._select_fit_per_sensor(
                    self.offsets_second_guess,
                    self.second_guess_linfit,
                    self.second_guess_expfit,
                )

        # Now detrend the initial drift guess with the fit:
        self.offsets_detrended = self.offsets.copy() - self.fit

    def calc_second_guess_shared_fluctuating_component(self):
        # Calculate a second guess shared fluctuating component from the
        # detrended offset time series.
        n = len(self.offsets.sn)
        # pre-allocate the output DataArray
        self.second_guess_shared_fluct_comp = self.offsets.copy()
        self.second_guess_shared_fluct_comp.data = (
            np.ones_like(self.second_guess_shared_fluct_comp) * np.nan
        )
        for ni in range(n):
            tt = self.select_triplet(ni)
            demeaned = tt - tt.groupby("depth").mean(dim="window")
            shared_component = demeaned.mean(dim="depth")
            self.second_guess_shared_fluct_comp[:, ni] = shared_component

    def calc_cleaned_offsets(self):
        # Calculate cleaned offset time series by removing the 2nd guess shared
        # fluctuating component.
        self.offsets_clean = self.offsets - self.second_guess_shared_fluct_comp

    def fit_cleaned_offsets(self):
        # Fit the clean offset time series. This is the final sensor drift.
        self.drift_linfit = xr.apply_ufunc(
            linfit_ufunc,
            self.offsets_clean,
            input_core_dims=[["window"]],
            output_core_dims=[["window"]],
            vectorize=True,
        )

        if self.fit_mode == "linear":
            self.drift_fit = self.drift_linfit
            self.fit_type = xr.DataArray(
                np.array(["lin"] * self.drift_linfit.sizes["depth"]),
                dims=("depth",),
                coords=self._sensor_axis_coords(self.drift_linfit),
            )
        else:
            self.drift_expfit = xr.apply_ufunc(
                functools.partial(
                    expfit_ufunc,
                    tau0=self.tau0,
                    tau_bounds=self.tau_bounds,
                    beta_bounds=self.beta_bounds,
                ),
                self.offsets_clean,
                input_core_dims=[["window"]],
                output_core_dims=[["window"]],
                vectorize=True,
            )
            self.drift_exp_params = xr.apply_ufunc(
                functools.partial(
                    _fit_cvhg16_params,
                    tau0=self.tau0,
                    tau_bounds=self.tau_bounds,
                    beta_bounds=self.beta_bounds,
                ),
                self.offsets_clean,
                input_core_dims=[["window"]],
                output_core_dims=[["param"]],
                vectorize=True,
                dask_gufunc_kwargs={"output_sizes": {"param": 5}},
            ).assign_coords(param=list(EXP_PARAM_NAMES))
            if self.fit_mode == "exp":
                self.drift_fit = self.drift_expfit
                self.fit_type = xr.DataArray(
                    np.array(["exp"] * self.drift_expfit.sizes["depth"]),
                    dims=("depth",),
                    coords=self._sensor_axis_coords(self.drift_expfit),
                )
            else:  # "auto"
                self.drift_fit, self.fit_type = self._select_fit_per_sensor(
                    self.offsets_clean,
                    self.drift_linfit,
                    self.drift_expfit,
                )

    @staticmethod
    def _sensor_axis_coords(da):
        return {c: da.coords[c] for c in ("depth", "sn") if c in da.coords}

    def _select_fit_per_sensor(self, offsets, linfit, expfit):
        """Pick linfit or expfit per sensor using the CvHG16 R² criterion.

        Returns (fit DataArray with the same dims as ``linfit`` /
        ``expfit``, fit_type DataArray along the sensor axis).
        """
        types = []
        for di in range(offsets.sizes["depth"]):
            types.append(
                lin_or_exp(
                    offsets.isel(depth=di),
                    linfit.isel(depth=di),
                    expfit.isel(depth=di),
                    return_type=True,
                )
            )
        fit_type = xr.DataArray(
            np.array(types),
            dims=("depth",),
            coords=self._sensor_axis_coords(linfit),
        )
        fit = xr.where(fit_type == "exp", expfit, linfit)
        return fit, fit_type

    def drift_to_dataarray(self):
        """Note: the old version of this function lets you evaluate the fits to
        the actual length of the time series. Bypassing this for now as we are
        using the full time series for fitting but will need to bring this back
        in.
        """
        out = self.drift_fit.copy()
        out.name = f"{self.mooring_name} sensor drift"
        out.coords["time"] = (["window"], np.array(self.time_window))
        self.drift = out

    def drift_to_netcdf(self, path, suffix=None):
        self.drift_to_dataarray()
        # config = io.load_config()
        # savename = config.data.aux.thermistors.drift[f"{self.mooring_name}"]
        # savename.parent.mkdir(exist_ok=True)
        # if suffix is not None:
        #     savename = savename.parent.joinpath(
        #         savename.stem + "_" + suffix + savename.suffix
        #     )
        savename = f"drift_{self.mooring_name}"
        if suffix is not None:
            savename = f"{savename}_{suffix}"
        savename = f"{savename}.nc"
        savepath = path.joinpath(savename)
        print("saving to", savepath)
        self.drift.to_netcdf(savepath, mode="w")

    def plot_sensor_drift_offsets(self):
        # Plot various stages of the offset time series and their shared
        # fluctuating component.
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
        fig.suptitle(f"Sensor drift {self.mooring_name}", fontsize=14)
        # Create 2x1 subfigs. Working with subfigures here to be able to do
        # titles per row.
        subfigs = fig.subfigures(nrows=2, ncols=1)
        subfigs[0].suptitle("Offset estimates")
        subfigs[1].suptitle("Shared fluctuating component")
        # Create 1x3 subplots per subfig.
        axtop = subfigs[0].subplots(nrows=1, ncols=5, sharey=True)
        axbot = subfigs[1].subplots(nrows=1, ncols=4, sharey=True)
        axall = [axtop, axbot]

        offset_options = dict(
            vmin=-0.01, vmax=0.01, cmap="RdBu", y="depth", add_colorbar=False
        )
        options_cb = offset_options.copy()
        options_cb["add_colorbar"] = True
        options_cb["cbar_kwargs"] = dict(aspect=25, shrink=0.8, label=r"$\Delta$T [°C]")
        self.offsets_initial.plot(ax=axtop[0], **offset_options)
        axtop[0].set(title="1st guess")

        self.offsets.plot(ax=axtop[1], **offset_options)
        axtop[1].set(title="Outliers removed")

        self.first_guess_shared_fluct_comp.plot(ax=axbot[1], **offset_options)
        axbot[1].set(title="1st guess")

        self.offsets_second_guess.plot(ax=axtop[2], **offset_options)
        axtop[2].set(title="2nd guess")

        self.offsets_detrended.plot(ax=axtop[3], **offset_options)
        axtop[3].set(title="Detrended")

        self.offsets_clean.plot(ax=axtop[4], **options_cb)
        axtop[4].set(title="Cleaned")

        self.second_guess_shared_fluct_comp.plot(ax=axbot[2], **options_cb)
        axbot[2].set(title="2nd guess (from detrended)")

        for axi in [axtop[0], axbot[0]]:
            axi.invert_yaxis()
        for axi in axbot[[0, 3]]:
            axi.set_visible(False)
        for axi in np.append(axtop[1:], axbot[2:]):
            axi.set(ylabel="")

    def plot_offsets_violin(self):
        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(5, 5), constrained_layout=True
        )
        mask = ~np.isnan(self.offsets_clean.data)
        filtered_data = [d[m] for d, m in zip(self.offsets_clean.data.T, mask.T)]
        ax.violinplot(
            filtered_data,
            self.offsets_clean.depth.data,
            points=50,
            widths=5,
            showmeans=False,
            vert=False,
            showextrema=False,
            showmedians=True,
            bw_method=0.5,
        )
        ax.set(
            xlabel=r"$\Delta \Theta$ [K]",
            title=f"Sensor Drift {self.mooring_name}",
        )
        ax.grid()
        ax.invert_yaxis()
        ax.set(ylabel="depth [m]")

    def plot_components(self, zi):
        fig, ax = gv.plot.quickfig()
        self.offsets.isel(depth=[zi - 1, zi, zi + 1]).plot(
            hue="depth", ax=ax, color="k", alpha=0.7
        )
        self.first_guess_shared_fluct_comp.isel(depth=[zi - 1, zi, zi + 1]).plot(
            hue="depth", ax=ax, color="b", alpha=0.3
        )
        self.offsets_second_guess.isel(depth=[zi - 1, zi, zi + 1]).plot(
            hue="depth", ax=ax, color="r", alpha=0.3
        )
        self.second_guess_linfit.isel(depth=[zi - 1, zi, zi + 1]).plot(
            hue="depth", ax=ax, color="g", alpha=0.3
        )
        self.second_guess_expfit.isel(depth=[zi - 1, zi, zi + 1]).plot(
            hue="depth", ax=ax, color="orange", alpha=0.3
        )

    def plot_drift_sensor_and_neighbors(self, zi):
        fig, (ax0, ax1) = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(7.5, 6),
            constrained_layout=True,
            sharex=True,
        )
        tt = self.select_triplet(zi)
        tt.plot(hue="depth", linewidth=0.75, ax=ax0)
        self.second_guess_shared_fluct_comp.isel(depth=zi).plot(color="k", ax=ax0)
        ax0.set(xlabel="", title="")
        ax0.grid()
        self.offsets.isel(depth=zi).plot(linestyle="", marker="+", color="b", ax=ax1)
        self.offsets_clean.isel(depth=zi).plot(
            linestyle="", marker="o", color="k", ax=ax1
        )
        self.drift_linfit.isel(depth=zi).plot(color="r", linestyle="--", ax=ax1)
        self.drift_expfit.isel(depth=zi).plot(color="r", linestyle="-", ax=ax1)
        r2fitlin = calculate_r2(
            self.offsets_clean.isel(depth=zi), self.drift_linfit.isel(depth=zi)
        )
        r2fitexp = calculate_r2(
            self.offsets_clean.isel(depth=zi), self.drift_expfit.isel(depth=zi)
        )
        anno_opts = dict(backgroundcolor="w")
        ax1.annotate(
            f"Rlin$^2$={r2fitlin.data:1.3f}",
            xy=(0.02, 0.05),
            xycoords="axes fraction",
            **anno_opts,
        )
        ax1.annotate(
            f"Rexp$^2$={r2fitexp.data:1.3f}",
            xy=(0.22, 0.05),
            xycoords="axes fraction",
            **anno_opts,
        )
        if hasattr(self, "drift_exp_params"):
            params = self.drift_exp_params.isel(depth=zi)
            tau = float(params.sel(param="tau").data)
            beta = float(params.sel(param="beta").data)
            if np.isfinite(tau) and np.isfinite(beta):
                label = rf"$\tau$={tau:.1f}d, $\beta$={beta:.2f}"
            else:
                label = r"$\tau$=—, $\beta$=—"
            ax1.annotate(
                label,
                xy=(0.42, 0.05),
                xycoords="axes fraction",
                **anno_opts,
            )
        ax1.grid()
        ax1.set(title="")
        fig.suptitle(
            f"SN {self.offsets.sn.isel(depth=zi).data} @ {self.offsets.depth.isel(depth=zi).data:3.1f}m"
        )

    def plot_drift_all_sensors(self, figdir, suffix=None):
        n = len(self.offsets.sn)
        print(f"Saving figures to {figdir}")
        for ni in range(n):
            sn = self.offsets.sn.isel(depth=ni).data
            self.plot_drift_sensor_and_neighbors(zi=ni)
            savename = f"{self.mooring_name}_fit_{sn}"
            if suffix is not None:
                savename += "_" + suffix
            gv.plot.png(savename, figdir=figdir, verbose=False)
            plt.close()


def linfit_ufunc(ti, n_output=None):
    """Linearly fit time series.

    The time series may have NaNs. The returned time series containing the
    linear fit has the same length as the original time series and keeps the
    NaNs in place.

    Parameters
    ----------
    ti : xr.DataArray
        Time series

    Returns
    -------
    fit : xr.DataArray
        Fit results evaluated at original data points.
    """
    n = np.arange(len(ti))
    good = ~np.isnan(ti)
    tig = ti[good]
    ng = n[good]
    fit = np.polyfit(ng, tig, 1)
    p = np.poly1d(fit)
    return p(n)


def exp_function(t, t0, m, A, beta, tau):
    """CvHG16 Eq. 5.

    ΔT(t) = ΔT₀ + m·t + A · γ(1/β, (t/τ)^β) / β

    with γ the lower incomplete gamma function, related to scipy's
    regularised gammainc by γ(a, x) = Γ(a) · gammainc(a, x).
    """
    a = 1.0 / beta
    z = (np.asarray(t, dtype=float) / tau) ** beta
    lower_gamma = scipy.special.gamma(a) * scipy.special.gammainc(a, z)
    return t0 + m * t + A * lower_gamma / beta


def calculate_r2(x, xfit):
    xm = np.mean(x)
    SSres = np.sum((x - xfit) ** 2)
    SStot = np.sum((x - xm) ** 2)
    return 1 - SSres / SStot


EXP_PARAM_NAMES = ("t0", "m", "A", "beta", "tau")


def _fit_cvhg16_params(
    x,
    tau0=20.0,
    tau_bounds=(5.0, 180.0),
    beta_bounds=(1.0 / 3.0, 3.0),
    A_scan_factor=1.5,
    A_scan_iters=40,
):
    """Fit CvHG16 Eq. 5 to a 1-d offset series, returning the parameters.

    Shared core between ``expfit_ufunc`` (which evaluates the best-R² fit
    back onto the full index range) and diagnostic code that needs
    ``[t0, m, A, beta, tau]``. Returns a length-5 NaN vector when the
    series has fewer than 5 finite points, every ``curve_fit`` attempt in
    the β / A scan raised, or no candidate yielded a finite R².
    """
    n = np.arange(len(x))
    good = ~np.isnan(x)
    xg = x[good]
    ng = n[good]

    # Five free parameters in exp_function, so we need at least five
    # residuals for curve_fit to have any chance.
    if xg.size < 5:
        return np.full(5, np.nan)

    lin = scipy.stats.linregress(ng, xg)
    m0 = float(np.clip(lin.slope, -1e-4, 1e-4))
    t0 = float(np.clip(lin.intercept, -0.2, 0.2))

    lb = [-0.2, -1e-4, -np.inf, beta_bounds[0], tau_bounds[0]]
    ub = [0.2, 1e-4, np.inf, beta_bounds[1], tau_bounds[1]]

    best = None
    for beta0 in (0.5, 2.0):
        A_seed = 0.005 * m0
        for _ in range(A_scan_iters):
            try:
                popt, _ = scipy.optimize.curve_fit(
                    exp_function,
                    ng,
                    xg,
                    p0=[t0, m0, A_seed, beta0, tau0],
                    bounds=(lb, ub),
                    maxfev=2000,
                )
            except (RuntimeError, ValueError):
                popt = None
            if popt is not None:
                r2 = calculate_r2(xg, exp_function(ng, *popt))
                if np.isfinite(r2) and (best is None or r2 > best[0]):
                    best = (r2, popt)
            A_seed *= A_scan_factor

    if best is None:
        return np.full(5, np.nan)
    return np.asarray(best[1], dtype=float)


def expfit_ufunc(
    x,
    tau0=20.0,
    tau_bounds=(5.0, 180.0),
    beta_bounds=(1.0 / 3.0, 3.0),
    A_scan_factor=1.5,
    A_scan_iters=40,
):
    """Fit CvHG16 Eq. 5 to a 1-d offset series.

    Implements the paper's algorithm (JTECH-D-15-0243.1 §4a): seed from a
    linear regression, then run Nelder-Mead-style bounded least-squares
    with β seeds {0.5, 2.0}, scanning the relaxation amplitude A by a
    factor of ``A_scan_factor`` for up to ``A_scan_iters`` steps. The
    returned fit is the best-R² candidate evaluated at every index of
    ``x`` (including indices where ``x`` is NaN). Returns all-NaN if the
    series has fewer than 5 finite points or every attempt raised.

    ``tau0`` / ``tau_bounds`` are the main knobs for RBR/SBE vs. NIOZ
    instruments — NIOZ needs τ₀ ≈ 2 d, RBR/SBE ≈ 20 d.
    """
    popt = _fit_cvhg16_params(
        x,
        tau0=tau0,
        tau_bounds=tau_bounds,
        beta_bounds=beta_bounds,
        A_scan_factor=A_scan_factor,
        A_scan_iters=A_scan_iters,
    )
    if np.any(np.isnan(popt)):
        return np.full_like(x, np.nan, dtype=float)
    return exp_function(np.arange(len(x)), *popt)


def lin_or_exp(x, xlin, xexp, return_type=False):
    xlin_orig = xlin.copy()
    xexp_orig = xexp.copy()
    if type(xlin) == xr.DataArray:
        xlin = xlin.data
    if type(xexp) == xr.DataArray:
        xexp = xexp.data
    good = ~np.isnan(x.data)
    xg = x.data[good]
    R2lin = calculate_r2(xg, xlin[good])
    Rlin = np.sqrt(R2lin)
    R2exp = calculate_r2(xg, xexp[good])
    Rexp = np.sqrt(R2exp)
    use_exp = Rexp > Rlin + 0.3 * (1 - Rlin)
    if return_type:
        if use_exp:
            return "exp"
        else:
            return "lin"
    else:
        # print(Rlin, Rexp, use_exp)
        if use_exp:
            # print(
            #     f"{x.sn.data:6.0f} @ {x.depth.data:4.0f}m: using exponential fit"
            # )
            return xexp_orig
        else:
            return xlin_orig


def calc_eps_thorpe_scale(t, lon, lat, S0=35.13):
    """Calculate turbulent dissipation based on the Thorpe scales.

    Parameters
    ----------
    t : xr.DataArray
        In-situ temperature calculated from thermistor dataset.

    Returns
    -------
    eps : xr.DataArray
        Thorpe scale-based turbulent dissipation [W/kg].
    Lt : xr.DataArray
        Thorpe scale [m].
    """

    def thorpe_scale_calcs(t, lon, lat, S):
        N2_method = "teos"
        eps_t, N2_t, diag = mx.overturn.nan_eps_overturn(
            t.depth.data,
            t.data,
            SP=S,
            lon=lon,
            lat=lat,
            N2_method=N2_method,
            return_diagnostics=True,
        )
        return eps_t, diag["Lt"]

    results = []
    for g, ti in tqdm.tqdm(t.groupby("time")):
        ti = ti.squeeze()
        try:
            eps, Lt = thorpe_scale_calcs(ti, lon, lat, S0)
        except:
            eps = np.zeros_like(ti.data) * np.nan
            Lt = np.ones_like(ti.data) * (-1)
        results.append((eps, Lt))

    eps_results = np.array([result_i[0] for result_i in results])
    Lt_results = np.array([result_i[1] for result_i in results])

    eps_out = xr.DataArray(eps_results.T, coords=t.coords, dims=t.dims)

    Lt_out = xr.DataArray(Lt_results.T, coords=t.coords, dims=t.dims)
    return eps_out, Lt_out


def _ignore_hidden_files(files):
    return [f for f in files if not f.name.startswith(".")]


def _parse_path(path):
    return Path(path) if isinstance(path, str) else path
