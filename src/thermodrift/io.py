#!/usr/bin/env python
# coding: utf-8
"""
I/O functions.

Generate a processing object with `ProcessThermistorMooring`.
"""

from pathlib import Path
import yaml
from box import Box
import numpy as np
import pandas as pd
import rbrmoored
# import sbemoored


def dummy():
    print("meow!!!")


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


def load_config_box(configfile) -> Box:
    """Load the yaml config file as Box object which is basically a dict with
    dot access.

    Returns
    -------
    cfg : Box
        Config parameters dictionary with dot access.

    See also
    --------
    `load_config`
    """

    with open(configfile, "r") as ymlfile:
        cfg = Box(yaml.safe_load(ymlfile))
    cfg.path.root = configfile.parent.parent.resolve()
    # cfg.path.data = cfg.path.root.joinpath(cfg.path.data)
    # cfg.path.fig = cfg.path.root.joinpath(cfg.path.fig)

    # parse datetimes
    for time in ["start_time", "end_time"]:
        cfg[time] = np.datetime64(cfg[time])

    return cfg


def mooring_sheet_load(path):
    mooring_info = pd.read_csv(path, sep=",", header=0, index_col="SN")
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
    elif path.suffix == ".xlsx":
        df = sensor_sheet_xlsx_load(path)
    df = sensor_sheet_rename_columns(df)
    df = sensor_sheet_columns_to_dt64(df)
    sensor_sheet_unify_sensor_type(df)
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
            "CTD Rosette Calibration": "ctd_cal",
            "Pre-Deployment Time Calibration": "time_cal1",
            "Post-Deployment Time Calibration": "time_cal2",
            "Post-Deployment UTC Time": "clock_read_utc",
            "Post-Deployment Logger Time": "clock_read_logger",
            "Type": "type",
            "SN": "sn",
            "Pre-Deployment Notes": "pre_notes",
            "Post-Deployment Notes": "post_notes",
        }
    )
    return df


def sensor_sheet_unify_sensor_type(df):
    """Parse sensor type in sensor spreadsheet. Will change various sensor
    descriptions to either `rbr` or `sbe`.

    Parameters
    ----------
    df : pandas.DataFrame
        Sensor information.
    """
    for i, v in df.type.items():
        if "56" in v or "Seabird" in v or "SBE" in v:
            df.type.at[i] = "sbe"
        elif "rbr" in v or "RBR" in v or "Solo" in v or "solo" in v:
            df.type.at[i] = "rbr"
        else:
            raise ValueError(f'Could not parse instrument type "{v}" in sensor sheet.')


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
        "ctd_cal",
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

    proc_db = sensor_sheet[["time_cal1", "time_cal2"]].copy()
    n = proc_db.index.shape[0]

    proc_db = proc_db.assign(processed=np.tile(False, n))
    proc_db = proc_db.assign(raw_data_exists=proc_db.processed.copy())
    proc_db = proc_db.assign(figure_exists=proc_db.processed.copy())
    proc_db = proc_db.assign(comment=np.tile("ok", n))

    proc_db = proc_db.sort_index()
    return proc_db


def get_file_name(sn, data_dir, extension="rsk"):
    files = list(data_dir.glob(f"{sn:06}*.{extension}"))
    if len(files) == 1:
        return files[0]
    elif len(files) == 0:
        return None
    else:
        raise OSError(f"more than one file for SN{sn} in {data_raw}")


def proc_db_update(proc_db, data_raw, data_out, figure_out):
    for g, v in proc_db.groupby("SN"):
        try:
            f = get_file_name(g, data_dir=data_raw, extension="rsk")
            if f is not None:
                proc_db.raw_data_exists.at[g] = True
            f = get_file_name(g, data_dir=data_out, extension="nc")
            if f is not None:
                proc_db.processed.at[g] = True
            f = get_file_name(g, data_dir=figure_out, extension="png")
            if f is not None:
                proc_db.figure_exists.at[g] = True
        except:
            pass


class ProcessThermistorMooring:
    """Process thermistors on one mooring."""

    def __init__(self, config_file):
        """Initialize processing object for one mooring.

        Parameters
        ----------
        config_file : pathlib.Path or str
            .yaml configuration file
        """
        self.cfg = load_config_box(config_file)
        self.meta = self.cfg.meta
        self.load_sensor_info()
        self.load_mooring_info()
        self.proc_db = proc_db_generate(self.sensor_info)

    def load_sensor_info(self):
        sensor_sheet_path = self.cfg.path.root.joinpath(self.cfg.path.sensors)
        self.sensor_info = sensor_sheet_load(sensor_sheet_path)

    def load_mooring_info(self):
        mooring_sheet_path = self.cfg.path.root.joinpath(self.cfg.path.mooring)
        self.mooring_info = mooring_sheet_load(mooring_sheet_path)


def _parse_path(path):
    return Path(path) if isinstance(path, str) else path
