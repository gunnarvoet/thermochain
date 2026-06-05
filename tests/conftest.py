import pathlib
import shutil

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml


# Determine paths and make them available in tests using fixtures.
# Anything function decorated with @pytest.fixture will be availabe as input
# variable in other test files.
@pytest.fixture
def rootdir():
    return pathlib.Path(__file__).parent.resolve()


@pytest.fixture
def config_file_path(rootdir):
    return rootdir / "data/config.yml"


@pytest.fixture
def synthetic_l1_dir(tmp_path):
    """Minimal L1-gridded thermistor dataset for pipeline smoke tests.

    Writes four 3-day netCDF files, each 1-minute cadence, 12 depths / sn.
    Temperatures are a smooth exponential profile plus small wiggles so
    the background fit has something to work with. No real CTD cal, no
    real drift — just enough structure to exercise the pipeline.

    The 4 × 3-day layout produces 1 + 3 + 3 + 1 = 8 windows, enough for
    the 4-parameter exponential fit inside fit_second_guess to converge.
    12 sensors keeps the default polydeg=8 polynomial well-conditioned.
    """
    n_depth = 12
    sn = np.array([72100 + i for i in range(n_depth)])
    depth = np.linspace(1000.0, 1200.0, n_depth)
    t_mean = 4.0 + 6.0 * np.exp(-(depth - 1000.0) / 200.0)

    def _build_file(day_start, day_end):
        rng = np.random.default_rng(int(day_start))
        times = np.arange(
            np.datetime64(f"2024-01-{day_start:02d}T00:00"),
            np.datetime64(f"2024-01-{day_end:02d}T00:00"),
            np.timedelta64(1, "m"),
        )
        arr = (
            t_mean[None, :]
            + 0.02 * rng.standard_normal((times.size, sn.size))
        )
        da = xr.DataArray(
            arr,
            dims=("time", "depth"),
            coords={
                "time": times,
                "depth": ("depth", depth),
                "sn": ("depth", sn),
            },
            name="t",
        )
        return da

    spans = [(1, 4), (4, 7), (7, 10), (10, 13)]
    for start, end in spans:
        path = (
            tmp_path
            / f"mavs0_gridded_2024-01-{start:02d}_to_2024-01-{end:02d}.nc"
        )
        _build_file(start, end).to_netcdf(path)
    return tmp_path


@pytest.fixture
def synthetic_l1_dir_mixed(tmp_path):
    """L1 grid dir containing two co-located segment prefixes.

    Mirrors the real MOTIVE_B grid dir, which holds both
    `motive_b_deep_L1_*.nc` and `motive_b_shallow_L1_*.nc`. Used to
    test the `file_pattern` filter in `sensor_drift`. Both segments
    carry the same 12-sensor structure so the pipeline runs whether or
    not the filter is applied.
    """
    n_depth = 12
    sn = np.array([72100 + i for i in range(n_depth)])
    depth = np.linspace(1000.0, 1200.0, n_depth)
    t_mean = 4.0 + 6.0 * np.exp(-(depth - 1000.0) / 200.0)

    def _build_file(day_start, day_end):
        rng = np.random.default_rng(int(day_start))
        times = np.arange(
            np.datetime64(f"2024-01-{day_start:02d}T00:00"),
            np.datetime64(f"2024-01-{day_end:02d}T00:00"),
            np.timedelta64(1, "m"),
        )
        arr = (
            t_mean[None, :]
            + 0.02 * rng.standard_normal((times.size, sn.size))
        )
        return xr.DataArray(
            arr,
            dims=("time", "depth"),
            coords={
                "time": times,
                "depth": ("depth", depth),
                "sn": ("depth", sn),
            },
            name="t",
        )

    spans = [(1, 4), (4, 7), (7, 10), (10, 13)]
    for start, end in spans:
        stamp = f"2024-01-{start:02d}_to_2024-01-{end:02d}"
        da = _build_file(start, end)
        da.to_netcdf(tmp_path / f"motive_b_deep_L1_{stamp}.nc")
        da.to_netcdf(tmp_path / f"motive_b_shallow_L1_{stamp}.nc")
    return tmp_path


@pytest.fixture
def segmented_mooring(tmp_path, rootdir):
    """A self-contained mooring: config + sheets + per-sensor L1 files.

    Reuses the known-good tests/data sheets (they already instantiate
    ProcessThermistorMooring), adds a ``segment`` column splitting the
    first three SNs into ``deep`` and the rest into ``shallow``, and writes
    one procl1 NetCDF per deep SN so grid_l1 has input. Returns the
    config path.

    The config is written at ``tmp_path / "run" / "config.yml"`` so that
    ``configfile.parent.parent == tmp_path``, matching the ``data/`` tree
    at ``tmp_path / "data/"``.
    """
    data = tmp_path / "data"
    (data / "proc" / "l1").mkdir(parents=True)

    shutil.copy(rootdir / "data/sensor_sheet.csv", data / "sensor_sheet.csv")
    moor = pd.read_csv(rootdir / "data/mooring_sheet.csv")
    moor["segment"] = ["deep"] * 3 + ["shallow"] * (len(moor) - 3)
    moor.to_csv(data / "mooring_sheet.csv", index=False)
    deep_sns = [int(s) for s in moor["SN"].iloc[:3]]

    times = np.arange(
        np.datetime64("2024-01-01T00:00"),
        np.datetime64("2024-01-05T00:00"),
        np.timedelta64(1, "m"),
    )
    for i, sn in enumerate(deep_sns):
        da = xr.DataArray(
            4.0 + 0.01 * i + 0.001 * np.sin(np.arange(times.size) / 50.0),
            dims="time",
            coords={"time": times},
            name="t",
        )
        da.attrs.update(
            {"sampling period in s": 60.0, "units": "degree_C", "long_name": "temperature"}
        )
        da.to_netcdf(data / "proc" / "l1" / f"mavs3__rbr__{sn:06d}_L1.nc")

    cfg = {
        "info": "segmented test",
        "meta": {"mooring_name": "mavs3", "project": "TestProj", "PI": "x", "email": "x"},
        "path": {
            "fig": "fig/",
            "data": {
                "raw": {"rbr": "data/raw/", "sbe": "data/raw/"},
                "proc": "data/proc/",
                "grid": "data/grid/",
            },
            "sensors": "data/sensor_sheet.csv",
            "mooring": "data/mooring_sheet.csv",
            "aux": "data/aux/",
        },
        "start_time": "2024-01-01 00:00:00",
        "end_time": "2024-01-05 00:00:00",
        "ignore_sns": [],
        "gridding": {"dt": "10s", "max_gap": "30s", "chunk": "2D"},
        "segments": {
            "deep": {"select": {"segment": "deep"}, "gridding": {"dt": "10s", "max_gap": "30s"}},
            "shallow": {"select": {"segment": "shallow"}},
        },
    }
    cfgdir = tmp_path / "run"
    cfgdir.mkdir()
    cfgpath = cfgdir / "config.yml"
    with open(cfgpath, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfgpath


@pytest.fixture
def segmented_mooring_excluded(tmp_path, rootdir):
    """Like ``segmented_mooring`` but with the first deep SN flagged exclude==1.

    Uses SN 72219 (already in tests/data/sensor_sheet.csv with type=RBR Deep)
    as the first deep sensor so the exclude flag can be set on a row that
    loads cleanly through sensor_sheet_load.  The mooring sheet is built from
    scratch with three deep SNs and one shallow SN.

    The fixture writes per-sensor L1 files for ALL three deep SNs (so the
    only reason the excluded sensor is absent from the grid is the exclude
    filter, not a missing input file).  Returns a tuple
    ``(config_path, excluded_sn)`` so tests can assert the SN is absent.
    """
    data = tmp_path / "data"
    (data / "proc" / "l1").mkdir(parents=True)

    # Build a mooring sheet where the first deep SN (72219) also lives in
    # the sensor sheet so that flagging it there is meaningful.
    # The other two deep SNs (201844, 202306) are also in the sensor sheet.
    deep_sns = [72219, 201844, 202306]
    shallow_sns = [392]
    excluded_sn = deep_sns[0]  # 72219

    moor_rows = (
        [{"type": "RBR Solo3", "SN": sn, "height": float(i + 1), "depth": float(1430 - i), "segment": "deep"}
         for i, sn in enumerate(deep_sns)]
        + [{"type": "SBE 56", "SN": sn, "height": float(i + 10), "depth": float(1420 - i), "segment": "shallow"}
           for i, sn in enumerate(shallow_sns)]
    )
    moor = pd.DataFrame(moor_rows)
    moor.to_csv(data / "mooring_sheet.csv", index=False)

    # Copy and augment sensor sheet: add exclude column, flag excluded_sn
    sensor_df = pd.read_csv(rootdir / "data/sensor_sheet.csv")
    if "exclude" not in sensor_df.columns:
        sensor_df["exclude"] = 0
    sensor_df.loc[sensor_df["SN"] == excluded_sn, "exclude"] = 1
    sensor_df.to_csv(data / "sensor_sheet.csv", index=False)

    times = np.arange(
        np.datetime64("2024-01-01T00:00"),
        np.datetime64("2024-01-05T00:00"),
        np.timedelta64(1, "m"),
    )
    # Write L1 files for ALL deep SNs (including the excluded one)
    for i, sn in enumerate(deep_sns):
        da = xr.DataArray(
            4.0 + 0.01 * i + 0.001 * np.sin(np.arange(times.size) / 50.0),
            dims="time",
            coords={"time": times},
            name="t",
        )
        da.attrs.update(
            {"sampling period in s": 60.0, "units": "degree_C", "long_name": "temperature"}
        )
        da.to_netcdf(data / "proc" / "l1" / f"mavs3__rbr__{sn:06d}_L1.nc")

    cfg = {
        "info": "segmented test excluded",
        "meta": {"mooring_name": "mavs3", "project": "TestProj", "PI": "x", "email": "x"},
        "path": {
            "fig": "fig/",
            "data": {
                "raw": {"rbr": "data/raw/", "sbe": "data/raw/"},
                "proc": "data/proc/",
                "grid": "data/grid/",
            },
            "sensors": "data/sensor_sheet.csv",
            "mooring": "data/mooring_sheet.csv",
            "aux": "data/aux/",
        },
        "start_time": "2024-01-01 00:00:00",
        "end_time": "2024-01-05 00:00:00",
        "ignore_sns": [],
        "gridding": {"dt": "10s", "max_gap": "30s", "chunk": "2D"},
        "segments": {
            "deep": {"select": {"segment": "deep"}, "gridding": {"dt": "10s", "max_gap": "30s"}},
            "shallow": {"select": {"segment": "shallow"}},
        },
    }
    cfgdir = tmp_path / "run"
    cfgdir.mkdir()
    cfgpath = cfgdir / "config.yml"
    with open(cfgpath, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfgpath, excluded_sn


@pytest.fixture
def cal_mooring(tmp_path):
    """Self-contained mooring for cut_and_cal: config + sheets + L0 + offsets.

    Two deep SNs (scalar_pre_only) and one shallow SN (linear_interp).
    L0 spans 2024-11-20..12-01 at 60 s; the deployment window
    2024-11-22..11-30 trims both ends so the cut logic is exercised.
    Pre/post offsets are distinct so an interpolated value is detectable.
    Returns the config path.
    """
    deep_sns = [301111, 301222]
    shallow_sns = [302333]
    all_sns = deep_sns + shallow_sns

    data = tmp_path / "data"
    (data / "proc" / "l0").mkdir(parents=True)
    (data / "aux" / "cal_results").mkdir(parents=True)

    # sensor sheet (raw column names; sensor_sheet_load renames them)
    # Includes all columns required by sensor_sheet_columns_to_dt64 (time_cal1/2,
    # clock_read_utc/logger) plus the CTD cal columns needed by cut_and_cal.
    sensor_rows = []
    for sn in all_sns:
        sensor_rows.append({
            "SN": sn,
            "Type": "RBR Solo",
            "Pre-Deployment CTD Calibration Time": "2024-11-20 12:00:00",
            "Pre-Deployment CTD Calibration Cast": 1,
            "Post-Deployment CTD Calibration Time": "2024-12-02 12:00:00",
            "Post-Deployment CTD Calibration Cast": 5,
            "Pre-Deployment Time Calibration": "2024-11-18 10:00:00",
            "Post-Deployment Time Calibration": "2024-12-05 10:00:00",
            "Post-Deployment UTC Time": "2024-12-05 10:00:00",
            "Post-Deployment Logger Time": "2024-12-05 10:00:05",
            "exclude": 0,
        })
    pd.DataFrame(sensor_rows).to_csv(data / "sensor_sheet.csv", index=False)

    # mooring sheet with segment column
    moor_rows = (
        [{"type": "RBR Solo", "SN": sn, "depth": 4300.0 - i, "segment": "deep"}
         for i, sn in enumerate(deep_sns)]
        + [{"type": "RBR Solo", "SN": sn, "depth": 2000.0, "segment": "shallow"}
           for sn in shallow_sns]
    )
    pd.DataFrame(moor_rows).to_csv(data / "mooring_sheet.csv", index=False)

    # per-sensor L0 (one file each; glob is *{sn:06}*.nc)
    times = np.arange(
        np.datetime64("2024-11-20T00:00"),
        np.datetime64("2024-12-01T00:00"),
        np.timedelta64(1, "m"),
    )
    for i, sn in enumerate(all_sns):
        da = xr.DataArray(
            4.0 + 0.01 * i + 0.001 * np.sin(np.arange(times.size) / 50.0),
            dims="time", coords={"time": times}, name="t",
        )
        da.attrs.update({"sampling period in s": 60.0, "units": "degree_C",
                         "long_name": "temperature", "SN": sn})
        da.to_netcdf(data / "proc" / "l0" / f"testproj_a__rbr__{sn:06d}_L0.nc")

    # pre / post offsets NetCDFs (distinct values per side)
    def _offsets(fname, base):
        ds = xr.Dataset(
            {"offset": ("sn", [base + 0.001 * k for k in range(len(all_sns))])},
            coords={"sn": all_sns, "cast": ("sn", [1] * len(all_sns))},
        )
        ds.to_netcdf(data / "aux" / "cal_results" / fname)
    _offsets("motive_cruise1_cal_offsets.nc", base=0.100)   # pre
    _offsets("motive_cruise2_cal_offsets.nc", base=0.200)   # post

    cfg = {
        "info": "cal test",
        "meta": {"mooring_name": "A", "project": "TestProj", "PI": "x", "email": "x"},
        "path": {
            "fig": "fig/",
            "data": {"raw": {"rbr": "data/raw/", "sbe": "data/raw/"},
                     "proc": "data/proc/", "grid": "data/grid/"},
            "sensors": "data/sensor_sheet.csv",
            "mooring": "data/mooring_sheet.csv",
            "aux": "data/aux/",
        },
        "start_time": "2024-11-22 00:00:00",
        "end_time": "2024-11-30 00:00:00",
        "ignore_sns": [],
        "gridding": {"dt": "10s", "max_gap": "30s", "chunk": "2D"},
        "calibration": {
            "method": "linear_interp",
            "offsets_pre": "data/aux/cal_results/motive_cruise1_cal_offsets.nc",
            "offsets_post": "data/aux/cal_results/motive_cruise2_cal_offsets.nc",
        },
        "segments": {
            "deep": {"select": {"segment": "deep"},
                     "calibration": {"method": "scalar_pre_only"}},
            "shallow": {"select": {"segment": "shallow"}},
        },
    }
    cfgdir = tmp_path / "run"
    cfgdir.mkdir()
    cfgpath = cfgdir / "config.yml"
    with open(cfgpath, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfgpath


@pytest.fixture
def drift_mooring(tmp_path):
    """Self-contained mooring for fit_drift: config + sheets + synthetic gridded-L1 deep chunks.

    Six deep sensors over 8 days at 1 min, written as four 2-day chunks under
    grid/l1/ named like grid_l1 output (testproj_a_deep_L1_<stamp>.nc). One
    shallow sensor exists but its segment is not a drift segment. Returns the
    config path.
    """
    deep_sns = [301111, 301222, 301333, 301444, 301555, 301666]
    shallow_sns = [302777]
    all_sns = deep_sns + shallow_sns

    data = tmp_path / "data"
    gridl1 = data / "grid" / "l1"
    gridl1.mkdir(parents=True)
    (data / "aux").mkdir(parents=True)

    # sheets — raw column names; sensor_sheet_load renames + dt64-casts the
    # time/CTD-cal columns, so they must be present for Mooring(...) to build.
    pd.DataFrame([
        {
            "SN": sn,
            "Type": "RBR Solo",
            "Pre-Deployment CTD Calibration Time": "2024-11-20 12:00:00",
            "Pre-Deployment CTD Calibration Cast": 1,
            "Post-Deployment CTD Calibration Time": "2024-12-02 12:00:00",
            "Post-Deployment CTD Calibration Cast": 5,
            "Pre-Deployment Time Calibration": "2024-11-18 10:00:00",
            "Post-Deployment Time Calibration": "2024-12-05 10:00:00",
            "Post-Deployment UTC Time": "2024-12-05 10:00:00",
            "Post-Deployment Logger Time": "2024-12-05 10:00:05",
            "exclude": 0,
        }
        for sn in all_sns
    ]).to_csv(data / "sensor_sheet.csv", index=False)
    moor_rows = (
        [{"type": "RBR Solo", "SN": sn, "depth": 4300.0 - 4.0 * i, "segment": "deep"}
         for i, sn in enumerate(deep_sns)]
        + [{"type": "RBR Solo", "SN": sn, "depth": 2000.0, "segment": "shallow"}
           for sn in shallow_sns]
    )
    pd.DataFrame(moor_rows).to_csv(data / "mooring_sheet.csv", index=False)

    # synthetic gridded-L1 deep array: dims (depth, time), sn coord on depth
    start = np.datetime64("2024-11-22T00:00")
    end = np.datetime64("2024-11-30T00:00")
    times = np.arange(start, end, np.timedelta64(1, "m"))
    depths = [4300.0 - 4.0 * i for i in range(len(deep_sns))]
    elapsed_d = (times - start) / np.timedelta64(1, "D")
    # smooth background in depth + time + a tiny per-sensor linear drift
    rows = []
    for i, _sn in enumerate(deep_sns):
        base = 4.0 - 0.001 * i + 1e-3 * np.sin(elapsed_d / 1.5)
        drift = 1e-4 * i * elapsed_d
        rows.append(base + drift)
    full = xr.DataArray(
        np.vstack(rows),
        dims=("depth", "time"),
        coords={"depth": depths, "time": times, "sn": ("depth", deep_sns)},
        name="t",
    )
    full.attrs.update({"units": "°C", "long_name": "temperature"})
    # real gridded-L1 deep arrays are depth-ascending; the background spline
    # fit (UnivariateSpline over depth) requires monotonic-increasing depth.
    full = full.sortby("depth")
    chunk = np.timedelta64(2, "D")
    for ti in np.arange(start, end, chunk, dtype="datetime64[s]"):
        stamp = np.datetime_as_string(np.datetime64(ti, "s")).replace("-", "").replace(":", "")
        # half-open [ti, ti+chunk) like real grid_l1 chunks; an inclusive slice
        # duplicates the boundary-midnight sample into two chunks and skews the
        # per-window mean off a whole second (un-serialisable by the nc3 writer).
        sub = full.sel(time=slice(ti, ti + chunk - np.timedelta64(1, "ns")))
        sub.to_netcdf(gridl1 / f"testproj_a_deep_L1_{stamp}.nc")

    cfg = {
        "info": "drift test",
        "meta": {"mooring_name": "A", "project": "TestProj", "PI": "x", "email": "x"},
        "path": {
            "fig": "fig/",
            "data": {"raw": {"rbr": "data/raw/", "sbe": "data/raw/"},
                     "proc": "data/proc/", "grid": "data/grid/"},
            "sensors": "data/sensor_sheet.csv",
            "mooring": "data/mooring_sheet.csv",
            "aux": "data/aux/",
        },
        "start_time": "2024-11-22 00:00:00",
        "end_time": "2024-11-30 00:00:00",
        "ignore_sns": [],
        "gridding": {"dt": "10s", "max_gap": "30s", "chunk": "2D"},
        "segments": {
            "deep": {"select": {"segment": "deep"}, "drift": True},
            "shallow": {"select": {"segment": "shallow"}},
        },
        "drift_parameters": {
            "label": "testfit",
            "exclude": 1.0e-3,
            "polydeg": 2,
            "outliers_polydeg": 2,
            "use_spline": False,
            "fit_mode": "linear",
            "iterate_subtract": False,
        },
    }
    cfgdir = tmp_path / "run"
    cfgdir.mkdir()
    cfgpath = cfgdir / "config.yml"
    with open(cfgpath, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfgpath


@pytest.fixture
def l2_mooring(tmp_path):
    """Self-contained mooring for make_l2 + grid_l2.

    Three deep sensors over 8 days at 1 min (per-sensor L1) + one shallow
    (non-drift) sensor; a drift product drift_testproj_a_testfit.nc in aux
    with dims (depth, window), sn on depth, time on window. Drift slope is
    1e-4 * i per day so deep sensor i=0 has zero drift. Returns the config
    path.
    """
    deep_sns = [301111, 301222, 301333]
    shallow_sns = [302444]
    all_sns = deep_sns + shallow_sns

    data = tmp_path / "data"
    procl1 = data / "proc" / "l1"
    procl1.mkdir(parents=True)
    (data / "proc" / "l2").mkdir(parents=True)
    (data / "aux").mkdir(parents=True)

    # sheets — include all columns required by sensor_sheet_columns_to_dt64
    pd.DataFrame([
        {
            "SN": sn,
            "Type": "RBR Solo",
            "Pre-Deployment CTD Calibration Time": "2024-11-20 12:00:00",
            "Pre-Deployment CTD Calibration Cast": 1,
            "Post-Deployment CTD Calibration Time": "2024-12-02 12:00:00",
            "Post-Deployment CTD Calibration Cast": 5,
            "Pre-Deployment Time Calibration": "2024-11-18 10:00:00",
            "Post-Deployment Time Calibration": "2024-12-05 10:00:00",
            "Post-Deployment UTC Time": "2024-12-05 10:00:00",
            "Post-Deployment Logger Time": "2024-12-05 10:00:05",
            "exclude": 0,
        }
        for sn in all_sns
    ]).to_csv(data / "sensor_sheet.csv", index=False)
    depths = [4300.0 - 4.0 * i for i in range(len(deep_sns))]
    moor_rows = (
        [{"type": "RBR Solo", "SN": sn, "height": float(i + 1), "depth": depths[i],
          "segment": "deep"} for i, sn in enumerate(deep_sns)]
        + [{"type": "RBR Solo", "SN": sn, "height": 50.0, "depth": 2000.0,
            "segment": "shallow"} for sn in shallow_sns]
    )
    pd.DataFrame(moor_rows).to_csv(data / "mooring_sheet.csv", index=False)

    # per-sensor L1 for the deep sensors
    start = np.datetime64("2024-11-22T00:00")
    end = np.datetime64("2024-11-30T00:00")
    times = np.arange(start, end, np.timedelta64(1, "m"))
    for i, sn in enumerate(deep_sns):
        da = xr.DataArray(
            4.0 - 0.001 * i + 1e-3 * np.sin(np.arange(times.size) / 500.0),
            dims="time", coords={"time": times}, name="t",
        )
        da.attrs.update({"sampling period in s": 60.0, "units": "degree_C",
                         "long_name": "temperature", "SN": sn})
        da.to_netcdf(procl1 / f"testproj_a__rbr__{sn:06d}_L1.nc")

    # drift product: dims (depth, window); sn on depth, time on window
    windows = np.arange(start, end, np.timedelta64(1, "D")) + np.timedelta64(12, "h")
    elapsed_d = (windows - start) / np.timedelta64(1, "D")
    drift_vals = np.outer(1e-4 * np.arange(len(deep_sns)), elapsed_d)  # (depth, window)
    drift = xr.DataArray(
        drift_vals,
        dims=("depth", "window"),
        coords={
            "depth": depths,
            "sn": ("depth", deep_sns),
            "window": np.arange(windows.size),
            "time": ("window", windows),
        },
        name="drift",
    )
    drift.to_netcdf(data / "aux" / "drift_testproj_a_testfit.nc")

    cfg = {
        "info": "l2 test",
        "meta": {"mooring_name": "A", "project": "TestProj", "PI": "x", "email": "x"},
        "path": {
            "fig": "fig/",
            "data": {"raw": {"rbr": "data/raw/", "sbe": "data/raw/"},
                     "proc": "data/proc/", "grid": "data/grid/"},
            "sensors": "data/sensor_sheet.csv",
            "mooring": "data/mooring_sheet.csv",
            "aux": "data/aux/",
        },
        "start_time": "2024-11-22 00:00:00",
        "end_time": "2024-11-30 00:00:00",
        "ignore_sns": [],
        # native L1 sampling here is 60s; max_gap must be >= it, else every
        # interval is treated as a gap (grid_thermistors._insert_gap_nans).
        "gridding": {"dt": "10s", "max_gap": "120s", "chunk": "2D"},
        "segments": {
            "deep": {"select": {"segment": "deep"}, "drift": True},
            "shallow": {"select": {"segment": "shallow"}},
        },
        "drift_parameters": {"label": "testfit"},
    }
    cfgdir = tmp_path / "run"
    cfgdir.mkdir()
    cfgpath = cfgdir / "config.yml"
    with open(cfgpath, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfgpath


@pytest.fixture
def ctd_cal_mooring(tmp_path):
    """Self-contained project for compute_ctd_offsets.

    Pre casts 1 (2 sensors) + 4 (1 sensor) -> cruise1 offsets; post cast 2
    (2 sensors) -> cruise2 offsets. Pre pool files are pre-sliced to the
    cast window; post pool files span the full post-cal period (sliced to
    the cast period by the stage). CTD casts each have one ~6-min stop.
    Returns the config path.
    """
    data = tmp_path / "data"
    pre_pool = data / "aux" / "pre_cal_cast_data"
    post_pool = data / "aux" / "post_cal" / "l0"
    ctd_dir = data / "cruises"
    pre_pool.mkdir(parents=True)
    post_pool.mkdir(parents=True)
    (ctd_dir / "cruise1").mkdir(parents=True)
    (ctd_dir / "cruise2").mkdir(parents=True)
    (data / "aux" / "cal_results").mkdir(parents=True)

    # --- sensor sheet (raw column names; cast assignments drive selection) ---
    pre1_sns = [301111, 301222]      # cast 1
    pre4_sns = [301333]              # cast 4
    post2_sns = [301111, 301222]     # cast 2 (same sensors, post side)
    rows = []
    for sn in [301111, 301222, 301333]:
        pre_cast = 1 if sn in pre1_sns else 4
        rows.append({
            "SN": sn, "Type": "RBR Solo", "exclude": 0,
            "Pre-Deployment CTD Calibration Time": "2024-11-12 12:00:00",
            "Pre-Deployment CTD Calibration Cast": pre_cast,
            "Post-Deployment CTD Calibration Time": "2025-12-08 12:00:00",
            "Post-Deployment CTD Calibration Cast": 2 if sn in post2_sns else "",
            "Pre-Deployment Time Calibration": "2024-11-10 10:00:00",
            "Post-Deployment Time Calibration": "2025-12-15 10:00:00",
            "Post-Deployment UTC Time": "2025-12-15 10:00:00",
            "Post-Deployment Logger Time": "2025-12-15 10:00:05",
        })
    pd.DataFrame(rows).to_csv(data / "sensor_sheet.csv", index=False)
    pd.DataFrame(
        [{"type": "RBR Solo", "SN": sn, "depth": 4300.0 - 4.0 * i, "segment": "deep"}
         for i, sn in enumerate([301111, 301222, 301333])]
    ).to_csv(data / "mooring_sheet.csv", index=False)

    # --- synthetic CTD casts: descend / ~6-min stop / ascend at 1 Hz ---
    def make_ctd(day, stop_p, t1_val, t2_val):
        times = np.arange(
            np.datetime64(f"{day}T00:00:00"),
            np.datetime64(f"{day}T00:20:00"),
            np.timedelta64(1, "s"),
        )
        n = times.size
        p = np.concatenate([
            np.linspace(0.0, stop_p, n // 3),
            np.full(n - 2 * (n // 3), stop_p),
            np.linspace(stop_p, 0.0, n // 3),
        ])
        at_stop = np.isclose(p, stop_p)
        t1 = np.where(at_stop, t1_val, 5.0)
        t2 = np.where(at_stop, t2_val, 5.0)
        return xr.Dataset(
            {"p": ("time", p), "t1": ("time", t1), "t2": ("time", t2)},
            coords={"time": times},
        ), times, at_stop

    ctd_specs = {
        ("cruise1", 1): ("2024-11-08", 1000.0, 4.00, 4.01),
        ("cruise1", 4): ("2024-11-12", 4040.0, 2.00, 2.01),
        ("cruise2", 2): ("2025-12-08", 4040.0, 2.00, 2.02),
    }
    stop_windows = {}
    for (cruise, cast), (day, stop_p, t1v, t2v) in ctd_specs.items():
        ds, times, at_stop = make_ctd(day, stop_p, t1v, t2v)
        cdir = ctd_dir / cruise
        ds.to_netcdf(cdir / f"{cruise}_cast_{cast:03d}.nc")
        sidx = np.flatnonzero(at_stop)
        stop_windows[(cruise, cast)] = (times[sidx[0] + 30], times[sidx[-1] - 30])

    # --- cal-cast pools: each assigned sensor records over its cast window ---
    def write_sensor(pool, fname, day, sn, offset):
        times = np.arange(
            np.datetime64(f"{day}T00:00:00"),
            np.datetime64(f"{day}T00:20:00"),
            np.timedelta64(2, "s"),
        )
        # thermistor = (ctd at stop) - offset, so kernel recovers `offset`
        da = xr.DataArray(5.0 - offset + 0.0 * np.arange(times.size),
                          dims="time", coords={"time": times}, name="t")
        da.attrs["SN"] = sn
        da.to_netcdf(pool / fname)

    offsets_truth = {301111: 0.0030, 301222: 0.0042, 301333: 0.0051}
    for sn in pre1_sns:
        write_sensor(pre_pool, f"motive_a__rbr__{sn:06d}_pre_ctd_cal.nc",
                     "2024-11-08", sn, offsets_truth[sn])
    for sn in pre4_sns:
        write_sensor(pre_pool, f"motive_a__rbr__{sn:06d}_pre_ctd_cal.nc",
                     "2024-11-12", sn, offsets_truth[sn])
    for sn in post2_sns:
        write_sensor(post_pool, f"motive_post_cal__rbr__{sn:06d}_L0.nc",
                     "2025-12-08", sn, offsets_truth[sn] + 1e-4)

    # --- cal_stops.csv (one row per selected stop per cast) ---
    def iso(t):
        return pd.Timestamp(t).isoformat()
    stop_rows = []
    for (cruise, cast), src in [
        (("cruise1", 1), "pre"), (("cruise1", 4), "pre"), (("cruise2", 2), "post")
    ]:
        s, e = stop_windows[(cruise, cast)]
        stop_rows.append({
            "source": src, "cast": cast,
            "stop_start": iso(s), "stop_end": iso(e),
            "ctd_file": f"{cruise}/{cruise}_cast_{cast:03d}.nc",
            "ref_temp": "",
        })
    pd.DataFrame(stop_rows).to_csv(data / "cal_stops.csv", index=False)

    cfg = {
        "info": "ctd cal test",
        "meta": {"mooring_name": "A", "project": "MOTIVE", "PI": "x", "email": "x"},
        "path": {
            "fig": "fig/",
            "data": {"raw": {"rbr": "data/raw/", "sbe": "data/raw/"},
                     "proc": "data/proc/", "grid": "data/grid/"},
            "sensors": "data/sensor_sheet.csv",
            "mooring": "data/mooring_sheet.csv",
            "aux": "data/aux/",
            "ctd": "data/cruises/",
            "cal_stops": "data/cal_stops.csv",
        },
        "start_time": "2024-11-22 00:00:00",
        "end_time": "2025-12-10 00:00:00",
        "ignore_sns": [],
        "gridding": {"dt": "10s", "max_gap": "30s", "chunk": "2D"},
        "calibration": {
            "method": "linear_interp",
            "cal_casts_pre": "data/aux/pre_cal_cast_data/",
            "cal_casts_post": "data/aux/post_cal/l0/",
            "offsets_pre": "data/aux/cal_results/motive_cruise1_cal_offsets.nc",
            "offsets_post": "data/aux/cal_results/motive_cruise2_cal_offsets.nc",
        },
        "segments": {"deep": {"select": {"segment": "deep"}}},
    }
    cfgdir = tmp_path / "run"
    cfgdir.mkdir()
    cfgpath = cfgdir / "config.yml"
    with open(cfgpath, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfgpath
