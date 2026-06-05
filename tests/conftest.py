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
