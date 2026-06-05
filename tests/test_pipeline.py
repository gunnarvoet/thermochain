# tests/test_pipeline.py
import numpy as np
import pytest

from thermodrift.pipeline import parse_gridding


def test_parse_gridding_converts_strings_to_timedelta64():
    out = parse_gridding({"dt": "2s", "max_gap": "10s", "chunk": "2D"})
    assert out["dt"] == np.timedelta64(2, "s")
    assert out["max_gap"] == np.timedelta64(10, "s")
    assert out["chunk"] == np.timedelta64(2, "D")


def test_parse_gridding_merges_defaults():
    out = parse_gridding({"dt": "2s"}, defaults={"dt": "10s", "max_gap": "30s", "chunk": "2D"})
    assert out["dt"] == np.timedelta64(2, "s")        # override wins
    assert out["max_gap"] == np.timedelta64(30, "s")  # from defaults


def test_parse_gridding_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown gridding keys"):
        parse_gridding({"dt": "2s", "max_gpa": "10s"})


import pandas as pd  # noqa: E402

from thermodrift.pipeline import resolve_segment_sns  # noqa: E402


def _mooring_df():
    return pd.DataFrame(
        {"type": ["rbr"] * 4, "depth": [100, 200, 300, 400], "segment": ["deep", "deep", "shallow", "shallow"]},
        index=pd.Index([111, 222, 333, 444], name="SN"),
    )


def test_resolve_segment_by_column():
    sns = resolve_segment_sns(_mooring_df(), {"segment": "deep"}, root=".")
    assert sorted(sns) == [111, 222]


def test_resolve_segment_by_explicit_sns():
    sns = resolve_segment_sns(_mooring_df(), {"sns": [333, 444]}, root=".")
    assert sorted(sns) == [333, 444]


def test_resolve_segment_missing_column_raises():
    df = _mooring_df().drop(columns="segment")
    with pytest.raises(ValueError, match="requires a 'segment' column"):
        resolve_segment_sns(df, {"segment": "deep"}, root=".")


def test_resolve_segment_unknown_spec_raises():
    with pytest.raises(ValueError, match="one of segment/sns/layout"):
        resolve_segment_sns(_mooring_df(), {"foo": "bar"}, root=".")


import thermodrift  # noqa: E402
from thermodrift.pipeline import Mooring  # noqa: E402


def test_mooring_exported_at_package_root():
    assert thermodrift.Mooring is Mooring


def test_mooring_parses_segments_and_gridding(segmented_mooring):
    m = Mooring(segmented_mooring)
    assert set(m.segments_cfg) == {"deep", "shallow"}
    assert m.gridding["deep"]["dt"] == np.timedelta64(10, "s")
    assert m.gridding["shallow"]["chunk"] == np.timedelta64(2, "D")


def test_mooring_segment_sensors_returns_three_deep(segmented_mooring):
    m = Mooring(segmented_mooring)
    assert len(m.segment_sensors("deep")) == 3


from pathlib import Path  # noqa: E402 (already imported at top of pipeline.py, idempotent here)

import xarray as xr  # noqa: E402


def test_grid_l1_writes_chunks_and_is_idempotent(segmented_mooring):
    m = Mooring(segmented_mooring)
    grid_dir = Path(m.cfg.path.root) / "data" / "grid"

    summary = m.grid_l1(segments=["deep"])
    assert summary["deep"]["written"] == 2          # 4-day span / 2D chunk
    assert summary["deep"]["skipped"] == 0
    files = sorted(grid_dir.glob("testproj_mavs3_deep_L1_*.nc"))
    assert len(files) == 2

    da = xr.open_dataarray(files[0])
    assert da.sizes["depth"] == 3
    assert set(int(s) for s in da.sn.values) == set(int(s) for s in m.segment_sensors("deep").index)

    summary2 = m.grid_l1(segments=["deep"])
    assert summary2["deep"]["written"] == 0
    assert summary2["deep"]["skipped"] == 2


def test_grid_l1_overwrite_forces_rewrite(segmented_mooring):
    m = Mooring(segmented_mooring)
    m.grid_l1(segments=["deep"])
    summary = m.grid_l1(segments=["deep"], overwrite=True)
    assert summary["deep"]["written"] == 2


def test_status_summary_reports_grid_progress(segmented_mooring):
    m = Mooring(segmented_mooring)
    before = m.status_summary()
    assert before.loc["deep", "n"] == 3
    assert before.loc["deep", "gridL1"] == "0/2"

    m.grid_l1(segments=["deep"])
    after = m.status_summary()
    assert after.loc["deep", "gridL1"] == "2/2"


def test_ignore_sns_includes_sheet_exclude(segmented_mooring_excluded):
    cfgpath, excluded_sn = segmented_mooring_excluded
    m = Mooring(cfgpath)
    assert excluded_sn in m._ignore_sns()


def test_grid_l1_drops_sheet_excluded_sensor(segmented_mooring_excluded):
    cfgpath, excluded_sn = segmented_mooring_excluded
    m = Mooring(cfgpath)
    m.grid_l1(segments=["deep"])
    grid_dir = Path(m.cfg.path.root) / "data" / "grid"
    da = xr.open_dataarray(sorted(grid_dir.glob("testproj_mavs3_deep_L1_*.nc"))[0])
    assert excluded_sn not in set(int(s) for s in da.sn.values)
    assert da.sizes["depth"] == 2   # 3 deep minus 1 excluded


def test_ignore_sns_handles_string_exclude(segmented_mooring):
    m = Mooring(segmented_mooring)
    sn = int(m.segment_sensors("deep").index[0])
    m.sensor_info = m.sensor_info.copy()
    m.sensor_info["exclude"] = "0"
    m.sensor_info.loc[sn, "exclude"] = "1"
    assert sn in m._ignore_sns()


def test_parse_gridding_raises_on_missing_required_keys():
    with pytest.raises(ValueError, match="missing required keys"):
        parse_gridding({"dt": "2s"})


from thermodrift.pipeline import load_cal_offsets  # noqa: E402


def _write_offsets(path, sns, offsets, casts, cruise_in_name="cruise1"):
    ds = xr.Dataset(
        {"offset": ("sn", offsets)},
        coords={"sn": list(sns), "cast": ("sn", list(casts))},
    )
    fpath = path / f"motive_{cruise_in_name}_cal_offsets.nc"
    ds.to_netcdf(fpath)
    return fpath


def test_load_cal_offsets_indexed_sorted_with_provenance(tmp_path):
    fpath = _write_offsets(tmp_path, sns=[333, 111, 222], offsets=[0.3, 0.1, 0.2], casts=[2, 1, 1])
    a = load_cal_offsets(fpath)
    assert list(a.sn.values) == [111, 222, 333]          # sorted by sn
    assert float(a.sel(sn=222).data) == 0.2
    assert str(a.sel(sn=111).source_cruise.item()) == "cruise1"
    assert int(a.sel(sn=333).source_cast.item()) == 2


def test_load_cal_offsets_missing_file_returns_none(tmp_path):
    assert load_cal_offsets(tmp_path / "does_not_exist.nc") is None


from thermodrift.pipeline import cal_diagnostic_attrs  # noqa: E402


def _offsets_da(sn=111, offset=0.15, cruise="cruise1", cast=1):
    return xr.DataArray(
        [offset],
        dims="sn",
        coords={
            "sn": [sn],
            "source_cruise": ("sn", [cruise]),
            "source_cast": ("sn", [cast]),
        },
    )


def test_cal_diagnostic_attrs_present_offset():
    pre = _offsets_da(sn=111, offset=0.15, cruise="cruise1", cast=1)
    attrs = cal_diagnostic_attrs(
        111, pre, None, cal_method="scalar_pre_only",
        pre_applied=True, post_applied=False,
        t_pre=np.datetime64("2024-11-22T05:00"), t_post=None,
    )
    assert attrs["cal_method"] == "scalar_pre_only"
    assert attrs["pre_cal_offset"] == 0.15
    assert attrs["pre_cal_applied"] == 1
    assert attrs["pre_cal_cruise"] == "cruise1"
    assert attrs["pre_cal_cast"] == 1
    assert attrs["pre_cal_time"].startswith("2024-11-22")
    # missing post side -> NaN / sentinel
    assert np.isnan(attrs["post_cal_offset"])
    assert attrs["post_cal_applied"] == 0
    assert attrs["post_cal_cast"] == -1
    assert attrs["post_cal_time"] == ""


def test_cal_diagnostic_attrs_sn_absent_from_offsets():
    pre = _offsets_da(sn=999)
    attrs = cal_diagnostic_attrs(
        111, pre, None, cal_method="none",
        pre_applied=False, post_applied=False,
    )
    assert np.isnan(attrs["pre_cal_offset"])
    assert attrs["pre_cal_applied"] == 0


def _l1_for(l1dir, sn):
    files = list(l1dir.glob(f"*{sn:06d}*_L1.nc"))
    assert len(files) == 1, f"expected 1 L1 for {sn}, got {files}"
    return xr.open_dataarray(files[0])


def test_cut_and_cal_deep_scalar_pre_only(cal_mooring):
    m = Mooring(cal_mooring)
    l1dir = Path(m.cfg.path.data.procl1)
    summary = m.cut_and_cal(segments=["deep"])
    assert summary["deep"]["written"] == 2
    deep_sn = sorted(m.segment_sensors("deep").index)[0]
    da = _l1_for(l1dir, deep_sn)
    assert da.time.values[0] >= np.datetime64("2024-11-22T00:00")
    assert da.time.values[-1] <= np.datetime64("2024-11-30T00:00")
    assert da.attrs["cal_method"] == "scalar_pre_only"
    assert da.attrs["pre_cal_applied"] == 1
    assert da.attrs["post_cal_applied"] == 0


def test_cut_and_cal_shallow_linear_interp(cal_mooring):
    m = Mooring(cal_mooring)
    l1dir = Path(m.cfg.path.data.procl1)
    m.cut_and_cal(segments=["shallow"])
    sn = sorted(m.segment_sensors("shallow").index)[0]
    da = _l1_for(l1dir, sn)
    assert da.attrs["cal_method"] == "linear_interp"
    assert da.attrs["pre_cal_applied"] == 1
    assert da.attrs["post_cal_applied"] == 1


def test_cut_and_cal_idempotent_then_overwrite(cal_mooring):
    m = Mooring(cal_mooring)
    first = m.cut_and_cal(segments=["deep"])
    assert first["deep"]["written"] == 2 and first["deep"]["skipped"] == 0
    second = m.cut_and_cal(segments=["deep"])
    assert second["deep"]["written"] == 0 and second["deep"]["skipped"] == 2
    third = m.cut_and_cal(segments=["deep"], overwrite=True)
    assert third["deep"]["written"] == 2


def test_cut_and_cal_respects_ignore_sns(cal_mooring):
    m = Mooring(cal_mooring)
    ignore_sn = int(sorted(m.segment_sensors("deep").index)[0])
    m.cfg.ignore_sns = [ignore_sn]
    summary = m.cut_and_cal(segments=["deep"])
    assert summary["deep"]["written"] == 1
    l1dir = Path(m.cfg.path.data.procl1)
    assert not list(l1dir.glob(f"*{ignore_sn:06d}*_L1.nc"))
