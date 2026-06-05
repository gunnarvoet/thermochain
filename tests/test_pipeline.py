# tests/test_pipeline.py
import shutil

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


def test_status_per_sensor_reports_l0_l1_cal_method(cal_mooring):
    m = Mooring(cal_mooring)
    before = m.status()
    deep_sn = int(sorted(m.segment_sensors("deep").index)[0])
    assert bool(before.loc[deep_sn, "l0"]) is True
    assert bool(before.loc[deep_sn, "l1"]) is False
    assert before.loc[deep_sn, "segment"] == "deep"

    m.cut_and_cal(segments=["deep"])
    after = m.status()
    assert bool(after.loc[deep_sn, "l1"]) is True
    assert after.loc[deep_sn, "cal_method"] == "scalar_pre_only"


def test_status_summary_reports_l1_count(cal_mooring):
    m = Mooring(cal_mooring)
    assert m.status_summary().loc["deep", "l1"] == "0/2"
    m.cut_and_cal(segments=["deep"])
    assert m.status_summary().loc["deep", "l1"] == "2/2"


from thermodrift.pipeline import correct_drift, DriftParameters  # noqa: E402


def test_drift_parameters_defaults_mirror_sensor_drift():
    dp = DriftParameters()
    assert dp.fit_mode == "auto"
    assert dp.iterate_mode == "restore"
    assert dp.polydeg == 8
    assert dp.use_spline is False
    assert dp.tau0 == 20.0
    assert dp.manual_outlier_sns == []


def test_drift_parameters_from_dict_merges_overrides():
    dp = DriftParameters.from_dict({"fit_mode": "linear", "spline_smooth": 5e-6})
    assert dp.fit_mode == "linear"
    assert dp.spline_smooth == 5e-6
    # untouched key keeps its default
    assert dp.outliers_polydeg == 8


def test_drift_parameters_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown drift_parameters keys"):
        DriftParameters.from_dict({"spline_smoothh": 5e-6})


def test_drift_parameters_rejects_bad_fit_mode():
    with pytest.raises(ValueError, match="fit_mode must be"):
        DriftParameters.from_dict({"fit_mode": "quadratic"})


def test_drift_parameters_rejects_bad_iterate_mode():
    with pytest.raises(ValueError, match="iterate_mode must be"):
        DriftParameters.from_dict({"iterate_mode": "blend"})


def test_drift_parameters_as_dict_roundtrips_for_sensor_drift():
    dp = DriftParameters.from_dict({"fit_mode": "linear", "manual_outlier_sns": [236109]})
    d = dp.as_dict()
    assert d["fit_mode"] == "linear"
    assert d["manual_outlier_sns"] == [236109]
    assert set(d) == {
        "exclude", "polydeg", "outliers_polydeg", "use_spline", "spline_smooth",
        "exclude_sn", "tau0", "tau_bounds", "beta_bounds", "fit_mode",
        "iterate_subtract", "iterate_mode", "amplitude_threshold_mK", "manual_outlier_sns",
    }


from thermodrift.pipeline import drift_provenance_attrs  # noqa: E402


def test_drift_provenance_attrs_are_netcdf_safe():
    dp = DriftParameters.from_dict(
        {"fit_mode": "auto", "use_spline": True, "spline_smooth": 5e-6,
         "tau_bounds": (5.0, 180.0), "manual_outlier_sns": [236109], "exclude": 1e-3}
    )
    attrs = drift_provenance_attrs(dp, label="spline_slowtau")
    assert attrs["drift_label"] == "spline_slowtau"
    assert attrs["drift_param_fit_mode"] == "auto"
    # bool -> int (NetCDF has no bool scalar attr)
    assert attrs["drift_param_use_spline"] == 1
    assert isinstance(attrs["drift_param_use_spline"], int)
    assert attrs["drift_param_spline_smooth"] == 5e-6
    # 2-tuple bounds -> _lo / _hi floats
    assert attrs["drift_param_tau_bounds_lo"] == 5.0
    assert attrs["drift_param_tau_bounds_hi"] == 180.0
    # list -> int64 array
    np.testing.assert_array_equal(attrs["drift_param_manual_outlier_sns"], np.array([236109]))
    # None -> empty-string sentinel
    assert attrs["drift_param_exclude_sn"] == ""
    # scalar exclude stays a float
    assert attrs["drift_param_exclude"] == 1e-3
    # no value is a tuple / bool / None (round-trips through to_netcdf)
    for v in attrs.values():
        assert not isinstance(v, (tuple, bool, type(None)))


import types  # noqa: E402

from thermodrift.pipeline import drift_diag_bundle  # noqa: E402


def _fake_sd(with_exp, with_pass1=True):
    """Minimal stand-in for a fitted sensor_drift (only the attrs the bundle reads)."""
    depth = [4300.0, 4296.0, 4292.0]
    window = [0, 1, 2]
    arr = lambda v: xr.DataArray(  # noqa: E731
        np.full((len(depth), len(window)), v),
        dims=("depth", "window"),
        coords={"depth": depth, "window": window},
    )
    sd = types.SimpleNamespace(
        offsets_clean=arr(0.0),
        drift_linfit=arr(1.0),
        drift_fit=arr(2.0),
        fit_type=xr.DataArray(["linear"] * len(depth), dims="depth", coords={"depth": depth}),
        fit_mode="auto" if with_exp else "linear",
        tau_bounds=(5.0, 180.0),
        iteration_count=1,
        flagged_outlier_sns=[236109, 236127],
    )
    if with_pass1:
        sd.drift_fit_pass1 = arr(3.0)
        sd.offsets_clean_pass1 = arr(4.0)
    if with_exp:
        sd.drift_expfit = arr(5.0)
        sd.drift_exp_params = xr.DataArray(
            np.zeros((len(depth), 2)), dims=("depth", "param"),
            coords={"depth": depth, "param": ["tau", "beta"]},
        )
    return sd


def test_drift_diag_bundle_linear_has_no_exp_vars():
    dp = DriftParameters.from_dict({"fit_mode": "linear"})
    bundle = drift_diag_bundle(_fake_sd(with_exp=False), dp, label="spline_lin")
    assert set(bundle.data_vars) >= {
        "offsets_clean", "drift_linfit", "drift_fit", "fit_type",
        "drift_fit_pass1", "offsets_clean_pass1",
    }
    assert "drift_expfit" not in bundle.data_vars
    assert bundle.attrs["fit_mode"] == "linear"
    assert bundle.attrs["drift_label"] == "spline_lin"
    assert bundle.attrs["drift_param_fit_mode"] == "linear"
    # iterate_subtract=False (no pass1 attrs) -> *_pass1 guarded out
    no_iter = drift_diag_bundle(_fake_sd(with_exp=False, with_pass1=False), dp, label="spline_lin")
    assert "drift_fit_pass1" not in no_iter.data_vars
    assert "offsets_clean_pass1" not in no_iter.data_vars


def test_drift_diag_bundle_auto_includes_exp_vars():
    dp = DriftParameters.from_dict({"fit_mode": "auto", "tau0": 20.0, "tau_bounds": (5.0, 180.0)})
    bundle = drift_diag_bundle(_fake_sd(with_exp=True), dp, label="spline_slowtau")
    assert "drift_expfit" in bundle.data_vars
    assert "drift_exp_params" in bundle.data_vars
    assert bundle.attrs["tau_bounds_lo"] == 5.0
    assert bundle.attrs["tau_bounds_hi"] == 180.0
    np.testing.assert_array_equal(bundle.attrs["flagged_outlier_sns"], np.array([236109, 236127]))


def test_fit_drift_writes_products_and_returns_sensor_drift(drift_mooring):
    m = Mooring(drift_mooring)
    aux = Path(m.cfg.path.root) / "data" / "aux"

    out = m.fit_drift()                       # config default: label=testfit, deep only
    assert set(out) == {"deep"}
    sd = out["deep"]
    assert isinstance(sd, thermodrift.io.sensor_drift)   # returned for diagnostics

    drift_f = aux / "drift_testproj_a_testfit.nc"
    diag_f = aux / "diag_testproj_a_testfit.nc"
    assert drift_f.exists() and diag_f.exists()

    da = xr.open_dataarray(drift_f)
    assert da.attrs["drift_label"] == "testfit"
    assert da.attrs["drift_param_fit_mode"] == "linear"
    assert "window" in da.dims and "depth" in da.dims


def test_fit_drift_skips_non_drift_segments(drift_mooring):
    m = Mooring(drift_mooring)
    # shallow has no drift: true -> requesting it is an error
    with pytest.raises(ValueError, match="not a drift segment"):
        m.fit_drift(segments=["shallow"])


def test_fit_drift_label_and_param_override(drift_mooring):
    m = Mooring(drift_mooring)
    aux = Path(m.cfg.path.root) / "data" / "aux"
    m.fit_drift(drift_parameters={"fit_mode": "linear"}, label="alt")
    assert (aux / "drift_testproj_a_alt.nc").exists()
    assert (aux / "diag_testproj_a_alt.nc").exists()


def test_fit_drift_rejects_unknown_param_key(drift_mooring):
    m = Mooring(drift_mooring)
    with pytest.raises(ValueError, match="unknown drift_parameters keys"):
        m.fit_drift(drift_parameters={"spline_smoothh": 1e-6}, label="bad")


def test_fit_drift_idempotent_then_overwrite(drift_mooring):
    m = Mooring(drift_mooring)
    first = m.fit_drift()
    assert first["deep"] is not None
    second = m.fit_drift()                    # products exist -> skip, sd is None
    assert second["deep"] is None
    third = m.fit_drift(overwrite=True)       # forced refit -> sd returned
    assert third["deep"] is not None


def test_status_summary_drift_column(drift_mooring):
    m = Mooring(drift_mooring)
    before = m.status_summary()
    assert before.loc["deep", "drift"] == "-"        # nothing fit yet
    assert before.loc["shallow", "drift"] == "-"     # non-drift segment

    m.fit_drift()                                     # writes label "testfit"
    after = m.status_summary()
    assert "testfit" in after.loc["deep", "drift"]
    assert after.loc["shallow", "drift"] == "-"


def test_fit_drift_label_arg_overrides_drift_parameters_label(drift_mooring):
    m = Mooring(drift_mooring)
    aux = Path(m.cfg.path.root) / "data" / "aux"
    # label= arg wins over a label inside drift_parameters (docstring contract),
    # and a stray "label" key in the override dict must not reach DriftParameters.
    m.fit_drift(drift_parameters={"fit_mode": "linear", "label": "ignored"}, label="winner")
    assert (aux / "drift_testproj_a_winner.nc").exists()
    assert not (aux / "drift_testproj_a_ignored.nc").exists()


def test_correct_drift_subtracts_interpolated_drift():
    times = np.arange(
        np.datetime64("2024-11-22T00:00"),
        np.datetime64("2024-11-24T00:00"),
        np.timedelta64(1, "h"),
    )
    sensor = xr.DataArray(np.full(times.size, 5.0), dims="time", coords={"time": times})
    # drift defined at two window centres (sn, time); linear 0 -> 1 over the 2 days
    wtimes = np.array(
        [np.datetime64("2024-11-22T00:00"), np.datetime64("2024-11-24T00:00")]
    )
    drift = xr.DataArray(
        [[0.0, 1.0]], dims=("sn", "time"), coords={"sn": [301111], "time": wtimes}
    )
    out = correct_drift(sensor, 301111, drift)
    # start: drift 0 -> unchanged
    assert float(out.isel(time=0)) == pytest.approx(5.0)
    # midpoint: drift 0.5 -> 5.0 - 0.5
    assert float(out.sel(time="2024-11-23T00:00")) == pytest.approx(4.5)


def test_correct_drift_extrapolates_before_first_window():
    # sensor samples that start BEFORE the first drift window centre
    times = np.array(
        [np.datetime64("2024-11-21T18:00"), np.datetime64("2024-11-23T00:00")]
    )
    sensor = xr.DataArray([5.0, 5.0], dims="time", coords={"time": times})
    wtimes = np.array(
        [np.datetime64("2024-11-22T00:00"), np.datetime64("2024-11-24T00:00")]
    )
    drift = xr.DataArray(
        [[0.0, 1.0]], dims=("sn", "time"), coords={"sn": [301111], "time": wtimes}
    )
    out = correct_drift(sensor, 301111, drift)
    # the early sample extrapolates to a slightly negative drift (no NaN)
    assert not np.isnan(float(out.isel(time=0)))


def _only(paths):
    paths = list(paths)
    assert len(paths) == 1, f"expected 1 file, got {paths}"
    return paths[0]


def test_make_l2_writes_per_sensor_l2_with_attrs(l2_mooring):
    m = Mooring(l2_mooring)
    procl2 = m._procl2_dir()
    procl1 = m._procl1_dir()

    summary = m.make_l2()                     # config default: label testfit, deep only
    assert set(summary) == {"deep"}
    assert summary["deep"]["written"] == 3

    deep_sns = sorted(int(s) for s in m.segment_sensors("deep").index)
    # zero-drift sensor (i=0): L2 == L1
    sn0 = deep_sns[0]
    l1 = xr.open_dataarray(_only(procl1.glob(f"*__{sn0:06d}_L1.nc")))
    l2 = xr.open_dataarray(_only(procl2.glob(f"*__{sn0:06d}_L2.nc")))
    # zero drift: L2 values equal L1; ignore the extra provenance coords
    # (sn/depth/window) that correct_drift carries through from the drift product
    xr.testing.assert_allclose(l1, l2.reset_coords(drop=True), atol=1e-12)
    assert l2.attrs["sn"] == sn0
    assert l2.attrs["SN"] == sn0             # L1 attrs carried over (copy)
    # nonzero-drift sensor: a correction was applied
    sn2 = deep_sns[2]
    l1b = xr.open_dataarray(_only(procl1.glob(f"*__{sn2:06d}_L1.nc")))
    l2b = xr.open_dataarray(_only(procl2.glob(f"*__{sn2:06d}_L2.nc")))
    assert float((l1b - l2b).max()) > 0.0


def test_make_l2_rejects_non_drift_segment(l2_mooring):
    m = Mooring(l2_mooring)
    with pytest.raises(ValueError, match="not a drift segment"):
        m.make_l2(segments=["shallow"])


def test_make_l2_idempotent_then_overwrite(l2_mooring):
    m = Mooring(l2_mooring)
    first = m.make_l2()
    assert first["deep"]["written"] == 3 and first["deep"]["skipped"] == 0
    second = m.make_l2()
    assert second["deep"]["written"] == 0 and second["deep"]["skipped"] == 3
    third = m.make_l2(overwrite=True)
    assert third["deep"]["written"] == 3


def test_make_l2_respects_ignore_sns(l2_mooring):
    m = Mooring(l2_mooring)
    drop = int(sorted(m.segment_sensors("deep").index)[0])
    m.cfg.ignore_sns = [drop]
    summary = m.make_l2()
    assert summary["deep"]["written"] == 2
    assert not list(m._procl2_dir().glob(f"*{drop:06d}*_L2.nc"))


def test_make_l2_drift_label_override(l2_mooring):
    m = Mooring(l2_mooring)
    aux = m._aux_dir()
    shutil.copy(aux / "drift_testproj_a_testfit.nc", aux / "drift_testproj_a_alt.nc")
    summary = m.make_l2(drift_label="alt")
    assert summary["deep"]["written"] == 3


def test_make_l2_missing_drift_product_raises(l2_mooring):
    m = Mooring(l2_mooring)
    with pytest.raises(FileNotFoundError, match="drift product not found"):
        m.make_l2(drift_label="does_not_exist")


def test_grid_l2_writes_chunks_for_drift_segment(l2_mooring):
    m = Mooring(l2_mooring)
    m.make_l2()                              # per-sensor L2 first
    summary = m.grid_l2()
    assert set(summary) == {"deep"}
    assert summary["deep"]["chunks"] == 4    # 8 days / 2D chunk
    assert summary["deep"]["written"] == 4

    gridl2 = m._gridl2_dir()
    assert gridl2 == m._grid_dir() / "l2"
    files = sorted(gridl2.glob("testproj_a_deep_L2_*.nc"))
    assert len(files) == 4
    da = xr.open_dataarray(files[0])
    assert "depth" in da.dims and "time" in da.dims
    assert "sn" in da.coords


def test_grid_l2_rejects_non_drift_segment(l2_mooring):
    m = Mooring(l2_mooring)
    with pytest.raises(ValueError, match="not a drift segment"):
        m.grid_l2(segments=["shallow"])


def test_grid_l2_idempotent_then_overwrite(l2_mooring):
    m = Mooring(l2_mooring)
    m.make_l2()
    first = m.grid_l2()
    assert first["deep"]["written"] == 4 and first["deep"]["skipped"] == 0
    second = m.grid_l2()
    assert second["deep"]["written"] == 0 and second["deep"]["skipped"] == 4
    third = m.grid_l2(overwrite=True)
    assert third["deep"]["written"] == 4


def test_status_includes_l2_column(l2_mooring):
    m = Mooring(l2_mooring)
    deep_sn = int(sorted(m.segment_sensors("deep").index)[0])
    before = m.status()
    assert bool(before.loc[deep_sn, "l2"]) is False
    m.make_l2()
    after = m.status()
    assert bool(after.loc[deep_sn, "l2"]) is True


def test_status_summary_l2_and_gridl2_columns(l2_mooring):
    m = Mooring(l2_mooring)
    before = m.status_summary()
    assert before.loc["deep", "l2"] == "0/3"
    assert before.loc["deep", "gridL2"] == "0/4"
    assert before.loc["shallow", "l2"] == "-"        # non-drift segment
    assert before.loc["shallow", "gridL2"] == "-"

    m.make_l2()
    m.grid_l2()
    after = m.status_summary()
    assert after.loc["deep", "l2"] == "3/3"
    assert after.loc["deep", "gridL2"] == "4/4"
