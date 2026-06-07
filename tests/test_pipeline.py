# tests/test_pipeline.py
import shutil

import numpy as np
import pytest

from thermochain.pipeline import parse_gridding


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

from thermochain.pipeline import resolve_segment_sns  # noqa: E402


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


import thermochain  # noqa: E402
from thermochain.pipeline import Mooring  # noqa: E402


def test_mooring_exported_at_package_root():
    assert thermochain.Mooring is Mooring


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
    grid_root = Path(m.cfg.path.root) / "data" / "grid"
    grid_dir = grid_root / "l1"

    summary = m.grid_l1(segments=["deep"])
    assert summary["deep"]["written"] == 2          # 4-day span / 2D chunk
    assert summary["deep"]["skipped"] == 0
    files = sorted(grid_dir.glob("testproj_mavs3_deep_L1_*.nc"))
    assert len(files) == 2
    # chunks land in grid/l1/ (where fit_drift reads), not bare grid/
    assert sorted(grid_root.glob("testproj_mavs3_deep_L1_*.nc")) == []

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
    grid_dir = Path(m.cfg.path.root) / "data" / "grid" / "l1"
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


from thermochain.pipeline import load_cal_offsets  # noqa: E402


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


from thermochain.pipeline import cal_diagnostic_attrs  # noqa: E402


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


from thermochain.pipeline import correct_drift, DriftParameters  # noqa: E402


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
        "two_step_shared", "max_triplet_gap_m",
    }


from thermochain.pipeline import drift_provenance_attrs  # noqa: E402


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

from thermochain.pipeline import drift_diag_bundle  # noqa: E402


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
    assert isinstance(sd, thermochain.io.sensor_drift)   # returned for diagnostics

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


from thermochain.pipeline import detect_cal_stops  # noqa: E402


def _synthetic_cast(stop_p=4000.0, stop_t=2.0):
    """A CTD cast: descend, hold a flat plateau ~6 min, ascend (1 Hz)."""
    times = np.arange(
        np.datetime64("2025-12-08T00:00:00"),
        np.datetime64("2025-12-08T00:20:00"),
        np.timedelta64(1, "s"),
    )
    n = times.size
    p = np.concatenate([
        np.linspace(0.0, stop_p, n // 3),                 # descend
        np.full(n - 2 * (n // 3), stop_p),                # plateau
        np.linspace(stop_p, 0.0, n // 3),                 # ascend
    ])
    t = np.where(np.isclose(p, stop_p), stop_t, 5.0)      # cold + steady at the stop
    return xr.Dataset(
        {"p": ("time", p), "t1": ("time", t), "t2": ("time", t)},
        coords={"time": times},
    )


def test_detect_cal_stops_finds_the_plateau():
    ctd = _synthetic_cast(stop_p=4000.0, stop_t=2.0)
    stops = detect_cal_stops(ctd, p_std_thresh=0.5, min_duration="60s")
    assert isinstance(stops, pd.DataFrame)
    assert len(stops) == 1
    row = stops.iloc[0]
    assert row["mean_p"] == pytest.approx(4000.0, abs=1.0)
    assert row["mean_t"] == pytest.approx(2.0, abs=1e-6)
    assert row["duration_s"] >= 60.0
    # window brackets the plateau
    assert np.datetime64(row["stop_start"]) >= np.datetime64("2025-12-08T00:06:00")
    assert np.datetime64(row["stop_end"]) <= np.datetime64("2025-12-08T00:14:00")


def test_detect_cal_stops_drops_short_plateaus():
    ctd = _synthetic_cast()
    # a 10-minute minimum is longer than the ~6-min plateau -> nothing kept
    stops = detect_cal_stops(ctd, min_duration="600s")
    assert len(stops) == 0


def test_detect_cal_stops_empty_when_never_stationary():
    times = np.arange(
        np.datetime64("2025-12-08T00:00:00"),
        np.datetime64("2025-12-08T00:05:00"),
        np.timedelta64(1, "s"),
    )
    p = np.linspace(0.0, 4000.0, times.size)              # monotonic, never flat
    ctd = xr.Dataset(
        {"p": ("time", p), "t1": ("time", p * 0), "t2": ("time", p * 0)},
        coords={"time": times},
    )
    assert len(detect_cal_stops(ctd)) == 0


from thermochain.io import rbr_ctd_cal_find_offset  # noqa: E402


def test_compute_ctd_offsets_writes_both_files(ctd_cal_mooring):
    m = Mooring(ctd_cal_mooring)
    summary = m.compute_ctd_offsets()
    assert set(summary) == {"pre", "post"}
    assert summary["pre"]["written"] == 3      # casts 1+4 -> 3 sensors
    assert summary["post"]["written"] == 2     # cast 2 -> 2 sensors

    pre = xr.open_dataset(m._offsets_out_path("pre"))
    assert list(pre.data_vars) == ["offset"]
    assert set(pre.coords) >= {"sn", "cast", "cal_temp"}
    assert "temp" in pre.coords
    np.testing.assert_array_equal(pre.temp.values, pre.cal_temp.values)
    assert list(pre.sn.values) == sorted(pre.sn.values.tolist())   # sorted by sn
    assert set(int(c) for c in pre.cast.values) == {1, 4}
    post = xr.open_dataset(m._offsets_out_path("post"))
    assert set(int(c) for c in post.cast.values) == {2}


def test_compute_ctd_offsets_matches_kernel(ctd_cal_mooring):
    """The stage's offset == a direct rbr_ctd_cal_find_offset on the same stack."""
    m = Mooring(ctd_cal_mooring)
    m.compute_ctd_offsets()
    pre = xr.open_dataarray(m._offsets_out_path("pre"))

    # rebuild cast-1 stack by hand and call the kernel directly
    stops = m._load_cal_stops()
    row = stops[(stops["source"] == "pre") & (stops["cast"] == 1)].iloc[0]
    ctd = xr.open_dataset(m._ctd_dir() / row["ctd_file"])
    pool = m._cal_cast_pool_dir("pre")
    sns = [301111, 301222]
    cals = [xr.open_dataarray(next(pool.glob(f"*{sn:06d}*.nc"))) for sn in sns]
    time = cals[0].time.copy()
    c = xr.concat([ci.interp_like(time) for ci in cals], dim="n")
    c["sn"] = (("n"), sns)
    ts = slice(np.datetime64(row["stop_start"]), np.datetime64(row["stop_end"]))
    res = rbr_ctd_cal_find_offset(ts, c, ctd)
    expected = res.isel(m=0).mean_diff
    for sn in sns:
        got = float(pre.sel(sn=sn))
        exp = float(expected.sel(sn=sn))
        assert got == pytest.approx(exp, abs=1e-12)


def test_compute_ctd_offsets_idempotent_then_overwrite(ctd_cal_mooring):
    m = Mooring(ctd_cal_mooring)
    first = m.compute_ctd_offsets()
    assert first["pre"]["written"] == 3 and first["pre"]["skipped"] == 0
    second = m.compute_ctd_offsets()
    assert second["pre"]["written"] == 0 and second["pre"]["skipped"] == 1
    third = m.compute_ctd_offsets(overwrite=True)
    assert third["pre"]["written"] == 3


def test_compute_ctd_offsets_single_source(ctd_cal_mooring):
    m = Mooring(ctd_cal_mooring)
    summary = m.compute_ctd_offsets(sources=["post"])
    assert set(summary) == {"post"}
    assert m._offsets_out_path("post").exists()
    assert not m._offsets_out_path("pre").exists()


def test_compute_ctd_offsets_honors_cal_ignore_list(ctd_cal_mooring):
    """A sensor with a valid pool file is still excluded if cal-ignored."""
    m = Mooring(ctd_cal_mooring)
    # 301111 IS assigned to pre cast 1 and has a valid pool file, but mark it
    # cal-ignored for the pre source -> it must not appear in the pre offsets.
    m.cfg.calibration.cal_ignore_sns_pre = [301111]
    summary = m.compute_ctd_offsets(sources=["pre"])
    assert summary["pre"]["written"] == 2          # was 3 (301111 dropped)
    pre = xr.open_dataarray(m._offsets_out_path("pre"))
    assert 301111 not in pre.sn.values.tolist()
    assert 301222 in pre.sn.values.tolist()
    pre.close()


def test_compute_ctd_offsets_skips_degenerate_pool_file(ctd_cal_mooring):
    """A pool file with no data variables is skipped, not fatal."""
    m = Mooring(ctd_cal_mooring)
    pool = m._cal_cast_pool_dir("pre")
    # overwrite 301222's pre-cast-1 pool file with a degenerate (no-data-var,
    # 0-length time) dataset, mimicking the real motive_c__rbr__235218 file.
    bad = next(pool.glob("*301222*.nc"))
    xr.Dataset(coords={"time": np.array([], dtype="datetime64[ns]")}).to_netcdf(bad)
    summary = m.compute_ctd_offsets(sources=["pre"])
    # 301222 dropped from cast 1; 301111 (cast1) + 301333 (cast4) remain
    assert summary["pre"]["written"] == 2
    pre = xr.open_dataarray(m._offsets_out_path("pre"))
    assert 301222 not in pre.sn.values.tolist()
    pre.close()


def test_compute_ctd_offsets_skips_sensor_with_no_cast_overlap(ctd_cal_mooring):
    """A pool file that is non-empty but has no samples in the cast period is skipped."""
    m = Mooring(ctd_cal_mooring)
    pool = m._cal_cast_pool_dir("pre")
    # overwrite 301222's pre-cast-1 pool file with data on a DIFFERENT day,
    # so it is non-empty globally but has zero overlap with the cast-1 window.
    bad = next(pool.glob("*301222*.nc"))
    times = np.arange(
        np.datetime64("2099-01-01T00:00:00"),
        np.datetime64("2099-01-01T00:10:00"),
        np.timedelta64(2, "s"),
    )
    xr.DataArray(np.full(times.size, 4.9), dims="time",
                 coords={"time": times}, name="t").to_netcdf(bad)
    summary = m.compute_ctd_offsets(sources=["pre"])
    # 301222 dropped from cast 1 (no overlap); 301111 (cast1) + 301333 (cast4) remain
    assert summary["pre"]["written"] == 2
    pre = xr.open_dataarray(m._offsets_out_path("pre"))
    assert 301222 not in pre.sn.values.tolist()
    pre.close()


def test_compute_ctd_offsets_missing_cal_stops_column_raises(ctd_cal_mooring):
    """A cal_stops.csv lacking a required column gives a clear, named error."""
    m = Mooring(ctd_cal_mooring)
    csv = Path(m.cfg.path.root) / m.cfg.path.cal_stops
    df = pd.read_csv(csv).drop(columns=["source"])
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="source"):
        m.compute_ctd_offsets()


def test_compute_ctd_offsets_missing_ctd_path_raises(ctd_cal_mooring):
    """An absent path.ctd config key gives a clear error, not a BoxKeyError."""
    m = Mooring(ctd_cal_mooring)
    del m.cfg.path["ctd"]
    with pytest.raises((KeyError, ValueError), match="ctd"):
        m.compute_ctd_offsets()


def test_compute_ctd_offsets_missing_cal_stops_path_raises(ctd_cal_mooring):
    """An absent path.cal_stops config key gives a clear error."""
    m = Mooring(ctd_cal_mooring)
    del m.cfg.path["cal_stops"]
    with pytest.raises((KeyError, ValueError), match="cal_stops"):
        m.compute_ctd_offsets()


from thermochain.pipeline import STAGE_ORDER  # noqa: E402
from thermochain.io import ProcessThermistorMooring  # noqa: E402


def test_process_l0_delegates_to_run_proc_all(cal_mooring, monkeypatch):
    """process_l0 is the L0-stage alias and forwards verbatim to run_proc_all."""
    m = Mooring(cal_mooring)
    called = []
    monkeypatch.setattr(m, "run_proc_all", lambda: called.append("ran"))
    m.process_l0()
    assert called == ["ran"]
    # exposed under the L0 vocabulary name (sanity: present on the class)
    assert hasattr(ProcessThermistorMooring, "run_proc_all")


def test_run_executes_chain_end_to_end(l2_mooring):
    """run(stages=[...]) over the L2 fixture chains make_l2 -> grid_l2."""
    m = Mooring(l2_mooring)
    out = m.run(stages=["make_l2", "grid_l2"], segments=["deep"])
    assert set(out) == {"make_l2", "grid_l2"}
    assert out["make_l2"]["deep"]["written"] == 3
    assert out["grid_l2"]["deep"]["written"] == 4
    gridl2 = m._gridl2_dir()
    files = sorted(gridl2.glob("testproj_a_deep_L2_*.nc"))
    assert len(files) == 4
    da = xr.open_dataarray(files[0])
    assert da.sizes["depth"] == 3          # all three deep sensors gridded


def test_run_honors_canonical_order_and_subset(l2_mooring, monkeypatch):
    """A reversed/partial stages= runs only those stages, in STAGE_ORDER."""
    m = Mooring(l2_mooring)
    calls = []
    for name in STAGE_ORDER:
        orig = getattr(m, name)

        def make(n, o):
            def wrapper(*a, **k):
                calls.append(n)
                return o(*a, **k)
            return wrapper

        monkeypatch.setattr(m, name, make(name, orig))
    # request the subset reversed; only these two run, in canonical order
    m.run(stages=["grid_l2", "make_l2"], segments=["deep"])
    assert calls == ["make_l2", "grid_l2"]


def test_run_idempotent_then_overwrite(l2_mooring):
    """Second run skips existing outputs; overwrite=True forces recompute."""
    m = Mooring(l2_mooring)
    m.run(stages=["make_l2", "grid_l2"], segments=["deep"])
    second = m.run(stages=["make_l2"], segments=["deep"])
    assert second["make_l2"]["deep"]["written"] == 0
    assert second["make_l2"]["deep"]["skipped"] == 3
    third = m.run(stages=["make_l2"], segments=["deep"], overwrite=True)
    assert third["make_l2"]["deep"]["written"] == 3


def test_run_rejects_unknown_stage(cal_mooring):
    m = Mooring(cal_mooring)
    with pytest.raises(ValueError, match="unknown stage"):
        m.run(stages=["frobnicate"])
