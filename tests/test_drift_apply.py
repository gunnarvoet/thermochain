import numpy as np
import xarray as xr

from thermochain.io import (
    EXP_PARAM_NAMES,
    evaluate_drift_model,
    exp_function,
    sensor_drift,
)


def _fit(synthetic_l1_dir):
    # Matches the existing drift tests' construction (default file_pattern).
    sd = sensor_drift(
        mooring_name="test",
        l1_grid_dir=synthetic_l1_dir,
        run_all=True,
    )
    sd.drift_to_dataarray()
    return sd


def test_evaluate_linear_is_exact_line():
    idx = np.array([-0.5, 0.0, 1.0, 2.5, 10.0])
    slope, intercept = 0.002, -0.01
    out = evaluate_drift_model(idx, "lin", (slope, intercept), None)
    np.testing.assert_allclose(out, intercept + slope * idx)


def test_evaluate_exp_matches_exp_function_for_nonneg_index():
    idx = np.array([0.0, 0.3, 1.0, 5.0, 30.0])
    params = (0.01, 1e-5, -0.05, 1.5, 20.0)  # t0, m, A, beta, tau
    out = evaluate_drift_model(idx, "exp", None, params)
    np.testing.assert_allclose(out, exp_function(idx, *params))


def test_evaluate_exp_negative_index_is_finite_linear_part():
    idx = np.array([-0.25, -0.1])
    t0, m, A, beta, tau = 0.01, 1e-5, -0.05, 1.5, 20.0
    out = evaluate_drift_model(idx, "exp", None, (t0, m, A, beta, tau))
    # gamma term held at 0 for idx < 0 -> drift = t0 + m*idx, finite & real
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, t0 + m * idx)


def test_drift_product_carries_fit_parameters(synthetic_l1_dir):
    sd = _fit(synthetic_l1_dir)
    da = sd.drift
    assert "fit_type" in da.coords
    assert "lin_slope" in da.coords
    assert "lin_intercept" in da.coords
    # lin params reproduce the evaluated linear fit at the window centres
    expected0 = sd.drift_linfit.isel(window=0).values
    np.testing.assert_allclose(da.coords["lin_intercept"].values, expected0)


def test_drift_product_evaluates_back_to_drift_fit_at_centres(synthetic_l1_dir):
    # Evaluating the stored parameters at integer window indices must
    # reproduce drift_fit (the model sampled at those same indices).
    sd = _fit(synthetic_l1_dir)
    da = sd.drift.swap_dims({"depth": "sn", "window": "time"})
    n_win = da.sizes["time"]
    idx = np.arange(n_win, dtype=float)
    for sn in da.sn.values:
        d = da.sel(sn=sn)
        ftype = str(d["fit_type"].values)
        lin = (float(d["lin_slope"]), float(d["lin_intercept"]))
        exp = (
            [float(d[f"exp_{p}"]) for p in EXP_PARAM_NAMES]
            if "exp_t0" in d.coords
            else None
        )
        got = evaluate_drift_model(idx, ftype, lin, exp)
        np.testing.assert_allclose(got, d.values, rtol=1e-6, atol=1e-9)


# --- Task 3: applying the drift via closed-form evaluation -----------------

from thermochain.pipeline import correct_drift  # noqa: E402


def _uniform_drift_da(values_per_sn, t0, dt_ns, fit_type, lin, exp=None):
    """Build a minimal (sn, time) drift product with parameter coords.

    `values_per_sn`: dict sn -> 1d array over windows (the drift_fit trace).
    Window centres are uniformly spaced (so linear interp of a line is exact).
    """
    sns = list(values_per_sn)
    n_win = len(next(iter(values_per_sn.values())))
    times = t0 + np.arange(n_win) * dt_ns
    data = np.vstack([values_per_sn[sn] for sn in sns])
    da = xr.DataArray(
        data,
        dims=("sn", "time"),
        coords={"sn": sns, "time": times.astype("datetime64[ns]")},
        name="drift",
    )
    da.coords["fit_type"] = ("sn", [fit_type] * len(sns))
    da.coords["lin_slope"] = ("sn", [lin[sn][0] for sn in sns])
    da.coords["lin_intercept"] = ("sn", [lin[sn][1] for sn in sns])
    if exp is not None:
        for i, pname in enumerate(EXP_PARAM_NAMES):
            da.coords[f"exp_{pname}"] = ("sn", [exp[sn][i] for sn in sns])
    return da


def _daily_centres(t0_str, n_win):
    t0 = np.datetime64(t0_str, "ns").astype("int64")
    dt_ns = np.timedelta64(1, "D").astype("timedelta64[ns]").astype("int64")
    return t0, dt_ns


def _hourly_sensor(t0, n_win, hour_step=6, start_hours=-3):
    # Native samples spanning the window range, including a few hours BEFORE
    # the first window centre (start_hours<0) so the start edge is exercised.
    h_ns = np.timedelta64(1, "h").astype("timedelta64[ns]").astype("int64")
    n_samp = (n_win * 24 - start_hours) // hour_step
    s = (t0 + (start_hours + np.arange(n_samp) * hour_step) * h_ns).astype(
        "datetime64[ns]"
    )
    return xr.DataArray(np.zeros(len(s)), dims="time", coords={"time": s})


def test_correct_drift_exp_uses_closed_form_and_diverges_from_interp():
    # THE load-bearing test: for an exponential fit sampled BETWEEN window
    # centres, the closed-form result must (a) match evaluate_drift_model and
    # (b) DIFFER from the legacy linear interp. (b) is what proves the new
    # code path actually runs — a fallback-only impl fails here.
    n_win = 10
    t0, dt_ns = _daily_centres("2025-01-01T00:00:00", n_win)
    params = (0.0, 0.0, -0.05, 0.5, 5.0)  # t0,m,A,beta,tau — strong early curvature
    idx = np.arange(n_win, dtype=float)
    trace = evaluate_drift_model(idx, "exp", None, params)  # knot values = drift_fit
    drift = _uniform_drift_da(
        {55: trace}, t0, dt_ns, "exp",
        lin={55: (0.0, 0.0)},  # unused for exp dispatch
        exp={55: list(params)},
    )
    sensor = _hourly_sensor(t0, n_win)
    new = correct_drift(sensor, 55, drift)

    samp_int = sensor["time"].values.astype("datetime64[ns]").astype("int64")
    frac = (samp_int - t0) / dt_ns
    expected_drift = evaluate_drift_model(frac, "exp", None, params)
    np.testing.assert_allclose((sensor - new).values, expected_drift, rtol=1e-9,
                               atol=1e-12)
    assert np.all(np.isfinite(new.values))  # start-edge samples stay finite

    legacy = sensor - drift.sel(sn=55).interp_like(
        sensor.time, kwargs=dict(fill_value="extrapolate")
    )
    # Closed-form must visibly differ from linear interp somewhere between knots.
    assert np.max(np.abs(new.values - legacy.values)) > 1e-6


def test_correct_drift_linear_equals_interp():
    # Sanity: for a linear fit on uniform centres, closed-form == legacy interp.
    n_win = 10
    t0, dt_ns = _daily_centres("2025-01-01T12:00:00", n_win)
    slope, intercept = 1e-4, 5e-3
    trace = intercept + slope * np.arange(n_win, dtype=float)
    drift = _uniform_drift_da(
        {101: trace}, t0, dt_ns, "lin", {101: (slope, intercept)}
    )
    sensor = _hourly_sensor(t0, n_win, hour_step=1, start_hours=0)
    new = correct_drift(sensor, 101, drift)
    legacy = sensor - drift.sel(sn=101).interp_like(
        sensor.time, kwargs=dict(fill_value="extrapolate")
    )
    np.testing.assert_allclose(new.values, legacy.values, rtol=1e-9, atol=1e-12)


def test_correct_drift_falls_back_without_params():
    # A drift product lacking parameter coords must reproduce the legacy path
    # EXACTLY (not merely "not crash").
    n_win = 5
    t0, dt_ns = _daily_centres("2025-01-01T00:00:00", n_win)
    times = (t0 + np.arange(n_win) * dt_ns).astype("datetime64[ns]")
    drift = xr.DataArray(
        np.linspace(0, 1e-3, n_win)[None, :],
        dims=("sn", "time"),
        coords={"sn": [7], "time": times},
        name="drift",
    )
    sensor = _hourly_sensor(t0, n_win, hour_step=3, start_hours=0)
    new = correct_drift(sensor, 7, drift)
    legacy = sensor - drift.sel(sn=7).interp_like(
        sensor.time, kwargs=dict(fill_value="extrapolate")
    )
    np.testing.assert_allclose(new.values, legacy.values, rtol=1e-12, atol=0)
