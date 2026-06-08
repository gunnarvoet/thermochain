import numpy as np
import pytest
import xarray as xr

from thermochain.io import (
    EXP_PARAM_NAMES,
    evaluate_drift_model,
    exp_function,
    sensor_drift,
)

# Building a real sensor_drift exercises find_outliers' known z.argmax()
# DeprecationWarning (and xarray's GroupBy.apply PendingDeprecationWarning),
# which the suite escalates to errors. Matches the suppression used by the
# other drift test modules (e.g. test_iterate_subtract); the find_outliers
# cleanup is tracked separately.
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


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
