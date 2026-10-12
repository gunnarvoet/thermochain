"""Tests for thermochain.io.calc_eps_thorpe_scale."""

import mixsea as mx
import numpy as np
import pytest
import xarray as xr

from thermochain.io import calc_eps_thorpe_scale

LON, LAT = -63.2314, 38.5855


def _chain(profiles):
    depth = 2100.0 + 3.0 * np.arange(20)
    time = np.datetime64("2024-01-01") + np.arange(len(profiles)) * np.timedelta64(1, "s")
    return xr.DataArray(np.array(profiles).T, dims=("depth", "time"), coords=dict(depth=depth, time=time))


def _stable():
    return 4.0 - 0.03 * np.arange(20)  # temperature falls with depth, 0.01 K/m


def _with_overturns():
    T = _stable()
    T[[4, 5]] = T[[5, 4]]        # two-point overturn
    T[10:14] = T[10:14][::-1]    # four-point overturn
    return T


def test_defaults_reproduce_the_direct_mixsea_call():
    t = _chain([_with_overturns(), _stable()])
    eps, Lt = calc_eps_thorpe_scale(t, LON, LAT)
    e0, _, d0 = mx.overturn.nan_eps_overturn(
        t.depth.values, t.values[:, 0], SP=35.13, lon=LON, lat=LAT, N2_method="teos", return_diagnostics=True)
    np.testing.assert_array_equal(eps.values[:, 0], e0)
    np.testing.assert_array_equal(Lt.values[:, 0], d0["Lt"])
    assert eps.dims == ("depth", "time")
    assert np.isfinite(eps.values[[4, 5, 10, 13], 0]).all()
    assert np.isnan(eps.values[:, 1]).all()


def test_min_points_removes_the_two_point_overturn_only():
    eps, _ = calc_eps_thorpe_scale(_chain([_with_overturns()]), LON, LAT, min_points=3)
    assert np.isnan(eps.values[[4, 5], 0]).all()
    assert np.isfinite(eps.values[10:14, 0]).all()


def test_abutting_two_point_overturns_stay_two_patches():
    T = _stable()
    T[[4, 5]] = T[[5, 4]]
    T[[6, 7]] = T[[7, 6]]
    eps, _, d = calc_eps_thorpe_scale(_chain([T]), LON, LAT, min_points=3, return_diagnostics=True)
    np.testing.assert_array_equal(d.n_points.values[4:8, 0], [2, 2, 2, 2])
    assert np.isnan(eps.values[4:8, 0]).all()


def test_alpha_scales_eps_by_its_square():
    t = _chain([_with_overturns()])
    a, _ = calc_eps_thorpe_scale(t, LON, LAT, alpha=0.95)
    b, _ = calc_eps_thorpe_scale(t, LON, LAT, alpha=0.8)
    np.testing.assert_allclose(b.values[10:14, 0] / a.values[10:14, 0], (0.8 / 0.95) ** 2)


def test_bulk_n2_and_dnoise_pass_through():
    t = _chain([_with_overturns()])
    eps, _ = calc_eps_thorpe_scale(t, LON, LAT, S0=34.9603, N2_method="bulk", dnoise=6e-5)
    e0, _ = mx.overturn.nan_eps_overturn(
        t.depth.values, t.values[:, 0], SP=34.9603, lon=LON, lat=LAT, N2_method="bulk", dnoise=6e-5)
    np.testing.assert_array_equal(eps.values[:, 0], e0)
    teos, _ = calc_eps_thorpe_scale(t, LON, LAT, S0=34.9603)
    assert not np.allclose(eps.values[10:14, 0], teos.values[10:14, 0])


def test_diagnostics_count_points_and_flag_the_chain_ends():
    top = _stable()
    top[0:3] = top[0:3][::-1]            # touches the top sensor
    bottom = _stable()
    bottom[17:20] = bottom[17:20][::-1]  # touches the bottom sensor
    gap = _stable()
    gap[0] = np.nan                      # top sensor missing: the chain ends one sensor lower
    gap[1:4] = gap[1:4][::-1]
    t = _chain([_with_overturns(), top, bottom, gap])
    _, _, d = calc_eps_thorpe_scale(t, LON, LAT, return_diagnostics=True)
    n = d.n_points.values
    np.testing.assert_array_equal(n[[4, 5], 0], [2, 2])
    np.testing.assert_array_equal(n[10:14, 0], [4, 4, 4, 4])
    assert n[:, 0].sum() == 2 * 2 + 4 * 4
    assert d.ends_flag.dtype == bool
    assert not d.ends_flag.values[:, 0].any()
    assert d.ends_flag.values[0:3, 1].all() and not d.ends_flag.values[3:, 1].any()
    assert d.ends_flag.values[17:20, 2].all() and not d.ends_flag.values[:17, 2].any()
    assert d.ends_flag.values[1:4, 3].all() and not d.ends_flag.values[0, 3]
    assert n[0, 3] == 0 and (n[1:4, 3] == 3).all()
    np.testing.assert_allclose(d.Lp.values[10:14, 0], 12.0)


def test_a_failed_profile_returns_nan_eps_and_minus_one_Lt():
    unstable = _stable()[::-1].copy()  # temperature rises with depth everywhere
    eps, Lt = calc_eps_thorpe_scale(_chain([unstable]), LON, LAT)
    assert np.isnan(eps.values).all() and (Lt.values == -1).all()


def test_point_counts_refuse_a_chain_across_two_reference_pressures():
    depth = 950.0 + 10.0 * np.arange(20)  # 958 to 1151 dbar, two of mixsea's 1000 dbar bins
    t = xr.DataArray(_stable()[:, None], dims=("depth", "time"),
                     coords=dict(depth=depth, time=[np.datetime64("2024-01-01")]))
    with pytest.raises(ValueError, match="1000 dbar"):
        calc_eps_thorpe_scale(t, LON, LAT, min_points=3)
