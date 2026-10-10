"""Tests for thermochain.io.offsets_from_background_fit."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from thermochain.io import (  # noqa: E402
    correct_offset,
    find_outliers,
    offsets_from_background_fit,
)


def _synthetic_profile(n_depth=40, n_time=200, seed=0, edge_kick=0.0):
    """Smooth exponential T(z) plus small time noise.

    `edge_kick` adds an offset to the first and last sensor only — used
    to exercise the weight-based endpoint damping.
    """
    rng = np.random.default_rng(seed)
    z = np.linspace(1000.0, 1500.0, n_depth)
    t_mean = 4.0 + 6.0 * np.exp(-(z - 1000.0) / 200.0)
    t_mean[0] += edge_kick
    t_mean[-1] += edge_kick
    noise = rng.standard_normal((n_time, n_depth)) * 0.01
    arr = t_mean[None, :] + noise
    return xr.DataArray(
        arr,
        dims=("time", "depth"),
        coords={"time": np.arange(n_time), "depth": z},
    )


class TestOffsetsFromBackgroundFitWeights:
    def test_weights_none_matches_default(self):
        t = _synthetic_profile()
        a = offsets_from_background_fit(
            t, exclude=0.1, plot=False, spline=True, spline_smooth=1e-4
        )
        b = offsets_from_background_fit(
            t, exclude=0.1, plot=False, spline=True, spline_smooth=1e-4,
            weights=None,
        )
        np.testing.assert_array_equal(a.values, b.values)

    def test_endpoint_downweight_releases_endpoint_fit(self):
        # Kick the first and last sensor by 0.05°C off the smooth profile.
        # With uniform weights the spline chases them (small offset at the
        # endpoint). With endpoint weight 0.01 the spline ignores them
        # (offset at the endpoint approaches the full kick).
        kick = 0.05
        t = _synthetic_profile(edge_kick=kick)
        uniform = offsets_from_background_fit(
            t, exclude=0.5, plot=False, spline=True, spline_smooth=1e-4,
        )
        w = np.ones(t.depth.size)
        w[[0, -1]] = 0.01
        damped = offsets_from_background_fit(
            t, exclude=0.5, plot=False, spline=True, spline_smooth=1e-4,
            weights=w,
        )
        # The endpoint offset grows by an amount comparable to the kick.
        assert abs(damped.values[0]) > abs(uniform.values[0]) + 0.5 * kick
        assert abs(damped.values[-1]) > abs(uniform.values[-1]) + 0.5 * kick

    def test_wrong_length_raises(self):
        t = _synthetic_profile(n_depth=20)
        with pytest.raises(ValueError, match="weights must be length"):
            offsets_from_background_fit(
                t, exclude=0.1, plot=False, spline=True,
                weights=np.ones(10),
            )

    def test_negative_weights_rejected(self):
        t = _synthetic_profile(n_depth=20)
        w = np.ones(20)
        w[0] = -0.1
        with pytest.raises(ValueError, match="non-negative"):
            offsets_from_background_fit(
                t, exclude=0.1, plot=False, spline=True, weights=w,
            )


class TestOffsetsFromBackgroundFitPlot:
    @pytest.mark.parametrize("spline", [False, True])
    def test_plot_true_renders(self, spline):
        t = _synthetic_profile()
        offs = offsets_from_background_fit(t, exclude=0.1, spline=spline)
        ref = offsets_from_background_fit(
            t, exclude=0.1, spline=spline, plot=False
        )
        plt.close("all")
        np.testing.assert_array_equal(offs.values, ref.values)

    def test_correct_offset_default_plot(self):
        t = _synthetic_profile()
        corrected, offs = correct_offset(t, exclude=0.1, return_offsets=True)
        plt.close("all")
        assert corrected.shape == t.shape
        assert offs.dims == ("depth",)


def _chain_at(z0, n_depth=46, n_time=200, seed=0, outlier=None):
    """Same 200 m chain and temperatures, placed with its top at depth `z0`."""
    rng = np.random.default_rng(seed)
    dz = np.linspace(0.0, 200.0, n_depth)
    t_mean = 1.2 + 0.3 * np.exp(-dz / 80.0)
    if outlier is not None:
        t_mean[outlier] += 5e-3
    noise = rng.standard_normal((n_time, n_depth)) * 1e-4
    return xr.DataArray(
        t_mean[None, :] + noise,
        dims=("time", "depth"),
        coords={"time": np.arange(n_time), "depth": z0 + dz},
    )


def test_all_sensors_rejected_raises_clear_error():
    # A first exclusion criterion of zero rejects every sensor, which leaves
    # nothing for the second fit.
    t = _synthetic_profile()
    with pytest.raises(ValueError, match="no sensors left"):
        find_outliers(t, [0.0, 0.1], plot=False)


class TestDepthOffsetInvariance:
    """A degree-8 background fit must not depend on where the chain sits."""

    def test_outlier_mask_and_offsets_independent_of_depth(self):
        shallow = _chain_at(0.0, outlier=20)
        deep = _chain_at(4100.0, outlier=20)
        mask_shallow = find_outliers(shallow, [2e-3, 1e-3], plot=False)
        mask_deep = find_outliers(deep, [2e-3, 1e-3], plot=False)
        assert not bool(mask_shallow[20])
        np.testing.assert_array_equal(mask_shallow.values, mask_deep.values)

        kw = dict(exclude=[2e-3, 1e-3], plot=False, polydeg=8, outliers_polydeg=8)
        offs_shallow = offsets_from_background_fit(shallow, **kw)
        offs_deep = offsets_from_background_fit(deep, **kw)
        np.testing.assert_allclose(
            offs_shallow.values, offs_deep.values, rtol=0, atol=1e-9
        )
