"""Tests for thermochain.io.offsets_from_background_fit."""

import numpy as np
import pytest
import xarray as xr

from thermochain.io import offsets_from_background_fit


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
