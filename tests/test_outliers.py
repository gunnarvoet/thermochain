"""Tests for thermochain.io.find_outliers."""

import warnings

import numpy as np
import xarray as xr

from thermochain.io import find_outliers


def _synthetic_stratification(n_depth=40, n_time=200, seed=0):
    """Smooth T(z) profile plus small time variability. No outliers."""
    rng = np.random.default_rng(seed)
    z = np.linspace(1000.0, 1500.0, n_depth)
    t_mean = 4.0 + 6.0 * np.exp(-(z - 1000.0) / 200.0)
    noise = rng.standard_normal((n_time, n_depth)) * 0.01
    arr = t_mean[None, :] + noise
    time = np.arange(n_time)
    return xr.DataArray(
        arr,
        dims=("time", "depth"),
        coords={"time": time, "depth": z},
    )


class TestFindOutliers:
    def test_no_outliers_returns_all_true(self):
        t = _synthetic_stratification()
        mask = find_outliers(t, exclusion_criteria=0.1, plot=False)
        assert bool(mask.all())

    def test_single_outlier_flagged(self):
        t = _synthetic_stratification()
        # Force one sensor far off its background.
        t.values[:, 7] += 0.5
        mask = find_outliers(t, exclusion_criteria=0.05, plot=False)
        # Target sensor is flagged. We don't assert on neighbours because
        # the single-stage algorithm fits one polynomial over all sensors
        # including the outlier, and a degree-8 polynomial bends toward
        # a localised +0.5 perturbation — pulling 3–6 adjacent sensors
        # past the 0.05 band. The two-stage variant in
        # test_two_stage_criterion exists to avoid exactly this spillover.
        assert not bool(mask[7])

    def test_two_stage_criterion(self):
        t = _synthetic_stratification()
        # Gross outlier caught by first stage, subtle one by second.
        t.values[:, 3] += 0.2
        t.values[:, 25] += 0.02
        mask = find_outliers(
            t, exclusion_criteria=[0.1, 0.01], plot=False,
        )
        assert not bool(mask[3])
        assert not bool(mask[25])

    def test_second_fit_emits_no_deprecation_warning(self):
        # The second_fit branch locates the bottom-most sensor. It must do
        # so without xarray's argmin/argmax-without-dim DeprecationWarning.
        t = _synthetic_stratification()
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            find_outliers(t, exclusion_criteria=[0.1, 0.01], plot=False)

    def test_dropout_sensor_excluded(self):
        # Sensors with lots of NaNs (>0.1% of samples) are dropped.
        t = _synthetic_stratification()
        t.values[:, 2] = np.nan
        mask = find_outliers(t, exclusion_criteria=0.1, plot=False)
        assert not bool(mask[2])
