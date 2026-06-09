"""Tests for sensor_drift.remove_outliers' 3*std offset masking.

remove_outliers masks per-sensor offset outliers (median +/-3 sigma) as
NaN, mutating each depth group in place and counting how many it masked.
These tests pin that behaviour (the masked locations and the count) and
assert the masking runs without xarray's GroupBy.apply
PendingDeprecationWarning, so the apply -> map migration is verified to
preserve behaviour.
"""

import warnings

import numpy as np
import xarray as xr

from thermochain.io import sensor_drift


def _drift_with_offsets():
    """A bare sensor_drift carrying a hand-built offsets_initial.

    Three sensors over twenty windows; tiny distinct baseline per sensor
    plus a single gross spike on the middle sensor so exactly one point
    falls outside the median +/-3 sigma band.
    """
    n_window, n_depth = 20, 3
    base = np.linspace(-0.01, 0.01, n_window)
    arr = np.tile(base[:, None], (1, n_depth))
    spike_w, spike_d = 5, 1
    arr[spike_w, spike_d] = 1.0
    offsets_initial = xr.DataArray(
        arr,
        dims=("window", "depth"),
        coords={
            "window": np.arange(n_window),
            "depth": np.array([10.0, 20.0, 30.0]),
        },
    )
    sd = sensor_drift.__new__(sensor_drift)
    sd.offsets_initial = offsets_initial
    return sd, (spike_w, spike_d)


class TestRemoveOutliers:
    def test_masks_only_the_injected_spike(self):
        sd, (spike_w, spike_d) = _drift_with_offsets()
        sd.remove_outliers()
        nan_mask = np.isnan(sd.offsets.values)
        # Exactly the injected spike is masked, nothing else.
        assert sd.n_offset_outliers == 1
        assert nan_mask.sum() == 1
        assert nan_mask[spike_w, spike_d]

    def test_emits_no_groupby_apply_deprecation(self):
        sd, _ = _drift_with_offsets()
        with warnings.catch_warnings():
            warnings.simplefilter("error", PendingDeprecationWarning)
            sd.remove_outliers()
