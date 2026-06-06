"""Tests for the two-step shared-fluctuation refinement in sensor_drift.

CvHG16 forms a first-guess shared fluctuating component, fits an interim
drift, *detrends* the offsets with it, and recomputes the shared component
from the detrended series before differencing it out of the original
offsets. Historically the recompute read the (non-detrended) ``offsets``,
making the second guess identical to the first and the interim fit a
no-op. ``two_step_shared`` (default True) wires the detrended series into
the recompute; ``two_step_shared=False`` recovers the legacy single-pass
behaviour.
"""

import numpy as np
import pytest
import tqdm

from _synthetic import write_drift_l1_files
from thermochain.io import sensor_drift


pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


@pytest.fixture(autouse=True)
def _plain_tqdm(monkeypatch):
    monkeypatch.setattr(tqdm, "tqdm_notebook", tqdm.tqdm, raising=False)


@pytest.fixture
def l1_dir_with_drift(tmp_path):
    """L1 grid dir with a strong linear drift on one interior sensor."""
    write_drift_l1_files(tmp_path)
    return tmp_path


def _run(l1_dir, **drift_params):
    return sensor_drift(
        mooring_name="synthetic",
        l1_grid_dir=l1_dir,
        run_all=True,
        drift_parameters=dict(fit_mode="linear", **drift_params),
    )


def _amp(da):
    return float(da.max("window") - da.min("window"))


def test_two_step_recompute_differs_from_first_guess_by_default(l1_dir_with_drift):
    # Two-step is the default: the recomputed (second-guess) shared
    # component must differ from the first guess, otherwise the interim fit
    # never influences the result.
    sd = _run(l1_dir_with_drift)
    assert not np.allclose(
        sd.second_guess_shared_fluct_comp.values,
        sd.first_guess_shared_fluct_comp.values,
        equal_nan=True,
    )


def test_legacy_flag_collapses_second_to_first(l1_dir_with_drift):
    # two_step_shared=False recovers the single-pass behaviour: the second
    # guess is recomputed from self.offsets and equals the first guess.
    sd = _run(l1_dir_with_drift, two_step_shared=False)
    np.testing.assert_array_equal(
        sd.second_guess_shared_fluct_comp.values,
        sd.first_guess_shared_fluct_comp.values,
    )


def _slope_over_windows(da):
    """Least-squares slope of a 1-D (window,) series, ignoring NaNs."""
    w = np.arange(da.sizes["window"], dtype=float)
    y = da.values
    m = np.isfinite(y)
    return float(np.polyfit(w[m], y[m], 1)[0])


def test_two_step_reduces_drift_trend_in_shared_component(l1_dir_with_drift):
    # Mechanism test. The first-guess shared component at the drifting
    # sensor's depth carries the drift's trend (it is the triplet mean of
    # demeaned offsets, and demeaning does not remove the slope). The
    # interim-fit detrend strips that trend before the recompute, so the
    # two-step second-guess shared component has a smaller-magnitude trend
    # at the drifting depth. Drift is planted at drift_index=6
    # (write_drift_l1_files), which is depth index 6 (depth-ascending grid).
    sd = _run(l1_dir_with_drift)
    fg_slope = _slope_over_windows(sd.first_guess_shared_fluct_comp.isel(depth=6))
    sg_slope = _slope_over_windows(sd.second_guess_shared_fluct_comp.isel(depth=6))
    assert abs(sg_slope) < abs(fg_slope)


def test_drift_parameters_default_is_two_step():
    from thermochain.pipeline import DriftParameters

    assert DriftParameters().two_step_shared is True
    assert DriftParameters.from_dict({"two_step_shared": False}).two_step_shared is False


def test_two_step_flag_recorded_in_provenance_attrs():
    from thermochain.pipeline import DriftParameters, drift_provenance_attrs

    attrs = drift_provenance_attrs(DriftParameters(), label="testfit")
    assert attrs["drift_param_two_step_shared"] == 1
    attrs_off = drift_provenance_attrs(
        DriftParameters(two_step_shared=False), label="testfit"
    )
    assert attrs_off["drift_param_two_step_shared"] == 0
