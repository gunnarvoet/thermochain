"""Tests for iterative drift subtraction in sensor_drift.

The shared-fluctuation method assumes neighbours share common ocean
signal so the differencing isolates drift. When one sensor's drift is
much larger than its neighbours', the differencing leaks the drift into
the neighbour fits with opposite sign. Iterative subtraction snapshots
the pass-1 drift on flagged outliers, subtracts it from `offsets`, and
re-runs the neighbour-stack stages once.
"""

import numpy as np
import pytest
import tqdm
import xarray as xr

from thermodrift.io import sensor_drift


pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


@pytest.fixture(autouse=True)
def _plain_tqdm(monkeypatch):
    monkeypatch.setattr(tqdm, "tqdm_notebook", tqdm.tqdm, raising=False)


@pytest.fixture
def synthetic_l1_dir_with_drift(tmp_path):
    """Like synthetic_l1_dir but injects a strong linear drift on one
    interior sensor. Returns (l1_dir, drifting_sn) so tests can assert
    against the prescribed sensor without re-deriving its serial.

    Drift amplitude ~5 mK over a 12-day deployment, matching the
    MOTIVE A 236127 case that motivated this feature.
    """
    n_depth = 12
    sn = np.array([72100 + i for i in range(n_depth)])
    depth = np.linspace(1000.0, 1200.0, n_depth)
    t_mean = 4.0 + 6.0 * np.exp(-(depth - 1000.0) / 200.0)
    drift_index = 6
    drift_total_C = 5e-3

    t_start = np.datetime64("2024-01-01T00:00")
    t_end = np.datetime64("2024-01-13T00:00")
    deployment_seconds = float((t_end - t_start) / np.timedelta64(1, "s"))

    def _build_file(day_start, day_end):
        rng = np.random.default_rng(int(day_start))
        times = np.arange(
            np.datetime64(f"2024-01-{day_start:02d}T00:00"),
            np.datetime64(f"2024-01-{day_end:02d}T00:00"),
            np.timedelta64(1, "m"),
        )
        arr = (
            t_mean[None, :]
            + 0.02 * rng.standard_normal((times.size, sn.size))
        )
        elapsed_s = (times - t_start) / np.timedelta64(1, "s")
        drift = drift_total_C * (elapsed_s.astype(float) / deployment_seconds)
        arr[:, drift_index] += drift
        return xr.DataArray(
            arr,
            dims=("time", "depth"),
            coords={
                "time": times,
                "depth": ("depth", depth),
                "sn": ("depth", sn),
            },
            name="t",
        )

    spans = [(1, 4), (4, 7), (7, 10), (10, 13)]
    for start, end in spans:
        path = (
            tmp_path
            / f"mavs0_gridded_2024-01-{start:02d}_to_2024-01-{end:02d}.nc"
        )
        _build_file(start, end).to_netcdf(path)
    return tmp_path, int(sn[drift_index])


def _amp(da):
    return float(da.max("window") - da.min("window"))


class TestNoFlag:
    def test_iteration_no_op_when_amplitudes_below_threshold(
        self, synthetic_l1_dir
    ):
        # Synthetic data carries only ~20 mK random noise per sample, so
        # daily-window drift amplitudes are far under a 50 mK threshold.
        # Iteration should be a no-op and record an empty flag list.
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=50.0,
                fit_mode="linear",
            ),
        )
        assert sd.flagged_outlier_sns == []
        assert sd.iteration_count == 0
        assert not hasattr(sd, "drift_fit_pass1")


class TestManualOverride:
    def test_manual_sn_is_flagged_and_subtracted(self, synthetic_l1_dir):
        target_sn = 72106
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=50.0,
                manual_outlier_sns=[target_sn],
                fit_mode="linear",
            ),
        )
        assert target_sn in sd.flagged_outlier_sns
        assert sd.iteration_count == 1
        idx = int(np.where(sd.offsets.sn.values == target_sn)[0][0])
        diff = sd.offsets.isel(depth=idx) - sd.offsets_pass1.isel(depth=idx)
        assert float(np.abs(diff).max()) > 0

    def test_manual_sn_not_in_deployment_warns(self, synthetic_l1_dir):
        with pytest.warns(UserWarning, match="not in the deployment"):
            sd = sensor_drift(
                mooring_name="synthetic",
                l1_grid_dir=synthetic_l1_dir,
                run_all=True,
                drift_parameters=dict(
                    iterate_subtract=True,
                    amplitude_threshold_mK=50.0,
                    manual_outlier_sns=[999999],
                    fit_mode="linear",
                ),
            )
        # 999999 not present → no flags, no iteration
        assert sd.flagged_outlier_sns == []
        assert sd.iteration_count == 0


class TestSyntheticLargeDriftImprovesNeighbours:
    def test_neighbours_improve_after_iteration(
        self, synthetic_l1_dir_with_drift
    ):
        # The prescribed-drift sensor's own pass-1 fit is already good;
        # what matters is the leakage onto its neighbours. We expect
        # pass-2 amplitudes on the immediate neighbours to drop after
        # the outlier's drift is subtracted from the offsets.
        l1_dir, drift_sn = synthetic_l1_dir_with_drift
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=1.5,
                fit_mode="linear",
            ),
        )
        assert drift_sn in sd.flagged_outlier_sns
        assert sd.iteration_count == 1

        sn_arr = sd.drift_fit.sn.values
        drift_idx = int(np.where(sn_arr == drift_sn)[0][0])
        for nb_idx in (drift_idx - 1, drift_idx + 1):
            if nb_idx < 0 or nb_idx >= len(sn_arr):
                continue
            amp_pass1 = _amp(sd.drift_fit_pass1.isel(depth=nb_idx))
            amp_pass2 = _amp(sd.drift_fit.isel(depth=nb_idx))
            assert amp_pass2 < amp_pass1, (
                f"neighbour at depth {nb_idx}: pass-2 amplitude {amp_pass2:.3e} "
                f"not less than pass-1 {amp_pass1:.3e}"
            )


class TestIdempotentWhenDisabled:
    def test_no_pass1_attrs_when_iteration_disabled(self, synthetic_l1_dir):
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            run_all=True,
            drift_parameters=dict(iterate_subtract=False, fit_mode="linear"),
        )
        assert not hasattr(sd, "drift_fit_pass1")
        assert not hasattr(sd, "offsets_pass1")
        assert not hasattr(sd, "iteration_count")
        assert not hasattr(sd, "flagged_outlier_sns")


class TestOutputAttrs:
    def test_drift_to_dataarray_writes_provenance_attrs(
        self, synthetic_l1_dir_with_drift
    ):
        l1_dir, _ = synthetic_l1_dir_with_drift
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=1.5,
                fit_mode="linear",
            ),
        )
        sd.drift_to_dataarray()
        assert sd.drift.attrs["iteration_count"] == 1
        assert isinstance(sd.drift.attrs["flagged_outlier_sns"], np.ndarray)
        assert sd.drift.attrs["flagged_outlier_sns"].dtype == np.int64
        assert sd.drift.attrs["flagged_outlier_sns"].size >= 1

    def test_attrs_present_when_iteration_disabled(self, synthetic_l1_dir):
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            run_all=True,
            drift_parameters=dict(iterate_subtract=False, fit_mode="linear"),
        )
        sd.drift_to_dataarray()
        # drift_to_dataarray uses getattr defaults so the schema is stable
        assert sd.drift.attrs["iteration_count"] == 0
        assert sd.drift.attrs["flagged_outlier_sns"].size == 0

    def test_attrs_round_trip_through_netcdf(
        self, synthetic_l1_dir_with_drift, tmp_path
    ):
        l1_dir, _ = synthetic_l1_dir_with_drift
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=1.5,
                fit_mode="linear",
            ),
        )
        sd.drift_to_netcdf(path=tmp_path, suffix="iter")
        reloaded = xr.open_dataarray(tmp_path / "drift_synthetic_iter.nc")
        assert int(reloaded.attrs["iteration_count"]) == 1
        flagged = np.asarray(reloaded.attrs["flagged_outlier_sns"])
        assert flagged.size >= 1
        assert flagged.dtype.kind == "i"
