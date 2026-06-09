"""Tests for iterative drift subtraction in sensor_drift.

The shared-fluctuation method assumes neighbours share common ocean
signal so the differencing isolates drift. When one sensor's drift is
much larger than its neighbours', the differencing leaks the drift into
the neighbour fits with opposite sign. Iterative subtraction snapshots
the pass-1 drift on flagged outliers, subtracts it from `offsets`, and
re-runs the neighbour-stack stages once.
"""

import pathlib

import numpy as np
import pytest
import tqdm
import xarray as xr

from _synthetic import write_drift_l1_files
from thermochain.io import sensor_drift


@pytest.fixture(autouse=True)
def _plain_tqdm(monkeypatch):
    monkeypatch.setattr(tqdm, "tqdm_notebook", tqdm.tqdm, raising=False)


@pytest.fixture
def synthetic_l1_dir_with_drift(tmp_path):
    """Like synthetic_l1_dir but injects a strong linear drift on one
    interior sensor. Returns (l1_dir, drifting_sn) so tests can assert
    against the prescribed sensor without re-deriving its serial.

    Drift amplitude ~5 mK over a 12-day deployment, matching the
    MOTIVE A 236127 case that motivated this feature. The dataset
    builder lives in ``_synthetic.write_drift_l1_files`` so the pinned
    restore-mode baseline is generated from the same data.
    """
    drift_sn = write_drift_l1_files(tmp_path)
    return tmp_path, drift_sn


def _amp(da):
    return float(da.max("window") - da.min("window"))


# Restore-mode drift_fit pinned from the pre-`iterate_mode` code on the
# `synthetic_l1_dir_with_drift` dataset; guards the default path against
# behavioural drift. Regenerate only with deliberate intent.
BASELINE = pathlib.Path(__file__).parent / "data" / "iterate_restore_baseline.nc"


def _run_iter(l1_dir, drift_sn, mode, **extra):
    """Run sensor_drift with iterate_subtract on, forcing `drift_sn` as the
    sole flagged outlier, under the given iterate_mode. Extra drift
    parameters (e.g. two_step_shared) override the defaults."""
    params = dict(
        iterate_subtract=True,
        amplitude_threshold_mK=999.0,
        manual_outlier_sns=[drift_sn],
        fit_mode="linear",
        iterate_mode=mode,
    )
    params.update(extra)
    return sensor_drift(
        mooring_name="synthetic",
        l1_grid_dir=l1_dir,
        run_all=True,
        drift_parameters=params,
    )


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
        # After iteration, the outlier's offsets and drift_fit are
        # restored to pass-1 (iteration only updates neighbours).
        np.testing.assert_array_equal(
            sd.offsets.isel(depth=idx).values,
            sd.offsets_pass1.isel(depth=idx).values,
        )
        np.testing.assert_array_equal(
            sd.drift_fit.isel(depth=idx).values,
            sd.drift_fit_pass1.isel(depth=idx).values,
        )

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
        # Test the iteration mechanic in isolation by forcing exactly
        # the prescribed sensor as the outlier (manual override, with
        # auto-flag effectively disabled via a high threshold).
        # The auto-flag heuristic itself is exercised by other tests;
        # here the question is whether iteration improves the
        # neighbours' fits when given a known outlier.
        l1_dir, drift_sn = synthetic_l1_dir_with_drift
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=999.0,
                manual_outlier_sns=[drift_sn],
                fit_mode="linear",
            ),
        )
        assert sd.flagged_outlier_sns == [drift_sn]
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

    def test_outlier_drift_fit_preserved_at_pass1(
        self, synthetic_l1_dir_with_drift
    ):
        # The outlier's drift_fit must equal pass-1: pass-2 at the
        # outlier is the residual after pass-1 was subtracted, which
        # if saved would zero out the L2 correction for the very
        # sensor we're trying to correct.
        l1_dir, drift_sn = synthetic_l1_dir_with_drift
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=999.0,
                manual_outlier_sns=[drift_sn],
                fit_mode="linear",
            ),
        )
        sn_arr = sd.drift_fit.sn.values
        drift_idx = int(np.where(sn_arr == drift_sn)[0][0])
        np.testing.assert_array_equal(
            sd.drift_fit.isel(depth=drift_idx).values,
            sd.drift_fit_pass1.isel(depth=drift_idx).values,
        )
        np.testing.assert_array_equal(
            sd.drift_linfit.isel(depth=drift_idx).values,
            sd.drift_linfit_pass1.isel(depth=drift_idx).values,
        )
        # Also: offsets and offsets_clean at outlier are restored so
        # plot_drift_sensor_and_neighbors shows the original signal.
        np.testing.assert_array_equal(
            sd.offsets.isel(depth=drift_idx).values,
            sd.offsets_pass1.isel(depth=drift_idx).values,
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
        l1_dir, drift_sn = synthetic_l1_dir_with_drift
        # Force the flag via manual_outlier_sns so the provenance assertions
        # are independent of the recovered amplitude (under the two-step
        # default the drifter sits just below the 1.5 mK auto-threshold).
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=999.0,
                manual_outlier_sns=[drift_sn],
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
        l1_dir, drift_sn = synthetic_l1_dir_with_drift
        # Force the flag via manual_outlier_sns (see note above) so the
        # round-tripped iteration_count is deterministically 1.
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=999.0,
                manual_outlier_sns=[drift_sn],
                fit_mode="linear",
            ),
        )
        sd.drift_to_netcdf(path=tmp_path, suffix="iter")
        reloaded = xr.open_dataarray(tmp_path / "drift_synthetic_iter.nc")
        assert int(reloaded.attrs["iteration_count"]) == 1
        flagged = np.asarray(reloaded.attrs["flagged_outlier_sns"])
        assert flagged.size >= 1
        assert flagged.dtype.kind == "i"


class TestIterateModeValidation:
    def test_invalid_mode_raises(self, synthetic_l1_dir):
        with pytest.raises(ValueError, match="iterate_mode"):
            sensor_drift(
                mooring_name="synthetic",
                l1_grid_dir=synthetic_l1_dir,
                run_all=True,
                drift_parameters=dict(
                    iterate_subtract=True,
                    fit_mode="linear",
                    iterate_mode="nope",
                ),
            )

    def test_default_mode_is_restore(self, synthetic_l1_dir):
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            run_all=True,
            drift_parameters=dict(iterate_subtract=True, fit_mode="linear"),
        )
        assert sd.iterate_mode == "restore"


class TestRestoreBackwardCompat:
    def test_restore_matches_pinned_baseline(self, synthetic_l1_dir_with_drift):
        # Legacy-path guard: restore-mode drift_fit on the single-pass
        # shared-component path (two_step_shared=False) must reproduce the
        # baseline captured from the pre-iterate_mode implementation. This
        # pins the legacy behaviour bit-for-bit now that two-step is the
        # default; the two-step default path is covered by test_two_step_shared
        # and the refit design tests.
        l1_dir, drift_sn = synthetic_l1_dir_with_drift
        sd = _run_iter(l1_dir, drift_sn, "restore", two_step_shared=False)
        baseline = xr.open_dataarray(BASELINE)
        xr.testing.assert_allclose(sd.drift_fit, baseline)


class TestRefitMode:
    def test_refit_changes_flagged_sensor_only(self, synthetic_l1_dir_with_drift):
        l1_dir, drift_sn = synthetic_l1_dir_with_drift
        restore = _run_iter(l1_dir, drift_sn, "restore")
        refit = _run_iter(l1_dir, drift_sn, "refit")
        sn_arr = refit.drift_fit.sn.values
        flag_idx = int(np.where(sn_arr == drift_sn)[0][0])
        # The flagged sensor's drift differs between the two modes.
        assert not np.allclose(
            refit.drift_fit.isel(depth=flag_idx).values,
            restore.drift_fit.isel(depth=flag_idx).values,
        )
        # Every non-flagged sensor is bitwise identical: their offsets_clean
        # is untouched, so re-fitting yields the same pass-2 drift.
        for di in range(len(sn_arr)):
            if di == flag_idx:
                continue
            np.testing.assert_array_equal(
                refit.drift_fit.isel(depth=di).values,
                restore.drift_fit.isel(depth=di).values,
            )

    def test_refit_recovers_more_drift_at_flagged(
        self, synthetic_l1_dir_with_drift
    ):
        # Load-bearing design test. In restore mode the flagged sensor's own
        # drift contaminates the shared component it is differenced against,
        # so pass-1 under-recovers the planted ramp. refit recomputes the
        # cleaned offsets against the *pass-2* shared component (computed
        # after the drift was subtracted), recovering a larger fraction of
        # the true drift amplitude.
        l1_dir, drift_sn = synthetic_l1_dir_with_drift
        restore = _run_iter(l1_dir, drift_sn, "restore")
        refit = _run_iter(l1_dir, drift_sn, "refit")
        flag_idx = int(np.where(refit.drift_fit.sn.values == drift_sn)[0][0])
        amp_restore = _amp(restore.drift_fit.isel(depth=flag_idx))
        amp_refit = _amp(refit.drift_fit.isel(depth=flag_idx))
        assert amp_refit > amp_restore

    def test_refit_no_op_when_nothing_flagged(self, synthetic_l1_dir):
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=True,
                amplitude_threshold_mK=50.0,
                fit_mode="linear",
                iterate_mode="refit",
            ),
        )
        assert sd.flagged_outlier_sns == []
        assert sd.iteration_count == 0
