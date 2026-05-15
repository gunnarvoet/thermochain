"""End-to-end smoke test for thermodrift.io.sensor_drift.

Goal: exercise the full call graph on synthetic data so the refactor
cannot silently break wiring. Does NOT assert numerical drift values.
"""

import numpy as np
import pytest
import tqdm

from thermodrift.io import sensor_drift


# Pre-existing deprecation noise from thermodrift.io that refactor phase 1
# will clean up: find_outliers' z.argmax() without dim= (DeprecationWarning),
# and an xarray GroupBy.apply call (PendingDeprecationWarning). Ignore both
# so the smoke test stays focused on pipeline wiring.
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


@pytest.fixture(autouse=True)
def _plain_tqdm(monkeypatch):
    # thermodrift.io hardcodes tqdm.tqdm_notebook, which needs ipywidgets.
    # Swap it for the plain text bar so tests don't require a Jupyter env.
    # Refactor phase 1 will replace this with tqdm.auto.
    monkeypatch.setattr(tqdm, "tqdm_notebook", tqdm.tqdm, raising=False)


class TestSensorDriftSmoke:
    @pytest.mark.parametrize("iterate_subtract", [False, True])
    def test_constructs_and_runs_full_pipeline(
        self, synthetic_l1_dir, iterate_subtract
    ):
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            run_all=True,
            drift_parameters=dict(
                iterate_subtract=iterate_subtract,
                amplitude_threshold_mK=50.0,
            ),
        )
        # Step-by-step state is populated.
        assert hasattr(sd, "offsets_initial")
        assert hasattr(sd, "offsets")
        assert hasattr(sd, "first_guess_shared_fluct_comp")
        assert hasattr(sd, "offsets_second_guess")
        assert hasattr(sd, "second_guess_linfit")
        assert hasattr(sd, "second_guess_expfit")
        assert hasattr(sd, "offsets_clean")
        assert hasattr(sd, "drift_linfit")
        # Final drift has window + depth axes matching the cleaned offsets.
        # drift_linfit comes out (depth, window); offsets_clean is
        # (window, depth). Use .sizes so the comparison is dim-aware.
        assert sd.drift_linfit.sizes == sd.offsets_clean.sizes

    def test_parses_exclude_sn(self, synthetic_l1_dir):
        # exclude_sn only removes listed SNs from the background FIT
        # (offsets_from_background_fit applies xn * xn2) — the returned
        # offsets DataArray still spans every sensor. Assert only that
        # the kwarg is parsed onto the instance.
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            drift_parameters=dict(exclude_sn=[72100]),
            run_all=False,
        )
        assert sd.exclude_sn == [72100]
        assert 72100 in sd.offsets.sn.values

    def test_file_pattern_filters_grid_dir(self, synthetic_l1_dir_mixed):
        # Real MOTIVE_B grid dir holds both `motive_b_deep_L1_*.nc` and
        # `motive_b_shallow_L1_*.nc`; sensor_drift must accept a glob to
        # narrow the load to a single segment.
        sd = sensor_drift(
            mooring_name="motive_b",
            l1_grid_dir=synthetic_l1_dir_mixed,
            file_pattern="motive_b_deep_L1_*.nc",
            run_all=False,
        )
        names = [p.name for p in sd.files_gridded_level1]
        assert len(names) == 4
        assert all(n.startswith("motive_b_deep_L1_") for n in names)

    def test_default_window_length_is_one_day(self, synthetic_l1_dir):
        # synthetic_l1_dir spans 12 days at 1-min cadence; default
        # window_length=1D should produce 12 uniform windows.
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            run_all=False,
        )
        assert sd.offsets_initial.sizes["window"] == 12

    def test_window_length_two_days_straddles_files(self, synthetic_l1_dir):
        # synthetic_l1_dir's 3-day files don't align with 2-day windows,
        # so this also exercises the cross-file accumulation path.
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            window_length=np.timedelta64(2, "D"),
            run_all=False,
        )
        assert sd.offsets_initial.sizes["window"] == 6

    def test_window_length_rejects_non_positive(self, synthetic_l1_dir):
        with pytest.raises(ValueError, match="window_length must be positive"):
            sensor_drift(
                mooring_name="synthetic",
                l1_grid_dir=synthetic_l1_dir,
                window_length=np.timedelta64(0, "D"),
                run_all=False,
            )


class TestSensorDriftFitMode:
    """Mixed protection + target-behaviour — refactor step 5.2(d)."""

    @pytest.mark.parametrize("fit_mode", ["linear", "auto"])
    def test_drift_fit_shape_matches_offsets(self, synthetic_l1_dir, fit_mode):
        # Protection: drift_fit is populated with matching shape today
        # (current code ignores fit_mode and always uses linfit). After
        # the refactor, both modes still produce drift_fit of the same
        # shape.
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            drift_parameters=dict(fit_mode=fit_mode),
            run_all=True,
        )
        assert hasattr(sd, "drift_fit")
        assert sd.drift_fit.sizes == sd.offsets_clean.sizes

    def test_auto_mode_populates_fit_type(self, synthetic_l1_dir):
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            drift_parameters=dict(fit_mode="auto"),
            run_all=True,
        )
        assert hasattr(sd, "fit_type")
        assert set(np.unique(sd.fit_type)).issubset({"lin", "exp"})

    @pytest.mark.parametrize("fit_mode", ["auto", "exp"])
    def test_drift_exp_params_shape_and_coord(self, synthetic_l1_dir, fit_mode):
        # drift_exp_params backs the τ / β annotations on
        # plot_drift_sensor_and_neighbors and lets downstream notebooks
        # inspect the per-sensor exponential parameters.
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            drift_parameters=dict(fit_mode=fit_mode),
            run_all=True,
        )
        assert hasattr(sd, "drift_exp_params")
        assert sd.drift_exp_params.sizes["depth"] == sd.offsets_clean.sizes["depth"]
        assert sd.drift_exp_params.sizes["param"] == 5
        assert list(sd.drift_exp_params.param.values) == [
            "t0", "m", "A", "beta", "tau",
        ]

    def test_linear_mode_skips_exp_params(self, synthetic_l1_dir):
        # In linear mode no exponential fit is attempted, so the param
        # array is not populated — plot_drift_sensor_and_neighbors falls
        # back gracefully via the hasattr guard.
        sd = sensor_drift(
            mooring_name="synthetic",
            l1_grid_dir=synthetic_l1_dir,
            drift_parameters=dict(fit_mode="linear"),
            run_all=True,
        )
        assert not hasattr(sd, "drift_exp_params")
