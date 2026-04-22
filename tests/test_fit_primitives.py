"""Unit tests for thermodrift.io fit primitives."""

import functools

import numpy as np
import pytest
import xarray as xr

from thermodrift.io import (
    calculate_r2,
    exp_function,
    expfit_ufunc,
    lin_or_exp,
    linfit_ufunc,
)

from _synthetic import cvhg16_eq5, noisy_drift


class TestCalculateR2:
    """Protection tests — pre- and post-refactor."""

    def test_perfect_fit_returns_one(self):
        x = np.linspace(0.0, 1.0, 50)
        assert calculate_r2(x, x) == pytest.approx(1.0)

    def test_mean_prediction_returns_zero(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        xfit = np.full_like(x, x.mean())
        assert calculate_r2(x, xfit) == pytest.approx(0.0)

    def test_worse_than_mean_returns_negative(self):
        x = np.array([1.0, 2.0, 3.0])
        xfit = np.array([3.0, 2.0, 1.0])  # anti-correlated
        assert calculate_r2(x, xfit) < 0


class TestExpFunctionMatchesEq5:
    """Target-behaviour tests for refactor step 5.2(a).

    The current thermodrift.io.exp_function uses
    ``gammainc(1/β, t/τ)`` without the ``/β`` prefactor. This matches
    Eq. 5 only at β = 1. These tests pin the rewrite to the paper.
    """

    @pytest.fixture
    def time_days(self):
        return np.linspace(0.0, 100.0, 200)

    @pytest.fixture
    def params(self):
        # (t0, m, A) chosen to keep all three terms numerically visible.
        return dict(t0=0.01, m=1e-5, A=1e-3)

    def test_beta_one_matches_reference(self, time_days, params):
        # β = 1: γ(1, x) = 1 − e^{−x}, so both forms collapse to the same
        # expression. This passes against current code AND the rewrite.
        ref = cvhg16_eq5(time_days, beta=1.0, tau=20.0, **params)
        got = exp_function(time_days, **params, beta=1.0, tau=20.0)
        np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("beta", [0.5, 1.5, 2.0, 2.5])
    def test_beta_not_one_matches_reference(self, time_days, params, beta):
        ref = cvhg16_eq5(time_days, beta=beta, tau=20.0, **params)
        got = exp_function(time_days, **params, beta=beta, tau=20.0)
        np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-10)

    def test_at_t_zero_returns_t0(self, params):
        # γ(1/β, 0) = 0 for all β > 0, so exp_function(0, …) = t0.
        for beta in (0.5, 1.0, 2.0):
            val = exp_function(np.array([0.0]), **params, beta=beta, tau=20.0)
            assert val[0] == pytest.approx(params["t0"])

    def test_saturation_limit_beta_one(self, params):
        # β = 1 saturates to t0 + m*t + A as t → ∞.
        t = np.array([1e4])  # t/τ = 500 ⇒ fully saturated
        val = exp_function(t, **params, beta=1.0, tau=20.0)
        expected = params["t0"] + params["m"] * t[0] + params["A"]
        assert val[0] == pytest.approx(expected, rel=1e-6)


class TestExpfitUfuncRecovery:
    """Target-behaviour tests for refactor step 5.2(b).

    Depend on the post-refactor signature
    ``expfit_ufunc(x, tau0=20.0, tau_bounds=(5.0, 180.0), ...)``.
    """

    def test_recovers_slow_tau_parameters(self):
        # Slow relaxation — the MOTIVE use case.
        n = 120  # ~4 months of 1-d windows
        _, x = noisy_drift(
            n_windows=n,
            t0=0.0, m=1e-5, A=1e-3, beta=1.0, tau=20.0,
            noise_std=2e-4, seed=1,
        )
        fit = expfit_ufunc(x, tau0=20.0, tau_bounds=(5.0, 180.0))
        assert np.std(x - fit) < 4e-4

    def test_recovers_fast_tau_parameters(self):
        # NIOZ-style fast relaxation — keeps old behaviour reachable.
        n = 90
        _, x = noisy_drift(
            n_windows=n,
            t0=0.0, m=1e-5, A=2e-3, beta=1.0, tau=2.0,
            noise_std=1e-4, seed=2,
        )
        fit = expfit_ufunc(x, tau0=2.0, tau_bounds=(0.5, 30.0))
        assert np.std(x - fit) < 3e-4

    def test_returns_nan_on_all_nan_input(self):
        x = np.full(50, np.nan)
        fit = expfit_ufunc(x, tau0=20.0)
        assert np.all(np.isnan(fit))

    def test_preserves_length(self):
        _, x = noisy_drift(
            n_windows=100, t0=0.0, m=1e-5, A=1e-3, beta=1.0, tau=20.0,
            noise_std=2e-4, seed=3,
        )
        fit = expfit_ufunc(x, tau0=20.0)
        assert fit.shape == x.shape

    def test_preserves_length_with_nans(self):
        # NaNs in input — output length matches; NaN policy pinned by
        # test_runs_through_apply_ufunc below.
        _, x = noisy_drift(
            n_windows=100, t0=0.0, m=1e-5, A=1e-3, beta=1.0, tau=20.0,
            noise_std=2e-4, seed=4,
        )
        x[::11] = np.nan
        fit = expfit_ufunc(x, tau0=20.0)
        assert fit.shape == x.shape

    @pytest.mark.parametrize(
        "tau0, tau_bounds",
        [
            (20.0, (5.0, 180.0)),
            (2.0, (0.5, 30.0)),
        ],
    )
    def test_runs_through_apply_ufunc(self, tau0, tau_bounds):
        # This is how sensor_drift actually calls it. Exercise the
        # xr.apply_ufunc path to make sure lambda/partial wiring works.
        rng = np.random.default_rng(5)
        arr = np.stack(
            [
                cvhg16_eq5(np.arange(100), 0.0, 1e-5, 1e-3, 1.0, tau0)
                + rng.standard_normal(100) * 2e-4
                for _ in range(3)
            ],
            axis=1,
        )
        da = xr.DataArray(arr, dims=("window", "depth"))
        fitter = functools.partial(
            expfit_ufunc, tau0=tau0, tau_bounds=tau_bounds,
        )
        out = xr.apply_ufunc(
            fitter, da,
            input_core_dims=[["window"]],
            output_core_dims=[["window"]],
            vectorize=True,
        )
        # apply_ufunc moves core dims to the end, so shape may reorder;
        # sizes is dim-keyed and order-independent.
        assert out.sizes == da.sizes
        assert not np.all(np.isnan(out.values))


class TestLinfitUfunc:
    """Protection tests — pre- and post-consolidation.

    Pins the current behaviour that the fit is evaluated at every index
    of the input (no NaN gaps). This is the property the refactor should
    preserve when collapsing ``linfit`` / ``linfit_ufunc``.
    """

    def test_perfect_linear_recovered(self):
        n = 50
        x = 0.5 + 0.01 * np.arange(n)
        fit = linfit_ufunc(x)
        np.testing.assert_allclose(fit, x, atol=1e-12)

    def test_preserves_length(self):
        x = np.arange(25, dtype=float)
        fit = linfit_ufunc(x)
        assert fit.shape == x.shape

    def test_handles_internal_nans(self):
        # Input is exactly linear with a few NaN gaps; the fit of the
        # remaining points recovers the same line, and linfit_ufunc
        # evaluates that line at every index → output matches input
        # at non-NaN positions to within numerical precision.
        x = 0.5 + 0.01 * np.arange(30)
        x[5:8] = np.nan
        fit = linfit_ufunc(x)
        assert fit.shape == x.shape
        good = ~np.isnan(x)
        np.testing.assert_allclose(fit[good], x[good], atol=1e-12)


class TestLinOrExp:
    """Protection tests — CvHG16 algorithm step 5 selector."""

    @pytest.fixture
    def time_axis(self):
        return np.arange(100, dtype=float)

    def test_pure_linear_picks_linear(self, time_axis):
        # Signal is almost exactly linear; both fits are near-identical, so
        # the R² selector has no reason to prefer exp.
        rng = np.random.default_rng(10)
        x = xr.DataArray(
            1e-3 * time_axis + 5e-5 * rng.standard_normal(time_axis.size),
            dims=("window",),
        )
        xlin = 1e-3 * time_axis
        xexp = xlin + 1e-6  # indistinguishable from xlin
        assert lin_or_exp(x, xlin, xexp, return_type=True) == "lin"

    def test_clear_exponential_picks_exp(self, time_axis):
        # Signal is an early-saturating exponential; linear fits it poorly,
        # the exp fit is exact → selector picks exp.
        signal = cvhg16_eq5(time_axis, 0.0, 0.0, 1e-3, 1.0, 10.0)
        x = xr.DataArray(signal, dims=("window",))
        coeffs = np.polyfit(time_axis, signal, 1)
        xlin = np.polyval(coeffs, time_axis)
        xexp = signal
        assert lin_or_exp(x, xlin, xexp, return_type=True) == "exp"

    def test_threshold_formula(self, time_axis):
        # CvHG16 selector: pick exp iff R_exp > R_lin + 0.3 * (1 - R_lin),
        # where R = sqrt(calculate_r2(...)). Construct xlin/xexp with known
        # R² by controlling residual variance: R² = 1 - Var(residual)/Var(x).
        rng = np.random.default_rng(11)
        base_arr = rng.standard_normal(time_axis.size)
        base_arr -= base_arr.mean()
        var_base = base_arr.var()
        x = xr.DataArray(base_arr, dims=("window",))

        def prediction_with_r2(target_r2):
            noise = rng.standard_normal(base_arr.size)
            noise -= noise.mean()
            noise *= np.sqrt((1 - target_r2) * var_base / noise.var())
            return base_arr + noise

        # R_lin = 0.5 → threshold = 0.5 + 0.3 * 0.5 = 0.65.
        xlin = prediction_with_r2(0.25)
        xexp_above = prediction_with_r2(0.49)  # R_exp = 0.70 > 0.65
        xexp_below = prediction_with_r2(0.36)  # R_exp = 0.60 < 0.65

        # Sanity-check the construction so failures surface as "fixture
        # drifted" rather than "selector is wrong".
        assert calculate_r2(base_arr, xlin) == pytest.approx(0.25, abs=0.03)
        assert calculate_r2(base_arr, xexp_above) == pytest.approx(0.49, abs=0.03)
        assert calculate_r2(base_arr, xexp_below) == pytest.approx(0.36, abs=0.03)

        assert lin_or_exp(x, xlin, xexp_above, return_type=True) == "exp"
        assert lin_or_exp(x, xlin, xexp_below, return_type=True) == "lin"
