"""Unit tests for thermodrift.io fit primitives."""

import numpy as np
import pytest

from thermodrift.io import calculate_r2, exp_function

from _synthetic import cvhg16_eq5


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

    @pytest.mark.xfail(
        strict=True,
        reason="refactor step 5.2(a) — exp_function does not yet match Eq. 5 for β ≠ 1",
    )
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
