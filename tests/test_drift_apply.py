import numpy as np
import pytest

from thermochain.io import evaluate_drift_model, exp_function


def test_evaluate_linear_is_exact_line():
    idx = np.array([-0.5, 0.0, 1.0, 2.5, 10.0])
    slope, intercept = 0.002, -0.01
    out = evaluate_drift_model(idx, "lin", (slope, intercept), None)
    np.testing.assert_allclose(out, intercept + slope * idx)


def test_evaluate_exp_matches_exp_function_for_nonneg_index():
    idx = np.array([0.0, 0.3, 1.0, 5.0, 30.0])
    params = (0.01, 1e-5, -0.05, 1.5, 20.0)  # t0, m, A, beta, tau
    out = evaluate_drift_model(idx, "exp", None, params)
    np.testing.assert_allclose(out, exp_function(idx, *params))


def test_evaluate_exp_negative_index_is_finite_linear_part():
    idx = np.array([-0.25, -0.1])
    t0, m, A, beta, tau = 0.01, 1e-5, -0.05, 1.5, 20.0
    out = evaluate_drift_model(idx, "exp", None, (t0, m, A, beta, tau))
    # gamma term held at 0 for idx < 0 -> drift = t0 + m*idx, finite & real
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, t0 + m * idx)
