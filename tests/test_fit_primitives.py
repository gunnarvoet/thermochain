"""Unit tests for thermodrift.io fit primitives."""

import numpy as np
import pytest

from thermodrift.io import calculate_r2


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
