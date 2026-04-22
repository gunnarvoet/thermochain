"""Reference implementation of CvHG16 Eq. 5 for synthetic drift test data.

This is intentionally independent of thermodrift.io.exp_function so that
test_fit_primitives can use it as ground truth.

Reference: Cimatoribus, van Haren, Gostiaux 2016, JTECH,
doi:10.1175/JTECH-D-15-0243.1, Eq. 5.
"""

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import gammainc


def cvhg16_eq5(t, t0, m, A, beta, tau):
    """Analytical drift from CvHG16 Eq. 5.

    ΔT(t) = ΔT_0 + m * t + A * γ(1/β, (t/τ)^β) / β

    where γ is the lower incomplete gamma function, related to
    scipy's regularised gammainc by γ(a, x) = Γ(a) * gammainc(a, x).

    Parameters
    ----------
    t : array_like
        Time (days, matching the 1-day window cadence).
    t0 : float
        Systematic offset ΔT_0.
    m : float
        Long-term (asymptotic) drift rate.
    A : float
        Relaxation amplitude (= a·τ in the paper).
    beta : float
        Stretch exponent.
    tau : float
        Relaxation time constant (days).
    """
    t = np.asarray(t, dtype=float)
    a = 1.0 / beta
    x = (t / tau) ** beta
    lower_gamma = gamma_fn(a) * gammainc(a, x)
    return t0 + m * t + A * lower_gamma / beta


def noisy_drift(n_windows, t0, m, A, beta, tau, noise_std, seed=0):
    """CvHG16 drift plus white Gaussian noise on a 1-d-per-window grid."""
    t = np.arange(n_windows, dtype=float)
    signal = cvhg16_eq5(t, t0, m, A, beta, tau)
    rng = np.random.default_rng(seed)
    return t, signal + noise_std * rng.standard_normal(n_windows)
