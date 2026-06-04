"""Reference implementation of CvHG16 Eq. 5 for synthetic drift test data.

This is intentionally independent of thermodrift.io.exp_function so that
test_fit_primitives can use it as ground truth.

Reference: Cimatoribus, van Haren, Gostiaux 2016, JTECH,
doi:10.1175/JTECH-D-15-0243.1, Eq. 5.
"""

import numpy as np
import xarray as xr
from scipy.special import gamma as gamma_fn
from scipy.special import gammainc


def write_drift_l1_files(directory, drift_index=6, drift_total_C=5e-3):
    """Write a synthetic L1-gridded dataset with a strong linear drift on
    one interior sensor, mirroring the MOTIVE A 236127 case.

    Four 3-day netCDF files, 1-minute cadence, 12 depths / sn, plus a
    ``drift_total_C`` linear ramp planted on the ``drift_index`` sensor.
    Deterministic (per-file seeded RNG) so outputs are reproducible and a
    restore-mode baseline can be pinned.

    Returns the serial number of the drifting sensor.
    """
    n_depth = 12
    sn = np.array([72100 + i for i in range(n_depth)])
    depth = np.linspace(1000.0, 1200.0, n_depth)
    t_mean = 4.0 + 6.0 * np.exp(-(depth - 1000.0) / 200.0)

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
        arr = t_mean[None, :] + 0.02 * rng.standard_normal((times.size, sn.size))
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
        path = directory / f"mavs0_gridded_2024-01-{start:02d}_to_2024-01-{end:02d}.nc"
        _build_file(start, end).to_netcdf(path)
    return int(sn[drift_index])


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
