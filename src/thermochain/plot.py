# coding: utf-8
"""Plotting helpers for thermochain datasets.

This module currently holds quick-look plotting helpers for gridded
thermistor records. Drift-diagnostic plot routines defined alongside
`thermochain.io.sensor_drift` are slated to consolidate here in a future
module-restructuring pass.
"""

import gvpy as gv


logger = gv.misc.log()


def plot_dtdz(t):
    """Plot the vertical temperature gradient of a gridded thermistor record.

    Computes ``dT/dz`` from a gridded ``(depth, time)`` DataArray and
    renders it as a ``robust``-clipped pcolormesh with a diverging
    colormap and a concise date axis. The derivative is taken against
    ``-T`` rather than ``T`` because the gridded depth coordinate is
    positive downward, so the returned gradient has the natural sign
    (positive in a stably stratified ocean).

    Parameters
    ----------
    t : xr.DataArray
        Gridded thermistor temperature with dims ``(depth, time)``.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the plot, with the depth axis inverted so that
        the surface is at the top.
    """
    opts = dict(cbar_kwargs=dict(shrink=0.7, aspect=30))
    fig, ax = gv.plot.quickfig(w=10)
    # We take the derivative of negative temperature here because the depth
    # coordinate is positive but really should be negative subsurface. This way
    # we get the sign of the gradient right.
    dtdz = (-1 * t).differentiate(coord="depth")
    # dtdz.attrs["long_name"] = r"$d\\Theta/dz$"
    dtdz.attrs["unit"] = "K/m"
    dtdz.depth.attrs["unit"] = "m"
    dtdz.plot(robust=True, cmap="RdBu_r", **opts)
    ax.invert_yaxis()
    gv.plot.concise_date(ax)
    return ax
