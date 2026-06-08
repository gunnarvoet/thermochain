# coding: utf-8
"""Plotting helpers for thermochain datasets.

This module currently holds quick-look plotting helpers for gridded
thermistor records. Drift-diagnostic plot routines defined alongside
`thermochain.io.sensor_drift` are slated to consolidate here in a future
module-restructuring pass.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from ._log import log

logger = log()


def axstyle(ax=None, fontsize=10, grid=False):
    """Minimal house style: hide top/right spines, set tick/label fontsize."""
    if ax is None:
        ax = plt.gca()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    if grid:
        ax.grid(color="0.5", linewidth=0.25, alpha=0.8)
    return ax


def quickfig(w=6, h=4, r=1, c=1, fs=10, grid=False, yi=True, **kwargs):
    """Quick figure with the house axis style applied.

    With ``yi=False`` the y-axis is inverted (handy for plotting against
    depth). Mirrors the subset of ``gvpy.plot.quickfig`` used here.
    """
    fig, ax = plt.subplots(r, c, figsize=(w, h), constrained_layout=True, **kwargs)
    axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
    for a in axes:
        axstyle(a, fontsize=fs, grid=grid)
    if yi is False and not isinstance(ax, np.ndarray):
        ax.invert_yaxis()
    return fig, ax


def concise_date(ax=None, minticks=3, maxticks=10, show_offset=True, **kwargs):
    """Concise date ticks on the x-axis via matplotlib's ConciseDateFormatter."""
    if ax is None:
        ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=minticks, maxticks=maxticks)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator, show_offset=show_offset, **kwargs)
    )
    if ax.get_xlabel() == "time":
        ax.set_xlabel("")


def annotate_corner(text, ax, quadrant=1, fw="bold", fs=10, col="k",
                    background_circle=False, **kwargs):
    """Annotate a corner of ``ax`` (quadrants 1-4), optional circle backdrop."""
    locs = {1: ((0.02, 0.9), "left"), 2: ((0.02, 0.1), "left"),
            3: ((0.98, 0.1), "right"), 4: ((0.98, 0.9), "right")}
    loc, ha = locs[quadrant]
    bbox = None
    if background_circle:
        c = "w" if background_circle is True else background_circle
        bbox = dict(boxstyle="circle", edgecolor=c, facecolor=c)
    return ax.annotate(text, loc, xycoords="axes fraction", fontweight=fw,
                       fontsize=fs, color=col, ha=ha, bbox=bbox, **kwargs)


def save_png(name, figdir, dpi=300):
    """Save the current figure to ``figdir/name.png`` (dir created as needed)."""
    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figdir / f"{name}.png", dpi=dpi, bbox_inches="tight",
                facecolor="w", edgecolor="none")


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
    fig, ax = quickfig(w=10)
    # We take the derivative of negative temperature here because the depth
    # coordinate is positive but really should be negative subsurface. This way
    # we get the sign of the gradient right.
    dtdz = (-1 * t).differentiate(coord="depth")
    # dtdz.attrs["long_name"] = r"$d\\Theta/dz$"
    dtdz.attrs["unit"] = "K/m"
    dtdz.depth.attrs["unit"] = "m"
    dtdz.plot(robust=True, cmap="RdBu_r", **opts)
    ax.invert_yaxis()
    concise_date(ax)
    return ax
