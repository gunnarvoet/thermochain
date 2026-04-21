# coding: utf-8
"""
Plotting functions.
"""

from pathlib import Path
import tqdm
import yaml
from box import Box
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
import rbrmoored
import sbemoored
import mixsea as mx
import gvpy as gv


logger = gv.misc.log()


def plot_dtdz(t):
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
