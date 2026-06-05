# src/thermodrift/pipeline.py
"""Config-driven Mooring pipeline orchestration over thermodrift primitives."""

from pathlib import Path  # noqa: F401

import numpy as np  # noqa: F401
import pandas as pd

from .io import ProcessThermistorMooring, grid_thermistors  # noqa: F401

_GRIDDING_KEYS = {"dt", "max_gap", "chunk"}


def parse_gridding(block, defaults=None):
    """Validate a gridding block and convert dt/max_gap/chunk to np.timedelta64.

    Parameters
    ----------
    block : dict or None
        Per-segment gridding overrides (pandas-style duration strings).
    defaults : dict or None, optional
        Top-level gridding defaults; ``block`` keys override these.

    Returns
    -------
    dict
        Keys among {dt, max_gap, chunk} present after merge, as np.timedelta64.

    Raises
    ------
    ValueError
        If ``block`` contains a key outside {dt, max_gap, chunk}.
    """
    block = dict(block or {})
    unknown = set(block) - _GRIDDING_KEYS
    if unknown:
        raise ValueError(f"unknown gridding keys: {sorted(unknown)}")
    merged = dict(defaults or {})
    merged.update(block)
    return {k: pd.Timedelta(v).to_timedelta64() for k, v in merged.items() if k in _GRIDDING_KEYS}


def resolve_segment_sns(mooring_info, select, root):
    """Return a pandas.Index of SNs for one segment's ``select`` spec.

    ``select`` accepts exactly one of:
      - ``{"segment": name}`` — match the ``segment`` column (PRIMARY).
      - ``{"sns": [..]}``      — explicit serial-number list (fallback).
      - ``{"layout": file}``   — CSV with an ``SN`` column (fallback/migration),
        resolved relative to ``root``.

    Selection is structural only; quality drops (``ignore_sns``) are applied
    separately by the consuming stage.

    Parameters
    ----------
    mooring_info : pd.DataFrame
        Mooring sensor sheet with SN as the index.
    select : dict
        Selection spec with exactly one of the keys above.
    root : str or Path
        Root directory for resolving relative ``layout`` CSV paths.

    Returns
    -------
    pd.Index
        Index of integer serial numbers matching the selection.

    Raises
    ------
    ValueError
        If ``select`` uses ``segment`` but the mooring sheet has no ``segment`` column.
    ValueError
        If ``select`` contains none of the recognised keys.
    """
    select = dict(select or {})
    if "segment" in select:
        if "segment" not in mooring_info.columns:
            raise ValueError("select.segment requires a 'segment' column in the mooring sheet")
        return mooring_info.index[mooring_info["segment"] == select["segment"]]
    if "sns" in select:
        return pd.Index([int(s) for s in select["sns"]], name="SN")
    if "layout" in select:
        layout_path = Path(root).joinpath(select["layout"])
        return pd.Index([int(s) for s in pd.read_csv(layout_path)["SN"]], name="SN")
    raise ValueError(f"segment select needs one of segment/sns/layout: {select!r}")
