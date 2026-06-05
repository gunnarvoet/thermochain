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


class Mooring(ProcessThermistorMooring):
    """Config-driven moored-thermistor pipeline for one mooring.

    Extends the L0 processing base with segment resolution and the
    gridding stage. Built from a single YAML config (see the schema in
    plans/generalize-thermistor-pipeline.md).
    """

    def __init__(self, config_file, project_root=None, data_root=None):
        """Initialize the Mooring pipeline.

        Parameters
        ----------
        config_file : pathlib.Path or str
            YAML config file for this mooring.
        project_root : pathlib.Path or None, optional
            Forwarded to ``load_config_box``; defaults to
            ``configfile.parent.parent``.
        data_root : pathlib.Path or None, optional
            Forwarded to ``load_config_box``.
        """
        super().__init__(config_file, project_root=project_root, data_root=data_root)
        self.segments_cfg = {
            name: dict(seg)
            for name, seg in (self.cfg.get("segments", {}) or {}).items()
        }
        default_grid = dict(self.cfg.get("gridding", {}) or {})
        self.gridding = {
            name: parse_gridding(seg.get("gridding"), defaults=default_grid)
            for name, seg in self.segments_cfg.items()
        }

    def _segment_names(self, segments):
        """Validate and return a list of segment names.

        Parameters
        ----------
        segments : str, list of str, or None
            Segment(s) to validate. ``None`` returns all defined segments.

        Returns
        -------
        list of str
        """
        if segments is None:
            return list(self.segments_cfg)
        if isinstance(segments, str):
            segments = [segments]
        for s in segments:
            if s not in self.segments_cfg:
                raise KeyError(f"unknown segment {s!r}; defined: {list(self.segments_cfg)}")
        return list(segments)

    def _ignore_sns(self):
        """Return the configured ignore_sns list as integers.

        Returns
        -------
        list of int
        """
        return [int(s) for s in (self.cfg.get("ignore_sns", []) or [])]

    def segment_sensors(self, segment):
        """Return mooring_info rows for one segment (structural selection only).

        Parameters
        ----------
        segment : str
            Segment name as defined in the config ``segments`` block.

        Returns
        -------
        pd.DataFrame
            Subset of ``mooring_info`` whose SNs belong to ``segment``.
        """
        sns = resolve_segment_sns(
            self.mooring_info,
            self.segments_cfg[segment]["select"],
            self.cfg.path.root,
        )
        return self.mooring_info.loc[self.mooring_info.index.intersection(sns)]
