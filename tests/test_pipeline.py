# tests/test_pipeline.py
import numpy as np
import pytest

from thermodrift.pipeline import parse_gridding


def test_parse_gridding_converts_strings_to_timedelta64():
    out = parse_gridding({"dt": "2s", "max_gap": "10s", "chunk": "2D"})
    assert out["dt"] == np.timedelta64(2, "s")
    assert out["max_gap"] == np.timedelta64(10, "s")
    assert out["chunk"] == np.timedelta64(2, "D")


def test_parse_gridding_merges_defaults():
    out = parse_gridding({"dt": "2s"}, defaults={"dt": "10s", "max_gap": "30s", "chunk": "2D"})
    assert out["dt"] == np.timedelta64(2, "s")        # override wins
    assert out["max_gap"] == np.timedelta64(30, "s")  # from defaults


def test_parse_gridding_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown gridding keys"):
        parse_gridding({"dt": "2s", "max_gpa": "10s"})


import pandas as pd  # noqa: E402

from thermodrift.pipeline import resolve_segment_sns  # noqa: E402


def _mooring_df():
    return pd.DataFrame(
        {"type": ["rbr"] * 4, "depth": [100, 200, 300, 400], "segment": ["deep", "deep", "shallow", "shallow"]},
        index=pd.Index([111, 222, 333, 444], name="SN"),
    )


def test_resolve_segment_by_column():
    sns = resolve_segment_sns(_mooring_df(), {"segment": "deep"}, root=".")
    assert sorted(sns) == [111, 222]


def test_resolve_segment_by_explicit_sns():
    sns = resolve_segment_sns(_mooring_df(), {"sns": [333, 444]}, root=".")
    assert sorted(sns) == [333, 444]


def test_resolve_segment_missing_column_raises():
    df = _mooring_df().drop(columns="segment")
    with pytest.raises(ValueError, match="requires a 'segment' column"):
        resolve_segment_sns(df, {"segment": "deep"}, root=".")


def test_resolve_segment_unknown_spec_raises():
    with pytest.raises(ValueError, match="one of segment/sns/layout"):
        resolve_segment_sns(_mooring_df(), {"foo": "bar"}, root=".")


import thermodrift  # noqa: E402
from thermodrift.pipeline import Mooring  # noqa: E402


def test_mooring_exported_at_package_root():
    assert thermodrift.Mooring is Mooring


def test_mooring_parses_segments_and_gridding(segmented_mooring):
    m = Mooring(segmented_mooring)
    assert set(m.segments_cfg) == {"deep", "shallow"}
    assert m.gridding["deep"]["dt"] == np.timedelta64(10, "s")
    assert m.gridding["shallow"]["chunk"] == np.timedelta64(2, "D")


def test_mooring_segment_sensors_returns_three_deep(segmented_mooring):
    m = Mooring(segmented_mooring)
    assert len(m.segment_sensors("deep")) == 3


from pathlib import Path  # noqa: E402 (already imported at top of pipeline.py, idempotent here)

import xarray as xr  # noqa: E402


def test_grid_l1_writes_chunks_and_is_idempotent(segmented_mooring):
    m = Mooring(segmented_mooring)
    grid_dir = Path(m.cfg.path.root) / "data" / "grid"

    summary = m.grid_l1(segments=["deep"])
    assert summary["deep"]["written"] == 2          # 4-day span / 2D chunk
    assert summary["deep"]["skipped"] == 0
    files = sorted(grid_dir.glob("testproj_mavs3_deep_L1_*.nc"))
    assert len(files) == 2

    da = xr.open_dataarray(files[0])
    assert da.sizes["depth"] == 3
    assert set(int(s) for s in da.sn.values) == set(int(s) for s in m.segment_sensors("deep").index)

    summary2 = m.grid_l1(segments=["deep"])
    assert summary2["deep"]["written"] == 0
    assert summary2["deep"]["skipped"] == 2


def test_grid_l1_overwrite_forces_rewrite(segmented_mooring):
    m = Mooring(segmented_mooring)
    m.grid_l1(segments=["deep"])
    summary = m.grid_l1(segments=["deep"], overwrite=True)
    assert summary["deep"]["written"] == 2
