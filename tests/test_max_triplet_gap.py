"""Tests for the gap-aware neighbour selector (max_triplet_gap_m).

select_triplet picks neighbours by index adjacency. With max_triplet_gap_m
set, a neighbour farther than the threshold (in metres, from the depth
coordinate) is dropped, so a large mooring gap breaks the shared-fluctuation
neighbour chain instead of being averaged across. Default None is inert.
"""

import numpy as np
import pytest
import tqdm

from _synthetic import write_drift_l1_files
from thermochain.io import sensor_drift
from thermochain.pipeline import DriftParameters, drift_provenance_attrs


pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
    pytest.mark.filterwarnings("ignore::PendingDeprecationWarning"),
]


@pytest.fixture(autouse=True)
def _plain_tqdm(monkeypatch):
    monkeypatch.setattr(tqdm, "tqdm_notebook", tqdm.tqdm, raising=False)


def test_driftparameters_accepts_and_roundtrips_gap():
    p = DriftParameters.from_dict({"max_triplet_gap_m": 15.0})
    assert p.max_triplet_gap_m == 15.0
    assert p.as_dict()["max_triplet_gap_m"] == 15.0


def test_default_gap_is_none():
    assert DriftParameters().max_triplet_gap_m is None


def test_provenance_attr_emitted():
    # None -> empty string; a float -> the value (drift_provenance_attrs rules)
    a_none = drift_provenance_attrs(DriftParameters(), "lbl")
    assert a_none["drift_param_max_triplet_gap_m"] == ""
    a_set = drift_provenance_attrs(
        DriftParameters.from_dict({"max_triplet_gap_m": 15.0}), "lbl"
    )
    assert a_set["drift_param_max_triplet_gap_m"] == 15.0


# A gapped array: 4 sensors at 4 m spacing, a 46 m gap, then 3 at 4 m.
GAPPED_DEPTH = [1000.0, 1004.0, 1008.0, 1012.0, 1058.0, 1062.0, 1066.0]
GAP_LO_IDX = 3   # depth 1012, last sensor above the gap
GAP_HI_IDX = 4   # depth 1058, first sensor below the gap


def _run_gapped(l1_dir, **drift_params):
    return sensor_drift(
        mooring_name="synthetic",
        l1_grid_dir=l1_dir,
        run_all=True,
        drift_parameters=dict(fit_mode="linear", polydeg=2, outliers_polydeg=2,
                              **drift_params),
    )


@pytest.fixture
def gapped_l1_dir(tmp_path):
    write_drift_l1_files(tmp_path, drift_index=1, depth=GAPPED_DEPTH)
    return tmp_path


def test_none_keeps_cross_gap_triplet(gapped_l1_dir):
    sd = _run_gapped(gapped_l1_dir)  # max_triplet_gap_m default None
    # the sensor just above the gap reaches across to the one below it
    tt = sd.select_triplet(GAP_LO_IDX)
    assert GAPPED_DEPTH[GAP_HI_IDX] in tt.depth.values  # 1058.0 present


def test_gap_guard_breaks_chain(gapped_l1_dir):
    sd = _run_gapped(gapped_l1_dir, max_triplet_gap_m=15.0)
    # above-gap sensor: its triplet must not include any depth below the gap
    tt = sd.select_triplet(GAP_LO_IDX)
    centre = sd.offsets.depth.values[GAP_LO_IDX]
    assert np.all(np.abs(tt.depth.values - centre) <= 15.0)
    # and the below-gap edge sensor only pairs downward (within its cluster)
    tt2 = sd.select_triplet(GAP_HI_IDX)
    centre2 = sd.offsets.depth.values[GAP_HI_IDX]
    assert np.all(np.abs(tt2.depth.values - centre2) <= 15.0)


def test_none_reproduces_select_triplet_membership(gapped_l1_dir):
    sd = _run_gapped(gapped_l1_dir)
    # interior sensor returns a full 3-member triplet when guard is off
    tt = sd.select_triplet(2)
    assert tt.depth.size == 3


def test_gap_isolates_singleton(gapped_l1_dir):
    # With a 3.0 m threshold and 4 m sensor spacing, every interior sensor's
    # neighbours (4 m away) exceed the threshold, so the triplet collapses to
    # just the centre sensor — the intended singleton fallback.
    sd = _run_gapped(gapped_l1_dir, max_triplet_gap_m=3.0)
    # sensor at index 2 (depth 1008 m) is interior; both neighbours are 4 m away
    ni = 2
    centre_depth = sd.offsets.depth.values[ni]
    tt = sd.select_triplet(ni)
    assert tt.depth.size == 1
    assert tt.depth.values[0] == centre_depth
