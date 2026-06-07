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
