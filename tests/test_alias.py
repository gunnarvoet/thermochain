# tests/test_alias.py
import warnings

import pytest


def test_thermochain_is_importable():
    import thermochain

    assert hasattr(thermochain, "Mooring")


def test_thermodrift_alias_warns_and_reexports():
    with pytest.warns(DeprecationWarning, match="renamed thermochain"):
        import thermodrift
    import thermochain

    assert thermodrift.Mooring is thermochain.Mooring
    assert thermodrift.io.grid_thermistors is thermochain.io.grid_thermistors
    assert thermodrift.pipeline.detect_cal_stops is thermochain.pipeline.detect_cal_stops


def test_thermodrift_submodule_import_resolves():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import thermodrift.io  # submodule-import syntax must resolve
    import thermochain

    assert thermodrift.io is thermochain.io
