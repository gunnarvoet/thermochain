#!/usr/bin/env python

"""Tests for `thermochain` package."""

from pathlib import Path

import pandas as pd
import pytest

import thermochain


def test_read_rbr_solo_file():
    pass


def test_read_config_file():
    cfg = thermochain.io.load_config("tests/data/config.yml")


def test_read_config_file_with_fixture(config_file_path):
    # note: config_file_path has been defined as fixture in conftest.py
    cfg = thermochain.io.load_config(config_file_path)
    assert cfg["info"] == "test config file"


def test_read_sensor_config_csv(rootdir):
    # note: rootdir has been defined as fixture in conftest.py
    sensor_sheet = rootdir.joinpath("data/sensor_sheet.csv")
    assert sensor_sheet.exists()
    df = thermochain.io.sensor_sheet_csv_load(sensor_sheet)
    assert df.loc[376].Type == "Seabird 56"


def test_read_sensor_config_xlsx(rootdir):
    sensor_sheet = rootdir.joinpath("data/sensor_sheet.xlsx")
    assert sensor_sheet.exists()
    df = thermochain.io.sensor_sheet_xlsx_load(sensor_sheet)
    assert df.loc[376].Type == "Seabird 56"


@pytest.mark.parametrize(
    "input,expected",
    [("data/sensor_sheet.csv", "<M8[ns]"), ("data/sensor_sheet.xlsx", "<M8[ns]")],
)
def test_sensor_sheet_load(input, expected, rootdir):
    """Make sure we load & parse both .csv and .xlsx sensor sheets the right
    way.
    """
    sensor_sheet = rootdir.joinpath(input)
    df = thermochain.io.sensor_sheet_load(sensor_sheet)
    assert df.pre_ctd_cal_time.dtype == expected
    assert df.post_ctd_cal_time.dtype == expected
    assert df.loc[376].type == "sbe"
    assert df.loc[72219].type == "rbr"


def test_read_sensor_config_csv_fail(rootdir):
    sensor_sheet = rootdir.joinpath("data/sensor_sheet_fail.csv")
    assert sensor_sheet.exists()
    with pytest.raises(ValueError):
        df = thermochain.io.sensor_sheet_load(sensor_sheet)


def _make_sensor_sheet(rows):
    df = pd.DataFrame(rows).set_index("sn")
    df.index.name = "SN"
    return df


def _make_layout(rows):
    df = pd.DataFrame(rows).set_index("sn")
    df.index.name = "SN"
    return df


def test_validate_thermistor_metadata_clean():
    sensor_sheet = _make_sensor_sheet(
        [
            {"sn": 376, "type": "sbe", "Mooring": "MOTIVE A"},
            {"sn": 418, "type": "sbe", "Mooring": "MOTIVE B"},
            {"sn": 72144, "type": "rbr", "Mooring": "MOTIVE A"},
        ]
    )
    layouts = {
        "A": _make_layout(
            [
                {"sn": 376, "type": "sbe", "depth": 100.0},
                {"sn": 72144, "type": "rbr", "depth": 200.0},
            ]
        ),
        "B": _make_layout(
            [
                {"sn": 418, "type": "sbe", "depth": 150.0},
            ]
        ),
    }
    assert thermochain.io.validate_thermistor_metadata(sensor_sheet, layouts) == []


def test_validate_thermistor_metadata_catches_issues():
    sensor_sheet = _make_sensor_sheet(
        [
            {"sn": 376, "type": "sbe", "Mooring": "MOTIVE A"},
            {"sn": 418, "type": "sbe", "Mooring": "MOTIVE B"},
            # 6413 declared on A in sensor sheet but absent from layouts["A"]
            {"sn": 6413, "type": "sbe", "Mooring": "MOTIVE A"},
            # 72144 declares an unmapped mooring "Z"
            {"sn": 72144, "type": "rbr", "Mooring": "MOTIVE Z"},
            # 99999 has NaN mooring — should be ignored, not crash
            {"sn": 99999, "type": "rbr", "Mooring": float("nan")},
        ]
    )
    layouts = {
        "A": _make_layout(
            [
                {"sn": 376, "type": "sbe", "depth": 100.0},
                # 9999 in layout but not in sensor sheet
                {"sn": 9999, "type": "sbe", "depth": 250.0},
                # duplicate SN in layout A
                {"sn": 376, "type": "sbe", "depth": 110.0},
            ]
        ),
        "B": _make_layout(
            [
                # type mismatch: layout says rbr, sensor sheet says sbe
                {"sn": 418, "type": "rbr", "depth": 150.0},
            ]
        ),
    }
    warnings = thermochain.io.validate_thermistor_metadata(sensor_sheet, layouts)
    joined = "\n".join(warnings)
    assert "duplicate SN(s) [376]" in joined
    assert "SN 9999 not in sensor sheet" in joined
    assert "SN 6413 declared on mooring A but not in its layout" in joined
    assert "SN 72144 declares mooring 'Z' but no layout provided" in joined
    assert "SN 418 type 'rbr' != sensor-sheet type 'sbe'" in joined


def test_mooring_sheet_load(rootdir):
    mooring_sheet = rootdir.joinpath("data/mooring_sheet.csv")
    assert mooring_sheet.exists()
    mooring_info = thermochain.io.mooring_sheet_load(mooring_sheet)
    assert mooring_info.loc[72213].height == 6.4


def test_proc_db(rootdir):
    sensor_sheet = rootdir.joinpath("data/sensor_sheet.xlsx")
    df = thermochain.io.sensor_sheet_load(sensor_sheet)
    proc_db = thermochain.io.proc_db_generate(df)


def test_generate_processing_object(config_file_path):
    M = thermochain.io.ProcessThermistorMooring(config_file_path)
    assert M.sensor_info.loc[376].type == "sbe"


class TestResolvePathVars:
    def test_resolves_data_prefix(self):
        out = thermochain.io._resolve_path_vars(
            "$data/foo/bar.nc",
            data_root=Path("/abs/data"),
            project_root=Path("/abs/proj"),
        )
        assert out == Path("/abs/data/foo/bar.nc")

    def test_resolves_root_prefix(self):
        out = thermochain.io._resolve_path_vars(
            "$root/parameters/x.csv",
            data_root=Path("/abs/data"),
            project_root=Path("/abs/proj"),
        )
        assert out == Path("/abs/proj/parameters/x.csv")

    def test_recurses_dicts_and_lists(self):
        obj = {
            "a": "$data/x",
            "b": ["$root/y", "plain"],
            "c": {"d": "$data/z"},
        }
        out = thermochain.io._resolve_path_vars(
            obj, data_root=Path("/d"), project_root=Path("/p")
        )
        assert out == {
            "a": Path("/d/x"),
            "b": [Path("/p/y"), "plain"],
            "c": {"d": Path("/d/z")},
        }

    def test_passes_through_non_strings(self):
        assert thermochain.io._resolve_path_vars(
            42, Path("/d"), Path("/p")
        ) == 42
        assert thermochain.io._resolve_path_vars(
            None, Path("/d"), Path("/p")
        ) is None

    def test_raises_when_data_prefix_lacks_data_root(self):
        with pytest.raises(ValueError, match="data_root"):
            thermochain.io._resolve_path_vars(
                "$data/foo", data_root=None, project_root=Path("/p")
            )

    def test_data_root_unused_is_fine(self):
        # No $data/ in the input → data_root may be None.
        out = thermochain.io._resolve_path_vars(
            {"a": "$root/x"}, data_root=None, project_root=Path("/p")
        )
        assert out == {"a": Path("/p/x")}


class TestLoadConfigBoxWithVars:
    @pytest.fixture
    def vars_config_path(self, rootdir):
        return rootdir / "data/config_with_vars.yml"

    def test_resolves_data_and_root_prefixes(self, vars_config_path, tmp_path):
        data_root = tmp_path / "alt_data"
        cfg = thermochain.io.load_config_box(
            vars_config_path,
            project_root=tmp_path,
            data_root=data_root,
        )
        assert cfg.path.root == tmp_path
        assert cfg.path.data.proc == data_root / "proc/testmoor"
        assert cfg.path.data.raw.rbr == data_root / "raw/rbrsolo"
        assert cfg.path.data.raw.sbe == data_root / "raw/sbe56"
        assert cfg.path.sensors == tmp_path / "data/sensor_sheet.xlsx"
        for level in range(3):
            assert cfg.path.data[f"procl{level}"] == data_root / f"proc/testmoor/l{level}"

    def test_default_project_root_is_configfile_parent_parent(self, vars_config_path, tmp_path):
        # data_root must be supplied because the YAML uses $data/.
        cfg = thermochain.io.load_config_box(
            vars_config_path, data_root=tmp_path / "d"
        )
        assert cfg.path.root == vars_config_path.parent.parent.resolve()

    def test_raises_when_data_prefix_present_but_data_root_missing(self, vars_config_path):
        with pytest.raises(ValueError, match="data_root"):
            thermochain.io.load_config_box(vars_config_path)

    def test_existing_no_dollar_yaml_unchanged(self, config_file_path):
        # Regression: the original test config has no $-prefixes.
        cfg = thermochain.io.load_config_box(config_file_path)
        assert cfg.path.root == config_file_path.parent.parent.resolve()
        assert cfg.path.data.proc == cfg.path.root / "data/proc/mavs3"


class TestProcessThermistorMooringWithVars:
    def test_forwards_data_and_project_root(self, rootdir, tmp_path):
        import shutil

        proj = tmp_path / "proj"
        (proj / "data/raw/rbrsolo").mkdir(parents=True)
        (proj / "data/raw/sbe56").mkdir(parents=True)
        shutil.copy(
            rootdir / "data/sensor_sheet.xlsx", proj / "data/sensor_sheet.xlsx"
        )
        shutil.copy(
            rootdir / "data/mooring_sheet.csv", proj / "data/mooring_sheet.csv"
        )

        cfg_path = rootdir / "data/config_with_vars.yml"
        M = thermochain.io.ProcessThermistorMooring(
            cfg_path,
            project_root=proj,
            data_root=proj / "data",
        )
        assert M.cfg.path.root == proj
        assert M.cfg.path.data.proc == proj / "data/proc/testmoor"
        assert M.cfg.path.data.procl0 == proj / "data/proc/testmoor/l0"
        assert M.cfg.path.sensors == proj / "data/sensor_sheet.xlsx"
