#!/usr/bin/env python

"""Tests for `thermodrift` package."""

import pytest

import thermodrift


def test_read_rbr_solo_file():
    pass


def test_read_config_file():
    cfg = thermodrift.io.load_config("tests/data/config.yml")


def test_read_config_file_with_fixture(config_file_path):
    # note: config_file_path has been defined as fixture in conftest.py
    cfg = thermodrift.io.load_config(config_file_path)
    assert cfg["info"] == "test config file"


def test_read_sensor_config_csv(rootdir):
    # note: rootdir has been defined as fixture in conftest.py
    sensor_sheet = rootdir.joinpath("data/sensor_sheet.csv")
    assert sensor_sheet.exists()
    df = thermodrift.io.sensor_sheet_csv_load(sensor_sheet)
    assert df.loc[376].Type == "Seabird 56"


def test_read_sensor_config_xlsx(rootdir):
    sensor_sheet = rootdir.joinpath("data/sensor_sheet.xlsx")
    assert sensor_sheet.exists()
    df = thermodrift.io.sensor_sheet_xlsx_load(sensor_sheet)
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
    df = thermodrift.io.sensor_sheet_load(sensor_sheet)
    assert df.ctd_cal.dtype == expected
    assert df.loc[376].type == "sbe"
    assert df.loc[72219].type == "rbr"


def test_read_sensor_config_csv_fail(rootdir):
    sensor_sheet = rootdir.joinpath("data/sensor_sheet_fail.csv")
    assert sensor_sheet.exists()
    with pytest.raises(ValueError):
        df = thermodrift.io.sensor_sheet_load(sensor_sheet)


def test_mooring_sheet_load(rootdir):
    mooring_sheet = rootdir.joinpath("data/mooring_sheet.csv")
    assert mooring_sheet.exists()
    mooring_info = thermodrift.io.mooring_sheet_load(mooring_sheet)
    assert mooring_info.loc[72213].height == 6.4


def test_proc_db(rootdir):
    sensor_sheet = rootdir.joinpath("data/sensor_sheet.xlsx")
    df = thermodrift.io.sensor_sheet_load(sensor_sheet)
    proc_db = thermodrift.io.proc_db_generate(df)


def test_generate_processing_object(config_file_path):
    M = thermodrift.io.ProcessThermistorMooring(config_file_path)
    assert M.sensor_info.loc[376].type == "sbe"
