"""Describe / statistics tests for the Polars backend."""
import datetime

import polars as pl
import pytest

from data_profiling import ProfileReport


@pytest.fixture
def mixed_df():
    return pl.DataFrame(
        {
            "num": [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
            "flt": [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5],
            "cat": ["x", "y", "x", "z", "y", "x", "x", "y", "z", "x"],
            "flag": [True, False, True, True, False, None, True, False, True, True],
            "dt": [datetime.date(2020, 1, i + 1) for i in range(10)],
        }
    )


@pytest.fixture
def description(mixed_df):
    return ProfileReport(mixed_df, progress_bar=False).get_description()


def test_table_stats(description):
    assert description.table["n"] == 10
    assert description.table["n_var"] == 5
    assert description.table["n_cells_missing"] == 2  # one in num, one in flag


def test_types_detected(description):
    types = description.table["types"]
    assert types["Numeric"] == 2
    assert types["Categorical"] == 1
    assert types["Boolean"] == 1
    assert types["DateTime"] == 1


def test_numeric_statistics(description):
    num = description.variables["num"]
    assert num["type"] == "Numeric"
    assert num["count"] == 9
    assert num["n_missing"] == 1
    assert num["min"] == 1
    assert num["max"] == 10
    assert num["mean"] == pytest.approx(5.444444, rel=1e-4)
    assert "histogram" in num
    assert "5%" in num and "95%" in num


def test_numeric_monotonic_increasing():
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5]})
    desc = ProfileReport(df, progress_bar=False).get_description()
    assert desc.variables["x"]["monotonic"] == 2  # strictly increasing


def test_categorical_statistics(description):
    cat = description.variables["cat"]
    assert cat["type"] == "Categorical"
    assert cat["n_distinct"] == 3
    assert cat["count"] == 10
    assert "value_counts_without_nan" in cat


def test_boolean_statistics(description):
    flag = description.variables["flag"]
    assert flag["type"] == "Boolean"
    assert flag["n_missing"] == 1
    assert flag["top"] in (True, False)


def test_datetime_statistics(description):
    dt = description.variables["dt"]
    assert dt["type"] == "DateTime"
    assert dt["min"] == datetime.datetime(2020, 1, 1) or dt["min"] == datetime.date(
        2020, 1, 1
    )


def test_unique_detection():
    df = pl.DataFrame({"u": [1, 2, 3], "d": [1, 1, 2]})
    desc = ProfileReport(df, progress_bar=False).get_description()
    assert desc.variables["u"]["is_unique"] is True
    assert desc.variables["d"]["is_unique"] is False


def test_text_type_statistics():
    df = pl.DataFrame({"t": [f"sentence number {i} here" for i in range(80)]})
    desc = ProfileReport(df, progress_bar=False).get_description()
    t = desc.variables["t"]
    assert t["type"] == "Text"
    assert "max_length" in t
    assert "word_counts" in t or "n_characters" in t


def test_all_null_column():
    df = pl.DataFrame({"a": [1, 2, 3], "n": [None, None, None]})
    desc = ProfileReport(df, progress_bar=False).get_description()
    assert desc.variables["n"]["n_missing"] == 3


def test_constant_column():
    df = pl.DataFrame({"c": [5, 5, 5, 5]})
    desc = ProfileReport(df, progress_bar=False).get_description()
    assert desc.variables["c"]["n_distinct"] == 1
