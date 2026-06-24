"""Type detection tests for the native Polars typeset."""
import datetime

import polars as pl
import pytest

from polars_profiling.config import Settings
from polars_profiling.model.typeset import ProfilingTypeSet


@pytest.fixture
def typeset():
    return ProfilingTypeSet(Settings())


@pytest.mark.parametrize(
    "series, expected",
    [
        (pl.Series("x", [1, 2, 3, 4]), "Numeric"),
        (pl.Series("x", [1.5, 2.5, 3.5]), "Numeric"),
        (pl.Series("x", [True, False, True]), "Boolean"),
        (pl.Series("x", ["a", "b", "a", "c"]), "Categorical"),
        (
            pl.Series("x", [datetime.date(2020, 1, i + 1) for i in range(3)]),
            "DateTime",
        ),
        (pl.Series("x", [None, None, None], dtype=pl.Null), "Unsupported"),
    ],
)
def test_detect_type(typeset, series, expected):
    assert typeset.detect_type(series) == expected


def test_high_cardinality_string_is_text(typeset):
    values = [f"unique-sentence-number-{i}" for i in range(100)]
    series = pl.Series("x", values)
    assert typeset.detect_type(series) == "Text"


def test_low_cardinality_string_is_categorical(typeset):
    series = pl.Series("x", ["red", "green", "blue"] * 20)
    assert typeset.detect_type(series) == "Categorical"


def test_numeric_like_string_inferred_numeric(typeset):
    series = pl.Series("x", ["1", "2", "3", "4"])
    assert typeset.detect_type(series) == "Numeric"


def test_boolean_like_string_inferred_boolean(typeset):
    series = pl.Series("x", ["true", "false", "true"])
    assert typeset.detect_type(series) == "Boolean"


def test_type_schema_override():
    typeset = ProfilingTypeSet(Settings(), {"col": "categorical"})
    assert typeset.type_schema["col"] == "Categorical"
