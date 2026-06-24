"""A native, Polars-based type system for data-profiling.

This module replaces the previous ``visions``/pandas-based typeset. Variable
types are detected directly from Polars dtypes (with optional content-based
inference for string columns), keeping the whole pipeline free of any pandas
dependency.

The supported variable types are:

- ``Numeric``      integer / float / decimal columns
- ``Boolean``      boolean columns (or 2-valued boolean-like strings)
- ``DateTime``     date / datetime / time / duration columns
- ``Categorical``  low-cardinality string columns
- ``Text``         high-cardinality (free-text) string columns
- ``Unsupported``  everything else (lists, structs, binary, objects, ...)
"""
from __future__ import annotations

from typing import Dict, List, Optional

import polars as pl

from polars_profiling.config import Settings

# Variable type names (kept identical to the render map keys).
NUMERIC = "Numeric"
CATEGORICAL = "Categorical"
BOOLEAN = "Boolean"
DATETIME = "DateTime"
TEXT = "Text"
UNSUPPORTED = "Unsupported"

#: Canonical set of types that the summarizer/handler knows how to describe.
PROFILING_TYPES: List[str] = [
    NUMERIC,
    DATETIME,
    TEXT,
    CATEGORICAL,
    BOOLEAN,
    "URL",
    "Path",
    "File",
    "Image",
    "TimeSeries",
]

_BOOL_TRUE = {"true", "t", "yes", "y", "1"}
_BOOL_FALSE = {"false", "f", "no", "n", "0"}


class ProfilingTypeSet:
    """Detect data-profiling variable types from Polars dtypes."""

    #: The root/base type from which every other type derives.
    base_type: str = UNSUPPORTED

    def __init__(self, config: Settings, type_schema: Optional[dict] = None) -> None:
        self.config = config
        self.types: List[str] = list(PROFILING_TYPES)
        # User-provided overrides: {column_name: "Numeric"/"Categorical"/...}
        self.type_schema: Dict[str, str] = self._init_type_schema(type_schema or {})

    # -- type schema -----------------------------------------------------
    def _init_type_schema(self, type_schema: dict) -> Dict[str, str]:
        return {k: self._get_type(v) for k, v in type_schema.items()}

    def _get_type(self, type_name: str) -> str:
        for t in [self.base_type, *self.types]:
            if t.lower() == str(type_name).lower():
                return t
        raise ValueError(f"Type [{type_name}] not found.")

    # -- detection -------------------------------------------------------
    def detect_type(self, series: pl.Series) -> str:
        """Detect the variable type of a Polars Series."""
        dtype = series.dtype

        if dtype == pl.Boolean:
            return BOOLEAN
        if dtype.is_numeric():
            return NUMERIC
        if dtype.is_temporal():
            return DATETIME
        if dtype in (pl.String, pl.Utf8, pl.Categorical, pl.Enum):
            return self._detect_string_type(series)
        # Lists, structs, binary, nested or object dtypes are unsupported.
        return UNSUPPORTED

    # Kept for API compatibility with the previous typeset.
    def infer_type(self, series: pl.Series) -> str:
        return self.detect_type(series)

    def cast_to_inferred(self, series: pl.Series) -> pl.Series:
        return series

    def _detect_string_type(self, series: pl.Series) -> str:
        # Work on a string view of the (non-null) values.
        s = series.cast(pl.String, strict=False).drop_nulls()
        if s.len() == 0:
            return CATEGORICAL

        if self.config.infer_dtypes:
            inferred = self._infer_from_strings(s)
            if inferred is not None:
                return inferred

        return self._categorical_or_text(series)

    def _infer_from_strings(self, s: pl.Series) -> Optional[str]:
        """Content-based inference for string columns."""
        stripped = s.str.strip_chars()

        # Boolean-like (exactly the accepted true/false tokens).
        lowered = set(stripped.str.to_lowercase().unique().to_list())
        if lowered and lowered <= (_BOOL_TRUE | _BOOL_FALSE):
            return BOOLEAN

        # Numeric-like (every value parses as a number).
        as_num = stripped.cast(pl.Float64, strict=False)
        if as_num.null_count() == 0:
            return NUMERIC

        # Datetime-like (every value parses as a datetime).
        try:
            as_dt = stripped.str.to_datetime(strict=False)
            if as_dt.null_count() == 0:
                return DATETIME
        except Exception:
            pass

        return None

    def _categorical_or_text(self, series: pl.Series) -> str:
        n_distinct = series.drop_nulls().n_unique()
        threshold = self.config.vars.cat.cardinality_threshold
        if n_distinct <= threshold:
            return CATEGORICAL
        return TEXT
