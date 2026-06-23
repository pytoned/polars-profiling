"""Date/Datetime variable description for the Polars backend."""
from typing import Tuple

import numpy as np
import polars as pl

from data_profiling.config import Settings
from data_profiling.model.summary_algorithms import chi_square, histogram_compute


def _to_datetime(series: pl.Series) -> Tuple[pl.Series, pl.Series]:
    """Return (parsed_datetime, original_non_null) handling string inputs."""
    present = series.drop_nulls()
    if present.dtype.is_temporal():
        return present, present
    parsed = present.cast(pl.String, strict=False).str.to_datetime(strict=False)
    return parsed, present


def _epoch_seconds(series: pl.Series) -> np.ndarray:
    s = series.drop_nulls()
    if s.len() == 0:
        return np.array([])
    if s.dtype == pl.Date:
        s = s.cast(pl.Datetime)
    if s.dtype in (pl.Datetime, pl.Time, pl.Duration):
        return s.dt.epoch(time_unit="s").to_numpy().astype(np.int64)
    return s.cast(pl.Int64, strict=False).to_numpy()


def polars_describe_date_1d(
    config: Settings, series: pl.Series, summary: dict
) -> Tuple[Settings, pl.Series, dict]:
    """Describe a date/datetime series using native Polars operations."""
    parsed, original = _to_datetime(series)

    # Values that failed to parse (only relevant for string-typed inputs).
    invalid = original.filter(parsed.is_null()) if original.len() == parsed.len() else original.clear()
    parsed = parsed.drop_nulls()

    if parsed.len() == 0:
        summary.update({"min": None, "max": None, "range": 0})
        values = np.array([])
    else:
        summary.update({"min": parsed.min(), "max": parsed.max()})
        summary["range"] = summary["max"] - summary["min"]
        values = _epoch_seconds(parsed)

    if config.vars.num.chi_squared_threshold > 0.0:
        summary["chi_squared"] = chi_square(values)

    if len(values) > 0:
        summary.update(histogram_compute(config, values, parsed.n_unique()))

    n_invalid = invalid.len()
    summary.update(
        {
            "invalid_dates": invalid.n_unique() if n_invalid else 0,
            "n_invalid_dates": n_invalid,
            "p_invalid_dates": n_invalid / summary["n"] if summary["n"] else 0,
        }
    )
    return config, series, summary
