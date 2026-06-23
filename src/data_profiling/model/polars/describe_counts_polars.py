"""Base descriptions (counts, generic, supported) for the Polars backend."""
from typing import Tuple

import numpy as np
import polars as pl

from data_profiling.config import Settings
from data_profiling.utils.varseries import VarSeries


def _value_counts(series: pl.Series) -> Tuple[np.ndarray, np.ndarray, int]:
    """Return (values, counts, n_missing) for a Polars Series.

    Values are sorted by descending frequency; nulls are excluded from the
    returned arrays and accounted for in ``n_missing``.
    """
    vc = series.value_counts(sort=True)
    val_col, count_col = vc.columns[0], vc.columns[1]

    null_mask = vc[val_col].is_null()
    n_missing = int(vc.filter(null_mask)[count_col].sum() or 0)

    vc_nn = vc.filter(~null_mask)
    values = vc_nn[val_col].to_numpy()
    counts = vc_nn[count_col].to_numpy()
    return values, counts, n_missing


def polars_describe_counts(
    config: Settings, series: pl.Series, summary: dict
) -> Tuple[Settings, pl.Series, dict]:
    """Count the values of a series (with/without nulls, distinct, sorted)."""
    values, counts, n_missing = _value_counts(series)

    value_counts_without_nan = VarSeries(counts, index=values)
    summary["hashable"] = True
    summary["value_counts_without_nan"] = value_counts_without_nan

    try:
        summary["value_counts_index_sorted"] = value_counts_without_nan.sort_index(
            ascending=True
        )
        summary["ordering"] = True
    except TypeError:
        summary["ordering"] = False

    summary["n_missing"] = n_missing
    return config, series, summary


def polars_describe_generic(
    config: Settings, series: pl.Series, summary: dict
) -> Tuple[Settings, pl.Series, dict]:
    """Describe generic (type-independent) properties of a series."""
    length = series.len()

    summary.update(
        {
            "n": length,
            "p_missing": summary["n_missing"] / length if length > 0 else 0,
            "count": length - summary["n_missing"],
            "memory_size": series.estimated_size(),
        }
    )
    return config, series, summary


def polars_describe_supported(
    config: Settings, series: pl.Series, series_description: dict
) -> Tuple[Settings, pl.Series, dict]:
    """Describe a supported (distinct/unique) series."""
    count = series_description["count"]

    value_counts = series_description["value_counts_without_nan"]
    distinct_count = len(value_counts)
    # Number of values that occur exactly once.
    unique_count = int(np.sum(value_counts.values == 1)) if len(value_counts) else 0

    stats = {
        "n_distinct": distinct_count,
        "p_distinct": distinct_count / count if count > 0 else 0,
        "is_unique": unique_count == count and count > 0,
        "n_unique": unique_count,
        "p_unique": unique_count / count if count > 0 else 0,
    }
    stats.update(series_description)
    return config, series, stats
