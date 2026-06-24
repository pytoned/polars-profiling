"""Boolean variable description for the Polars backend."""
import math
from typing import Tuple

import polars as pl

from polars_profiling.config import Settings
from polars_profiling.model.polars.utils_polars import column_imbalance_score


def polars_describe_boolean_1d(
    config: Settings, series: pl.Series, summary: dict
) -> Tuple[Settings, pl.Series, dict]:
    """Describe a boolean series."""
    value_counts = summary["value_counts_without_nan"]
    if not value_counts.empty:
        summary.update(
            {"top": value_counts.index[0], "freq": value_counts.values[0]}
        )
        summary["imbalance"] = column_imbalance_score(
            value_counts.values, len(value_counts)
        )
    else:
        summary.update({"top": math.nan, "freq": 0, "imbalance": 0})

    return config, series, summary
