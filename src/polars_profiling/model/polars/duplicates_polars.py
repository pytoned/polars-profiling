"""Duplicate-row detection for the Polars backend."""
from typing import Any, Dict, Optional, Sequence, Tuple

import polars as pl

from polars_profiling.config import Settings
from polars_profiling.model.duplicates import get_duplicates


@get_duplicates.register(Settings, pl.DataFrame, Sequence)
def polars_get_duplicates(
    config: Settings, df: pl.DataFrame, supported_columns: Sequence
) -> Tuple[Dict[str, Any], Optional[pl.DataFrame]]:
    """Return the most frequent duplicate rows in the DataFrame."""
    n_head = config.duplicates.head
    metrics: Dict[str, Any] = {}

    if n_head == 0 or not supported_columns or df.height == 0:
        if n_head > 0:
            metrics["n_duplicates"] = 0
            metrics["p_duplicates"] = 0.0
        return metrics, None

    duplicates_key = config.duplicates.key
    if duplicates_key in df.columns:
        raise ValueError(
            f"Duplicates key ({duplicates_key}) may not be part of the DataFrame. "
            f"Either change the column name in the DataFrame or change the "
            f"'duplicates.key' parameter."
        )

    columns = list(supported_columns)
    grouped = (
        df.group_by(columns)
        .agg(pl.len().alias(duplicates_key))
        .filter(pl.col(duplicates_key) > 1)
    )

    metrics["n_duplicates"] = grouped.height
    metrics["p_duplicates"] = metrics["n_duplicates"] / df.height

    duplicated_rows = grouped.sort(duplicates_key, descending=True).head(n_head)
    return metrics, duplicated_rows
