"""Compute statistical descriptions of a Polars DataFrame."""
from typing import Any

import polars as pl

from data_profiling.config import Settings
from data_profiling.model.summarizer import BaseSummarizer
from data_profiling.model.typeset import ProfilingTypeSet
from data_profiling.utils.dataframe import sort_column_names


def polars_describe_1d(
    config: Settings,
    series: pl.Series,
    summarizer: BaseSummarizer,
    typeset: Any,
) -> dict:
    """Describe a single Polars Series (detect type, then type-specific stats)."""
    name = series.name

    if isinstance(typeset, ProfilingTypeSet) and name in typeset.type_schema:
        vtype = typeset.type_schema[name]
    elif config.infer_dtypes:
        vtype = typeset.infer_type(series)
    else:
        vtype = typeset.detect_type(series)

    if isinstance(typeset, ProfilingTypeSet):
        typeset.type_schema[name] = vtype

    summary = summarizer.summarize(config, series, dtype=vtype)
    summary["cast_type"] = None
    return summary


def polars_get_series_descriptions(
    config: Settings,
    df: pl.DataFrame,
    summarizer: BaseSummarizer,
    typeset: Any,
    pbar: Any,
) -> dict:
    """Describe every column of a Polars DataFrame."""
    series_description = {}
    for name in df.columns:
        pbar.set_postfix_str(f"Describe variable: {name}")
        series_description[name] = polars_describe_1d(
            config, df.get_column(name), summarizer, typeset
        )
        pbar.update()

    return sort_column_names(series_description, config.sort)
