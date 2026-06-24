"""Compute statistical description of datasets."""
from typing import Any

import polars as pl
from tqdm import tqdm

from polars_profiling.config import Settings
from polars_profiling.model.polars.summary_polars import (
    polars_describe_1d,
    polars_get_series_descriptions,
)
from polars_profiling.model.summarizer import BaseSummarizer


def describe_1d(
    config: Settings,
    series: Any,
    summarizer: BaseSummarizer,
    typeset: Any,
) -> dict:
    """Describe a single Polars Series."""
    if isinstance(series, pl.Series):
        return polars_describe_1d(config, series, summarizer, typeset)
    raise TypeError(f"Unsupported series type: {type(series)}")


def get_series_descriptions(
    config: Settings,
    df: Any,
    summarizer: BaseSummarizer,
    typeset: Any,
    pbar: tqdm,
) -> dict:
    if isinstance(df, pl.DataFrame):
        return polars_get_series_descriptions(config, df, summarizer, typeset, pbar)
    raise TypeError(f"Unsupported dataframe type: {type(df)}")
