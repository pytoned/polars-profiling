from typing import Any

import polars as pl

from polars_profiling.config import Settings
from polars_profiling.model.polars.dataframe_polars import polars_preprocess


def preprocess(config: Settings, df: Any) -> Any:
    """Validate and normalise the input DataFrame.

    Ensures column names follow the expected rules.

    Args:
        config: data-profiling Settings object
        df: a Polars DataFrame

    Returns:
        A preprocessed Polars DataFrame
    """
    if isinstance(df, pl.DataFrame):
        return polars_preprocess(config=config, df=df)
    raise NotImplementedError(
        f"`df` must be a `polars.DataFrame`, but got {type(df)}."
    )
