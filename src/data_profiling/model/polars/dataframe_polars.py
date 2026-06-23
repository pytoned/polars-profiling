"""DataFrame preprocessing for the Polars backend."""
import polars as pl

from data_profiling.config import Settings


def polars_preprocess(config: Settings, df: pl.DataFrame) -> pl.DataFrame:
    """Preprocess a Polars DataFrame.

    - Renames a reserved ``index`` column to ``df_index``.
    - Ensures column names are strings (always true for Polars).
    """
    if "index" in df.columns:
        df = df.rename({"index": "df_index"})
    return df
