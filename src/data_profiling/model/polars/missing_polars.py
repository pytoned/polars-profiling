"""Missing-data diagrams for the Polars backend.

The nullity data is computed natively with Polars; rendering is delegated to
the (matplotlib-based) visualisation layer.
"""
from typing import Optional

import numpy as np
import polars as pl

from data_profiling.config import Settings
from data_profiling.visualisation.missing import (
    plot_missing_bar,
    plot_missing_heatmap,
    plot_missing_matrix,
)
from data_profiling.utils.varseries import VarSeries


def missing_bar(config: Settings, df: pl.DataFrame) -> Optional[str]:
    nrows = df.height
    notnull = [nrows - df.get_column(c).null_count() for c in df.columns]
    return plot_missing_bar(
        config,
        notnull_counts=VarSeries(np.asarray(notnull), index=np.asarray(df.columns, dtype=object)),
        nrows=nrows,
        columns=list(df.columns),
    )


def missing_matrix(config: Settings, df: pl.DataFrame) -> Optional[str]:
    notnull = df.select(pl.all().is_not_null()).to_numpy()
    return plot_missing_matrix(
        config,
        columns=list(df.columns),
        notnull=notnull,
        nrows=df.height,
    )


def missing_heatmap(config: Settings, df: pl.DataFrame) -> Optional[str]:
    # Nullity (1 = missing) per cell, keeping only columns that vary.
    isnull = df.select(pl.all().is_null().cast(pl.Int8)).to_numpy().astype(float)
    variances = isnull.var(axis=0)
    keep = np.where(variances > 0)[0]
    if len(keep) == 0:
        return None

    isnull = isnull[:, keep]
    columns = [df.columns[i] for i in keep]

    with np.errstate(invalid="ignore"):
        corr_mat = np.corrcoef(isnull, rowvar=False)
    corr_mat = np.atleast_2d(corr_mat)

    mask = np.zeros_like(corr_mat)
    mask[np.triu_indices_from(mask)] = True
    return plot_missing_heatmap(
        config, corr_mat=corr_mat, mask=mask, columns=columns
    )
