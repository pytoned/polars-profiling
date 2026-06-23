"""Numeric variable description for the Polars backend (native Polars stats)."""
import math
from typing import Tuple

import numpy as np
import polars as pl

from data_profiling.config import Settings
from data_profiling.model.summary_algorithms import chi_square, histogram_compute


def _safe(value: object) -> float:
    return float(value) if value is not None else math.nan


def polars_describe_numeric_1d(
    config: Settings, series: pl.Series, summary: dict
) -> Tuple[Settings, pl.Series, dict]:
    """Describe a numeric series using native Polars aggregations."""
    chi_squared_threshold = config.vars.num.chi_squared_threshold
    quantiles = config.vars.num.quantiles

    present = series.drop_nulls()
    value_counts = summary["value_counts_without_nan"]
    idx = np.asarray(value_counts.index, dtype=float) if len(value_counts) else np.array([])
    cnt = np.asarray(value_counts.values, dtype=float) if len(value_counts) else np.array([])

    # Sign / special-value bookkeeping (weighted by frequency).
    summary["n_negative"] = int(cnt[idx < 0].sum()) if len(idx) else 0
    summary["p_negative"] = summary["n_negative"] / summary["n"] if summary["n"] else 0

    infinite_mask = np.isinf(idx) if len(idx) else np.array([], dtype=bool)
    summary["n_infinite"] = int(cnt[infinite_mask].sum()) if len(idx) else 0

    summary["n_zeros"] = int(cnt[idx == 0].sum()) if len(idx) else 0

    stats = summary

    if present.len() > 0:
        finite = present.filter(present.is_finite()) if present.dtype.is_float() else present
        stats.update(
            {
                "mean": _safe(present.mean()),
                "std": _safe(present.std(ddof=1)),
                "variance": _safe(present.var(ddof=1)),
                "min": present.min(),
                "max": present.max(),
                "kurtosis": _safe(present.kurtosis(fisher=True, bias=False)),
                "skewness": _safe(present.skew(bias=False)),
                "sum": present.sum(),
                "mad": _safe((present - present.median()).abs().median()),
            }
        )
    else:
        finite = present
        stats.update(
            {
                "mean": math.nan,
                "std": 0.0,
                "variance": 0.0,
                "min": math.nan,
                "max": math.nan,
                "kurtosis": 0.0,
                "skewness": 0.0,
                "sum": 0,
                "mad": math.nan,
            }
        )

    finite_idx = idx[~infinite_mask] if len(idx) else idx
    finite_cnt = cnt[~infinite_mask] if len(idx) else cnt

    if chi_squared_threshold > 0.0:
        stats["chi_squared"] = chi_square(finite_idx)

    stats["range"] = stats["max"] - stats["min"] if present.len() > 0 else 0

    # Quantiles (linear interpolation, to match the previous behaviour).
    for q in quantiles:
        key = f"{q:.0%}"
        stats[key] = (
            present.quantile(q, interpolation="linear") if present.len() > 0 else math.nan
        )

    stats["iqr"] = (
        stats["75%"] - stats["25%"]
        if stats.get("75%") is not None and stats.get("25%") is not None
        else math.nan
    )
    stats["cv"] = stats["std"] / stats["mean"] if stats["mean"] else math.nan
    stats["p_zeros"] = stats["n_zeros"] / summary["n"] if summary["n"] else 0
    stats["p_infinite"] = summary["n_infinite"] / summary["n"] if summary["n"] else 0

    # Monotonicity
    inc = bool(present.is_sorted(descending=False)) if present.len() > 0 else False
    dec = bool(present.is_sorted(descending=True)) if present.len() > 0 else False
    is_unique = present.n_unique() == present.len() if present.len() > 0 else False
    stats["monotonic_increase"] = inc
    stats["monotonic_decrease"] = dec
    stats["monotonic_increase_strict"] = inc and is_unique
    stats["monotonic_decrease_strict"] = dec and is_unique
    if stats["monotonic_increase_strict"]:
        stats["monotonic"] = 2
    elif stats["monotonic_decrease_strict"]:
        stats["monotonic"] = -2
    elif inc:
        stats["monotonic"] = 1
    elif dec:
        stats["monotonic"] = -1
    else:
        stats["monotonic"] = 0

    if len(finite_idx) > 0:
        stats.update(
            histogram_compute(
                config,
                finite_idx,
                summary["n_distinct"],
                weights=finite_cnt,
            )
        )

    return config, series, stats
