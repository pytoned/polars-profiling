"""Categorical variable description for the Polars backend."""
from typing import Tuple

import polars as pl

from data_profiling.config import Settings
from data_profiling.model.polars.text_helpers import (
    length_summary_vc,
    unicode_summary_vc,
    word_summary_vc,
)
from data_profiling.model.polars.utils_polars import column_imbalance_score
from data_profiling.model.summary_algorithms import chi_square, histogram_compute
from data_profiling.utils.varseries import VarSeries


def polars_describe_categorical_1d(
    config: Settings, series: pl.Series, summary: dict
) -> Tuple[Settings, pl.Series, dict]:
    """Describe a categorical series."""
    # Operate on a string view of the categories.
    str_series = series.drop_nulls().cast(pl.String, strict=False)

    value_counts: VarSeries = summary["value_counts_without_nan"].astype_index_str()
    summary["value_counts_without_nan"] = value_counts
    if "value_counts_index_sorted" in summary:
        summary["value_counts_index_sorted"] = value_counts.sort_index(ascending=True)

    summary["imbalance"] = column_imbalance_score(
        value_counts.values, len(value_counts)
    )

    if not config.vars.cat.redact:
        summary["first_rows"] = VarSeries(
            str_series.head(5).to_numpy(),
            index=range(min(5, str_series.len())),
        )

    if config.vars.num.chi_squared_threshold > 0.0:
        summary["chi_squared"] = chi_square(histogram=value_counts.values)

    if config.vars.cat.length:
        summary.update(length_summary_vc(value_counts))
        summary.update(
            histogram_compute(
                config,
                summary["length_histogram"].index,
                len(summary["length_histogram"]),
                name="histogram_length",
                weights=summary["length_histogram"].values,
            )
        )

    if config.vars.cat.characters:
        summary.update(unicode_summary_vc(value_counts))

    if config.vars.cat.words:
        summary.update(word_summary_vc(value_counts, config.vars.cat.stop_words))

    return config, series, summary
