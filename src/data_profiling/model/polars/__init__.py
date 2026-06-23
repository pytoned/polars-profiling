"""Polars backend for data-profiling.

Importing this package registers the Polars implementations (via
``multimethod`` and the summarizer map) for table stats, samples, duplicates,
correlations and missing-data diagrams.
"""
from data_profiling.model.polars.describe_boolean_polars import (
    polars_describe_boolean_1d,
)
from data_profiling.model.polars.describe_categorical_polars import (
    polars_describe_categorical_1d,
)
from data_profiling.model.polars.describe_counts_polars import (
    polars_describe_counts,
    polars_describe_generic,
    polars_describe_supported,
)
from data_profiling.model.polars.describe_date_polars import polars_describe_date_1d
from data_profiling.model.polars.describe_numeric_polars import (
    polars_describe_numeric_1d,
)
from data_profiling.model.polars.describe_text_polars import polars_describe_text_1d
from data_profiling.model.polars.summary_polars import (
    polars_describe_1d,
    polars_get_series_descriptions,
)

# Imported for their dispatch / multimethod registration side effects.
from data_profiling.model.polars import (  # noqa: E402,F401  isort:skip
    correlations_polars,
    dataframe_polars,
    duplicates_polars,
    missing_polars,
    sample_polars,
    table_polars,
)

__all__ = [
    "polars_describe_counts",
    "polars_describe_generic",
    "polars_describe_supported",
    "polars_describe_numeric_1d",
    "polars_describe_categorical_1d",
    "polars_describe_boolean_1d",
    "polars_describe_date_1d",
    "polars_describe_text_1d",
    "polars_describe_1d",
    "polars_get_series_descriptions",
]
