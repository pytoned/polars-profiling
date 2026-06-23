# mypy: ignore-errors

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Union

import numpy as np

from data_profiling.config import Settings
from data_profiling.model import BaseDescription
from data_profiling.model.handler import Handler
from data_profiling.utils.varseries import VarSeries
from data_profiling.model.polars import (
    polars_describe_boolean_1d,
    polars_describe_categorical_1d,
    polars_describe_counts,
    polars_describe_date_1d,
    polars_describe_generic,
    polars_describe_numeric_1d,
    polars_describe_supported,
    polars_describe_text_1d,
)


class BaseSummarizer(Handler):
    """A base summarizer

    Can be used to define custom summarizations
    """

    def summarize(self, config: Settings, series: Any, dtype: Any) -> dict:
        """Generates the summary for a given series"""
        return self.handle(str(dtype), config, series, {"type": str(dtype)})


class ProfilingSummarizer(BaseSummarizer):
    """A summarizer for Polars DataFrames."""

    def __init__(self, typeset: Any):
        self._summary_map = self._create_summary_map()
        super().__init__(self._summary_map, typeset)

    @property
    def summary_map(self) -> Dict[str, List[Callable]]:
        """Allows users to modify the summary map after initialization."""
        return self._summary_map

    def _create_summary_map(self) -> Dict[str, List[Callable]]:
        """Creates the summary map for Polars summarization."""
        summary_map = {
            "Unsupported": [
                polars_describe_counts,
                polars_describe_generic,
                polars_describe_supported,
            ],
            "Numeric": [polars_describe_numeric_1d],
            "DateTime": [polars_describe_date_1d],
            "Text": [polars_describe_text_1d],
            "Categorical": [polars_describe_categorical_1d],
            "Boolean": [polars_describe_boolean_1d],
            # Content-based types fall back to the text description.
            "URL": [polars_describe_text_1d],
            "Path": [polars_describe_text_1d],
            "File": [polars_describe_text_1d],
            "Image": [polars_describe_text_1d],
            "TimeSeries": [polars_describe_numeric_1d],
        }
        return summary_map


def format_summary(summary: Union[BaseDescription, dict]) -> dict:
    """Prepare summary for export to json file.

    Args:
        summary (Union[BaseDescription, dict]): summary to export

    Returns:
        dict: summary as dict
    """

    def fmt(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: fmt(va) for k, va in v.items()}
        else:
            if isinstance(v, VarSeries):
                return fmt(v.to_dict())
            elif (
                isinstance(v, tuple)
                and len(v) == 2
                and all(isinstance(x, np.ndarray) for x in v)
            ):
                return {"counts": v[0].tolist(), "bin_edges": v[1].tolist()}
            else:
                return v

    if isinstance(summary, BaseDescription):
        summary = asdict(summary)

    summary = {k: fmt(v) for k, v in summary.items()}
    return summary


def _redact_column(column: Dict[str, Any]) -> Dict[str, Any]:
    def redact_key(data: Dict[str, Any]) -> Dict[str, Any]:
        return {f"REDACTED_{i}": v for i, (_, v) in enumerate(data.items())}

    def redact_value(data: Dict[str, Any]) -> Dict[str, Any]:
        return {k: f"REDACTED_{i}" for i, (k, _) in enumerate(data.items())}

    keys_to_redact = [
        "block_alias_char_counts",
        "block_alias_values",
        "category_alias_char_counts",
        "category_alias_values",
        "character_counts",
        "script_char_counts",
        "value_counts_index_sorted",
        "value_counts_without_nan",
        "word_counts",
    ]

    values_to_redact = ["first_rows"]

    for field in keys_to_redact:
        if field not in column:
            continue
        is_dict = (isinstance(v, dict) for v in column[field].values())
        if any(is_dict):
            column[field] = {k: redact_key(v) for k, v in column[field].items()}
        else:
            column[field] = redact_key(column[field])

    for field in values_to_redact:
        if field not in column:
            continue
        is_dict = (isinstance(v, dict) for v in column[field].values())
        if any(is_dict):
            column[field] = {k: redact_value(v) for k, v in column[field].items()}
        else:
            column[field] = redact_value(column[field])

    return column


def redact_summary(summary: dict, config: Settings) -> dict:
    """Redact summary to export to json file.

    Args:
        summary (dict): summary to redact

    Returns:
        dict: redacted summary
    """
    for _, col in summary["variables"].items():
        if (config.vars.cat.redact and col["type"] == "Categorical") or (
            config.vars.text.redact and col["type"] == "Text"
        ):
            col = _redact_column(col)
    return summary
