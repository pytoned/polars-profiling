"""Main module of data-profiling (Polars edition).

.. include:: ../../README.md
"""
import warnings  # isort:skip # noqa

from data_profiling.compare_reports import compare  # isort:skip # noqa
from data_profiling.controller import polars_decorator  # isort:skip # noqa
from data_profiling.profile_report import ProfileReport  # isort:skip # noqa
from data_profiling.version import __version__  # isort:skip # noqa

# backend (registers the Polars implementations)
import data_profiling.model.polars  # isort:skip  # noqa

__all__ = [
    "polars_decorator",
    "ProfileReport",
    "__version__",
    "compare",
]
