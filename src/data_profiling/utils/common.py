"""Common util functions (e.g. missing in Python)."""

import collections.abc
import contextlib
import os
import platform
import subprocess
import zipfile
from datetime import datetime, timedelta

# Image type detection
from pathlib import Path
from typing import Mapping

import polars as pl
import requests

from data_profiling.version import __version__


def update(d: dict, u: Mapping) -> dict:
    """Recursively update a dict.

    Args:
        d: Dictionary to update.
        u: Dictionary with values to use.

    Returns:
        The merged dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def _copy(self, target):
    """Monkeypatch for pathlib

    Args:
        self:
        target:

    Returns:

    """
    import shutil

    assert self.is_file()
    shutil.copy(str(self), target)


Path.copy = _copy  # type: ignore


def extract_zip(outfile, effective_path):
    try:
        with zipfile.ZipFile(outfile) as z:
            z.extractall(effective_path)
    except zipfile.BadZipFile as e:
        raise ValueError("Bad zip file") from e


def convert_timestamp_to_datetime(timestamp: int) -> datetime:
    if timestamp >= 0:
        return datetime.fromtimestamp(timestamp)
    else:
        return datetime(1970, 1, 1) + timedelta(seconds=int(timestamp))


def analytics_features(
    dataframe: str, datatype: str, report_type: str, ncols: int, nrows: int, dbx: str
) -> None:
    """No-op. polars-profiling does not collect or send any telemetry."""
    return None


def is_running_in_databricks():
    mask = "DATABRICKS_RUNTIME_VERSION" in os.environ
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        return os.environ["DATABRICKS_RUNTIME_VERSION"]
    else:
        return str(mask)


def calculate_nrows(df):
    """
    Calculates the number of rows of a Polars DataFrame.

    Returns: int, number of rows
    """
    if isinstance(df, pl.DataFrame):
        return df.height
    return 0
