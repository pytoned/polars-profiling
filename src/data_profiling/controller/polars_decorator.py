"""This file adds the ``profile_report`` method on the Polars DataFrame."""
from polars import DataFrame

from data_profiling.profile_report import ProfileReport


def profile_report(df: DataFrame, **kwargs) -> ProfileReport:
    """Profile a Polars DataFrame.

    Args:
        df: The DataFrame to profile.
        **kwargs: Optional arguments for the ProfileReport object.

    Returns:
        A ProfileReport of the DataFrame.
    """
    return ProfileReport(df, **kwargs)


DataFrame.profile_report = profile_report
