"""Correlations, missing diagrams, duplicates, samples and alerts."""
import polars as pl
import pytest

from polars_profiling import ProfileReport


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # perfectly correlated with a
            "cat": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
            "miss": [1, None, 3, None, 5, None, 7, None, 9, None],
        }
    )


def test_pearson_correlation(df):
    report = ProfileReport(
        df,
        progress_bar=False,
        correlations={"pearson": {"calculate": True}},
    )
    corr = report.get_description().correlations
    assert "pearson" in corr
    matrix = corr["pearson"]
    # a and b are perfectly correlated
    labels = list(matrix.columns)
    ai, bi = labels.index("a"), labels.index("b")
    assert matrix.values[ai, bi] == pytest.approx(1.0, abs=1e-6)


def test_auto_correlation(df):
    report = ProfileReport(df, progress_bar=False)
    corr = report.get_description().correlations
    assert "auto" in corr


def test_spearman_kendall(df):
    report = ProfileReport(
        df,
        progress_bar=False,
        correlations={
            "spearman": {"calculate": True},
            "kendall": {"calculate": True},
        },
    )
    corr = report.get_description().correlations
    assert "spearman" in corr
    assert "kendall" in corr


def test_cramers(df):
    df2 = df.with_columns(pl.col("cat").alias("cat2"))
    report = ProfileReport(
        df2,
        progress_bar=False,
        correlations={"cramers": {"calculate": True}},
    )
    corr = report.get_description().correlations
    assert "cramers" in corr


def test_missing_diagrams(df):
    report = ProfileReport(df, progress_bar=False)
    missing = report.get_description().missing
    # bar/matrix/heatmap are enabled by default
    assert len(missing) >= 1
    for diagram in missing.values():
        assert diagram["matrix"] is not None


def test_duplicates():
    df = pl.DataFrame({"a": [1, 1, 2, 2, 3], "b": ["x", "x", "y", "y", "z"]})
    report = ProfileReport(df, progress_bar=False)
    desc = report.get_description()
    assert desc.duplicates is not None
    assert desc.duplicates.height >= 1


def test_no_duplicates():
    df = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    report = ProfileReport(df, progress_bar=False)
    desc = report.get_description()
    # No duplicate rows -> empty or None
    assert desc.duplicates is None or desc.duplicates.height == 0


def test_samples(df):
    report = ProfileReport(df, progress_bar=False)
    samples = report.get_description().sample
    ids = {s.id for s in samples}
    assert "head" in ids
    assert "tail" in ids


def test_alerts_present(df):
    report = ProfileReport(df, progress_bar=False)
    alerts = report.get_description().alerts
    # b is perfectly correlated with a and miss has 50% missing -> alerts expected
    assert len(alerts) >= 1


def test_interactions(df):
    report = ProfileReport(df, progress_bar=False, interactions={"continuous": True})
    scatter = report.get_description().scatter
    assert isinstance(scatter, dict)
