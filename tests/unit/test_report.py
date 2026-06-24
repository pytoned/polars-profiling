"""Report rendering / export / options tests."""
import json
import subprocess
import sys

import polars as pl
import pytest

from polars_profiling import ProfileReport, compare


def test_imports_without_setuptools():
    """The package must import without `pkg_resources`/`setuptools` present
    (it is not installed by default on Python 3.12+)."""
    code = (
        "import sys; sys.modules['pkg_resources'] = None;\n"
        "import polars as pl\n"
        "from polars_profiling import ProfileReport\n"
        "ProfileReport(pl.DataFrame({'a': [1, 2, 3]}), progress_bar=False).to_html()\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def df():
    return pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": ["x", "y", "x", "z", "y", "x", "x", "y", "z", "x"],
            "c": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1],
        }
    )


def test_to_html(df):
    html = ProfileReport(df, title="T", progress_bar=False).to_html()
    assert isinstance(html, str)
    assert "T" in html
    assert len(html) > 1000


def test_to_html_has_no_ydata(df):
    html = ProfileReport(df, progress_bar=False).to_html()
    assert "ydata" not in html.lower()
    assert "polars-profiling" in html


def test_to_file(df, tmp_path):
    out = tmp_path / "report.html"
    ProfileReport(df, progress_bar=False).to_file(str(out))
    assert out.exists()
    assert out.stat().st_size > 1000


def test_to_json(df):
    payload = ProfileReport(df, progress_bar=False).to_json()
    data = json.loads(payload)
    assert "table" in data
    assert "variables" in data


def test_minimal_mode(df):
    html = ProfileReport(df, minimal=True, progress_bar=False).to_html()
    assert len(html) > 1000


def test_explorative_mode(df):
    html = ProfileReport(df, explorative=True, progress_bar=False).to_html()
    assert len(html) > 1000


def test_disable_features(df):
    report = ProfileReport(
        df,
        progress_bar=False,
        correlations=None,
        missing_diagrams=None,
        interactions=None,
        samples=None,
        duplicates=None,
    )
    assert len(report.to_html()) > 1000


def test_default_title(df):
    report = ProfileReport(df, progress_bar=False)
    assert report.config.title == "Polars Profiling Report"


def test_dataframe_accessor(df):
    report = df.profile_report(progress_bar=False, title="Accessor")
    assert isinstance(report, ProfileReport)
    assert "Accessor" in report.to_html()


def test_compare(df):
    df2 = df.with_columns(pl.col("a") + 1)
    r1 = ProfileReport(df, title="A", progress_bar=False)
    r2 = ProfileReport(df2, title="B", progress_bar=False)
    html = compare([r1, r2]).to_html()
    assert "A" in html and "B" in html


def test_empty_dataframe_raises():
    with pytest.raises(ValueError):
        ProfileReport(pl.DataFrame({"a": []}), progress_bar=False)


def test_non_polars_input_raises():
    from typeguard import TypeCheckError

    with pytest.raises((TypeError, TypeCheckError)):
        ProfileReport({"a": [1, 2, 3]}, progress_bar=False)


def test_type_schema_override(df):
    report = ProfileReport(
        df, type_schema={"a": "categorical"}, progress_bar=False
    )
    assert report.get_description().variables["a"]["type"] == "Categorical"
