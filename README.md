# polars-profiling

<p align="center">
  <strong>Profile your <a href="https://pola.rs">Polars</a> DataFrames with a single line of code.</strong>
</p>

<p align="center">
  <a href="https://github.com/pytoned/polars-profiling/actions/workflows/tests.yml"><img alt="Build" src="https://github.com/pytoned/polars-profiling/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://pypi.org/project/polars-profiling/"><img alt="PyPI" src="https://img.shields.io/pypi/v/polars-profiling?color=blue&label=pypi"></a>
  <a href="https://github.com/pytoned/polars-profiling/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="Polars only" src="https://img.shields.io/badge/engine-polars--only-orange.svg">
</p>

`polars-profiling` generates a rich, interactive **HTML exploratory-data-analysis
report** from a Polars `DataFrame`. It is a fork of
[ydata-profiling](https://github.com/ydataai/ydata-profiling) (formerly
*pandas-profiling*) **rebuilt to run exclusively on Polars** — pandas, Spark,
and `visions` have been removed entirely, and every statistic is computed
natively with Polars expressions.

For each column you get type detection, descriptive statistics, quantiles,
histograms, common/extreme values, correlations, missing-value diagnostics,
interactions, duplicate-row detection, samples, and automatic data-quality
alerts.

---

## Why Polars-only?

| | ydata-profiling | **polars-profiling** |
| --- | --- | --- |
| Input | pandas / Spark | **Polars** |
| Compute engine | pandas / numpy | **native Polars expressions** |
| Heavy dependencies | pandas, visions, seaborn, statsmodels, phik | **none of them** |
| Telemetry | phones home | **none** |

The full data path — value counts, distinct/unique, numeric statistics,
quantiles, histograms, correlations, missing-data nullity, duplicates and
sampling — runs through Polars. There is **no `.to_pandas()`** anywhere in the
pipeline.

## Installation

```bash
pip install polars-profiling
```

## Quickstart

```python
import polars as pl
from data_profiling import ProfileReport

df = pl.read_csv("titanic.csv")

profile = ProfileReport(df, title="Titanic Dataset", explorative=True)
profile.to_file("titanic_report.html")
```

Or use the DataFrame accessor:

```python
df.profile_report().to_file("report.html")
```

### In a Jupyter notebook

```python
profile = ProfileReport(df, title="Profiling Report")
profile.to_notebook_iframe()
```

### Get the raw statistics (no HTML)

```python
description = ProfileReport(df, progress_bar=False).get_description()
print(description.table["types"])      # {'Numeric': 7, 'Text': 3, 'Categorical': 2}
print(description.variables["Age"]["mean"])
```

## Features

- **Type inference** — Numeric, Categorical, Boolean, DateTime, Text (with
  configurable, content-based inference for string columns).
- **Univariate statistics** — count, distinct/unique, missing, mean, std,
  variance, min/max, quantiles, MAD, skewness, kurtosis, monotonicity, zeros,
  negatives, infinities.
- **Histograms & frequency tables**, common and extreme values.
- **Correlations** — `auto`, Pearson, Spearman, Kendall and Cramér's V,
  computed natively.
- **Missing-value diagnostics** — count bar, nullity matrix and correlation
  heatmap.
- **Interactions** — pairwise scatter plots between continuous variables.
- **Duplicate-row detection** and **head/tail/random samples**.
- **Data-quality alerts** — high correlation, high cardinality, imbalance,
  missing values, zeros, uniqueness, constants and more.
- **Report comparison** — diff two datasets side by side.

## Configuration

`ProfileReport` accepts a familiar set of keyword arguments. A few common ones:

```python
ProfileReport(
    df,
    title="My Report",
    minimal=True,              # faster, fewer computations
    explorative=True,          # enable extra analyses
    correlations={"pearson": {"calculate": True}},
    missing_diagrams={"heatmap": False},
    samples={"head": 5, "tail": 5},
    type_schema={"col": "categorical"},  # override inferred types
    progress_bar=False,
)
```

### Comparing two reports

```python
from data_profiling import compare

report_a = ProfileReport(df_a, title="A")
report_b = ProfileReport(df_b, title="B")
compare([report_a, report_b]).to_file("comparison.html")
```

## Differences from ydata-profiling

- Input must be a **Polars `DataFrame`** (pandas/Spark are not accepted).
- `phi_k` correlation is not available (it is a pandas-only package); the other
  correlation methods are reimplemented natively.
- The Great Expectations export converts to pandas at that single boundary only
  (and requires the optional `great_expectations` package, which itself depends
  on pandas).
- No telemetry is collected or sent.

## License

MIT — see [LICENSE](LICENSE). This project builds on the work of
ydata-profiling and pandas-profiling.
