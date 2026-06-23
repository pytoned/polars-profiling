"""Correlations for the Polars backend (native, pandas-free)."""
import itertools
import warnings
from typing import List, Optional

import numpy as np
import polars as pl
from scipy import stats

from data_profiling.config import Settings


class _Labels:
    """A minimal, pandas-Index-like label container."""

    def __init__(self, labels: List[str]) -> None:
        self._labels = list(labels)

    def __iter__(self):
        return iter(self._labels)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, item):
        if isinstance(item, np.ndarray) and item.dtype == bool:
            return _Labels([x for x, keep in zip(self._labels, item) if keep])
        return self._labels[item]

    @property
    def values(self) -> np.ndarray:
        return np.asarray(self._labels, dtype=object)

    def tolist(self) -> list:
        return list(self._labels)


class _NullFrame:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask

    @property
    def values(self) -> np.ndarray:
        return self._mask

    def any(self) -> bool:
        return bool(self._mask.any())


class CorrelationMatrix:
    """A square, labelled correlation matrix without any pandas dependency."""

    def __init__(self, labels: List[str], matrix: np.ndarray) -> None:
        self._labels = list(labels)
        self.matrix = np.asarray(matrix, dtype=float)

    @property
    def columns(self) -> _Labels:
        return _Labels(self._labels)

    @property
    def index(self) -> _Labels:
        return _Labels(self._labels)

    @property
    def values(self) -> np.ndarray:
        return self.matrix

    @property
    def shape(self):
        return self.matrix.shape

    def __len__(self) -> int:
        return len(self._labels)

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.matrix, dtype=dtype)

    def isnull(self) -> _NullFrame:
        return _NullFrame(np.isnan(self.matrix))

    def to_dict(self) -> dict:
        return {
            row: {col: self.matrix[i, j] for j, col in enumerate(self._labels)}
            for i, row in enumerate(self._labels)
        }

    def to_html(self, classes: str = "", float_format=str, **kwargs) -> str:
        """Render the matrix as an HTML table (pandas-compatible signature)."""
        import html as _html
        import math

        def _fmt(value: float) -> str:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return "NaN"
            try:
                return float_format(value)
            except (TypeError, ValueError):
                return str(value)

        cls = f' class="dataframe {classes}"' if classes else ' class="dataframe"'
        header_cells = "".join(
            f"<th>{_html.escape(str(c))}</th>" for c in self._labels
        )
        rows = []
        for i, row_label in enumerate(self._labels):
            cells = "".join(
                f"<td>{_fmt(self.matrix[i, j])}</td>"
                for j in range(len(self._labels))
            )
            rows.append(f"<tr><th>{_html.escape(str(row_label))}</th>{cells}</tr>")
        return (
            f'<table border="1"{cls}>'
            f'<thead><tr style="text-align: right;"><th></th>{header_cells}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table>"
        )


# --- column selection ---------------------------------------------------
def _numeric_columns(summary: dict) -> List[str]:
    return [
        key
        for key, value in summary.items()
        if value["type"] in {"Numeric", "TimeSeries"} and value["n_distinct"] > 1
    ]


def _categorical_columns(config: Settings, summary: dict) -> List[str]:
    threshold = config.categorical_maximum_correlation_distinct
    return [
        key
        for key, value in summary.items()
        if value["type"] in {"Categorical", "Boolean"}
        and 1 < value["n_distinct"] <= threshold
    ]


# --- pairwise primitives ------------------------------------------------
def _to_float(df: pl.DataFrame, col: str) -> np.ndarray:
    return df.get_column(col).cast(pl.Float64, strict=False).to_numpy()


def _pairwise_numeric(a: np.ndarray, b: np.ndarray, method: str) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 2:
        return np.nan
    a, b = a[mask], b[mask]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if method == "pearson":
            r = stats.pearsonr(a, b)[0]
        elif method == "spearman":
            r = stats.spearmanr(a, b)[0]
        else:
            r = stats.kendalltau(a, b)[0]
    return float(r)


def _numeric_matrix(
    df: pl.DataFrame, summary: dict, method: str
) -> Optional[CorrelationMatrix]:
    cols = sorted(_numeric_columns(summary))
    if len(cols) <= 1:
        return None
    data = {c: _to_float(df, c) for c in cols}
    n = len(cols)
    matrix = np.ones((n, n))
    for i, j in itertools.combinations(range(n), 2):
        r = _pairwise_numeric(data[cols[i]], data[cols[j]], method)
        matrix[i, j] = matrix[j, i] = r
    return CorrelationMatrix(cols, matrix)


def pearson_compute(config: Settings, df: pl.DataFrame, summary: dict):
    return _numeric_matrix(df, summary, "pearson")


def spearman_compute(config: Settings, df: pl.DataFrame, summary: dict):
    return _numeric_matrix(df, summary, "spearman")


def kendall_compute(config: Settings, df: pl.DataFrame, summary: dict):
    return _numeric_matrix(df, summary, "kendall")


# --- Cramer's V ---------------------------------------------------------
def _cramers_corrected_stat(confusion: np.ndarray, correction: bool) -> float:
    if confusion.size == 0:
        return 0.0
    chi2 = stats.chi2_contingency(confusion, correction=correction)[0]
    n = confusion.sum()
    phi2 = chi2 / n
    r, k = confusion.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        phi2corr = max(0.0, phi2 - ((k - 1.0) * (r - 1.0)) / (n - 1.0))
        rcorr = r - ((r - 1.0) ** 2.0) / (n - 1.0)
        kcorr = k - ((k - 1.0) ** 2.0) / (n - 1.0)
        rkcorr = min((kcorr - 1.0), (rcorr - 1.0))
        corr = 1.0 if rkcorr == 0.0 else np.sqrt(phi2corr / rkcorr)
    return float(corr)


def _confusion_matrix(df: pl.DataFrame, col1: str, col2: str) -> np.ndarray:
    sub = df.select([pl.col(col1).cast(pl.String), pl.col(col2).cast(pl.String)]).drop_nulls()
    if sub.height == 0:
        return np.empty((0, 0))
    ct = sub.group_by([col1, col2]).agg(pl.len().alias("__count"))
    pivot = ct.pivot(values="__count", index=col1, on=col2).fill_null(0)
    return pivot.drop(col1).to_numpy()


def cramers_compute(
    config: Settings, df: pl.DataFrame, summary: dict
) -> Optional[CorrelationMatrix]:
    cols = sorted(_categorical_columns(config, summary))
    if len(cols) <= 1:
        return None
    n = len(cols)
    matrix = np.ones((n, n))
    for i, j in itertools.combinations(range(n), 2):
        confusion = _confusion_matrix(df, cols[i], cols[j])
        score = _cramers_corrected_stat(confusion, correction=True) if confusion.size else np.nan
        matrix[i, j] = matrix[j, i] = score
    return CorrelationMatrix(cols, matrix)


def phik_compute(config: Settings, df: pl.DataFrame, summary: dict) -> None:
    # phi_k relies on the pandas-based `phik` package; not supported in the
    # Polars-only build.
    return None


# --- Auto ---------------------------------------------------------------
def _discretize(df: pl.DataFrame, col: str, n_bins: int) -> pl.Series:
    s = df.get_column(col).cast(pl.Float64, strict=False)
    try:
        return s.qcut(n_bins, allow_duplicates=True).cast(pl.String)
    except Exception:
        return s.cast(pl.String)


def auto_compute(
    config: Settings, df: pl.DataFrame, summary: dict
) -> Optional[CorrelationMatrix]:
    numerical = _numeric_columns(summary)
    categorical = _categorical_columns(config, summary)
    columns = sorted(numerical + categorical)
    if len(columns) <= 1:
        return None

    n_bins = config.correlations["auto"].n_bins
    # Pre-compute float views (numeric) and discretized string views.
    float_view = {c: _to_float(df, c) for c in numerical}
    n = len(columns)
    matrix = np.ones((n, n))

    for i, j in itertools.combinations(range(n), 2):
        c1, c2 = columns[i], columns[j]
        both_numeric = c1 in numerical and c2 in numerical
        if both_numeric:
            score = _pairwise_numeric(float_view[c1], float_view[c2], "spearman")
        else:
            # Cramer's V; discretize numeric partners first.
            cols_for_ct = {}
            tmp = df.select(
                [
                    (_discretize(df, c, n_bins) if c in numerical else pl.col(c).cast(pl.String)).alias(c)
                    for c in (c1, c2)
                ]
            )
            confusion = _confusion_matrix(tmp, c1, c2)
            score = (
                _cramers_corrected_stat(confusion, correction=True)
                if confusion.size
                else np.nan
            )
        matrix[i, j] = matrix[j, i] = score

    return CorrelationMatrix(columns, matrix)
