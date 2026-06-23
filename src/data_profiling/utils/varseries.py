"""A minimal, numpy-backed ordered series container.

This replaces the small subset of the ``pandas.Series`` API that the report
and visualisation layers rely on (frequency tables, value counts, histograms),
allowing ``data-profiling`` to operate on Polars data without any pandas
dependency.

A :class:`VarSeries` holds an ``index`` (the labels/categories) and ``values``
(the associated counts/measures), both stored as numpy arrays. Only the
operations that are actually used across the rendering pipeline are
implemented.
"""
from __future__ import annotations

from typing import Any, Iterator, Tuple

import numpy as np


class _ILocIndexer:
    """Positional indexer mimicking ``pandas.Series.iloc``."""

    def __init__(self, parent: "VarSeries") -> None:
        self._parent = parent

    def __getitem__(self, item: Any) -> Any:
        index = self._parent.index
        values = self._parent.values
        if isinstance(item, slice):
            return VarSeries(values[item], index=index[item])
        result = values[item]
        if isinstance(item, (list, np.ndarray)):
            return VarSeries(result, index=index[item])
        return result


class VarSeries:
    """A lightweight ordered series (index + values) backed by numpy."""

    def __init__(self, values: Any, index: Any = None) -> None:
        self._values = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=object) \
            if _needs_object(values) else np.asarray(values)
        if index is None:
            self._index = np.arange(len(self._values))
        else:
            self._index = np.asarray(list(index) if not isinstance(index, np.ndarray) else index, dtype=object) \
                if _needs_object(index) else np.asarray(index)

    # -- core attributes -------------------------------------------------
    @property
    def index(self) -> np.ndarray:
        return self._index

    @index.setter
    def index(self, new_index: Any) -> None:
        self._index = np.asarray(list(new_index), dtype=object) if _needs_object(new_index) else np.asarray(new_index)

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def iloc(self) -> _ILocIndexer:
        return _ILocIndexer(self)

    @property
    def empty(self) -> bool:
        return len(self._values) == 0

    # -- dunder ----------------------------------------------------------
    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            # Positional slicing (supports reverse via [::-1])
            return VarSeries(self._values[item], index=self._index[item])
        if isinstance(item, np.ndarray) and item.dtype == bool:
            return VarSeries(self._values[item], index=self._index[item])
        if isinstance(item, (list, np.ndarray)):
            return VarSeries(self._values[item], index=self._index[item])
        return self._values[item]

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        pairs = ", ".join(f"{i!r}: {v!r}" for i, v in zip(self._index, self._values))
        return f"VarSeries({{{pairs}}})"

    # -- iteration helpers ----------------------------------------------
    def items(self) -> Iterator[Tuple[Any, Any]]:
        return zip(self._index.tolist(), self._values.tolist())

    def keys(self) -> np.ndarray:
        return self._index

    # -- reductions ------------------------------------------------------
    # ``*args``/``**kwargs`` absorb numpy's ``axis``/``out`` arguments so that
    # ``np.sum(varseries)`` and friends dispatch correctly.
    def sum(self, *args: Any, **kwargs: Any) -> Any:
        return self._values.sum() if len(self._values) else 0

    def max(self, *args: Any, **kwargs: Any) -> Any:
        return self._values.max() if len(self._values) else 0

    def min(self, *args: Any, **kwargs: Any) -> Any:
        return self._values.min() if len(self._values) else 0

    def count(self) -> int:
        return int(len(self._values))

    def nunique(self) -> int:
        return int(len(np.unique(self._values)))

    # -- transforms ------------------------------------------------------
    def head(self, n: int = 5) -> "VarSeries":
        return VarSeries(self._values[:n], index=self._index[:n])

    def sort_index(self, ascending: bool = True) -> "VarSeries":
        order = np.argsort(self._index, kind="stable")
        if not ascending:
            order = order[::-1]
        return VarSeries(self._values[order], index=self._index[order])

    def sort_values(self, ascending: bool = True) -> "VarSeries":
        order = np.argsort(self._values, kind="stable")
        if not ascending:
            order = order[::-1]
        return VarSeries(self._values[order], index=self._index[order])

    def to_dict(self) -> dict:
        return {k: v for k, v in zip(self._index.tolist(), self._values.tolist())}

    def to_numpy(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self._values, dtype=dtype)

    def astype_index_str(self) -> "VarSeries":
        """Return a copy with the index cast to strings."""
        return VarSeries(self._values, index=np.asarray([str(i) for i in self._index], dtype=object))


def _needs_object(values: Any) -> bool:
    """Heuristic to decide whether an object-dtype numpy array is required."""
    if isinstance(values, np.ndarray):
        return values.dtype == object
    try:
        sample = next(iter(values))
    except (TypeError, StopIteration):
        return False
    return isinstance(sample, str)
