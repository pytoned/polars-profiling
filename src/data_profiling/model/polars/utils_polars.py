"""Small numeric helpers for the Polars backend (no pandas)."""
from typing import Union

import numpy as np
from numpy import log2
from scipy.stats import entropy

from data_profiling.utils.varseries import VarSeries


def column_imbalance_score(
    value_counts: Union[VarSeries, np.ndarray], n_classes: int
) -> Union[float, int]:
    """Bounded (0..1) class-imbalance score based on entropy.

    A perfectly uniform distribution returns 0; a perfectly imbalanced one
    returns 1.
    """
    if n_classes > 1:
        counts = np.asarray(value_counts, dtype=float)
        return 1 - (entropy(counts, base=2) / log2(n_classes))
    return 0


def weighted_median(data: np.ndarray, weights: np.ndarray) -> float:
    """Weighted median of ``data`` given integer/float ``weights``."""
    data = np.asarray(data)
    weights = np.asarray(weights)

    s_data, s_weights = map(np.sort, [data, weights])
    midpoint = 0.5 * np.sum(s_weights)

    if s_weights[-1] > midpoint:
        w_median = data[weights == np.max(weights)][0]
    else:
        cs_weights = np.cumsum(s_weights)
        idx = np.where(cs_weights <= midpoint)[0][-1]
        if cs_weights[idx] == midpoint:
            w_median = np.mean(s_data[idx : idx + 2])
        else:
            w_median = s_data[idx + 1]
    return w_median
