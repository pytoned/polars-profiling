from typing import Optional, Tuple, Union

import numpy as np
from scipy.stats import chisquare

from data_profiling.config import Settings


def safe_histogram(
    values: np.ndarray,
    bins: Union[int, str, np.ndarray] = "auto",
    weights: Optional[np.ndarray] = None,
    density: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper to avoid
    ValueError: Too many bins for data range. Cannot create N finite-sized bins.
    """
    try:
        return np.histogram(values, bins=bins, weights=weights, density=density)
    except ValueError as exc:
        if "Too many bins for data range" in str(exc):
            try:
                return np.histogram(
                    values, bins="auto", weights=weights, density=density
                )
            except ValueError:
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    return np.array([]), np.array([])
                vmin = float(np.min(finite))
                vmax = float(np.max(finite))
                if vmin == vmax:
                    eps = 0.5 if vmin == 0 else abs(vmin) * 0.5
                    bin_edges = np.array([vmin - eps, vmin + eps])
                else:
                    bin_edges = np.array([vmin, vmax])
                return np.histogram(
                    values, bins=bin_edges, weights=weights, density=density
                )
        raise


def histogram_compute(
    config: Settings,
    finite_values: np.ndarray,
    n_unique: int,
    name: str = "histogram",
    weights: Optional[np.ndarray] = None,
) -> dict:
    stats = {}
    if len(finite_values) == 0:
        return {name: []}

    hist_config = config.plot.histogram

    # Compute data range
    finite = finite_values[np.isfinite(finite_values)]
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    data_range = vmax - vmin

    # Choose of Bins based on observed data values
    if data_range == 0:
        eps = 0.5 if vmin == 0 else abs(vmin) * 0.1
        bins = np.array([vmin - eps, vmin + eps])
    else:
        requested_bins = hist_config.bins if hist_config.bins > 0 else "auto"

        if isinstance(requested_bins, int):
            safe_bins = min(requested_bins, n_unique, hist_config.max_bins)

            safe_bins = max(1, safe_bins)

            bins = np.linspace(vmin, vmax, safe_bins + 1)
        else:
            bins = np.histogram_bin_edges(finite_values, bins="auto")
            if len(bins) - 1 > hist_config.max_bins:
                bins = np.linspace(vmin, vmax, hist_config.max_bins + 1)

    hist = np.histogram(
        finite_values,
        bins=bins,
        weights=weights,
        density=hist_config.density,
    )

    stats[name] = hist
    return stats


def chi_square(
    values: Optional[np.ndarray] = None,
    histogram: Optional[np.ndarray] = None,
) -> dict:
    # Case 1: histogram not passed → we compute it
    if histogram is None:
        if values is None:
            return {"statistic": 0, "pvalue": 0}

        # Try NumPy "auto" binning (may fail under NumPy 2)
        try:
            bins = np.histogram_bin_edges(values, bins="auto")
        except ValueError:
            # Fallback: basic 1-bin histogram covering the min→max range
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                return {"statistic": 0, "pvalue": 0}

            vmin = float(finite.min())
            vmax = float(finite.max())
            if vmin == vmax:
                bins = np.array([vmin - 0.5, vmin + 0.5])
            else:
                bins = np.array([vmin, vmax])

        histogram, _ = np.histogram(values, bins=bins)

    # Case 2: histogram exists but is empty
    if histogram.size == 0 or histogram.sum() == 0:
        return {"statistic": 0, "pvalue": 0}

    return dict(chisquare(histogram)._asdict())
