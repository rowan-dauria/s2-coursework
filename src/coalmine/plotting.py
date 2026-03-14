"""Visualisation helpers for coal mining accident analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from coalmine.constants import START_DATE


def plot_cumulative_accidents(
    dates: pd.DatetimeIndex,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Step plot of cumulative accident count over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    counts = np.arange(1, len(dates) + 1)
    ax.step(dates, counts, where="post", **kwargs)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative number of accidents")
    ax.set_title("Cumulative Coal Mining Accidents (1851–1962)")
    ax.grid(True)
    return ax


def plot_prior_posterior(
    h: np.ndarray,
    prior: np.ndarray,
    posterior: np.ndarray,
    ax: Axes | None = None,
) -> Axes:
    """Overlay prior and posterior PDFs for a rate parameter."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(h, prior, label="Prior", linestyle="--")
    ax.plot(h, posterior, label="Posterior")
    ax.set_xlabel("Rate $h$ (accidents/day)")
    ax.set_ylabel("Probability density")
    ax.legend()
    ax.grid(True)
    return ax


def plot_rate_history(
    intervals: np.ndarray,
    rates: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    ax: Axes | None = None,
) -> Axes:
    """Plot model-averaged rate with optional uncertainty bands.

    Parameters
    ----------
    intervals : ndarray
        Inter-accident intervals (used to build time axis).
    rates : ndarray
        Point-estimate rates at each time bin.
    lower, upper : ndarray, optional
        Credible-interval bounds for shading.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    times = np.insert(np.cumsum(intervals), 0, 0)
    dates = START_DATE + pd.to_timedelta(times[: len(rates)], unit="D")

    ax.step(dates, rates, where="post", label="Rate estimate")
    if lower is not None and upper is not None:
        ax.fill_between(
            dates, lower, upper, step="post", alpha=0.3, label="Credible interval"
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate (accidents/day)")
    ax.set_title("Model-Averaged Accident Rate")
    ax.legend()
    ax.grid(True)
    return ax
