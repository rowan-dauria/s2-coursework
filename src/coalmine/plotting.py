"""Visualisation helpers for coal mining accident analysis."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from coalmine.constants import START_DATE

# Colourblind-friendly palette (Tol bright)
CB_COLOURS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]


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
    ax.plot(h, prior, label="Prior", linestyle="--", color=CB_COLOURS[0])
    ax.plot(h, posterior, label="Posterior", color=CB_COLOURS[1])
    ax.set_xlabel("Rate $h$ (accidents/day)")
    ax.set_ylabel("Probability density")
    ax.legend()
    ax.grid(True)
    return ax


def plot_changepoint_prior_comparison(
    plain_samples: np.ndarray,
    even_samples: np.ndarray,
    k: int,
    L: float,
) -> Figure:
    """Compare marginal densities and min-gap distributions for two priors.

    Parameters
    ----------
    plain_samples, even_samples : (n_samples, k) arrays of change-point positions.
    k : number of change points.
    L : total interval length.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: marginal densities of each change point
    ax = axes[0]
    for j in range(k):
        c = CB_COLOURS[j % len(CB_COLOURS)]
        ax.hist(
            plain_samples[:, j], bins=80, density=True, alpha=0.35,
            color=c, label=f"Plain $s_{j+1}$",
        )
        ax.hist(
            even_samples[:, j], bins=80, density=True, alpha=0.35,
            color=c, histtype="step", linewidth=1.5,
            label=f"Even $s_{j+1}$",
        )
    ax.set_xlabel("Position (days)")
    ax.set_ylabel("Density")
    ax.set_title(f"Marginal change-point densities ($k={k}$)")
    ax.legend(fontsize=7, ncol=2)
    ax.set_xlim(0, L)

    # Panel 2: minimum gap distribution
    ax = axes[1]
    plain_gaps = np.diff(
        np.column_stack([np.zeros(len(plain_samples)), plain_samples,
                         np.full(len(plain_samples), L)]),
        axis=1,
    )
    even_gaps = np.diff(
        np.column_stack([np.zeros(len(even_samples)), even_samples,
                         np.full(len(even_samples), L)]),
        axis=1,
    )
    ax.hist(plain_gaps.min(axis=1), bins=80, density=True, alpha=0.5, label="Plain", color=CB_COLOURS[0])
    ax.hist(even_gaps.min(axis=1), bins=80, density=True, alpha=0.5, label="Even", color=CB_COLOURS[1])
    ax.set_xlabel("Minimum gap (days)")
    ax.set_ylabel("Density")
    ax.set_title(f"Distribution of minimum gap ($k={k}$)")
    ax.legend()

    fig.tight_layout()
    return fig


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

    ax.step(dates, rates, where="post", label="Rate estimate", color=CB_COLOURS[0])
    if lower is not None and upper is not None:
        ax.fill_between(
            dates, lower, upper, step="post", alpha=0.3, label="Credible interval",
            color=CB_COLOURS[0],
        )
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate (accidents/day)")
    ax.set_title("Model-Averaged Accident Rate")
    ax.legend()
    ax.grid(True)
    return ax
