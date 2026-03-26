"""Visualisation helpers for coal mining accident analysis."""

from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from coalmine.constants import START_DATE, L
from coalmine.analysis import gamma_prior_pdf, m0_posterior_pdf, m0_posterior_dist

# Colourblind-friendly palette (Tol bright)
CB_COLOURS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

FIGS_DIR = Path(__file__).resolve().parents[2] / "figs"


def savefig(fig: Figure, name: str, **kwargs) -> Path:
    """Save *fig* to the figs/ directory as a PDF and PNG.

    Returns the stem path (without extension) for reference.
    """
    FIGS_DIR.mkdir(exist_ok=True)
    defaults = dict(dpi=300, bbox_inches="tight")
    defaults.update(kwargs)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS_DIR / f"{name}.{ext}", **defaults)
    return FIGS_DIR / name


def autosave(fn):
    """Decorator that adds a ``save_as`` kwarg to a plotting function.

    If ``save_as`` is passed, the returned Figure (or the Figure owning the
    returned Axes) is saved to ``figs/<save_as>.{pdf,png}`` automatically.

    Usage in a notebook::

        plot_cumulative_accidents(dates, save_as="q1_cumulative")
    """
    @wraps(fn)
    def wrapper(*args, save_as=None, **kwargs):
        result = fn(*args, **kwargs)
        if save_as is not None:
            fig = result if isinstance(result, Figure) else result.get_figure()
            savefig(fig, save_as)
        return result
    return wrapper


@autosave
def plot_cumulative_accidents(
    dates: pd.DatetimeIndex,
    mean_rate: float | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> Axes:
    """Step plot of cumulative accident count over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    counts = np.arange(1, len(dates) + 1)
    ax.step(dates, counts, where="post", **kwargs)
    if mean_rate is not None:
        t_days = np.array([0, L])
        mean_dates = START_DATE + pd.to_timedelta(t_days, unit="D")
        ax.plot(mean_dates, mean_rate * t_days, linestyle="--", color=CB_COLOURS[1],
                label=f"Mean rate ({mean_rate:.5f} acc/day)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative number of accidents")
    ax.set_title("Cumulative Coal Mining Accidents (1851–1962)")
    ax.legend()
    ax.grid(True)
    return ax


@autosave
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


@autosave
def plot_m0_prior_posterior(ax: Axes | None = None, n_points: int = 500) -> Axes:
    """Plot M0 prior and normalised posterior on a sensible h range."""
    dist = m0_posterior_dist()
    h = np.linspace(0, dist.mean() + 5 * dist.std(), n_points)
    return plot_prior_posterior(h, gamma_prior_pdf(h), m0_posterior_pdf(h), ax=ax)


@autosave
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


def set_corner_titles(
    fig: Figure,
    flat_samples: np.ndarray,
    param_labels: list[str],
    param_fmts: list[str],
) -> None:
    """Set diagonal titles on a corner plot with per-parameter formatting.

    Each diagonal panel gets a title of the form
    ``label = median ^{+upper}_{-lower}`` using the 16th/50th/84th percentiles.

    Parameters
    ----------
    fig : Figure
        The corner-plot figure (as returned by ``corner.corner``).
    flat_samples : (n_samples, ndim) array
        Flattened MCMC chain.
    param_labels : list of str
        LaTeX-style labels for each parameter.
    param_fmts : list of str
        Format specifiers (e.g. ``".4f"``, ``".0f"``) for each parameter.
    """
    ndim = flat_samples.shape[1]
    axes = np.array(fig.axes).reshape((ndim, ndim))
    for i in range(ndim):
        q_lo, q_mid, q_hi = np.percentile(flat_samples[:, i], [16, 50, 84])
        q_plus = q_hi - q_mid
        q_minus = q_mid - q_lo
        fmt = param_fmts[i]
        title = (
            f"{param_labels[i]} = {q_mid:{fmt}}"
            + r"$^{+"
            + f"{q_plus:{fmt}}"
            + r"}_{-"
            + f"{q_minus:{fmt}}"
            + r"}$"
        )
        axes[i, i].set_title(title)


@autosave
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
