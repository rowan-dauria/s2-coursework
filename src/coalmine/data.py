"""Data loading and preprocessing."""

from pathlib import Path

import numpy as np
import pandas as pd

from coalmine.constants import START_DATE, L, N


def load_intervals(
    path: str | Path = "data/coal_mining_accident_data.dat",
) -> np.ndarray:
    """Load inter-accident intervals in chronological order.

    The raw file is a 17×10 matrix read column-wise (Fortran order).

    Returns
    -------
    intervals : ndarray of shape (190,)
        Inter-accident intervals in days.
    """
    data_matrix = np.loadtxt(path)
    intervals = data_matrix.flatten(order="F")
    assert intervals.shape == (N - 1,), (
        f"Expected {N - 1} intervals, got {intervals.shape[0]}"
    )
    return intervals


def cumulative_accident_times(intervals: np.ndarray) -> np.ndarray:
    """Compute cumulative accident times from day zero.

    Returns
    -------
    times : ndarray of shape (191,)
        Times of each accident relative to the start of observation,
        with times[0] = 0 (first accident at the start).
    """
    return np.insert(np.cumsum(intervals), 0, 0)


def accident_dates(intervals: np.ndarray) -> pd.DatetimeIndex:
    """Convert inter-accident intervals to calendar dates.

    Returns
    -------
    dates : DatetimeIndex of length 191
    """
    times = cumulative_accident_times(intervals)
    return START_DATE + pd.to_timedelta(times, unit="D")


def mean_rate() -> float:
    """Overall mean accident rate N/L (accidents per day)."""
    return N / L
