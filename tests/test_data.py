"""Tests for data loading and preprocessing."""

import numpy as np
import pytest

from coalmine.constants import L, N
from coalmine.data import (
    accident_dates,
    cumulative_accident_times,
    load_intervals,
    mean_rate,
)


@pytest.fixture
def intervals():
    return load_intervals()


def test_load_intervals_shape(intervals):
    assert intervals.shape == (N - 1,)


def test_load_intervals_non_negative(intervals):
    assert np.all(intervals >= 0)


def test_cumulative_starts_at_zero(intervals):
    times = cumulative_accident_times(intervals)
    assert times[0] == 0
    assert len(times) == N


def test_accident_dates_length(intervals):
    dates = accident_dates(intervals)
    assert len(dates) == N


def test_mean_rate():
    rate = mean_rate()
    assert rate == pytest.approx(N / L)
    assert 0 < rate < 1
