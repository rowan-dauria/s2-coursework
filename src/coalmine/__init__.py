"""Bayesian analysis of UK coal mining accident rates (1851-1962)."""

from coalmine.constants import START_DATE, L, N
from coalmine.data import load_intervals

__all__ = ["L", "N", "START_DATE", "load_intervals"]
