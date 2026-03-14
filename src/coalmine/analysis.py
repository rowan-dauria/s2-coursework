"""Statistical analysis: priors, posteriors, evidence, and likelihoods."""

import numpy as np
from scipy import special, integrate
from scipy.stats import gamma as gamma_dist

from coalmine.constants import ALPHA, BETA, L, N


def gamma_prior_pdf(h: np.ndarray, alpha: float = ALPHA, beta: float = BETA) -> np.ndarray:
    """Gamma(alpha, beta) prior PDF for rate heights h."""
    return gamma_dist.pdf(h, a=alpha, scale=1 / beta)


def gamma_prior_logpdf(h: np.ndarray, alpha: float = ALPHA, beta: float = BETA) -> np.ndarray:
    """Log of Gamma(alpha, beta) prior PDF for rate heights h."""
    return gamma_dist.logpdf(h, a=alpha, scale=1 / beta)


# ── M0: constant-rate model ──────────────────────────────────────────────


def m0_log_likelihood(h: float | np.ndarray) -> np.ndarray:
    """Log-likelihood for M0 (constant rate h) given N events in time L."""
    h = np.asarray(h)
    return N * np.log(h) - h * L


def m0_posterior_unnorm(h: np.ndarray) -> np.ndarray:
    """Unnormalised M0 posterior: likelihood × prior."""
    return np.exp(m0_log_likelihood(h) + gamma_prior_logpdf(h))


def m0_posterior_pdf(h: np.ndarray) -> np.ndarray:
    """Normalised M0 posterior — Gamma(alpha + N, beta + L)."""
    return gamma_dist.pdf(h, a=ALPHA + N, scale=1 / (BETA + L))


def m0_evidence() -> float:
    """Analytic evidence Z_0 for M0.

    Z_0 = beta^alpha / Gamma(alpha) × Gamma(alpha + N) / (beta + L)^(alpha + N)
    """
    log_z = (
        ALPHA * np.log(BETA)
        - special.gammaln(ALPHA)
        + special.gammaln(ALPHA + N)
        - (ALPHA + N) * np.log(BETA + L)
    )
    return np.exp(log_z)


def m0_evidence_numerical(
    h_upper: float = 0.05, epsrel: float = 1e-12
) -> tuple[float, float]:
    """Numerical evidence Z_0 via quadrature.

    Returns (evidence, absolute_error).
    """
    result, err = integrate.quad(m0_posterior_unnorm, 0, h_upper, epsrel=epsrel)
    return result, err


# ── M1: single change-point model ────────────────────────────────────────


def m1_log_likelihood(
    h0: float, h1: float, s: float, intervals: np.ndarray
) -> float:
    """Log-likelihood for M1 with rates h0, h1 and change point at time s.

    Parameters
    ----------
    h0, h1 : float
        Rates before and after the change point.
    s : float
        Change-point location (days from start of observation).
    intervals : ndarray
        Inter-accident intervals.
    """
    times = np.insert(np.cumsum(intervals), 0, 0)
    n0 = np.searchsorted(times, s, side="right") - 1  # events before s
    n1 = N - 1 - n0  # remaining events (N-1 intervals → N events, but first at t=0)
    # More carefully: N total events at times[0..N-1]
    # events in [0, s): times < s
    n0 = np.sum(times[:-1] < s)  # number of accidents before change point
    n1 = (N - 1) - n0  # accidents at or after change point (from intervals)

    return n0 * np.log(h0) + n1 * np.log(h1) - h0 * s - h1 * (L - s)
