"""Statistical analysis: priors, posteriors, evidence, and likelihoods."""

import numpy as np
from scipy import integrate, special
from scipy.stats import gamma as gamma_dist

from coalmine.constants import ALPHA, BETA, L, N


def gamma_prior_pdf(
    h: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> np.ndarray:
    """Gamma(alpha, beta) prior PDF for rate heights h."""
    return gamma_dist.pdf(h, a=alpha, scale=1 / beta)


def gamma_prior_logpdf(
    h: np.ndarray,
    alpha: float = ALPHA,
    beta: float = BETA,
) -> np.ndarray:
    """Log of Gamma(alpha, beta) prior PDF for rate heights h."""
    return gamma_dist.logpdf(h, a=alpha, scale=1 / beta)


# ── M0: constant-rate model ──────────────────────────────────────────────


def m0_log_likelihood(h: float | np.ndarray) -> np.ndarray:
    """Log-likelihood for M0 (constant rate h) given N events in time L."""
    h = np.asarray(h)
    return N * np.log(h) - h * L


def m0_posterior_pdf(h: np.ndarray) -> np.ndarray:
    """Normalised M0 posterior — Gamma(alpha + N, beta + L)."""
    return gamma_dist.pdf(h, a=ALPHA + N, scale=1 / (BETA + L))


def m0_posterior_dist():
    """Return the scipy distribution object for the M0 posterior."""
    return gamma_dist(a=ALPHA + N, scale=1 / (BETA + L))


def m0_log_evidence() -> float:
    """Analytic log-evidence log(Z_0) for M0.

    log Z_0 = alpha*log(beta) - gammaln(alpha)
            + gammaln(alpha+N) - (alpha+N)*log(beta+L)
    """
    return (
        ALPHA * np.log(BETA)
        - special.gammaln(ALPHA)
        + special.gammaln(ALPHA + N)
        - (ALPHA + N) * np.log(BETA + L)
    )


def m0_log_evidence_numerical(h_upper: float = 0.05, n_points: int = 10_000) -> float:
    """Numerical log-evidence log(Z_0) for M0 via the trapezium rule.

    Shifts the log-integrand by its peak value before exponentiating to
    avoid underflow, then adds the shift back.
    """
    h = np.linspace(0, h_upper, n_points)
    log_vals = m0_log_likelihood(h[1:]) + gamma_prior_logpdf(h[1:])
    log_peak = log_vals.max()
    shifted = np.zeros(n_points)
    shifted[1:] = np.exp(log_vals - log_peak)
    return np.log(integrate.trapezoid(shifted, h)) + log_peak


# ── M1: single change-point model ────────────────────────────────────────


# ── Change-point prior sampling ──────────────────────────────────────────


def sample_order_stats(
    k: int,
    n_samples: int,
    L: float = L,
    even: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample k change points from an order-statistics prior.

    Parameters
    ----------
    even : bool
        If False, draw k uniform on [0, L] and sort (plain order statistics).
        If True, draw 2k+1 uniform on [0, L], sort, and take the
        even-numbered order statistics (1-indexed: 2nd, 4th, ...).

    Returns (n_samples, k).
    """
    if rng is None:
        rng = np.random.default_rng()
    n_draw = 2 * k + 1 if even else k
    samples = rng.uniform(0, L, size=(n_samples, n_draw))
    samples.sort(axis=1)
    if even:
        samples = samples[:, 1::2]
    return samples


def m1_log_likelihood(h0: float, h1: float, s: float, intervals: np.ndarray) -> float:
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
