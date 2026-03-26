"""MCMC convergence diagnostics: Gelman-Rubin, effective sample size, thinning."""

from __future__ import annotations

from dataclasses import dataclass

import emcee
import numpy as np


def gelman_rubin(chain: np.ndarray) -> np.ndarray:
    r"""Compute the Gelman-Rubin :math:`\hat{R}` statistic for an ensemble chain.

    Parameters
    ----------
    chain : ndarray, shape (n_steps, n_chains, n_params)
        MCMC chain from ``emcee`` (via ``sampler.get_chain()``).

    Returns
    -------
    r_hat : ndarray, shape (n_params,)
        :math:`\hat{R}` for each parameter. Values close to 1.0 indicate
        convergence; values above ~1.01 suggest the chains have not mixed.
    """
    n, m, d = chain.shape  # steps, chains, params

    # Per-chain means and variances
    chain_means = chain.mean(axis=0)  # (m, d)
    chain_vars = chain.var(axis=0, ddof=1)  # (m, d)

    # Within-chain variance W
    w = chain_vars.mean(axis=0)  # (d,)

    # Between-chain variance B
    b = n * np.var(chain_means, axis=0, ddof=1)  # (d,)

    # Pooled variance estimate and R-hat
    var_hat = (1 - 1 / n) * w + b / n
    r_hat = np.sqrt(var_hat / w)
    return r_hat


def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    r"""Estimate the effective sample size of a 1D chain via autocorrelation.

    Uses the initial positive sequence estimator: sums autocorrelation pairs
    until the first negative pair, then applies the standard ESS formula.

    Parameters
    ----------
    x : ndarray, shape (n,)
        1D array of MCMC samples (after burn-in).
    max_lag : int, optional
        Maximum lag to consider. Defaults to ``n // 2``.

    Returns
    -------
    ess : float
        Estimated effective sample size.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n // 2

    x_centred = x - x.mean()
    var = np.var(x_centred, ddof=0)
    if var == 0:
        return float(n)

    # Normalised autocorrelation via FFT
    fft_x = np.fft.fft(x_centred, n=2 * n)
    acf_full = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (var * n)

    # Initial positive sequence: sum consecutive pairs until negative
    tau = 1.0
    for lag in range(1, max_lag, 2):
        pair_sum = acf_full[lag] + (acf_full[lag + 1] if lag + 1 < n else 0.0)
        if pair_sum < 0:
            break
        tau += 2.0 * pair_sum

    return n / tau


@dataclass
class EmceeDiagnostics:
    """Diagnostics summary for an emcee sampler."""

    tau: np.ndarray
    thin: int
    r_hat: np.ndarray
    flat_samples: np.ndarray
    ess: np.ndarray
    labels: list[str]

    def print_summary(self) -> None:
        """Print a formatted diagnostics summary."""
        print("Autocorrelation times:")
        for label, t in zip(self.labels, self.tau):
            print(f"  {label}: {t:.1f}")
        print(f"Thinning by {self.thin} (max autocorrelation time)\n")

        print("Gelman-Rubin R-hat:")
        for label, rh in zip(self.labels, self.r_hat):
            print(f"  {label}: {rh:.4f}")
        if np.all(self.r_hat < 1.01):
            print("All R-hat < 1.01: chains have converged.\n")
        else:
            print("WARNING: some R-hat >= 1.01, chains may not have converged.\n")

        print(f"Samples after thinning: {self.flat_samples.shape[0]}")
        print("Effective sample sizes (post-thinning):")
        for label, e in zip(self.labels, self.ess):
            print(f"  {label}: {e:.0f}")


def diagnose_emcee(
    sampler: emcee.EnsembleSampler,
    labels: list[str],
) -> EmceeDiagnostics:
    """Run convergence diagnostics on an emcee sampler.

    Computes autocorrelation times, thins the chain accordingly,
    calculates Gelman-Rubin R-hat, and reports effective sample sizes.

    Parameters
    ----------
    sampler : emcee.EnsembleSampler
        A sampler that has already been run (production samples only;
        burn-in should have been discarded via ``sampler.reset()``).
    labels : list of str
        Parameter names for display.

    Returns
    -------
    EmceeDiagnostics
        Dataclass with tau, thin, r_hat, flat_samples, ess, labels.
    """
    try:
        tau = sampler.get_autocorr_time()
    except emcee.autocorr.AutocorrError:
        tau = sampler.get_autocorr_time(quiet=True)

    thin = max(1, int(np.ceil(tau.max())))
    chain = sampler.get_chain()
    r_hat = gelman_rubin(chain)
    flat_samples = sampler.get_chain(flat=True, thin=thin)
    ess = np.array(
        [
            effective_sample_size(flat_samples[:, i])
            for i in range(flat_samples.shape[1])
        ]
    )

    return EmceeDiagnostics(
        tau=tau,
        thin=thin,
        r_hat=r_hat,
        flat_samples=flat_samples,
        ess=ess,
        labels=labels,
    )


@dataclass
class RJMCMCDiagnostics:
    """Diagnostics summary for an RJMCMC chain."""

    burn_in: int
    thin: int
    chain_post: list
    k_trace: np.ndarray
    ess_k: float

    def print_summary(self) -> None:
        """Print a formatted diagnostics summary."""
        print(f"Burn-in: {self.burn_in}")
        print(f"Post-burn-in samples: {len(self.chain_post)}")
        print(f"Thinning factor for k: {self.thin}")
        print(f"Thinned samples: {len(self.k_trace)}")
        print(f"Effective sample size for k: {self.ess_k:.0f}")


def diagnose_rjmcmc(
    chain: list,
    burn_in: int,
    thin: int | None = None,
) -> RJMCMCDiagnostics:
    """Run convergence diagnostics on an RJMCMC chain.

    Parameters
    ----------
    chain : list of ndarray
        Full RJMCMC chain (list of state vectors with varying dimension).
    burn_in : int
        Number of initial samples to discard.
    thin : int, optional
        Thinning factor. If None, estimated from the autocorrelation of
        the model-index trace.

    Returns
    -------
    RJMCMCDiagnostics
        Dataclass with burn_in, thin, chain_post, k_trace, ess_k.
    """
    chain_post = chain[burn_in:]
    k_full = np.array([(len(state) - 1) // 2 for state in chain_post])

    if thin is None:
        ess_raw = effective_sample_size(k_full.astype(float))
        thin = max(1, int(np.ceil(len(k_full) / ess_raw)))

    k_trace = k_full[::thin]
    chain_post_thinned = chain_post[::thin]
    ess_k = effective_sample_size(k_trace.astype(float))

    return RJMCMCDiagnostics(
        burn_in=burn_in,
        thin=thin,
        chain_post=chain_post_thinned,
        k_trace=k_trace,
        ess_k=ess_k,
    )
