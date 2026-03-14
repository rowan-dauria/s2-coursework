"""Tests for statistical analysis functions."""

import numpy as np
import pytest

from coalmine.analysis import (
    gamma_prior_pdf,
    m0_evidence,
    m0_evidence_numerical,
    m0_posterior_pdf,
)


def test_gamma_prior_integrates_to_one():
    h = np.linspace(0, 0.1, 10_000)
    integral = np.trapezoid(gamma_prior_pdf(h), h)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_m0_posterior_integrates_to_one():
    h = np.linspace(0, 0.02, 10_000)
    integral = np.trapezoid(m0_posterior_pdf(h), h)
    assert integral == pytest.approx(1.0, abs=1e-3)


def test_m0_evidence_analytic_vs_numerical():
    z_analytic = m0_evidence()
    z_numerical, _ = m0_evidence_numerical()
    assert z_analytic == pytest.approx(z_numerical, rel=1e-6)
