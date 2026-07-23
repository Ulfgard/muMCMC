"""Tests for ``muMCMC.validation.mixture``: the Gaussian mixture as its own unit.

``log_prob`` is checked against a hand-built mixture and for normalization,
``fit`` for recovering a well-separated bimodal cloud, ``sample`` for reproducing
the mixture moments, and ``conditional`` against the analytic Gaussian
conditional (K=1) plus normalization and self-consistency of the mixture case.
"""
import math

import torch

from muMCMC.validation.mixture import GaussianMixture, ConditionalGaussianMixture

torch.set_default_dtype(torch.float64)


def _mix2d():
    """A fixed 2-component mixture in 2D, built by hand (no fit)."""
    weights = torch.tensor([0.7, 0.3])
    means = torch.tensor([[-1.0, 0.5], [2.0, -1.0]])
    covs = torch.stack([
        torch.tensor([[0.5, 0.2], [0.2, 0.4]]),
        torch.tensor([[0.3, -0.1], [-0.1, 0.6]]),
    ])
    return GaussianMixture(weights, means, torch.linalg.cholesky(covs)), weights, means, covs


# --------------------------------------------------------------------------- #
#  log_prob                                                                    #
# --------------------------------------------------------------------------- #

def test_log_prob_matches_component_logsumexp():
    gm, w, mu, covs = _mix2d()
    z = torch.tensor([[0.0, 0.0], [-1.0, 0.5], [3.0, -2.0]])
    comp = [torch.distributions.MultivariateNormal(mu[k], covs[k]).log_prob(z) for k in range(2)]
    expected = torch.logsumexp(torch.stack([math.log(float(w[k])) + comp[k] for k in range(2)]), 0)
    assert torch.allclose(gm.log_prob(z), expected, atol=1e-10)


def test_log_prob_normalizes():
    gm, *_ = _mix2d()
    g = torch.linspace(-6, 7, 400)
    xx, yy = torch.meshgrid(g, g, indexing="ij")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], -1)
    dens = torch.exp(gm.log_prob(grid)).reshape(400, 400)
    integral = torch.trapezoid(torch.trapezoid(dens, g, dim=1), g)
    assert abs(float(integral) - 1.0) < 1e-3


# --------------------------------------------------------------------------- #
#  fit                                                                         #
# --------------------------------------------------------------------------- #

def test_fit_recovers_separated_bimodal():
    g = torch.Generator().manual_seed(0)
    n = 6000
    comp = torch.rand(n, generator=g) < 0.6
    z = torch.where(comp, -4.0, 4.0) + 0.5 * torch.randn(n, generator=g)
    gm = GaussianMixture.fit(z[:, None], 2, generator=torch.Generator().manual_seed(1))

    order = torch.argsort(gm.means[:, 0])
    means = gm.means[order, 0]
    weights = gm.weights[order]
    variances = gm.covs[order, 0, 0]
    assert abs(float(means[0]) + 4.0) < 0.15 and abs(float(means[1]) - 4.0) < 0.15
    assert abs(float(weights[0]) - 0.6) < 0.05
    assert abs(float(variances[0]) - 0.25) < 0.05 and abs(float(variances[1]) - 0.25) < 0.05


def test_fit_single_component_is_sample_moments():
    g = torch.Generator().manual_seed(2)
    z = torch.randn(2000, 3, generator=g) @ torch.tensor(
        [[1.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.7]])
    gm = GaussianMixture.fit(z, 1)
    assert torch.allclose(gm.means[0], z.mean(0), atol=1e-6)
    assert torch.allclose(gm.covs[0], torch.cov(z.T) + 1e-6 * torch.eye(3), atol=1e-6)


# --------------------------------------------------------------------------- #
#  sample                                                                      #
# --------------------------------------------------------------------------- #

def test_sample_reproduces_mixture_moments():
    gm, w, mu, covs = _mix2d()
    z = gm.sample(200000, generator=torch.Generator().manual_seed(3))
    mean_true = (w[:, None] * mu).sum(0)
    second = sum(float(w[k]) * (covs[k] + torch.outer(mu[k], mu[k])) for k in range(2))
    cov_true = second - torch.outer(mean_true, mean_true)
    assert torch.allclose(z.mean(0), mean_true, atol=0.02)
    assert torch.allclose(torch.cov(z.T), cov_true, atol=0.05)


# --------------------------------------------------------------------------- #
#  conditional                                                                 #
# --------------------------------------------------------------------------- #

def test_conditional_single_component_matches_analytic():
    mu = torch.tensor([0.3, -0.7, 1.1])
    cov = torch.tensor([[1.0, 0.3, 0.2], [0.3, 0.8, -0.1], [0.2, -0.1, 0.6]])
    gm = GaussianMixture(torch.ones(1), mu[None], torch.linalg.cholesky(cov)[None])

    a, b = [0, 1], [2]
    z_a = torch.tensor([[0.5, -0.5], [1.0, 0.0], [-0.2, 0.9]])
    cond = gm.conditional(a, b, z_a)

    Saa, Sab = cov[:2, :2], cov[:2, 2:]
    A = torch.linalg.solve(Saa, Sab)
    mu_cond = mu[2:] + (z_a - mu[:2]) @ A
    S_cond = cov[2:, 2:] - Sab.T @ A
    assert torch.allclose(cond.means[:, 0, :], mu_cond, atol=1e-9)
    assert torch.allclose(cond.scale_tril[0] @ cond.scale_tril[0].T, S_cond, atol=1e-9)
    assert torch.allclose(cond.log_weights, torch.zeros(3, 1), atol=1e-12)


def test_conditional_mixture_normalizes():
    gm, *_ = _mix2d()                                  # a=coord0, b=coord1
    z_a = torch.tensor([[0.0]])
    cond = gm.conditional([0], [1], z_a)
    grid = torch.linspace(-6, 6, 2000)
    dens = torch.exp(cond.log_prob(grid.reshape(1, -1, 1)))[0]
    assert abs(float(torch.trapezoid(dens, grid)) - 1.0) < 1e-3


def test_conditional_sample_matches_density_mean():
    gm, *_ = _mix2d()
    z_a = torch.tensor([[1.5]])
    cond = gm.conditional([0], [1], z_a)
    draws = cond.sample(200000, generator=torch.Generator().manual_seed(4))
    mean_analytic = float((torch.exp(cond.log_weights[0]) * cond.means[0, :, 0]).sum())
    assert draws.shape == (1, 200000, 1)
    assert abs(float(draws.mean()) - mean_analytic) < 0.02
