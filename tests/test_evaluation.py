"""Tests for ``muMCMC.evaluation`` -- BAR evidence and posterior density.

The pure BAR core ``_bar_root`` is checked against its defining equation and a
case with a closed-form answer.  The public ``PosteriorEvaluation`` is checked
on a conjugate Gaussian whose evidence and posterior are known analytically:

    prior       p(y)   = N(0, I)          (per-name Normal(0, 1))
    likelihood  p(x|y) = N(x; y, I)
    posterior   p(y|x) = N(x/2, I/2)
    evidence    p(x)   = N(x; 0, 2I)

so ``log p(x) = -||x||^2/4 - (d/2) log(2π) - (d/2) log 2``.
"""
import math

import torch
import pytest
from pyro.distributions import Normal

from muMCMC.MCMCSampler import MCMCSampler
from muMCMC.spaces import UnconstrainedSpace
from muMCMC.evaluation import PosteriorEvaluation, _bar_root, _bar_gaussian

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
#  Conjugate Gaussian model + a minimal sampler exposing evaluate_model.       #
# --------------------------------------------------------------------------- #

class _Sampler(MCMCSampler):
    """Bare sampler: only ``evaluate_model``/``space`` are exercised here."""

    def __init__(self, space, potential_fn):
        super().__init__(potential_fn=potential_fn, space=space,
                         requires_metric=False)

    def init(self, z):            raise NotImplementedError
    def step(self, s):            raise NotImplementedError
    def end_warmup(self):         raise NotImplementedError


def _gaussian_model(x):
    """Return (sampler, x, logZ_true) for the conjugate Gaussian at observed x."""
    d = x.shape[0]
    names = [f"y{i}" for i in range(d)]
    space = UnconstrainedSpace(names, priors={n: Normal(0.0, 1.0) for n in names})
    const = 0.5 * d * math.log(2 * math.pi)

    def potential_fn(theta):                      # U_lik = -log N(x; theta, I)
        return 0.5 * ((x - theta) ** 2).sum(-1) + const

    sampler = _Sampler(space, potential_fn)
    logZ_true = (-0.25 * (x ** 2).sum()
                 - 0.5 * d * math.log(2 * math.pi)
                 - 0.5 * d * math.log(2.0))
    return sampler, names, float(logZ_true)


def _posterior_samples(x, names, K, n, seed=0):
    """Draw exact posterior samples N(x/2, I/2), grouped by chain."""
    g = torch.Generator().manual_seed(seed)
    d = x.shape[0]
    z = x / 2.0 + torch.randn(K, n, d, generator=g) / math.sqrt(2.0)
    return {name: z[..., i] for i, name in enumerate(names)}


def _linear_gaussian_model(x_obs, sigma):
    """y=(y0,y1), prior N(0,I), likelihood N(x_obs; y0+y1, sigma^2).

    A small sigma makes the sum y0+y1 tightly determined, so the posterior is
    strongly correlated and each coordinate is individually predictive. Returns
    (sampler, names, mu_post, Sigma_post).
    """
    names = ["y0", "y1"]
    space = UnconstrainedSpace(names, priors={n: Normal(0.0, 1.0) for n in names})
    const = 0.5 * math.log(2 * math.pi * sigma ** 2)

    def potential_fn(theta):                      # U_lik = -log N(x_obs; y0+y1, sigma^2)
        return 0.5 * (x_obs - theta.sum(-1)) ** 2 / sigma ** 2 + const

    sampler = _Sampler(space, potential_fn)
    a = torch.tensor([1.0, 1.0])
    Sigma_post = torch.linalg.inv(torch.eye(2) + torch.outer(a, a) / sigma ** 2)
    mu_post = Sigma_post @ (a * x_obs / sigma ** 2)
    return sampler, names, mu_post, Sigma_post


# --------------------------------------------------------------------------- #
#  BAR core                                                                    #
# --------------------------------------------------------------------------- #

def test_bar_root_recovers_constant_log_ratio():
    # If W is a constant c on every draw (q̂ equals the posterior up to scale),
    # the estimating equation returns exactly c, independent of n1, n0.
    c = 1.2345
    est = _bar_root(torch.full((500,), c), torch.full((300,), c))
    assert abs(est - c) < 1e-8


def test_bar_root_satisfies_estimating_equation():
    torch.manual_seed(0)
    W_post = torch.randn(400) * 1.5 + 0.3
    W_q = torch.randn(250) * 1.5 - 0.2
    est = _bar_root(W_post, W_q)
    n1, n0 = W_post.numel(), W_q.numel()
    b = math.log(n1 / n0) - est                   # invert logZ = log(n1/n0) - b
    residual = torch.sigmoid(torch.cat([W_post, W_q]) + b).sum() - n1
    assert abs(float(residual)) < 1e-6


# --------------------------------------------------------------------------- #
#  Evidence  (acceptance test 1)                                               #
# --------------------------------------------------------------------------- #

def test_log_evidence_matches_gaussian():
    x = torch.tensor([1.0, -0.5, 0.5, 2.0, -1.5])
    sampler, names, logZ_true = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=8, n=4000, seed=1)

    gen = torch.Generator().manual_seed(2)
    ev = PosteriorEvaluation(sampler, samples, generator=gen)

    err = abs(ev.log_evidence - logZ_true)
    se = ev.diagnostics["log_evidence_se"]
    # q̂ is (near) exact here, so recovery is tight; gate on 3 SE with a small
    # floor against a vanishing SE.
    assert err < max(3.0 * se, 0.02), (ev.log_evidence, logZ_true, se)


# --------------------------------------------------------------------------- #
#  Posterior density round-trip  (acceptance test 5)                           #
# --------------------------------------------------------------------------- #

def test_log_posterior_matches_gaussian_posterior():
    x = torch.tensor([1.0, -0.5, 0.5, 2.0, -1.5])
    d = x.shape[0]
    sampler, names, logZ_true = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=8, n=4000, seed=3)

    gen = torch.Generator().manual_seed(4)
    ev = PosteriorEvaluation(sampler, samples, generator=gen)

    # A batch of evaluation points, not the draws themselves.
    torch.manual_seed(5)
    y = x / 2.0 + 0.3 * torch.randn(16, d)
    y_dict = {name: y[:, i] for i, name in enumerate(names)}
    lp = ev.log_posterior(y_dict)

    true_post = torch.distributions.MultivariateNormal(
        x / 2.0, covariance_matrix=0.5 * torch.eye(d))
    lp_true = true_post.log_prob(y)

    # logZ cancels in differences -> the density *shape* must match exactly.
    assert torch.allclose(lp - lp[0], lp_true - lp_true[0], atol=1e-9)
    # The residual constant is exactly the evidence error; small here.
    assert abs(float((lp - lp_true).mean()) - (logZ_true - ev.log_evidence)) < 1e-9
    assert (lp - lp_true).std() < 1e-9            # constant offset
    assert abs(float((lp - lp_true).mean())) < 0.05


def test_log_posterior_marginal_matches_gaussian_marginal():
    # Conjugate Gaussian: the posterior N(x/2, I/2) factorizes, so the marginal
    # over a subset of coordinates is N(x_a/2, I_a/2). Marginalize one coord out
    # of three via the conditional-Gaussian importance sampler.
    x = torch.tensor([1.0, -0.5, 0.5])
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=4, n=2000, seed=16)
    ev = PosteriorEvaluation(sampler, samples,
                             generator=torch.Generator().manual_seed(17))

    a = names[:2]                                  # keep y0, y1; marginalize y2
    torch.manual_seed(18)
    ya = x[:2] / 2.0 + 0.3 * torch.randn(16, 2)
    y_dict = {a[0]: ya[:, 0], a[1]: ya[:, 1]}
    lp, ess = ev.log_posterior(y_dict, max_marginal=5000, return_ess=True,
                               generator=torch.Generator().manual_seed(19))

    true = torch.distributions.MultivariateNormal(
        x[:2] / 2.0, covariance_matrix=0.5 * torch.eye(2))
    lp_true = true.log_prob(ya)

    # The posterior-informed proposal is near exact here, so the estimate is
    # tight pointwise and the weight ESS is a large fraction of the draws.
    assert torch.max(torch.abs(lp - lp_true)) < 0.05
    assert float(ess.min()) > 0.3 * 5000


def test_log_posterior_marginal_consistent_across_prior_weight():
    # The defensive mixture is unbiased for every prior_weight: pure conditional
    # (0), 50/50 (0.5), and pure prior (1) all recover the analytic marginal.
    x = torch.tensor([1.0, -0.5, 0.5])
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=4, n=2000, seed=20)
    ev = PosteriorEvaluation(sampler, samples,
                             generator=torch.Generator().manual_seed(21))
    ya = x[:2] / 2.0 + 0.2 * torch.randn(8, 2, generator=torch.Generator().manual_seed(22))
    y_dict = {names[0]: ya[:, 0], names[1]: ya[:, 1]}
    lp_true = torch.distributions.MultivariateNormal(
        x[:2] / 2.0, covariance_matrix=0.5 * torch.eye(2)).log_prob(ya)

    for alpha in (0.0, 0.5, 1.0):
        lp = ev.log_posterior(y_dict, max_marginal=20000, prior_weight=alpha,
                              generator=torch.Generator().manual_seed(23))
        assert torch.max(torch.abs(lp - lp_true)) < 0.1, alpha


def test_log_posterior_prior_weight_out_of_range_raises():
    x = torch.tensor([1.0, -0.5, 0.5])
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=2, n=500, seed=24)
    ev = PosteriorEvaluation(sampler, samples, generator=torch.Generator().manual_seed(25))
    with pytest.raises(ValueError):
        ev.log_posterior({names[0]: torch.zeros(3)}, prior_weight=1.5)


def test_log_posterior_marginal_adaptive_reaches_target_ess():
    x = torch.tensor([1.0, -0.5, 0.5])
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=4, n=2000, seed=26)
    ev = PosteriorEvaluation(sampler, samples, generator=torch.Generator().manual_seed(27))
    ya = x[:2] / 2.0 + 0.2 * torch.randn(8, 2, generator=torch.Generator().manual_seed(28))
    y_dict = {names[0]: ya[:, 0], names[1]: ya[:, 1]}
    lp, ess = ev.log_posterior(y_dict, target_ess=3000, max_marginal=100000,
                               return_ess=True, generator=torch.Generator().manual_seed(29))
    assert float(ess.min()) >= 3000
    lp_true = torch.distributions.MultivariateNormal(
        x[:2] / 2.0, covariance_matrix=0.5 * torch.eye(2)).log_prob(ya)
    assert torch.max(torch.abs(lp - lp_true)) < 0.1


def test_log_posterior_marginal_adaptive_stops_at_max():
    # An unreachable target stops at max_marginal and still returns a finite
    # estimate (prior_weight > 0 keeps the weights bounded).
    x = torch.tensor([1.0, -0.5, 0.5])
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=2, n=1000, seed=33)
    ev = PosteriorEvaluation(sampler, samples, generator=torch.Generator().manual_seed(34))
    y_dict = {names[0]: torch.zeros(4), names[1]: torch.zeros(4)}
    lp, ess = ev.log_posterior(y_dict, target_ess=1e9, max_marginal=4000,
                               return_ess=True, generator=torch.Generator().manual_seed(35))
    assert float(ess.min()) < 1e9
    assert ess.shape == (4,) and torch.isfinite(lp).all()


def test_log_posterior_marginal_correlated_predictive_block():
    # Marginalize the predictive, correlated coordinate y1. The prior would be a
    # poor proposal, but the q̂ conditional tracks y1 ~ x_obs - y0, so the weight
    # ESS stays high and the marginal matches N(mu_post[0], Sigma_post[00]).
    x_obs, sigma = 1.5, 0.3
    sampler, names, mu_post, Sigma_post = _linear_gaussian_model(x_obs, sigma)

    g = torch.Generator().manual_seed(30)
    L = torch.linalg.cholesky(Sigma_post)
    zc = mu_post + torch.randn(4, 4000, 2, generator=g) @ L.T
    samples = {names[i]: zc[..., i] for i in range(2)}
    ev = PosteriorEvaluation(sampler, samples,
                             generator=torch.Generator().manual_seed(31))

    sd0 = Sigma_post[0, 0].sqrt()
    y0 = mu_post[0] + torch.linspace(-1.5, 1.5, 12) * sd0
    lp, ess = ev.log_posterior({"y0": y0}, max_marginal=8000, return_ess=True,
                               generator=torch.Generator().manual_seed(32))
    lp_true = torch.distributions.Normal(mu_post[0], sd0).log_prob(y0)

    assert torch.max(torch.abs(lp - lp_true)) < 0.1
    assert float(ess.min()) > 0.3 * 8000


def test_log_posterior_marginal_requires_a_name():
    x = torch.tensor([1.0, -0.5, 0.5])
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=2, n=500, seed=6)
    ev = PosteriorEvaluation(sampler, samples, generator=torch.Generator().manual_seed(7))
    with pytest.raises(ValueError):
        ev.log_posterior({"not_a_name": torch.zeros(4)})


def test_diagnostics_shape_and_keys():
    x = torch.tensor([0.5, -1.0])
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=4, n=1000, seed=8)
    ev = PosteriorEvaluation(sampler, samples, n_q=1500,
                             generator=torch.Generator().manual_seed(9))
    d = ev.diagnostics
    assert d["n1"] == 4000 and d["n0"] == 1500
    assert d["per_chain_log_evidence"].shape == (4,)
    assert math.isfinite(d["log_evidence_se"])
    assert set(d["W_percentiles"]) == {0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99}


def test_diagnostics_single_chain_omits_se():
    x = torch.tensor([0.5, -1.0])
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=1, n=2000, seed=10)
    ev = PosteriorEvaluation(sampler, samples,
                             generator=torch.Generator().manual_seed(11))
    d = ev.diagnostics
    assert "log_evidence_se" not in d
    assert d["per_chain_log_evidence"].shape == (1,)


def test_bar_gaussian_matches_pooled_estimate():
    # The free-standing core on the pooled draws reproduces log_evidence.
    x = torch.tensor([1.0, -0.5, 0.5, 2.0, -1.5])
    d = x.shape[0]
    sampler, names, _ = _gaussian_model(x)
    samples = _posterior_samples(x, names, K=8, n=4000, seed=14)
    gen = torch.Generator().manual_seed(15)
    ev = PosteriorEvaluation(sampler, samples, generator=gen)

    # Same seed, same draws, same fit -> reproduces the pooled estimate exactly.
    z = ev._z.reshape(-1, d)
    est = _bar_gaussian(z, ev._log_target, generator=torch.Generator().manual_seed(15))
    assert abs(est - ev.log_evidence) < 1e-6
