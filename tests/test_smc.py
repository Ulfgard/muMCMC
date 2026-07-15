"""Tests for the adaptive tempered SMC driver.

Two layers: unit tests for the schedule / resampling primitives (ESS control,
systematic resampling, the likelihood-only potential), and statistical-recovery
tests that transport a population from prior to posterior and check the moments,
the evidence against a closed form, and -- the actual point -- recovery of a
well-separated bimodal posterior with the correct relative mass.
"""
import math

import torch
import pytest

from muMCMC import RMHMC, SMC, UnconstrainedSpace
from muMCMC.SMC import _systematic_resample
from pyro.distributions import Normal

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
#  models / fixtures                                                          #
# --------------------------------------------------------------------------- #

def gaussian_1d(lam, mu):
    """Gaussian likelihood in theta with precision ``lam`` and mean ``mu``:
    U_lik = 1/2 lam (theta - mu)^2, constant likelihood metric G_lik = lam.
    Paired with a N(0,1) prior (metric I) this is conjugate, so the posterior
    and evidence are known in closed form."""
    def model(theta):
        U = 0.5 * lam * (theta[..., 0] - mu) ** 2
        G = lam * torch.eye(1, dtype=theta.dtype).expand(*theta.shape[:-1], 1, 1)
        return U, G
    return model


def gaussian_1d_space():
    return UnconstrainedSpace(
        ["x"],
        priors={"x": Normal(0.0, 1.0)},
        prior_metric_fn=lambda theta: torch.eye(1, dtype=theta.dtype).expand(
            *theta.shape[:-1], 1, 1),
    )


def bimodal_1d(m, s):
    """Symmetric two-well likelihood at +/- m (unnormalized mixture), with a
    constant identity likelihood metric -- enough for RMHMC to mutate within a
    well; the tempering is what has to populate both."""
    def model(theta):
        t = theta[..., 0]
        left = -0.5 * ((t - m) / s) ** 2
        right = -0.5 * ((t + m) / s) ** 2
        U = -torch.logsumexp(torch.stack([left, right], dim=-1), dim=-1)
        G = torch.eye(1, dtype=theta.dtype).expand(*theta.shape[:-1], 1, 1)
        return U, G
    return model


def make_smc(model, space, *, step_size=0.3, num_steps=8, num_particles=None, **smc_kw):
    sampler = RMHMC(model, space, step_size=step_size, num_steps=num_steps,
                    adapt_step_size=False)
    return SMC(sampler, **smc_kw)


def _ess_at(u_lik, d):
    a = torch.logsumexp(-d * u_lik, dim=0)
    b = torch.logsumexp(-2.0 * d * u_lik, dim=0)
    return float(torch.exp(2.0 * a - b))


# --------------------------------------------------------------------------- #
#  potential_likelihood: likelihood only                                      #
# --------------------------------------------------------------------------- #

def test_potential_likelihood_is_likelihood_only():
    lam, mu = 3.0, 2.0
    space = gaussian_1d_space()
    sampler = RMHMC(gaussian_1d(lam, mu), space, adapt_step_size=False)
    z = torch.tensor([[0.5], [-1.0], [2.0]])

    u_lik = sampler.potential_likelihood(z)
    assert u_lik.shape == (3,)
    assert torch.allclose(u_lik, 0.5 * lam * (z[..., 0] - mu) ** 2)

    # evaluate_model's potential adds the N(0,1) prior (no Jacobian in the
    # identity space): U - U_lik == -log prior == 1/2 z^2 + 1/2 log(2 pi).
    U, _ = sampler.evaluate_model(z, beta=1.0)
    u_prior = 0.5 * z[..., 0] ** 2 + 0.5 * math.log(2 * math.pi)
    assert torch.allclose(U - u_lik, u_prior)


# --------------------------------------------------------------------------- #
#  schedule: ESS control                                                      #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("scale", [0.3, 1.0, 3.0])
def test_next_dbeta_hits_ess_target(scale):
    torch.manual_seed(0)
    space = gaussian_1d_space()
    smc = make_smc(gaussian_1d(2.0, 1.0), space, ess_target=0.5)
    N = 2000
    u_lik = scale * torch.randn(N)
    d = smc._next_dbeta(u_lik, max_dbeta=1.0)
    if d < 1.0 - 1e-9:                                  # interior solution
        assert _ess_at(u_lik, d) == pytest.approx(0.5 * N, rel=0.05)
    else:                                               # full jump still fine
        assert _ess_at(u_lik, 1.0) >= 0.5 * N


def test_next_dbeta_takes_full_step_when_ess_allows():
    space = gaussian_1d_space()
    smc = make_smc(gaussian_1d(2.0, 1.0), space, ess_target=0.5)
    u_lik = 1e-3 * torch.randn(500)                     # nearly flat -> ESS ~ N
    assert smc._next_dbeta(u_lik, max_dbeta=0.2) == 0.2


# --------------------------------------------------------------------------- #
#  systematic resampling                                                       #
# --------------------------------------------------------------------------- #

def test_systematic_resample_shapes_and_range():
    torch.manual_seed(0)
    W = torch.rand(50)
    W = W / W.sum()
    idx = _systematic_resample(W)
    assert idx.shape == (50,) and idx.dtype == torch.long
    assert int(idx.min()) >= 0 and int(idx.max()) <= 49


def test_systematic_resample_point_mass_selects_only_survivor():
    W = torch.zeros(10)
    W[3] = 1.0
    assert torch.equal(_systematic_resample(W), torch.full((10,), 3, dtype=torch.long))


def test_systematic_resample_matches_weights_in_expectation():
    torch.manual_seed(0)
    W = torch.tensor([0.1, 0.6, 0.3])
    counts = torch.zeros(3)
    trials = 4000
    for _ in range(trials):
        idx = _systematic_resample(W)
        counts += torch.bincount(idx, minlength=3).to(counts.dtype)
    freq = counts / (trials * W.shape[0])
    assert torch.allclose(freq, W, atol=0.02)


# --------------------------------------------------------------------------- #
#  statistical recovery                                                        #
# --------------------------------------------------------------------------- #

def test_recovers_gaussian_posterior_and_evidence():
    torch.manual_seed(0)
    lam, mu = 3.0, 2.0
    space = gaussian_1d_space()
    smc = make_smc(gaussian_1d(lam, mu), space, num_mcmc_steps=5)

    samples = smc.run_smc(4000, disable_progbar=True)
    x = samples["x"]
    assert x.shape == (1, 4000)

    # conjugate posterior: precision 1+lam, mean lam*mu/(1+lam), var 1/(1+lam)
    post_mean = lam * mu / (1.0 + lam)
    post_var = 1.0 / (1.0 + lam)
    assert float(x.mean()) == pytest.approx(post_mean, abs=0.05)
    assert float(x.var()) == pytest.approx(post_var, rel=0.15)

    # log evidence: log Z = -1/2 log(1+lam) - 1/2 lam*mu^2/(1+lam)
    log_Z = -0.5 * math.log(1.0 + lam) - 0.5 * lam * mu ** 2 / (1.0 + lam)
    diag = smc.diagnostics()
    assert float(diag["log_evidence_estimate"]) == pytest.approx(log_Z, abs=0.15)

    # schedule: strictly increasing from 0 to exactly 1
    betas = diag["betas"][:, 0]
    assert float(betas[0]) == 0.0 and float(betas[-1]) == pytest.approx(1.0)
    assert bool((betas[1:] > betas[:-1]).all())
    # ESS held near the target (0.5 * M) at every interior stage
    for e in diag["ess"][:-1, 0]:
        assert float(e) == pytest.approx(0.5 * 4000, rel=0.25)


def test_recovers_bimodal_posterior_with_balanced_mass():
    torch.manual_seed(0)
    m, s = 2.0, 0.5
    space = UnconstrainedSpace(
        ["x"],
        priors={"x": Normal(0.0, 2.0)},
        prior_metric_fn=lambda theta: 0.25 * torch.eye(1, dtype=theta.dtype).expand(
            *theta.shape[:-1], 1, 1),
    )
    smc = make_smc(bimodal_1d(m, s), space, step_size=0.25, num_steps=8,
                   num_mcmc_steps=8)

    samples = smc.run_smc(4000, disable_progbar=True)
    x = samples["x"]

    # both wells populated, near +/- m
    near_pos = (x - m).abs() < 1.0
    near_neg = (x + m).abs() < 1.0
    assert float(near_pos.to(torch.float64).mean()) > 0.3
    assert float(near_neg.to(torch.float64).mean()) > 0.3
    # symmetric target -> roughly balanced mass between the modes
    frac_pos = float((x > 0).to(torch.float64).mean())
    assert frac_pos == pytest.approx(0.5, abs=0.12)


# --------------------------------------------------------------------------- #
#  parallel populations: per-chain diagnostics, evidence, R-hat                #
# --------------------------------------------------------------------------- #

def test_multi_chain_diagnostics_and_rhat():
    torch.manual_seed(0)
    lam, mu = 3.0, 2.0
    space = gaussian_1d_space()
    smc = make_smc(gaussian_1d(lam, mu), space, num_mcmc_steps=5)

    samples = smc.run_smc(1000, num_chains=4, disable_progbar=True)
    assert samples["x"].shape == (4, 1000)

    diag = smc.diagnostics()
    assert diag["log_evidence"].shape == (4,)
    assert diag["betas"].shape[1] == 4 and diag["ess"].shape[1] == 4

    # combined evidence matches the closed form; per-chain spread is finite
    log_Z = -0.5 * math.log(1.0 + lam) - 0.5 * lam * mu ** 2 / (1.0 + lam)
    assert float(diag["log_evidence_estimate"]) == pytest.approx(log_Z, abs=0.15)
    assert float(diag["log_evidence_se"]) >= 0.0

    # unimodal target: chains agree, so R-hat ~ 1
    assert float(diag["r_hat"]["x"]) == pytest.approx(1.0, abs=0.1)

    # posterior mean recovered (pooled over chains)
    post_mean = lam * mu / (1.0 + lam)
    assert float(samples["x"].mean()) == pytest.approx(post_mean, abs=0.05)
