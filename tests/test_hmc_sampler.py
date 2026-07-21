"""Tests for the Euclidean HMC sampler.

The target is a shifted-Gaussian likelihood under an N(0,1) prior, so the
posterior is a Gaussian with analytically known moments -- the same setup the
RMHMC/NUTS regression tests use, giving a non-constant potential (real
gradient).  Tests are seed-fixed and sized to run on CPU in CI.
"""
import torch
import pytest
from pyro.distributions import Normal

from muMCMC import HMC, PT, UnconstrainedSpace

torch.set_default_dtype(torch.float64)

NAMES = ["a", "b"]
COMMON_KEYS = {"accept_rate", "num_divergences", "step_size"}

# Likelihood N(MU, SIGMA2) per dim, prior N(0, 1) per dim.  Posterior is
# Gaussian: precision 1 + 1/SIGMA2, mean MU / (SIGMA2 + 1).
MU = torch.tensor([1.0, -0.5])
SIGMA2 = 1.0
POST_MEAN = MU / (SIGMA2 + 1.0)                 # [0.5, -0.25]
POST_STD = (SIGMA2 / (SIGMA2 + 1.0)) ** 0.5     # ~0.7071


def _space():
    return UnconstrainedSpace(NAMES, priors={n: Normal(0.0, 1.0) for n in NAMES})


def _model(theta):
    return 0.5 * (((theta - MU) ** 2) / SIGMA2).sum(-1)


def test_hmc_recovers_known_gaussian():
    torch.manual_seed(0)
    s = HMC(_model, _space(), step_size=0.2, num_steps=5)
    out = s.run_mcmc(torch.zeros(2), num_samples=500, num_warmup_steps=500,
                     num_chains=6, disable_progbar=True)
    for i, n in enumerate(NAMES):
        x = out[n]
        assert x.shape == (6, 500)
        assert abs(float(x.mean()) - float(POST_MEAN[i])) < 0.1
        assert abs(float(x.std()) - POST_STD) < 0.1


def test_hmc_dense_mass_matrix_recovers_gaussian():
    torch.manual_seed(0)
    s = HMC(_model, _space(), step_size=0.2, num_steps=5,
            mass_matrix=torch.tensor([[2.0, 0.3], [0.3, 0.5]]))
    out = s.run_mcmc(torch.zeros(2), num_samples=500, num_warmup_steps=500,
                     num_chains=6, disable_progbar=True)
    for i, n in enumerate(NAMES):
        x = out[n]
        assert abs(float(x.mean()) - float(POST_MEAN[i])) < 0.1
        assert abs(float(x.std()) - POST_STD) < 0.1


def test_hmc_mass_matrix_shape_validated():
    # A wrong-shaped user mass matrix is caught in init (d = 2 here).
    s = HMC(_model, _space(), mass_matrix=torch.eye(3))
    with pytest.raises(ValueError):
        s.run_mcmc(torch.zeros(2), num_samples=1, num_warmup_steps=0,
                   num_chains=1, disable_progbar=True)


def test_hmc_common_diagnostics_schema():
    torch.manual_seed(0)
    s = HMC(_model, _space(), step_size=0.25, num_steps=6)
    s.run_mcmc(torch.zeros(2), num_samples=40, num_warmup_steps=40,
               num_chains=3, disable_progbar=True)
    d = s.diagnostics()
    assert COMMON_KEYS <= set(d)
    for k in COMMON_KEYS:
        assert torch.is_tensor(d[k]) and d[k].shape == (3,)
    assert d["num_divergences"].dtype == torch.long


def test_hmc_warmup_zero_keeps_constructor_step_size():
    torch.manual_seed(0)
    s = HMC(_model, _space(), step_size=0.37, num_steps=4, adapt_step_size=True)
    s.run_mcmc(torch.zeros(2), num_samples=20, num_warmup_steps=0,
               num_chains=3, disable_progbar=True)
    # adapter never updated -> step_size left at the constructor value.
    assert torch.allclose(s.step_size, torch.full((3,), 0.37))


def test_hmc_adaptation_moves_toward_target_accept():
    torch.manual_seed(0)
    # A deliberately large initial step should be shrunk by dual averaging so
    # the post-warmup acceptance lands near the target.
    s = HMC(_model, _space(), step_size=2.0, num_steps=8,
            target_accept_prob=0.7)
    s.run_mcmc(torch.zeros(2), num_samples=300, num_warmup_steps=400,
               num_chains=4, disable_progbar=True)
    acc = float(s.diagnostics()["accept_rate"].mean())
    assert abs(acc - 0.7) < 0.15


def test_hmc_under_pt_recovers_gaussian():
    # Exercises the tempered reorder path: PT permutes swapped configurations
    # across temperature slots, and the target chain must recover the posterior.
    torch.manual_seed(0)
    base = HMC(_model, _space(), step_size=0.2, num_steps=5)
    pt = PT(base, betas=torch.tensor([0.0, 0.25, 0.5, 1.0]))
    out = pt.run_mcmc(torch.zeros(2), num_samples=500, num_warmup_steps=300,
                      num_chains=3, disable_progbar=True)
    for i, n in enumerate(NAMES):
        assert abs(float(out[n].mean()) - float(POST_MEAN[i])) < 0.12


def test_hmc_state_caches_consistent_potential_and_grad():
    # The state carries the endpoint's tempered potential/gradient, so after any
    # step they must match a fresh evaluation at the state's position.
    torch.manual_seed(0)
    s = HMC(_model, _space(), step_size=0.2, num_steps=5, adapt_step_size=False)
    q0 = torch.zeros(3, 2)
    state = s.init(q0)
    for _ in range(5):
        state = s.step(state)
    U_fresh, _, g_fresh = s.evaluate_model(state.q, grad=True)
    assert torch.allclose(state.U.value, U_fresh.value, atol=1e-10)
    assert torch.allclose(state.grad.value, g_fresh.value, atol=1e-10)


def test_hmc_divergence_count_is_per_chain():
    torch.manual_seed(0)
    # A tiny threshold forces steps to register as divergences.
    s = HMC(_model, _space(), step_size=0.5, num_steps=4,
            adapt_step_size=False, divergence_threshold=1e-6)
    s.run_mcmc(torch.zeros(2), num_samples=15, num_warmup_steps=5,
               num_chains=3, disable_progbar=True)
    nd = s.diagnostics()["num_divergences"]
    assert nd.shape == (3,) and nd.dtype == torch.long
    assert int(nd.sum()) > 0
