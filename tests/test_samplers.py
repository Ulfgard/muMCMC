"""Self-contained smoke + regression tests for the samplers and adapters.

The target is a shifted-Gaussian likelihood under an N(0,1) prior, so the
posterior is a Gaussian with analytically known moments -- a non-constant
potential (real gradient) rather than a forgiving flat one.  Tests are
seed-fixed and sized to run on CPU in CI.  NUTS uses a single chain to avoid
multiprocessing; RMHMC is batched in-process so it uses several chains.
"""
import torch
import pytest
from pyro.distributions import Normal

from riemann_mcmc import (
    RMHMC,
    NUTS,
    UnconstrainedSpace,
    DualAveraging,
    REINFORCEAdapter,
)

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


def _rmhmc_model(theta):
    """Shifted-Gaussian likelihood + a mild position-dependent SPD metric.
    The metric does not change the target (only the geometry)."""
    U = 0.5 * (((theta - MU) ** 2) / SIGMA2).sum(-1)
    G = torch.eye(theta.shape[-1], dtype=theta.dtype) \
        + 0.3 * theta[..., :, None] * theta[..., None, :]
    return U, G


def _nuts_model(theta):
    return 0.5 * (((theta - MU) ** 2) / SIGMA2).sum(-1)


def test_rmhmc_recovers_known_gaussian():
    torch.manual_seed(0)
    s = RMHMC(_rmhmc_model, _space(), step_size=0.3, num_steps=5, fp_max_iter=20)
    out = s.run_mcmc(torch.zeros(2), num_samples=250, num_warmup_steps=150,
                     num_chains=4, disable_progbar=True)
    for i, n in enumerate(NAMES):
        x = out[n]
        assert x.shape == (4, 250)
        assert abs(float(x.mean()) - float(POST_MEAN[i])) < 0.1
        assert abs(float(x.std()) - POST_STD) < 0.1


def test_nuts_recovers_known_gaussian():
    torch.manual_seed(0)
    s = NUTS(_nuts_model, _space())
    out = s.run_mcmc(torch.zeros(2), num_samples=500, num_warmup_steps=300,
                     num_chains=1, disable_progbar=True)
    for i, n in enumerate(NAMES):
        x = out[n]
        assert abs(float(x.mean()) - float(POST_MEAN[i])) < 0.1
        assert abs(float(x.std()) - POST_STD) < 0.1


def test_common_diagnostics_schema():
    torch.manual_seed(0)
    r = RMHMC(_rmhmc_model, _space(), step_size=0.3, num_steps=5, fp_max_iter=20)
    r.run_mcmc(torch.zeros(2), num_samples=40, num_warmup_steps=40,
               num_chains=3, disable_progbar=True)
    dr = r.diagnostics()
    assert COMMON_KEYS <= set(dr)
    for k in COMMON_KEYS:
        assert torch.is_tensor(dr[k]) and dr[k].shape == (3,)
    assert dr["num_divergences"].dtype == torch.long

    n = NUTS(_nuts_model, _space())
    n.run_mcmc(torch.zeros(2), num_samples=40, num_warmup_steps=40,
               num_chains=1, disable_progbar=True)
    dn = n.diagnostics()
    assert set(dn) == COMMON_KEYS
    for k in COMMON_KEYS:
        assert torch.is_tensor(dn[k]) and dn[k].shape == (1,)


def test_rmhmc_divergence_count_is_per_chain():
    torch.manual_seed(0)
    # Tiny threshold forces many steps to register as divergences.
    r = RMHMC(_rmhmc_model, _space(), step_size=0.3, num_steps=4,
              fp_max_iter=20, divergence_threshold=1e-6)
    r.run_mcmc(torch.zeros(2), num_samples=15, num_warmup_steps=5,
               num_chains=3, disable_progbar=True)
    nd = r.diagnostics()["num_divergences"]
    assert nd.shape == (3,) and nd.dtype == torch.long
    assert int(nd.sum()) > 0


def test_warmup_zero_keeps_constructor_step_size():
    torch.manual_seed(0)
    s = RMHMC(_rmhmc_model, _space(), step_size=0.37, num_steps=4,
              fp_max_iter=20, adapt_step_size=True)
    s.run_mcmc(torch.zeros(2), num_samples=20, num_warmup_steps=0,
               num_chains=3, disable_progbar=True)
    # adapter never updated -> step_size left at the constructor value.
    assert torch.allclose(s.step_size, torch.full((3,), 0.37))


def test_dual_averaging_no_update_roundtrip():
    c = torch.tensor([-1.2, 0.0, 0.7])
    da = DualAveraging()
    da.prox_center = c
    da.reset()
    # never stepped: the running average equals the prox-center exactly.
    assert torch.equal(torch.as_tensor(da.get_state()[1]), c)


def test_reinforce_adapter_no_update_roundtrip():
    step_size0 = torch.tensor([0.05, 0.3, 1.7, 0.9])
    ad = REINFORCEAdapter(4, sigma=0.1)
    ad.prox_center = torch.log(step_size0)
    ad.reset()
    frozen = torch.exp(ad.get_state()[1])
    assert torch.allclose(frozen, step_size0)
