"""Tests for the explicit Lagrangian Monte Carlo sampler (Lan et al. e-RMLMC).

Integrator anchors: the constant-metric limit reduces exactly to HMC, the map
is time-reversible, and the accumulated log-Jacobian matches a finite-difference
Jacobian of the full (q, v) -> (q', v') map. Then a statistical-recovery test on
a curved metric (which leaves the target unchanged).
"""
import torch
import pytest
from pyro.distributions import Normal

from muMCMC import LMC, HMC, LMCState, UnconstrainedSpace

torch.set_default_dtype(torch.float64)

NAMES = ["a", "b"]
MU = torch.tensor([1.0, -0.5])
SIGMA2 = 1.0
POST_MEAN = MU / (SIGMA2 + 1.0)
POST_STD = (SIGMA2 / (SIGMA2 + 1.0)) ** 0.5


def _space():
    return UnconstrainedSpace(NAMES, priors={n: Normal(0.0, 1.0) for n in NAMES})


def _curved_model(theta):
    """Shifted-Gaussian likelihood with a mild position-dependent SPD metric.
    The metric changes the geometry, not the target."""
    U = 0.5 * (((theta - MU) ** 2) / SIGMA2).sum(-1)
    G = torch.eye(theta.shape[-1], dtype=theta.dtype) \
        + 0.3 * theta[..., :, None] * theta[..., None, :]
    return U, G


# --------------------------------------------------------------------------- #
#  integrator                                                                 #
# --------------------------------------------------------------------------- #

def test_constant_metric_reduces_to_hmc():
    # With G = M constant the Christoffels and the Jacobian vanish, so LMC with
    # velocity v0 = M^-1 p0 must reproduce HMC (mass M) with momentum p0.
    M = torch.tensor([[2.0, 0.3], [0.3, 1.0]])

    def lmc_model(theta):
        return 0.5 * ((theta - MU) ** 2).sum(-1), M.expand(*theta.shape[:-1], 2, 2)

    def hmc_model(theta):
        return 0.5 * ((theta - MU) ** 2).sum(-1)

    lmc = LMC(lmc_model, _space(), step_size=0.2, num_steps=6, adapt_step_size=False)
    hmc = HMC(hmc_model, _space(), step_size=0.2, num_steps=6,
              mass_matrix=M, adapt_step_size=False)

    q0 = torch.tensor([[0.3, -0.2], [0.1, 0.4]])
    p0 = torch.tensor([[0.5, -0.3], [0.2, 0.7]])
    sl, sh = lmc.init(q0), hmc.init(q0)
    sh.p = p0.clone()
    sl.v = (torch.linalg.inv(M) @ p0[..., None])[..., 0]
    sl.log_jac = torch.zeros(2)
    for _ in range(6):
        sh = hmc.integration_step(sh)
    for _ in range(6):
        sl = lmc.integration_step(sl)

    assert torch.allclose(sl.q, sh.q, atol=1e-10)
    assert float(sl.log_jac.abs().max()) < 1e-12


def _integrate(lmc, q, v, num_steps):
    s = LMCState(q.clone(), v.clone(), None, None, torch.zeros(q.shape[0]))
    for _ in range(num_steps):
        s = lmc.integration_step(s)
    return s


def test_integrator_is_reversible():
    lmc = LMC(_curved_model, _space(), step_size=0.15, num_steps=8,
              adapt_step_size=False)
    lmc.step_size = torch.tensor([0.15])
    q0 = torch.tensor([[0.4, -0.3]])
    v0 = torch.tensor([[0.6, 0.2]])

    fwd = _integrate(lmc, q0, v0, 8)
    back = _integrate(lmc, fwd.q, -fwd.v, 8)

    assert torch.allclose(back.q, q0, atol=1e-10)
    assert torch.allclose(back.v, -v0, atol=1e-10)


def test_log_jacobian_matches_finite_difference():
    lmc = LMC(_curved_model, _space(), step_size=0.15, num_steps=5,
              adapt_step_size=False)
    lmc.step_size = torch.tensor([0.15])

    def run(qv):
        s = _integrate(lmc, qv[:2].unsqueeze(0), qv[2:].unsqueeze(0), 5)
        return torch.cat([s.q[0], s.v[0]]), float(s.log_jac[0])

    qv0 = torch.tensor([0.4, -0.3, 0.6, 0.2])
    _, log_jac = run(qv0)
    h = 1e-6
    J = torch.zeros(4, 4)
    for i in range(4):
        dp, dm = qv0.clone(), qv0.clone()
        dp[i] += h
        dm[i] -= h
        J[:, i] = (run(dp)[0] - run(dm)[0]) / (2 * h)
    assert abs(log_jac - float(torch.linalg.slogdet(J)[1])) < 1e-7


# --------------------------------------------------------------------------- #
#  geometry: Omega and the force term, pinned to an independent reference      #
# --------------------------------------------------------------------------- #
#
# The integrator tests above check the *Metropolis machinery* -- reversibility,
# the self-consistent log-Jacobian, the constant-metric reduction to HMC.  None
# of them pins the *curvature* terms: a flipped sign in Omega's antisymmetric
# part (D + J - J^T) or in the force's +1/2 log det G leaves the map reversible
# and its Jacobian self-consistent, so LMC still samples the right target under
# Metropolis correction -- only with lower acceptance.  The constant-metric
# reduction cannot see them either (both vanish when G is constant).  These
# tests compare Omega and the force against an independent reference that
# materialises the rank-3 dG the production code deliberately avoids, so a wrong
# curvature fails here instead of silently costing acceptance.


def _geometry_reference(model, q_row, v_row):
    """Independent ``Omega(q, v)`` and ``G^-1 grad phi`` at one point ``(d,)``.

    Builds the full third-order derivative ``dG[a, b, c] = d G_ab / d q_c`` by
    autograd (the direct route ``_geometry`` sidesteps), then assembles
    ``Omega = 1/2 G^-1 (D + J - J^T)`` with ``J_lj = d(G v)_l / d q_j`` and
    ``D_lj = (v . grad) G_lj``, and the force ``G^-1 grad(U + 1/2 log det G)``.
    """
    G_of   = lambda x: model(x.unsqueeze(0))[1][0]
    phi_of = lambda x: (model(x.unsqueeze(0))[0]
                        + 0.5 * torch.logdet(model(x.unsqueeze(0))[1]))[0]
    G  = G_of(q_row)
    dG = torch.autograd.functional.jacobian(G_of, q_row)          # (d, d, d)
    J  = torch.einsum("lkj,k->lj", dG, v_row)                     # d(G v)_l / d q_j
    D  = torch.einsum("ljc,c->lj", dG, v_row)                     # (v . grad) G
    omega = 0.5 * torch.linalg.solve(G, D + J - J.transpose(-1, -2))
    force = torch.linalg.solve(G, torch.autograd.functional.jacobian(phi_of, q_row))
    return omega, force


def _armed_geometry():
    """An LMC over an identity, prior-free space (so ``phi = U + 1/2 log det G``
    isolates the geometry and the free metric is exactly the model's ``G``),
    with fixed positions/velocities the reference is checked against."""
    lmc = LMC(_curved_model, UnconstrainedSpace(NAMES), step_size=0.1,
              num_steps=1, adapt_step_size=False)
    lmc.step_size = torch.full((3,), 0.1)
    torch.manual_seed(0)
    return lmc, torch.randn(3, 2), torch.randn(3, 2)


def test_omega_matches_christoffel_reference():
    # Pins Omega's antisymmetric assembly: a flipped or dropped J^T term is
    # invisible to reversibility and to the self-consistent log-Jacobian.
    lmc, q, v = _armed_geometry()
    _, omega = lmc._geometry(q)
    om_prod = omega(v)
    for i in range(q.shape[0]):
        om_ref, _ = _geometry_reference(_curved_model, q[i], v[i])
        assert torch.allclose(om_prod[i], om_ref, atol=1e-9)


def test_force_matches_reference_including_logdet_term():
    # Pins G^-1 grad(U + 1/2 log det G): the metric inverse (not G) and the
    # +1/2 log det G sign, neither of which the constant-metric reduction sees.
    lmc, q, v = _armed_geometry()
    force_prod, _ = lmc._geometry(q)
    for i in range(q.shape[0]):
        _, force_ref = _geometry_reference(_curved_model, q[i], v[i])
        assert torch.allclose(force_prod[i], force_ref, atol=1e-9)


# --------------------------------------------------------------------------- #
#  sampler                                                                    #
# --------------------------------------------------------------------------- #

def test_lmc_recovers_gaussian_on_curved_metric():
    torch.manual_seed(0)
    s = LMC(_curved_model, _space(), step_size=0.3, num_steps=5,
            adapt_step_size=False)
    out = s.run_mcmc(torch.zeros(2), num_samples=2000, num_warmup_steps=200,
                     num_chains=6, disable_progbar=True)
    for i, n in enumerate(NAMES):
        assert abs(float(out[n].mean()) - float(POST_MEAN[i])) < 0.1
        assert abs(float(out[n].std()) - POST_STD) < 0.1


def test_lmc_common_diagnostics_schema():
    torch.manual_seed(0)
    s = LMC(_curved_model, _space(), step_size=0.3, num_steps=4)
    s.run_mcmc(torch.zeros(2), num_samples=40, num_warmup_steps=40,
               num_chains=3, disable_progbar=True)
    d = s.diagnostics()
    for k in ("accept_rate", "num_divergences", "step_size"):
        assert torch.is_tensor(d[k]) and d[k].shape == (3,)
    assert d["num_divergences"].dtype == torch.long
