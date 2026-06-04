"""Bottom-up tests for the RMHMC implicit-midpoint integrator.

Built from the lowest level upward so each layer rests on a verified one:

1. ``_hamiltonian``  -- the value H = U + 1/2 p^T G^-1 p + 1/2 log det G, checked
   against dense linear algebra.
2. ``_midpoint_map`` -- the fixed-point map F(z_k).  Its only gradient is
   dH/dq at the midpoint; we verify both the position update formula and that
   gradient against an *independent* finite difference of H.  The test model has
   genuine q-dependence in BOTH the likelihood and the metric, so the gradient
   exercises the metric's log-det and kinetic q-terms (a metric that were
   silently detached would fail the finite-difference check).
3. ``_implicit_midpoint_step`` -- the Picard solve: the returned endpoint must
   satisfy the implicit-midpoint equations, per-chain convergence is
   independent, the step is time-reversible, and it preserves phase-space volume
   and the symplectic form (finite-difference Jacobian).
4. The integrator property that motivates the whole scheme: on a quadratic
   Hamiltonian (Gaussian target, *constant* metric) the implicit midpoint rule
   conserves H *exactly* -- to the fixed-point tolerance, independent of the step
   size -- because it preserves quadratic invariants.
"""
import torch
import pytest

from muMCMC.RMHMC import (
    RMHMC,
    _hamiltonian,
    _midpoint_map,
    _implicit_midpoint_step,
)
from muMCMC.spaces import UnconstrainedSpace

torch.set_default_dtype(torch.float64)

D = 3

# Deterministic SPD matrices (diagonally dominant) and a mean, so the models
# below are fixed and reproducible without seeding.
A_QUAD = torch.tensor([[2.0, 0.3, 0.1],
                       [0.3, 3.0, 0.2],
                       [0.1, 0.2, 1.5]])
B_CONST = torch.tensor([[1.5, 0.2, 0.0],
                        [0.2, 2.0, 0.1],
                        [0.0, 0.1, 1.0]])
MU = torch.tensor([1.0, -0.5, 0.3])


def make_eval(model_fn, fp_tol=1e-12, fp_max_iter=200):
    """evaluate_model for an identity space (no prior): the pulled-back metric
    is exactly G_lik and the potential is exactly U_lik."""
    space = UnconstrainedSpace([f"x{i}" for i in range(D)])
    s = RMHMC(model_fn, space, fp_tol=fp_tol, fp_max_iter=fp_max_iter)
    return s.evaluate_model


def model_qdep(theta):
    """Likelihood AND metric genuinely depend on q (metric is a rank-1 update,
    always SPD).  Used wherever we need real dH/dq through the metric."""
    U = 0.5 * ((theta - MU) ** 2).sum(-1)
    n = theta.shape[-1]
    G = torch.eye(n, dtype=theta.dtype) + 0.3 * theta[..., :, None] * theta[..., None, :]
    return U, G


def model_gauss_const(theta):
    """Quadratic potential with a *constant* metric -> quadratic Hamiltonian."""
    U = 0.5 * torch.einsum("...i,ij,...j->...", theta, A_QUAD, theta)
    n = theta.shape[-1]
    return U, B_CONST.expand(*theta.shape[:-1], n, n)


# ========================================================================== #
#  1. _hamiltonian                                                           #
# ========================================================================== #

def test_hamiltonian_matches_dense():
    ev = make_eval(model_qdep)
    torch.manual_seed(0)
    q = torch.randn(4, D)
    p = torch.randn(4, D)
    U, metric = ev(q)
    G = model_qdep(q)[1]
    Ginv_p = torch.linalg.solve(G, p[..., None])[..., 0]
    expected = U + 0.5 * (p * Ginv_p).sum(-1) + 0.5 * torch.logdet(G)
    H = _hamiltonian(q, p, U, metric)
    assert H.shape == (4,)
    assert torch.allclose(H, expected, atol=1e-10)


def test_hamiltonian_ignores_position_argument():
    # Docstring: q is passed for interface symmetry but unused (U/metric are
    # pre-evaluated).  Passing a different q must not change the result.
    ev = make_eval(model_qdep)
    torch.manual_seed(1)
    q = torch.randn(3, D)
    p = torch.randn(3, D)
    U, metric = ev(q)
    assert torch.equal(_hamiltonian(q, p, U, metric),
                       _hamiltonian(q + 5.0, p, U, metric))


# ========================================================================== #
#  2. _midpoint_map                                                          #
# ========================================================================== #

def _random_phase(N, seed):
    torch.manual_seed(seed)
    return (torch.randn(N, D), torch.randn(N, D),
            torch.randn(N, D), torch.randn(N, D))


def test_midpoint_map_position_update_formula():
    ev = make_eval(model_qdep)
    q, p, q_k, p_k = _random_phase(2, seed=2)
    eps = torch.full((2,), 0.2)
    F_q, _ = _midpoint_map(q, p, q_k, p_k, eps, ev)

    q_mid = 0.5 * (q + q_k)
    _, metric_mid = ev(q_mid)
    expected = q + (eps.unsqueeze(-1) / 2.0) * metric_mid.inv_metric_times_vec(p + p_k)
    assert torch.allclose(F_q, expected, atol=1e-10)


def test_midpoint_map_momentum_gradient_matches_finite_difference():
    # F_p = p - eps * dH/dq|_mid, so the implied gradient is (p - F_p)/eps.
    # Check it against a finite difference of H at the midpoint -- this is the
    # real test that the metric's q-dependence flows into the gradient.
    ev = make_eval(model_qdep)
    q, p, q_k, p_k = _random_phase(2, seed=3)
    eps = torch.full((2,), 0.2)
    _, F_p = _midpoint_map(q, p, q_k, p_k, eps, ev)
    dHdq_used = (p - F_p) / eps.unsqueeze(-1)

    q_mid = 0.5 * (q + q_k)
    p_mid = 0.5 * (p + p_k)

    def H_of(qm):
        U, m = ev(qm)
        return _hamiltonian(qm, p_mid, U, m)

    h = 1e-5
    dHdq_fd = torch.zeros(2, D)
    for j in range(D):
        qp = q_mid.clone(); qp[:, j] += h
        qm = q_mid.clone(); qm[:, j] -= h
        dHdq_fd[:, j] = (H_of(qp) - H_of(qm)) / (2 * h)

    assert torch.allclose(dHdq_used, dHdq_fd, atol=1e-6)


# ========================================================================== #
#  3. _implicit_midpoint_step                                               #
# ========================================================================== #

def test_step_endpoint_satisfies_implicit_midpoint_equations():
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(3, seed=4)
    eps = torch.full((3,), 0.2)
    q1, p1, iters, residual = _implicit_midpoint_step(q, p, eps, ev, 200, 1e-12)
    # the converged endpoint is a fixed point of the midpoint map
    F_q, F_p = _midpoint_map(q, p, q1, p1, eps, ev)
    assert torch.allclose(q1, F_q, atol=1e-8)
    assert torch.allclose(p1, F_p, atol=1e-8)
    assert torch.all(residual < 1e-8)
    assert iters.shape == (3,) and residual.shape == (3,)


def test_step_per_chain_convergence_is_independent():
    # One batch, two chains: a small step converges quickly; a huge step never
    # converges.  The freeze-mask must keep them independent.
    ev = make_eval(model_gauss_const)
    torch.manual_seed(5)
    q = torch.randn(2, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    eps = torch.tensor([0.2, 3.0])
    q1, p1, iters, residual = _implicit_midpoint_step(q, p, eps, ev, 50, 1e-10)
    assert int(iters[0]) < 50 and float(residual[0]) < 1e-9      # converged
    assert int(iters[1]) == 50                                    # never converged
    assert torch.isfinite(q1[0]).all() and torch.isfinite(p1[0]).all()


def test_step_is_time_reversible():
    # Symmetric integrator: stepping forward, flipping p, and stepping again
    # returns to the start with reversed momentum.
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(3, seed=6)
    for eps_val in (0.1, 0.3, 0.5):
        eps = torch.full((3,), eps_val)
        q1, p1, _, _ = _implicit_midpoint_step(q, p, eps, ev, 200, 1e-12)
        q2, p2, _, _ = _implicit_midpoint_step(q1, -p1, eps, ev, 200, 1e-12)
        assert torch.allclose(q2, q, atol=1e-9)
        assert torch.allclose(p2, -p, atol=1e-9)


def test_step_preserves_volume_and_symplectic_form():
    # Finite-difference Jacobian M of the single-step map on phase space.
    # A symplectic integrator satisfies M^T Omega M = Omega (hence det M = 1),
    # independent of the Hamiltonian -- the volume property.
    ev = make_eval(model_qdep, fp_tol=1e-13, fp_max_iter=300)
    torch.manual_seed(7)
    z0 = torch.randn(2 * D)
    eps = torch.full((1,), 0.3)

    def step_map(z):
        q = z[:D].reshape(1, D)
        p = z[D:].reshape(1, D)
        q1, p1, _, _ = _implicit_midpoint_step(q, p, eps, ev, 300, 1e-13)
        return torch.cat([q1.reshape(-1), p1.reshape(-1)])

    h = 1e-6
    M = torch.zeros(2 * D, 2 * D)
    for j in range(2 * D):
        zp = z0.clone(); zp[j] += h
        zm = z0.clone(); zm[j] -= h
        M[:, j] = (step_map(zp) - step_map(zm)) / (2 * h)

    Omega = torch.zeros(2 * D, 2 * D)
    Omega[:D, D:] = torch.eye(D)
    Omega[D:, :D] = -torch.eye(D)

    assert abs(float(torch.det(M)) - 1.0) < 1e-6                       # volume
    assert float((M.T @ Omega @ M - Omega).abs().max()) < 1e-6        # symplectic


# ========================================================================== #
#  4. Exact conservation on a quadratic Hamiltonian                          #
# ========================================================================== #

def _H_at(ev, q, p):
    U, metric = ev(q)
    return _hamiltonian(q, p, U, metric)


@pytest.mark.parametrize("eps_val", [0.05, 0.1, 0.3, 0.7])
def test_quadratic_hamiltonian_conserved_at_any_step_size(eps_val):
    # Gaussian target + constant metric => quadratic H.  The implicit midpoint
    # rule preserves quadratic invariants, so dH is at the solver tolerance for
    # every step size at which the solve converges (the trajectory itself is
    # only a Cayley approximation -- it is H that is exact).
    ev = make_eval(model_gauss_const)
    torch.manual_seed(0)
    q = torch.randn(1, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    H0 = _H_at(ev, q, p)

    eps = torch.full((1,), eps_val)
    q1, p1, _, residual = _implicit_midpoint_step(q, p, eps, ev, 300, 1e-12)
    assert float(residual) < 1e-9                       # solver converged
    assert abs(float(_H_at(ev, q1, p1) - H0)) < 1e-8    # H exactly conserved


def test_quadratic_hamiltonian_conserved_over_many_steps():
    ev = make_eval(model_gauss_const)
    torch.manual_seed(0)
    q = torch.randn(1, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    H0 = _H_at(ev, q, p)

    eps = torch.full((1,), 0.3)
    for _ in range(25):
        q, p, _, residual = _implicit_midpoint_step(q, p, eps, ev, 300, 1e-12)
        assert float(residual) < 1e-9
    assert abs(float(_H_at(ev, q, p) - H0)) < 1e-7      # no drift over the run
