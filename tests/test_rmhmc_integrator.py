"""Bottom-up tests for the RMHMC implicit-midpoint integrator.

Built from the lowest level upward so each layer rests on a verified one:

1. ``_hamiltonian``  -- the value H = U + 1/2 p^T G^-1 p + 1/2 log det G, checked
   against dense linear algebra.
2. The oracle ``_midpoint_map`` -- the implicit-midpoint equations on (q, p),
   written here independently of the integrator, which solves them with p
   eliminated. Its only gradient is dH/dq at the midpoint; we verify both the
   position update formula and that gradient against a finite difference of H.
   The test model has genuine q-dependence in BOTH the likelihood and the
   metric, so the gradient exercises the metric's log-det and kinetic q-terms.
   ``_midpoint_terms`` -- the integrator's own gradient, (eps^2/2) grad V minus
   half the metric's quadratic term, checked against a finite difference too.
3. ``_implicit_midpoint_step`` -- the preconditioned Picard solve: the returned
   endpoint must satisfy the (q, p) implicit-midpoint equations, per-chain
   convergence is independent, the step is time-reversible, and it preserves
   phase-space volume and the symplectic form (finite-difference Jacobian).
4. The integrator property that motivates the whole scheme: on a quadratic
   Hamiltonian (Gaussian target, *constant* metric) the implicit midpoint rule
   conserves H *exactly* -- to the fixed-point tolerance, independent of the step
   size -- because it preserves quadratic invariants.
"""
import math

import torch
import pytest

from muMCMC.RMHMC import (
    RMHMC,
    _hamiltonian,
    _midpoint_terms,
    _implicit_midpoint_step,
)
from muMCMC._solvers import FixedPointSolver
from muMCMC.spaces import UnnormalizedSpace

torch.set_default_dtype(torch.float64)


def _fp(kind="picard", **kw):
    """A FixedPointSolver, the object the integrator now takes."""
    return FixedPointSolver(kind, **kw)

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
    space = UnnormalizedSpace([f"x{i}" for i in range(D)])
    s = RMHMC(model_fn, space, step_size=0.1, fp_tol=fp_tol, fp_max_iter=fp_max_iter)
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


def _midpoint_map(q, p, q_k, p_k, eps, evaluate_model):
    """Oracle: the implicit-midpoint fixed-point map F(z_k) = (F_q, F_p) on the
    endpoint z_k = (q_k, p_k),

        q_mid = (q + q_k)/2,  p_mid = (p + p_k)/2
        F_q   = q + (eps/2) G(q_mid)^-1 (p + p_k)
        F_p   = p - eps dH/dq|_(q_mid, p_mid),

    whose fixed point is the step endpoint. Independent of the integrator, which
    eliminates p from these equations before solving."""
    q_mid = (0.5 * (q + q_k)).detach().requires_grad_(True)
    p_mid = 0.5 * (p + p_k)
    with torch.enable_grad():
        potential, metric = evaluate_model(q_mid)
        H = _hamiltonian(q_mid, p_mid, potential.value, metric)
        (dHdq,) = torch.autograd.grad(H.sum(), q_mid)
    e = eps.unsqueeze(-1)
    with torch.no_grad():
        F_q = q + (e / 2.0) * metric.inv_metric_times_vec(p + p_k)
        F_p = p - e * dHdq
    return F_q, F_p


def _assert_fixed_point(q, p, q1, p1, eps, ev, atol=1e-8):
    """The endpoint satisfies the (q, p) implicit-midpoint equations."""
    F_q, F_p = _midpoint_map(q, p, q1, p1, eps, ev)
    assert torch.allclose(q1, F_q, atol=atol)
    assert torch.allclose(p1, F_p, atol=atol)


# ========================================================================== #
#  1. _hamiltonian                                                           #
# ========================================================================== #

def test_hamiltonian_matches_dense():
    ev = make_eval(model_qdep)
    torch.manual_seed(0)
    q = torch.randn(4, D)
    p = torch.randn(4, D)
    potential, metric = ev(q)
    G = model_qdep(q)[1]
    Ginv_p = torch.linalg.solve(G, p[..., None])[..., 0]
    expected = potential.value + 0.5 * (p * Ginv_p).sum(-1) + 0.5 * torch.logdet(G)
    H = _hamiltonian(q, p, potential.value, metric)
    assert H.shape == (4,)
    assert torch.allclose(H, expected, atol=1e-10)


def test_hamiltonian_ignores_position_argument():
    # Docstring: q is passed for interface symmetry but unused (U/metric are
    # pre-evaluated).  Passing a different q must not change the result.
    ev = make_eval(model_qdep)
    torch.manual_seed(1)
    q = torch.randn(3, D)
    p = torch.randn(3, D)
    potential, metric = ev(q)
    assert torch.equal(_hamiltonian(q, p, potential.value, metric),
                       _hamiltonian(q + 5.0, p, potential.value, metric))


# ========================================================================== #
#  2. The oracle, and the integrator's midpoint terms                       #
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
        potential, m = ev(qm)
        return _hamiltonian(qm, p_mid, potential.value, m)

    h = 1e-5
    dHdq_fd = torch.zeros(2, D)
    for j in range(D):
        qp = q_mid.clone(); qp[:, j] += h
        qm = q_mid.clone(); qm[:, j] -= h
        dHdq_fd[:, j] = (H_of(qp) - H_of(qm)) / (2 * h)

    assert torch.allclose(dHdq_used, dHdq_fd, atol=1e-6)


def test_midpoint_terms_match_finite_difference():
    # G(qb) dq is a plain matvec; the gradient term is the derivative of
    # S(qb) = (eps^2/2) V(qb) - (1/4) dq^T G(qb) dq at fixed dq, with
    # V = U + (1/2) log det G, which is where the metric's q-dependence and the
    # 1/2 in front of Gamma enter. Both against an independent evaluation.
    ev = make_eval(model_qdep)
    q0, q1, _, _ = _random_phase(2, seed=8)
    eps = torch.tensor([0.2, 0.5])
    G_dq, grad_S = _midpoint_terms(q0, q1, eps, ev)

    dq, qb = q1 - q0, 0.5 * (q0 + q1)
    G = model_qdep(qb)[1]
    assert torch.allclose(G_dq, (G @ dq.unsqueeze(-1)).squeeze(-1), atol=1e-12)

    def S_of(x):
        potential, m = ev(x)
        Gx = model_qdep(x)[1]
        V = potential.value + 0.5 * torch.logdet(Gx)
        return 0.5 * eps * eps * V - 0.25 * torch.einsum("ni,nij,nj->n", dq, Gx, dq)

    h = 1e-5
    fd = torch.zeros(2, D)
    for j in range(D):
        xp = qb.clone(); xp[:, j] += h
        xm = qb.clone(); xm[:, j] -= h
        fd[:, j] = (S_of(xp) - S_of(xm)) / (2 * h)
    assert torch.allclose(grad_S, fd, atol=1e-6)


def test_midpoint_terms_are_the_eliminated_midpoint_equations():
    # Eliminating p: any q1 is the endpoint from the start momentum
    # p0 = (G dq + grad S)/eps, the root of the residual, and the momentum the
    # integrator reads off it is p1 = (G dq - grad S)/eps. The oracle must
    # then hold (q1, p1) fixed from (q0, p0).
    ev = make_eval(model_qdep)
    q0, _, q1, _ = _random_phase(3, seed=9)
    eps = torch.full((3,), 0.3)
    e = eps.unsqueeze(-1)
    G_dq, grad_S = _midpoint_terms(q0, q1, eps, ev)
    p0 = (G_dq + grad_S) / e
    p1 = (G_dq - grad_S) / e
    _assert_fixed_point(q0, p0, q1, p1, eps, ev, atol=1e-10)


# ========================================================================== #
#  3. _implicit_midpoint_step                                               #
# ========================================================================== #

def test_step_endpoint_satisfies_implicit_midpoint_equations():
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(3, seed=4)
    eps = torch.full((3,), 0.2)
    q1, p1, iters, residual = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=200, tol=1e-12))
    # the converged endpoint is a fixed point of the (q, p) midpoint map
    _assert_fixed_point(q, p, q1, p1, eps, ev)
    assert torch.all(residual < 1e-8)
    assert iters.shape == (3,) and residual.shape == (3,)


def test_step_takes_the_start_metric_as_its_preconditioner():
    # Passing the metric at q reproduces the step that evaluates it itself, and
    # a warm start lands on the same endpoint, changing only the iteration
    # count.
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(3, seed=4)
    eps = torch.full((3,), 0.2)
    _, metric = ev(q)
    solver = _fp("picard", max_iter=200, tol=1e-12)
    cold = _implicit_midpoint_step(q, p, eps, ev, solver)
    given = _implicit_midpoint_step(q, p, eps, ev, solver, metric=metric)
    assert torch.equal(cold[0], given[0]) and torch.equal(cold[1], given[1])
    assert torch.equal(cold[2], given[2])

    warm = _implicit_midpoint_step(q, p, eps, ev, solver, q_init=cold[0])
    assert torch.allclose(warm[0], cold[0], atol=1e-10)
    assert torch.allclose(warm[1], cold[1], atol=1e-10)
    assert torch.all(warm[2] < cold[2])


def test_preconditioned_picard_converges_at_second_order_in_the_step():
    # Preconditioned by G(q0), the iteration is a frozen-Jacobian Newton whose
    # contraction is O(eps^2): on the constant-metric quadratic model it is
    # exactly (eps^2/4) B^-1 A, so the iteration count at eps grows only
    # slowly, and a halved step converges in a fraction of the iterations that
    # an O(eps) contraction would need.
    ev = make_eval(model_gauss_const)
    torch.manual_seed(16)
    q = torch.randn(1, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    solver = _fp("picard", max_iter=200, tol=1e-10)
    rho = float(torch.linalg.eigvals(torch.linalg.solve(B_CONST, A_QUAD)).real.max())

    def iters_at(eps_val):
        _, _, it, r = _implicit_midpoint_step(q, p, torch.full((1,), eps_val), ev, solver)
        assert float(r) < 1e-10
        return int(it)

    for eps_val in (0.4, 0.8):
        contraction = 0.25 * eps_val * eps_val * rho
        assert contraction < 1.0
        # a linear contraction c reaches tol from O(1) in ~ log(tol)/log(c)
        expected = math.log(1e-10) / math.log(contraction)
        assert iters_at(eps_val) <= expected + 3


def test_step_per_chain_convergence_is_independent():
    # One batch, two chains: a small step converges quickly; a huge step never
    # converges.  The freeze-mask must keep them independent.
    ev = make_eval(model_gauss_const)
    torch.manual_seed(5)
    q = torch.randn(2, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    eps = torch.tensor([0.2, 3.0])
    q1, p1, iters, residual = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=50, tol=1e-10))
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
        q1, p1, _, _ = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=200, tol=1e-12))
        q2, p2, _, _ = _implicit_midpoint_step(q1, -p1, eps, ev, _fp("picard", max_iter=200, tol=1e-12))
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
        q1, p1, _, _ = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=300, tol=1e-13))
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
    potential, metric = ev(q)
    return _hamiltonian(q, p, potential.value, metric)


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
    q1, p1, _, residual = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=300, tol=1e-12))
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
        q, p, _, residual = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=300, tol=1e-12))
        assert float(residual) < 1e-9
    assert abs(float(_H_at(ev, q, p) - H0)) < 1e-7      # no drift over the run


# ========================================================================== #
#  5. Anderson solver: same fixed point, fewer iterations                    #
# ========================================================================== #

def test_anderson_reaches_same_endpoint_as_picard():
    # Both solvers attack the identical fixed-point equation, so the converged
    # endpoint must agree to the solver tolerance -- only the path (and the
    # iteration count) differs.
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(4, seed=11)
    eps = torch.full((4,), 0.3)

    qp, pp, _, rp = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=200, tol=1e-12))
    qa, pa, _, ra = _implicit_midpoint_step(q, p, eps, ev, _fp("anderson", max_iter=200, tol=1e-12, anderson_history=6))

    assert torch.all(rp < 1e-11) and torch.all(ra < 1e-11)   # both converged
    assert torch.allclose(qa, qp, atol=1e-9)
    assert torch.allclose(pa, pp, atol=1e-9)


def test_anderson_endpoint_satisfies_implicit_midpoint_equations():
    # The Anderson endpoint is a genuine fixed point of the midpoint map, not
    # just close to Picard's.
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(3, seed=12)
    eps = torch.full((3,), 0.25)
    q1, p1, _, residual = _implicit_midpoint_step(q, p, eps, ev, _fp("anderson", max_iter=200, tol=1e-12))
    _assert_fixed_point(q, p, q1, p1, eps, ev)
    assert torch.all(residual < 1e-8)


def test_anderson_is_time_reversible():
    # The endpoint is solver-independent, so Anderson inherits the integrator's
    # time-reversibility.
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(3, seed=13)
    for eps_val in (0.1, 0.3, 0.5):
        eps = torch.full((3,), eps_val)
        q1, p1, _, _ = _implicit_midpoint_step(q, p, eps, ev, _fp("anderson", max_iter=200, tol=1e-12))
        q2, p2, _, _ = _implicit_midpoint_step(q1, -p1, eps, ev, _fp("anderson", max_iter=200, tol=1e-12))
        assert torch.allclose(q2, q, atol=1e-9)
        assert torch.allclose(p2, -p, atol=1e-9)


def test_anderson_solves_linear_map_faster_than_picard():
    # Constant metric + quadratic potential => the midpoint map is affine, where
    # Anderson(m>=1) reaches the fixed point in far fewer iterations than the
    # linearly-convergent Picard iteration.
    ev = make_eval(model_gauss_const)
    torch.manual_seed(14)
    q = torch.randn(1, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    eps = torch.full((1,), 0.6)

    _, _, it_p, r_p = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=100, tol=1e-10))
    _, _, it_a, r_a = _implicit_midpoint_step(q, p, eps, ev, _fp("anderson", max_iter=100, tol=1e-10, anderson_history=D))
    assert float(r_p) < 1e-9 and float(r_a) < 1e-9          # both converge
    assert int(it_a) < int(it_p)                            # Anderson is faster


def test_anderson_default_history_is_the_solve_dimension():
    # A None history resolves to the row dimension, which here is dim(q) since
    # the unknown is the endpoint position. Checked behaviourally: an explicit
    # dim(q) has to reproduce the default exactly.
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(2, seed=15)
    eps = torch.full((2,), 0.3)
    auto = _implicit_midpoint_step(
        q, p, eps, ev, _fp("anderson", max_iter=200, tol=1e-12,
                           anderson_history=None))
    named = _implicit_midpoint_step(
        q, p, eps, ev, _fp("anderson", max_iter=200, tol=1e-12,
                           anderson_history=D))
    assert torch.equal(auto[2], named[2])              # same iteration counts
    assert torch.allclose(auto[0], named[0], atol=1e-12)
    assert torch.all(auto[3] < 1e-10)


# ========================================================================== #
#  6. Damping (under-relaxation): stabilises the near-imaginary spectrum     #
# ========================================================================== #

def _damped(kind, beta, **kw):
    """A solver of the given kind at damping beta, for parametrised tests."""
    return FixedPointSolver(kind, damping=beta, **kw)


def test_damping_rescues_a_step_size_where_undamped_diverges():
    # Constant metric + quadratic potential => the preconditioned iteration
    # matrix is -(eps^2/4) B^-1 A, with real negative eigenvalues. Here
    # lambda_max(B_CONST^-1 A_QUAD) ~ 1.58, so the undamped (beta=1) iteration
    # has spectral radius ~1.28 for eps=1.8 and cannot converge, while
    # under-relaxation (beta=0.5) maps the spectrum to 1 - 0.5(1 + mu), inside
    # the unit circle. Same eps, same solver, only beta differs.
    #
    # Picard only. A constant metric with a quadratic potential makes the
    # residual affine in q, so Anderson at the default history of dim(q)
    # solves it directly and damping cannot change the outcome.
    solver = "picard"
    ev = make_eval(model_gauss_const)
    torch.manual_seed(20)
    q = torch.randn(1, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    eps = torch.full((1,), 1.8)

    _, _, _, res_undamped = _implicit_midpoint_step( q, p, eps, ev, _damped(solver, 1.0, max_iter=100, tol=1e-9))
    q1, p1, _, res_damped = _implicit_midpoint_step( q, p, eps, ev, _damped(solver, 0.5, max_iter=100, tol=1e-9))

    assert float(res_undamped) > 1e-9                 # beta=1: does not converge
    assert float(res_damped) < 1e-9                   # beta<1: converges
    # ...and to a genuine fixed point of the (beta-independent) midpoint map.
    _assert_fixed_point(q, p, q1, p1, eps, ev)


@pytest.mark.parametrize("solver", ["picard", "anderson"])
def test_damping_reaches_same_endpoint_as_undamped(solver):
    # Where both converge, beta only rescales the path: the fixed point is
    # beta-independent, so the endpoints must agree.
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(3, seed=21)
    eps = torch.full((3,), 0.3)
    q1, p1, _, r1 = _implicit_midpoint_step( q, p, eps, ev, _damped(solver, 1.0, max_iter=300, tol=1e-12))
    qb, pb, _, rb = _implicit_midpoint_step( q, p, eps, ev, _damped(solver, 0.6, max_iter=300, tol=1e-12))
    assert torch.all(r1 < 1e-11) and torch.all(rb < 1e-11)
    assert torch.allclose(qb, q1, atol=1e-9)
    assert torch.allclose(pb, p1, atol=1e-9)


# ========================================================================== #
#  7. Solver fallback ladder                                                 #
# ========================================================================== #

def test_fallback_ladder_rescues_a_step_where_the_base_solver_diverges():
    # Same setup as the damping-rescue test: at eps=1.8 the undamped Picard base
    # does not converge. The fallback ladder re-solves the failed chain with
    # damping and lands it on a genuine fixed point of the midpoint map -- so a
    # step that would have been rejected (breaking detailed balance) is resolved.
    ev = make_eval(model_gauss_const)
    torch.manual_seed(20)
    q = torch.randn(1, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    eps = torch.full((1,), 1.8)

    _, _, _, r_base = _implicit_midpoint_step( q, p, eps, ev, _fp("picard", max_iter=100, tol=1e-9, damping=1.0))                 # no ladder
    q1, p1, _, r_lad = _implicit_midpoint_step( q, p, eps, ev, _fp("picard", max_iter=100, tol=1e-9, damping=1.0, fallback_damping=(0.5,), fallback_iter_scale=3))

    assert float(r_base) > 1e-9                        # base alone: does not converge
    assert float(r_lad) < 1e-9                         # ladder: rescued
    _assert_fixed_point(q, p, q1, p1, eps, ev)


def test_fallback_only_touches_unconverged_chains():
    # A batch mixing an easy chain (base converges) and a hard one (needs the
    # ladder): the easy chain's endpoint and iteration count are identical with
    # or without the ladder -- only the failed chain is re-solved.
    ev = make_eval(model_gauss_const)
    torch.manual_seed(20)
    q = torch.randn(1, D)
    _, metric = ev(q)
    p = metric.sample_momentum()
    q2 = torch.cat([q, q], 0)
    p2 = torch.cat([p, p], 0)
    eps = torch.tensor([0.3, 1.8])                     # easy, then diverges undamped

    qb, pb, ib, rb = _implicit_midpoint_step( q2, p2, eps, ev, _fp("picard", max_iter=100, tol=1e-9, damping=1.0))                # no ladder
    ql, _, il, rl = _implicit_midpoint_step( q2, p2, eps, ev, _fp("picard", max_iter=100, tol=1e-9, damping=1.0, fallback_damping=(0.5,), fallback_iter_scale=3))

    # easy chain untouched by the ladder
    assert torch.equal(qb[0], ql[0]) and torch.equal(ib[0], il[0])
    # hard chain: rejected by base, rescued by the ladder
    assert float(rb[1]) > 1e-9 and float(rl[1]) < 1e-9


def test_fallback_empty_schedule_is_a_plain_single_pass():
    ev = make_eval(model_qdep)
    q, p, _, _ = _random_phase(3, seed=41)
    eps = torch.full((3,), 0.3)
    q0, p0, i0, r0 = _implicit_midpoint_step(q, p, eps, ev, _fp("picard", max_iter=200, tol=1e-12))
    q1, p1, i1, r1 = _implicit_midpoint_step(
        q, p, eps, ev, _fp("picard", max_iter=200, tol=1e-12, fallback_damping=()))
    assert torch.equal(q0, q1) and torch.equal(p0, p1)
    assert torch.equal(i0, i1) and torch.equal(r0, r1)
