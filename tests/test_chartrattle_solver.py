"""Solver and integrator tests for ChartRATTLE, independent of the sampler.

The RATTLE step is exercised through :meth:`ChartRATTLE.integrate` on crafted
states, checking the properties that make it a valid MCMC proposal:

1. The position solve drives the orthogonality residual F(q1) to tolerance.
2. On an affine map (constant metric, linear F) the frozen-Jacobian
   preconditioner is the exact Jacobian, so the solve converges in one step.
3. The step is time-reversible: forward, flip p, forward returns to the start,
   and it is symplectic, hence volume-preserving on (q, p).
4. Anderson and Picard reach the same endpoint.
5. Per-chain convergence is independent, and a failed solve is flagged.
6. Energy is conserved to second order: halving the step at a fixed trajectory
   length quarters the energy error.
"""
import math

import torch
import pytest

from muMCMC.ChartRATTLE import (
    ChartRATTLE, ChartRATTLEState)
from muMCMC.RMHMC import _hamiltonian
from muMCMC.spaces import ConditionalLayer, LocationScaleLayer, NormalSpace

torch.set_default_dtype(torch.float64)


class FunnelLayer(ConditionalLayer):
    """Neal funnel x = e^{{σ θ / 2}} ε. phi(eps) = e^{{σ θ / 2}} eps, so the latent
    Jacobian is e^{{σ θ / 2}} I and W = (σ/2) ψ, both in closed form."""

    jvp_needs_grad = False

    def __init__(self, sigma, m):
        self.s = sigma
        self.m = m

    def _log_scale(self, theta):
        return (self.s * theta[:, 0] / 2)[:, None].expand(-1, self.m)

    def forward(self, theta, eps):
        return torch.exp(self._log_scale(theta)) * eps

    def inverse(self, theta, y):
        return torch.exp(-self._log_scale(theta)) * y

    def forward_with_jvp(self, theta, eps):
        log_diag = self._log_scale(theta)
        y = torch.exp(log_diag) * eps
        return y, (self.s / 2.0) * y.unsqueeze(-1), torch.diag_embed(torch.exp(log_diag))

    def inverse_with_jvp(self, theta, y):
        log_diag = self._log_scale(theta)
        eps = torch.exp(-log_diag) * y
        return (eps, (self.s / 2.0) * eps.unsqueeze(-1),
                torch.diag_embed(torch.exp(-log_diag)))


class AffineLayer(ConditionalLayer):
    """phi_theta(eps) = c + A theta + B eps with B lower triangular, so
    ψ(theta) = B⁻¹(x − c) − (B⁻¹A) theta. W = B⁻¹A and the latent Jacobians are
    constant, and the induced posterior is exactly Gaussian."""

    jvp_needs_grad = False

    def __init__(self, A, B, c):
        self.A, self.B, self.c = A, B, c
        self.Binv = torch.linalg.inv(B)
        self.W_const = self.Binv @ A

    def forward(self, theta, eps):
        return self.c + theta @ self.A.mT + eps @ self.B.mT

    def inverse(self, theta, y):
        return (y - self.c) @ self.Binv.mT - theta @ self.W_const.mT

    def forward_with_jvp(self, theta, eps):
        N = theta.shape[0]
        return (self.forward(theta, eps), self.A.expand(N, *self.A.shape),
                self.B.expand(N, *self.B.shape))

    def inverse_with_jvp(self, theta, y):
        N = theta.shape[0]
        return (self.inverse(theta, y),
                self.W_const.expand(N, *self.W_const.shape),
                self.Binv.expand(N, *self.Binv.shape))

    def posterior(self, x):
        """Mean and covariance of p(theta | x) for this layer at ``x``."""
        d = self.Binv @ (x - self.c)
        n = self.W_const.shape[-1]
        G = torch.eye(n) + self.W_const.mT @ self.W_const
        Sigma = torch.linalg.inv(G)
        return Sigma @ (self.W_const.mT @ d), Sigma

def _funnel_sampler(sigma=2.0, m=4, seed=0, prior_sd=None, **kw):
    torch.manual_seed(seed)
    c, x = FunnelLayer(sigma, m), torch.randn(m)
    space = NormalSpace(["v"], sigma=prior_sd or 1.0)
    return ChartRATTLE(c, x, space, adapt_step_size=False, **kw)


def _affine_sampler(n=2, m=5, seed=0, **kw):
    torch.manual_seed(seed)
    A = torch.randn(m, n)
    B = torch.eye(m) + 0.15 * torch.randn(m, m)
    B = B @ B.transpose(-2, -1)
    c, x = AffineLayer(A, B, torch.zeros(m)), torch.randn(m)
    space = NormalSpace([f"v{i}" for i in range(n)])
    return ChartRATTLE(c, x, space, adapt_step_size=False, **kw), n


def _seed_state(sampler, eta):
    """A momentum-carrying start state at ``eta`` (as sample_momentum builds it)."""
    U, metric, psi, W, grad_V = sampler.evaluate_model(eta, grad=True)
    st = ChartRATTLEState(eta, None, U, metric, psi, W, grad_V, None)
    return sampler.sample_momentum(st)


def _restart(st, q, p):
    """A fresh trajectory state at (q, p) reusing st's endpoint bundle."""
    return ChartRATTLEState(q, p, st.U, st.metric, st.psi, st.W, st.grad_V, None)


def _H(st):
    return _hamiltonian(st.q, st.p, st.U.value, st.metric)


# ========================================================================== #
#  1. Position solve: orthogonality residual                                #
# ========================================================================== #

def test_position_solve_reaches_orthogonality_tolerance():
    # Recompute the RATTLE position residual at the endpoint from the constraint
    # itself, not from the solver's own stopping value:
    #   F(q1) = (q1 − q0) − β W0ᵀ(ψ(q1) − ψ0) − hp0 + (h²/2)∇V(q0).
    # At convergence F(q1) is at tolerance, so q1 sits on the RATTLE update.
    h = 0.2
    s = _funnel_sampler(step_size=h, num_steps=1, fp_tol=1e-12, fp_max_iter=200)
    torch.manual_seed(1)
    st = _seed_state(s, torch.randn(6, 1))
    out = s.integrate(_restart(st, st.q.clone(), st.p.clone()), torch.full((6,), h))

    psi1 = s._chart.psi(out.q)
    corr = (st.W.transpose(-2, -1) @ (psi1 - st.psi).unsqueeze(-1)).squeeze(-1)
    F = (out.q - st.q) - s.beta * corr - h * st.p + 0.5 * h * h * st.grad_V
    assert float(F.abs().max()) < 1e-11


# ========================================================================== #
#  2. Affine map: one-step convergence, exact Jacobian preconditioner       #
# ========================================================================== #

def test_affine_solve_converges_in_one_iteration():
    # F is affine with Jacobian G_M(q0), which is exactly the preconditioner, so
    # the frozen-Jacobian step is a Newton step and lands in a single iteration.
    s, n = _affine_sampler(step_size=0.5, num_steps=1, fp_tol=1e-12, fp_max_iter=50)
    torch.manual_seed(2)
    st = _seed_state(s, torch.randn(5, n))
    s.integrate(st, torch.full((5,), 0.5))
    assert int(s._step_iters.max()) == 1
    assert float(s._step_residual.max()) < 1e-11


# ========================================================================== #
#  3. Time reversibility                                                     #
# ========================================================================== #

@pytest.mark.parametrize("h", [0.1, 0.3, 0.5])
def test_step_is_time_reversible(h):
    s = _funnel_sampler(step_size=h, num_steps=1, fp_tol=1e-12, fp_max_iter=300)
    torch.manual_seed(3)
    st = _seed_state(s, torch.randn(5, 1))
    hh = torch.full((5,), h)

    fwd = s.integrate(_restart(st, st.q.clone(), st.p.clone()), hh)
    back = s.integrate(_restart(fwd, fwd.q.clone(), -fwd.p.clone()), hh)
    assert torch.allclose(back.q, st.q, atol=1e-9)
    assert torch.allclose(back.p, -st.p, atol=1e-9)


@pytest.mark.parametrize("prior_sd", [None, [0.5, 2.0]])
def test_step_preserves_volume_and_symplectic_form(prior_sd):
    # The step is the variational integrator of the discrete Lagrangian
    #   S_h(q0,q1) = (q1-q0)^T M (q1-q0)/(2h) + beta||psi1-psi0||^2/(2h)
    #                - (h/2)[V0+V1],
    # so it is symplectic: J^T Omega J = Omega, hence det J = 1. That volume
    # property is the half of Metropolis exactness reversibility does not give,
    # so it is worth pinning independently. Running it with an anisotropic M too
    # checks that M reached every place it belongs, since a mismatch between the
    # metric and the kinetic term would break the identity. Finite differences,
    # since the step detaches and autograd cannot see through it.
    n, m, h = 2, 3, 0.3
    torch.manual_seed(11)
    A = 0.7 * torch.randn(m, n)
    B = torch.eye(m) + 0.3 * torch.diag(torch.tensor([1.0, -1.0, 0.5]))
    Sigma = B @ B.transpose(-2, -1)
    # Nonlinear in q through both the mean and the scale, so a step that was only
    # accidentally symplectic (e.g. on a constant metric) would show up here.
    chart = LocationScaleLayer.from_covariance(
        lambda q: torch.tanh(q @ A.transpose(-2, -1)),
        lambda q: torch.exp(0.6 * q[:, 0])[:, None, None] * Sigma)
    x = torch.randn(m)
    space = NormalSpace([f"v{i}" for i in range(n)], sigma=prior_sd or 1.0)
    s = ChartRATTLE(chart, x, space, step_size=h, num_steps=1,
                    adapt_step_size=False, solver="anderson",
                    fp_tol=1e-14, fp_max_iter=500)
    s.init(torch.zeros(1, n))                     # seeds the solver diagnostics

    def step_map(z):
        st = _seed_state(s, z[:n].reshape(1, n))
        out = s.integrate(_restart(st, st.q, z[n:].reshape(1, n)), torch.full((1,), h))
        return torch.cat([out.q.reshape(-1), out.p.reshape(-1)])

    z0 = torch.cat([torch.randn(n), torch.randn(n)])
    d = 1e-6
    J = torch.stack([(step_map(z0 + d * e) - step_map(z0 - d * e)) / (2 * d)
                     for e in torch.eye(2 * n)], dim=1)

    Omega = torch.zeros(2 * n, 2 * n)
    Omega[:n, n:] = torch.eye(n)
    Omega[n:, :n] = -torch.eye(n)

    assert abs(float(torch.det(J)) - 1.0) < 1e-6                        # volume
    assert float((J.T @ Omega @ J - Omega).abs().max()) < 1e-6          # symplectic


def test_affine_step_is_time_reversible():
    s, n = _affine_sampler(step_size=0.4, num_steps=1, fp_tol=1e-12)
    torch.manual_seed(4)
    st = _seed_state(s, torch.randn(4, n))
    hh = torch.full((4,), 0.4)
    fwd = s.integrate(_restart(st, st.q.clone(), st.p.clone()), hh)
    back = s.integrate(_restart(fwd, fwd.q.clone(), -fwd.p.clone()), hh)
    assert torch.allclose(back.q, st.q, atol=1e-9)
    assert torch.allclose(back.p, -st.p, atol=1e-9)


# ========================================================================== #
#  4. Anderson and Picard reach the same endpoint                           #
# ========================================================================== #

@pytest.mark.parametrize("solver", ["anderson", "newton"])
def test_solvers_match_the_picard_endpoint(solver):
    # All three rules solve the same position equation, so the endpoint is the
    # solver's business only through the iteration count.
    def run(kind):
        s = _funnel_sampler(step_size=0.3, num_steps=1, solver=kind,
                            fp_tol=1e-12, fp_max_iter=300)
        torch.manual_seed(5)
        st = _seed_state(s, torch.randn(6, 1))
        out = s.integrate(st, torch.full((6,), 0.3))
        return out.q, out.p
    qp, pp = run("picard")
    qs, ps = run(solver)
    assert torch.allclose(qs, qp, atol=1e-9)
    assert torch.allclose(ps, pp, atol=1e-9)


@pytest.mark.parametrize("solver", ["picard", "anderson", "newton"])
def test_a_prior_metric_keeps_the_step_reversible(solver):
    # The prior block M of the discrete Lagrangian measures the q block, so F,
    # the momentum line and G_M all carry it. If they had not moved together the
    # step would stop being self-adjoint. M is the identity in the chart, so a
    # prior of another width is what varies it in the chain's own coordinates.
    h = 0.3
    s = _funnel_sampler(step_size=h, num_steps=1, solver=solver, fp_tol=1e-12,
                        fp_max_iter=300, prior_sd=1.0 / math.sqrt(7.0))
    torch.manual_seed(31)
    st = _seed_state(s, torch.randn(5, 1))
    hh = torch.full((5,), h)
    fwd = s.integrate(_restart(st, st.q.clone(), st.p.clone()), hh)
    back = s.integrate(_restart(fwd, fwd.q.clone(), -fwd.p.clone()), hh)
    assert torch.allclose(back.q, st.q, atol=1e-9)
    assert torch.allclose(back.p, -st.p, atol=1e-9)


# ========================================================================== #
#  5. Per-chain independence and failed-solve flag                          #
# ========================================================================== #

def test_per_chain_convergence_is_independent():
    # One easy chain and one at a step so large the solve cannot converge in the
    # iteration budget: the easy chain still converges, the hard one is flagged.
    s = _funnel_sampler(sigma=3.0, step_size=0.1, num_steps=1, solver="picard",
                        fp_tol=1e-10, fp_max_iter=15)
    torch.manual_seed(6)
    eta = torch.tensor([[0.0], [0.0]])
    st = _seed_state(s, eta)
    # Force a huge step on chain 1 only.
    hh = torch.tensor([0.1, 6.0])
    s.integrate(st, hh)
    assert float(s._step_residual[0]) < 1e-9
    assert not (s._step_residual[1] <= s._fp_tol)     # chain 1 failed (or non-finite)


# ========================================================================== #
#  6. Second-order energy conservation                                      #
# ========================================================================== #

def test_energy_error_is_second_order_in_step():
    # Fixed trajectory length, halving the step: a second-order integrator
    # quarters the energy error. Check the ratio is near 4 over two halvings.
    def dH_at(step, num_steps):
        s = _funnel_sampler(sigma=2.0, m=5, step_size=step, num_steps=num_steps,
                            solver="picard", fp_tol=1e-12, fp_max_iter=300)
        torch.manual_seed(7)
        st = _seed_state(s, torch.zeros(64, 1))
        H0 = _H(st)
        hh = torch.full((64,), step)
        prop = st
        for _ in range(num_steps):
            prop = s.integrate(prop, hh)
        return (_H(prop) - H0).abs().median()

    coarse = dH_at(0.08, 10)
    mid = dH_at(0.04, 20)
    fine = dH_at(0.02, 40)
    assert 3.0 < float(coarse / mid) < 5.0
    assert 3.0 < float(mid / fine) < 5.0
