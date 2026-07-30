"""Sampler-level tests for ChartRATTLE: the operator interface around the
integrator, plus statistical recovery on targets with a closed form.

Adaptation is out of scope here (finicky against the solver-convergence cliff),
so every sampler runs at a fixed step. The integrator internals live in
test_chartrattle_solver.py. Recovery is checked against the exact Gaussian
induced by an affine map and against a quadrature reference for the funnel;
parallel tempering rides on the TemperedAffine / TemperedMetric that
evaluate_model returns.
"""
import math

import torch
import pytest

from muMCMC.ChartRATTLE import ChartRATTLE, ChartRATTLEState, ChartConstraint
from muMCMC.spaces import UnconstrainedSpace
from muMCMC.PT import PT

torch.set_default_dtype(torch.float64)


def _eye(n):
    """Identity prior metric. ChartRATTLE requires the space to supply M, and
    these targets all want the plain identity."""
    return lambda th: torch.eye(n).expand(th.shape[0], n, n)


def _N01():
    """Explicit standard-normal prior, the latent of the plain non-centered
    parameterization. ChartRATTLE reads the prior off the space, so it is named
    rather than assumed."""
    return torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))


class FunnelChart(ChartConstraint):
    def __init__(self, sigma, x):
        super().__init__(x)
        self.s = sigma

    def psi(self, eta):
        return torch.exp(-self.s * eta[:, 0] / 2)[:, None] * self.x

    def log_abs_det_B(self, eta):
        return 0.5 * self.x.shape[-1] * self.s * eta[:, 0]

    def psi_with_jvp(self, eta):
        eps = self.psi(eta)
        return eps, (self.s / 2.0) * eps.unsqueeze(-1), self.log_abs_det_B(eta)


class AffineChart(ChartConstraint):
    def __init__(self, A, B, c, x):
        super().__init__(x)
        Binv = torch.linalg.inv(B)
        self.W_const = Binv @ A
        self.d = Binv @ (x - c)
        self.ldB = torch.linalg.slogdet(B).logabsdet

    def psi(self, eta):
        return self.d - eta @ self.W_const.transpose(-2, -1)

    def log_abs_det_B(self, eta):
        return self.ldB.expand(eta.shape[0])

    def psi_with_jvp(self, eta):
        N = eta.shape[0]
        return self.psi(eta), self.W_const.expand(N, *self.W_const.shape), self.log_abs_det_B(eta)

    def posterior(self):
        n = self.W_const.shape[-1]
        G = torch.eye(n) + self.W_const.transpose(-2, -1) @ self.W_const
        Sigma = torch.linalg.inv(G)
        mu = Sigma @ (self.W_const.transpose(-2, -1) @ self.d)
        return mu, Sigma


def _funnel_sampler(sigma=2.0, m=4, seed=0, **kw):
    torch.manual_seed(seed)
    c = FunnelChart(sigma, torch.randn(m))
    kw.setdefault("adapt_step_size", False)
    kw.setdefault("step_size", 0.1)
    return ChartRATTLE(c, UnconstrainedSpace(["v"], priors={"v": _N01()},
                                          prior_metric_fn=_eye(1)), **kw)


def _endpoint_state(sampler, eta):
    # init() seeds the diagnostic accumulators and builds the endpoint bundle.
    return sampler.sample_momentum(sampler.init(eta))


def _restart(st, q, p):
    return ChartRATTLEState(q, p, st.U, st.metric, st.psi, st.W, st.grad_V, None)


# ========================================================================== #
#  init                                                                      #
# ========================================================================== #

def test_init_sizes_step_size_and_resets_counters():
    s = _funnel_sampler(step_size=0.25, num_steps=4)
    state = s.init(torch.zeros(5, 1))
    assert s.step_size.shape == (5,)
    assert torch.allclose(s.step_size, torch.full((5,), 0.25))
    assert s._step == 0
    assert torch.equal(s._accepted, torch.zeros(5, dtype=torch.long))
    assert state.U is not None and state.metric is not None
    assert state.p is None                        # momentum drawn in step()
    assert torch.allclose(s._residual_sum, torch.zeros(5))


# ========================================================================== #
#  sample_momentum: draws from the chart metric                             #
# ========================================================================== #

def test_sample_momentum_covariance_is_the_chart_metric():
    # p ~ N(0, G_M(q)): the empirical covariance over many draws matches G_M.
    torch.manual_seed(0)
    A = torch.randn(4, 2)
    B = torch.eye(4) + 0.2 * torch.randn(4, 4)
    B = B @ B.transpose(-2, -1)
    c = AffineChart(A, B, torch.zeros(4), torch.randn(4))
    s = ChartRATTLE(c, UnconstrainedSpace(["a", "b"],
                                          priors={"a": _N01(), "b": _N01()},
                                          prior_metric_fn=_eye(2)),
                    step_size=0.1, adapt_step_size=False)
    N = 40000
    state = s.sample_momentum(s.init(torch.zeros(N, 2)))
    G = state.metric.value[0]
    emp = (state.p.transpose(0, 1) @ state.p) / N
    assert torch.allclose(emp, G, rtol=0.03, atol=0.02)   # Monte Carlo covariance


# ========================================================================== #
#  accept: failed solve rejected                                            #
# ========================================================================== #

def test_failed_solve_is_rejected_even_when_energy_matches():
    s = _funnel_sampler(step_size=0.1, num_steps=1, fp_tol=1e-8)
    old = _endpoint_state(s, torch.zeros(3, 1))
    new = _restart(old, old.q.clone(), old.p.clone())
    s._step_residual = torch.full((3,), 1e-3)     # solve failed (>> fp_tol)
    out = s.accept(new, old)
    assert torch.allclose(out.q, old.q)           # rejected despite delta_H ~ 0
    assert torch.equal(s._num_divergences, torch.ones(3, dtype=torch.long))


# ========================================================================== #
#  reorder                                                                   #
# ========================================================================== #

def test_reorder_relabels_position_and_drops_the_model():
    # A swap moves the configuration q (and its momentum p) to a new temperature
    # slot. The model is re-evaluated there at the next step, so it is dropped.
    s = _funnel_sampler(num_steps=2)
    state = s.sample_momentum(s.init(torch.randn(3, 1)))
    perm = torch.tensor([2, 0, 1])
    r = state.reorder(perm)
    assert torch.equal(r.q, state.q[perm])
    assert torch.equal(r.p, state.p[perm])
    assert r.U is None and r.metric is None and r.W is None and r.psi is None
    assert r.grad_V is None


# ========================================================================== #
#  step: composes num_steps substeps + accept                              #
# ========================================================================== #

def test_step_runs_exactly_num_steps_substeps():
    s = _funnel_sampler(num_steps=4)
    state = s.init(torch.zeros(2, 1))
    calls = {"n": 0}
    original = s.integrate

    def counting(x, step_size):
        calls["n"] += 1
        return original(x, step_size)

    s.integrate = counting
    s.step(state)
    assert calls["n"] == 4


def test_step_keeps_only_the_fields_that_outlive_a_transition():
    # q and p carry the chain and U.lik is the PT swap statistic, so those three
    # survive. The metric, geometry and force are trajectory scratch rebuilt by
    # sample_momentum, so carrying them would only risk them going stale behind a
    # PT relabeling.
    s = _funnel_sampler(num_steps=3)
    out = s.step(s.init(torch.zeros(2, 1)))
    assert out.q.shape == (2, 1)
    assert out.p is not None and out.U is not None
    assert out.metric is None and out.psi is None and out.W is None
    assert out.grad_V is None and out.dq is None


# ========================================================================== #
#  constructor validation                                                   #
# ========================================================================== #

def test_invalid_solver_raises():
    with pytest.raises(ValueError, match="unknown solver"):
        _funnel_sampler(solver="secant")


def test_invalid_anderson_history_raises():
    with pytest.raises(ValueError, match="anderson_history"):
        _funnel_sampler(solver="anderson", anderson_history=0)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_damping_raises(bad):
    with pytest.raises(ValueError, match="damping"):
        _funnel_sampler(damping=bad)


# ========================================================================== #
#  statistical recovery                                                      #
# ========================================================================== #

def test_recovers_affine_gaussian_posterior():
    # An affine map induces an exact Gaussian q-marginal. The chain must match
    # its mean and covariance.
    torch.manual_seed(1)
    A = torch.randn(5, 2)
    B = torch.eye(5) + 0.2 * torch.randn(5, 5)
    B = B @ B.transpose(-2, -1)
    c = AffineChart(A, B, torch.zeros(5), torch.randn(5))
    mu, Sigma = c.posterior()

    s = ChartRATTLE(c, UnconstrainedSpace(["a", "b"],
                                          priors={"a": _N01(), "b": _N01()},
                                          prior_metric_fn=_eye(2)),
                    step_size=0.4, num_steps=12,
                    adapt_step_size=False, solver="anderson", fp_tol=1e-10)
    out = s.run_mcmc(torch.zeros(2), num_samples=500, num_warmup_steps=200,
                     num_chains=64, disable_progbar=True)
    draws = torch.stack([out["a"], out["b"]], dim=-1).reshape(-1, 2)
    assert torch.allclose(draws.mean(0), mu, atol=0.03)
    assert torch.allclose(torch.cov(draws.T), Sigma, atol=0.05)
    assert int(s.diagnostics()["num_divergences"].sum()) == 0


def test_recovers_affine_gaussian_posterior_under_a_space_prior():
    # Same affine chart, but the space carries a Normal(m0, s0) prior instead of
    # the standard-normal latent. The exact posterior precision becomes
    # S0^-1 + W^T W, so a chain that ignored the prior would miss both moments.
    torch.manual_seed(1)
    A = torch.randn(5, 2)
    B = torch.eye(5) + 0.2 * torch.randn(5, 5)
    B = B @ B.transpose(-2, -1)
    c = AffineChart(A, B, torch.zeros(5), torch.randn(5))

    m0 = torch.tensor([0.8, -0.5])
    s0 = torch.tensor([1.5, 0.6])
    priors = {n: torch.distributions.Normal(m0[i], s0[i])
              for i, n in enumerate(["a", "b"])}

    W = c.W_const
    S0_inv = torch.diag(1.0 / s0 ** 2)
    Lam = S0_inv + W.transpose(-2, -1) @ W
    Sigma = torch.linalg.inv(Lam)
    mu = Sigma @ (S0_inv @ m0 + W.transpose(-2, -1) @ c.d)

    # The prior really does move the target, so the test has teeth.
    mu_flat, _ = c.posterior()
    assert float((mu - mu_flat).abs().max()) > 0.1

    s = ChartRATTLE(c, UnconstrainedSpace(["a", "b"], priors=priors,
                                          prior_metric_fn=_eye(2)),
                    step_size=0.4, num_steps=12, adapt_step_size=False,
                    solver="anderson", fp_tol=1e-10)
    out = s.run_mcmc(torch.zeros(2), num_samples=500, num_warmup_steps=200,
                     num_chains=64, disable_progbar=True)
    draws = torch.stack([out["a"], out["b"]], dim=-1).reshape(-1, 2)
    assert torch.allclose(draws.mean(0), mu, atol=0.03)
    assert torch.allclose(torch.cov(draws.T), Sigma, atol=0.05)
    assert int(s.diagnostics()["num_divergences"].sum()) == 0


def test_prior_metric_moves_the_metric_and_not_the_potential():
    # M belongs to G_M alone. U is the target and must not notice it, or the
    # sampler would be answering a different question per preconditioner.
    torch.manual_seed(5)
    A = torch.randn(4, 2)
    B = torch.eye(4) + 0.2 * torch.randn(4, 4)
    B = B @ B.transpose(-2, -1)
    c = AffineChart(A, B, torch.zeros(4), torch.randn(4))
    names = ["a", "b"]
    priors = {n: _N01() for n in names}
    M = torch.diag(torch.tensor([3.0, 0.5]))

    def build(metric_fn):
        space = UnconstrainedSpace(names, priors=priors,
                                   prior_metric_fn=metric_fn)
        return ChartRATTLE(c, space, step_size=0.2, adapt_step_size=False)

    eta = torch.randn(6, 2)
    U_i, m_i, _, W = build(_eye(2)).evaluate_model(eta, grad=False)
    U_m, m_m, _, _ = build(
        lambda th: M.expand(th.shape[0], 2, 2)).evaluate_model(eta, grad=False)

    assert torch.allclose(U_m.value, U_i.value, atol=1e-12)      # target unmoved
    gram = W.transpose(-2, -1) @ W
    assert torch.allclose(m_i.value, torch.eye(2) + gram, atol=1e-10)
    assert torch.allclose(m_m.value, M + gram, atol=1e-10)


def test_prior_metric_matched_to_a_wide_prior_mixes():
    # A weak likelihood with a prior N(0, s^2) leaves a posterior about s wide.
    # With M = I the momentum is drawn at unit scale against it, so one chain
    # barely moves. M = s^-2 I is the scale the prior actually has.
    torch.manual_seed(0)
    n, m, S = 2, 4, 1.0e4
    A = 1e-3 * torch.randn(m, n)                       # weak coupling q -> x
    B = torch.eye(m) + 0.2 * torch.randn(m, m)
    B = B @ B.transpose(-2, -1)
    c = AffineChart(A, B, torch.zeros(m), torch.randn(m))
    names = ["a", "b"]
    wide = torch.distributions.Normal(torch.tensor(0.0),
                                      torch.tensor(math.sqrt(S)))
    priors = {k: wide for k in names}

    W = c.W_const
    sd_exact = torch.linalg.inv(
        torch.eye(n) / S + W.transpose(-2, -1) @ W).diagonal().sqrt()

    def per_chain_spread(metric_fn):
        space = UnconstrainedSpace(names, priors=priors,
                                   prior_metric_fn=metric_fn)
        s = ChartRATTLE(c, space, step_size=0.15, num_steps=10,
                        adapt_step_size=False, solver="anderson", fp_tol=1e-10)
        out = s.run_mcmc(torch.zeros(n), num_samples=400, num_warmup_steps=200,
                         num_chains=16, disable_progbar=True)
        # One chain, so this measures mixing rather than the spread of the
        # independent starts.
        sd = torch.stack([out[k][0] for k in names], dim=-1).std(0)
        return float((sd / sd_exact).mean())

    unmatched = per_chain_spread(_eye(n))
    matched = per_chain_spread(
        lambda th: torch.eye(n).expand(th.shape[0], n, n) / S)
    assert unmatched < 0.5           # M = I explores a fraction of the width
    assert matched > 0.8             # matched M gets most of the way across


def test_newton_recovers_the_affine_gaussian_posterior():
    # End to end on the newton rule, whose residual returns the Jacobian too.
    torch.manual_seed(1)
    A = torch.randn(5, 2)
    B = torch.eye(5) + 0.2 * torch.randn(5, 5)
    B = B @ B.transpose(-2, -1)
    c = AffineChart(A, B, torch.zeros(5), torch.randn(5))
    mu, Sigma = c.posterior()

    s = ChartRATTLE(c, UnconstrainedSpace(["a", "b"],
                                          priors={"a": _N01(), "b": _N01()},
                                          prior_metric_fn=_eye(2)),
                    step_size=0.4, num_steps=12, adapt_step_size=False,
                    solver="newton", fp_tol=1e-10)
    out = s.run_mcmc(torch.zeros(2), num_samples=500, num_warmup_steps=200,
                     num_chains=64, disable_progbar=True)
    draws = torch.stack([out["a"], out["b"]], dim=-1).reshape(-1, 2)
    assert torch.allclose(draws.mean(0), mu, atol=0.03)
    assert torch.allclose(torch.cov(draws.T), Sigma, atol=0.05)
    assert int(s.diagnostics()["num_divergences"].sum()) == 0


def test_recovers_funnel_posterior_against_quadrature():
    # Scalar q, so the exact posterior is 1-D and available by quadrature.
    sigma, m = 3.0, 6
    torch.manual_seed(7)
    eta_true = torch.randn(1)
    xobs = torch.exp(sigma * eta_true / 2) * torch.randn(m)
    c = FunnelChart(sigma, xobs)

    grid = torch.linspace(-8, 8, 8001)
    log_post = -(0.5 * grid * grid + 0.5 * torch.exp(-sigma * grid) * (xobs * xobs).sum()
                 + 0.5 * m * sigma * grid)
    w = torch.softmax(log_post, dim=0)
    mean_q = (w * grid).sum()
    sd_q = (w * (grid - mean_q) ** 2).sum().sqrt()

    s = ChartRATTLE(c, UnconstrainedSpace(["v"], priors={"v": _N01()},
                                          prior_metric_fn=_eye(1)),
                    step_size=0.08, num_steps=16,
                    adapt_step_size=False, solver="anderson", fp_tol=1e-9)
    out = s.run_mcmc(torch.zeros(1), num_samples=400, num_warmup_steps=200,
                     num_chains=64, disable_progbar=True)
    v = out["v"].reshape(-1)
    assert abs(float(v.mean()) - float(mean_q)) < 0.03
    assert abs(float(v.std()) - float(sd_q)) < 0.03
    assert int(s.diagnostics()["num_divergences"].sum()) == 0


# ========================================================================== #
#  parallel tempering                                                        #
# ========================================================================== #

def test_pt_runs_swaps_and_recovers_target_mean():
    # ChartRATTLE as a PT exploration kernel: evaluate_model returns U / G_M as a
    # TemperedAffine / TemperedMetric, so the swap statistic U.lik and the
    # retempering reorder ride on the base machinery. The target chain (β = 1)
    # recovers the funnel posterior mean.
    sigma, m = 3.0, 6
    torch.manual_seed(7)
    eta_true = torch.randn(1)
    xobs = torch.exp(sigma * eta_true / 2) * torch.randn(m)
    grid = torch.linspace(-8, 8, 8001)
    log_post = -(0.5 * grid * grid + 0.5 * torch.exp(-sigma * grid) * (xobs * xobs).sum()
                 + 0.5 * m * sigma * grid)
    mean_q = float((torch.softmax(log_post, 0) * grid).sum())

    kernel = ChartRATTLE(FunnelChart(sigma, xobs),
                         UnconstrainedSpace(["v"], priors={"v": _N01()},
                                            prior_metric_fn=_eye(1)),
                         step_size=0.06, num_steps=12, adapt_step_size=False,
                         solver="anderson", fp_tol=1e-9)
    # β > 0 throughout: β = 0 sends Σ/β -> ∞, outside the scale family.
    pt = PT(kernel, betas=torch.tensor([0.1, 0.3, 0.55, 1.0]))
    state = pt.init(torch.zeros(24, 1))
    for _ in range(400):
        state = pt.step(state)

    diag = pt.diagnostics()
    assert diag["swap_accept_rate"].shape == (3,)
    assert float(diag["swap_accept_rate"].min()) > 0.1   # every pair communicates
    assert abs(float(state.q.reshape(-1).mean()) - mean_q) < 0.05
