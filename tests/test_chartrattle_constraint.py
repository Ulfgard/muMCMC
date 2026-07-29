"""Constraint / evaluate_model tests for ChartRATTLE.

The geometry and potential rest on the inverse map ψ(η) = φ_η⁻¹(x). evaluate_model
turns a constraint into the RMHMC-shaped pieces (U a TemperedAffine, G_M a
TemperedMetric). Here we check, against closed forms, that:

1. G_M = I + β Wᵀ W and its Cholesky are what evaluate_model returns.
2. The explicit-Jacobian fast path agrees with the classic-autograd W, and ∇V
   matches a finite difference.
3. The potential U.value is exactly −log p(η | x): the funnel posterior, the
   affine Gaussian, and the β-tempered funnel.
4. U.lik = ½‖ψ‖² is the temperature-free swap statistic.
5. The returned pieces are detached (no autograd graph pinned).
"""
import torch
import pytest

from muMCMC.ChartRATTLE import ChartRATTLE, ChartConstraint, LocationScaleChart
from muMCMC.spaces import UnconstrainedSpace

torch.set_default_dtype(torch.float64)


class FunnelChart(ChartConstraint):
    """Neal funnel x = e^{σ η / 2} ε with η ~ N(0, 1). ψ = e^{−σ η / 2} x,
    W = (σ/2) ψ, log|det B| = (m/2) σ η. Temperature-free."""

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
    """φ_η(ε) = c + A η + B ε, so ψ(η) = B⁻¹(x − c) − (B⁻¹A) η. W = B⁻¹A and
    log|det B| are constant, and the induced posterior is exactly Gaussian."""

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


def _eval(constraint, eta, beta=1.0, grad=True):
    """evaluate_model through a minimal sampler at the given temperature."""
    n = eta.shape[-1]
    s = ChartRATTLE(constraint, UnconstrainedSpace([f"v{i}" for i in range(n)]),
                    step_size=0.1, adapt_step_size=False)
    s.beta = beta
    return s.evaluate_model(eta, grad=grad)


def _funnel_pair(sigma=3.0, m=5, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m)
    ls_mean = lambda eta: torch.zeros(eta.shape[0], m, dtype=eta.dtype)
    ls_cov = lambda eta: torch.exp(sigma * eta[:, 0])[:, None, None] * torch.eye(m, dtype=eta.dtype)
    return FunnelChart(sigma, x), LocationScaleChart(ls_mean, ls_cov, x), x


# ========================================================================== #
#  1. Metric and its Cholesky                                               #
# ========================================================================== #

def test_metric_is_identity_plus_gram_and_cholesky_matches():
    c, _, _ = _funnel_pair()
    eta = torch.randn(6, 1)
    _, metric, _, W = _eval(c, eta, grad=False)
    n = eta.shape[-1]
    G = torch.eye(n) + W.transpose(-2, -1) @ W                # beta = 1
    assert torch.allclose(metric.value, G, atol=1e-10)
    assert torch.allclose(metric.L @ metric.L.transpose(-2, -1), G, atol=1e-10)


def test_metric_scales_with_beta():
    c, _, _ = _funnel_pair()
    eta = torch.randn(4, 1)
    _, m1, _, W = _eval(c, eta, beta=1.0, grad=False)
    _, mb, _, _ = _eval(c, eta, beta=0.3, grad=False)
    n = eta.shape[-1]
    assert torch.allclose(mb.value, torch.eye(n) + 0.3 * W.transpose(-2, -1) @ W, atol=1e-10)


# ========================================================================== #
#  2. Explicit Jacobians vs the autograd default                            #
# ========================================================================== #

def test_explicit_jacobian_matches_autograd_default():
    # Same funnel through explicit psi_with_jvp and through the classic-autograd
    # W (LocationScaleChart). W, U and ∇V must agree.
    fun, ls, _ = _funnel_pair()
    eta = torch.randn(8, 1)
    Uf, mf, _, Wf, gf = _eval(fun, eta)
    Ul, ml, _, Wl, gl = _eval(ls, eta)
    assert torch.allclose(Wf, Wl, atol=1e-9)
    assert torch.allclose(Uf.value, Ul.value, atol=1e-9)
    assert torch.allclose(gf, gl, atol=1e-8)


def test_grad_V_matches_finite_difference():
    torch.manual_seed(1)
    A = torch.randn(4, 2)
    B = torch.eye(4) + 0.1 * torch.randn(4, 4)
    B = B @ B.transpose(-2, -1)                       # SPD, so det > 0
    c = AffineChart(A, B, torch.zeros(4), torch.randn(4))
    eta = torch.randn(3, 2)
    _, _, _, _, gV = _eval(c, eta)

    def V(e):
        U, metric, _, _ = _eval(c, e, grad=False)
        return U.value + 0.5 * metric.log_det_metric()
    h = 1e-6
    gfd = torch.zeros_like(eta)
    for j in range(eta.shape[-1]):
        ep = eta.clone(); ep[:, j] += h
        em = eta.clone(); em[:, j] -= h
        gfd[:, j] = (V(ep) - V(em)) / (2 * h)
    assert torch.allclose(gV, gfd, atol=1e-6)


# ========================================================================== #
#  3. The potential is exactly −log p(η | x)                                #
# ========================================================================== #

def test_potential_matches_funnel_posterior():
    fun, _, x = _funnel_pair(sigma=3.0, m=6, seed=2)
    eta = torch.randn(50, 1)
    U, _, _, _ = _eval(fun, eta, grad=False)
    e = eta[:, 0]
    neg_log_post = 0.5 * e * e + 0.5 * torch.exp(-fun.s * e) * (x * x).sum() \
        + 0.5 * x.shape[-1] * fun.s * e
    assert torch.allclose(U.value - U.value.mean(),
                          neg_log_post - neg_log_post.mean(), atol=1e-9)


def test_potential_is_the_exact_affine_gaussian():
    torch.manual_seed(3)
    A = torch.randn(5, 2)
    B = torch.eye(5) + 0.2 * torch.randn(5, 5)
    B = B @ B.transpose(-2, -1)
    c = AffineChart(A, B, torch.zeros(5), torch.randn(5))
    mu, Sigma = c.posterior()
    prec = torch.linalg.inv(Sigma)

    eta = torch.randn(40, 2)
    U, _, _, _ = _eval(c, eta, grad=False)
    d = eta - mu
    neg_log_gauss = 0.5 * torch.einsum("ni,ij,nj->n", d, prec, d)
    assert torch.allclose(U.value - U.value.mean(),
                          neg_log_gauss - neg_log_gauss.mean(), atol=1e-9)


def test_beta_tempers_the_data_fit_only():
    # U.value = base + β·½‖ψ‖²: the Mahalanobis is scaled by β, the base (prior +
    # log|det B|) is not, matching the β-tempered funnel posterior.
    fun, _, x = _funnel_pair(sigma=3.0, m=6, seed=2)
    eta = torch.randn(30, 1)
    U, _, _, _ = _eval(fun, eta, beta=0.4, grad=False)
    e = eta[:, 0]
    neg_log_post = 0.5 * e * e + 0.5 * x.shape[-1] * fun.s * e \
        + 0.4 * 0.5 * torch.exp(-fun.s * e) * (x * x).sum()
    assert torch.allclose(U.value, neg_log_post, atol=1e-9)


# ========================================================================== #
#  4. U.lik: temperature-free swap statistic                                #
# ========================================================================== #

def test_u_lik_is_half_psi_squared_and_temperature_free():
    fun, _, _ = _funnel_pair(sigma=3.0, m=5, seed=6)
    eta = torch.randn(8, 1)
    U1, _, psi, _ = _eval(fun, eta, beta=1.0, grad=False)
    Ub, _, _, _ = _eval(fun, eta, beta=0.2, grad=False)
    assert torch.allclose(U1.lik, 0.5 * (psi ** 2).sum(-1), atol=1e-10)
    assert torch.allclose(U1.lik, Ub.lik, atol=1e-10)     # β-free


def test_location_scale_log_abs_det_B_is_half_logdet_sigma():
    _, ls, _ = _funnel_pair(sigma=2.0, m=4, seed=4)
    eta = torch.randn(6, 1)
    ldB = ls.log_abs_det_B(eta)
    Sigma = ls.cov(eta)
    assert torch.allclose(ldB, 0.5 * torch.logdet(Sigma), atol=1e-10)


# ========================================================================== #
#  5. Detached                                                              #
# ========================================================================== #

def test_evaluate_model_output_is_detached():
    _, ls, _ = _funnel_pair()
    eta = torch.randn(4, 1)
    U, metric, psi, W, gV = _eval(ls, eta)
    for t in (U.value, U.lik, metric.value, metric.L, psi, W, gV):
        assert not t.requires_grad
