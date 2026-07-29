"""Constraint-layer tests for ChartRATTLE.

The chart geometry and potential rest on the inverse map ψ(η) = φ_η⁻¹(x). Here
we check, against closed forms, that:

1. The metric G_M = I + Wᵀ W and its Cholesky are what the endpoint returns.
2. The explicit-Jacobian fast path (psi_with_jvp overridden) agrees with the
   classic-autograd default (W by reverse pass, ∇V by autograd.grad).
3. ∇V matches an independent finite difference of V.
4. The chart target reproduces the closed-form posterior: for the funnel,
   V − ½ log det G_M equals −log p(η | x) up to a constant, and for an affine
   map the induced posterior is the exact Gaussian.
5. The endpoint bundle is detached (no autograd graph pinned).
"""
import torch
import pytest

from muMCMC.ChartRATTLE import ChartConstraint, LocationScaleChart

torch.set_default_dtype(torch.float64)


# ---- explicit-Jacobian test constraints ---------------------------------- #

class FunnelChart(ChartConstraint):
    """Neal funnel x = e^{σ η / 2} ε with η ~ N(0, 1). Closed-form geometry:
    ψ = e^{−σ η / 2} x, W = (σ/2) ψ, log|det B| = (m/2) σ η."""

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
        self.W_const = Binv @ A                      # (m, n)
        self.d = Binv @ (x - c)                      # (m,)
        self.ldB = torch.linalg.slogdet(B).logabsdet

    def psi(self, eta):
        return self.d - eta @ self.W_const.transpose(-2, -1)   # (N, m)

    def log_abs_det_B(self, eta):
        return self.ldB.expand(eta.shape[0])

    def psi_with_jvp(self, eta):
        N = eta.shape[0]
        return self.psi(eta), self.W_const.expand(N, *self.W_const.shape), self.log_abs_det_B(eta)

    def posterior(self):
        """Exact N(μ, Σ) of the η-marginal: Σ = (I + WᵀW)⁻¹, μ = Σ Wᵀ d."""
        n = self.W_const.shape[-1]
        G = torch.eye(n) + self.W_const.transpose(-2, -1) @ self.W_const
        Sigma = torch.linalg.inv(G)
        mu = Sigma @ (self.W_const.transpose(-2, -1) @ self.d)
        return mu, Sigma


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
    local, _ = c.endpoint(eta)
    n = eta.shape[-1]
    G = torch.eye(n) + local.W.transpose(-2, -1) @ local.W
    assert torch.allclose(local.chol_G @ local.chol_G.transpose(-2, -1), G, atol=1e-10)


# ========================================================================== #
#  2. Explicit Jacobians vs the autograd default                            #
# ========================================================================== #

def test_explicit_jacobian_matches_autograd_default():
    # Same funnel through the explicit psi_with_jvp and through the
    # classic-autograd W (LocationScaleChart). W, V and ∇V must agree.
    fun, ls, _ = _funnel_pair()
    eta = torch.randn(8, 1)
    lf, gf = fun.endpoint(eta)
    ll, gl = ls.endpoint(eta)
    assert torch.allclose(lf.W, ll.W, atol=1e-9)
    assert torch.allclose(lf.V, ll.V, atol=1e-9)
    assert torch.allclose(gf, gl, atol=1e-8)


def test_autograd_W_matches_analytic_funnel():
    # The autograd default recovers W = (σ/2) ψ exactly.
    fun, ls, _ = _funnel_pair()
    eta = torch.randn(5, 1)
    local, _ = ls.endpoint(eta)
    W_ana = (fun.s / 2.0) * fun.psi(eta).unsqueeze(-1)
    assert torch.allclose(local.W, W_ana, atol=1e-9)


# ========================================================================== #
#  3. ∇V against finite differences                                         #
# ========================================================================== #

def test_grad_V_matches_finite_difference():
    torch.manual_seed(1)
    A = torch.randn(4, 2)
    B = torch.eye(4) + 0.1 * torch.randn(4, 4)
    B = B @ B.transpose(-2, -1)                       # SPD, so det > 0
    c = AffineChart(A, B, torch.zeros(4), torch.randn(4))
    eta = torch.randn(3, 2)
    _, gV = c.endpoint(eta)

    h = 1e-6
    gfd = torch.zeros_like(eta)
    for j in range(eta.shape[-1]):
        ep = eta.clone(); ep[:, j] += h
        em = eta.clone(); em[:, j] -= h
        gfd[:, j] = (c.endpoint(ep)[0].V - c.endpoint(em)[0].V) / (2 * h)
    assert torch.allclose(gV, gfd, atol=1e-6)


# ========================================================================== #
#  4. Chart target reproduces the closed-form posterior                     #
# ========================================================================== #

def test_chart_target_matches_funnel_posterior():
    # V − ½ log det G_M is −log p(η | x) up to an additive constant.
    fun, _, x = _funnel_pair(sigma=3.0, m=6, seed=2)
    eta = torch.randn(50, 1)
    local, _ = fun.endpoint(eta)
    half_logdet_G = torch.log(local.chol_G.diagonal(dim1=-2, dim2=-1)).sum(-1)
    lhs = local.V - half_logdet_G

    e = eta[:, 0]
    neg_log_post = 0.5 * e * e + 0.5 * torch.exp(-fun.s * e) * (x * x).sum() \
        + 0.5 * x.shape[-1] * fun.s * e
    assert torch.allclose(lhs - lhs.mean(), neg_log_post - neg_log_post.mean(), atol=1e-9)


def test_affine_target_is_the_exact_gaussian():
    # For an affine map the η-marginal is Gaussian. Check that the chart target
    # exp(−V) √det G_M has that Gaussian's log-density up to a constant.
    torch.manual_seed(3)
    A = torch.randn(5, 2)
    B = torch.eye(5) + 0.2 * torch.randn(5, 5)
    B = B @ B.transpose(-2, -1)
    c = AffineChart(A, B, torch.zeros(5), torch.randn(5))
    mu, Sigma = c.posterior()
    prec = torch.linalg.inv(Sigma)

    eta = torch.randn(40, 2)
    local, _ = c.endpoint(eta)
    half_logdet_G = torch.log(local.chol_G.diagonal(dim1=-2, dim2=-1)).sum(-1)
    log_target = -(local.V - half_logdet_G)          # log exp(−V) √det G_M
    d = eta - mu
    log_gauss = -0.5 * torch.einsum("ni,ij,nj->n", d, prec, d)
    assert torch.allclose(log_target - log_target.mean(),
                          log_gauss - log_gauss.mean(), atol=1e-9)


def test_location_scale_log_abs_det_B_is_half_logdet_sigma():
    _, ls, _ = _funnel_pair(sigma=2.0, m=4, seed=4)
    eta = torch.randn(6, 1)
    ldB = ls.log_abs_det_B(eta)
    Sigma = ls.cov(eta)
    assert torch.allclose(ldB, 0.5 * torch.logdet(Sigma), atol=1e-10)


# ========================================================================== #
#  5. Endpoint bundle carries no autograd graph                             #
# ========================================================================== #

def test_endpoint_bundle_is_detached():
    _, ls, _ = _funnel_pair()
    eta = torch.randn(4, 1)
    local, gV = ls.endpoint(eta)
    for t in (local.eps, local.W, local.chol_G, local.V, gV):
        assert not t.requires_grad
