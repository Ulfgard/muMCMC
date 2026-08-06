"""Constraint / evaluate_model tests for ChartRATTLE.

The geometry and potential rest on the inverse map ψ(q) = φ_q⁻¹(x). evaluate_model
turns a constraint into the tempered pieces (U a TemperedAffine, G_M a
TemperedMetric). Here we check, against closed forms, that:

1. G_M = I + β Wᵀ W and its Cholesky are what evaluate_model returns.
2. The explicit-Jacobian fast path agrees with the classic-autograd W, and ∇V
   matches a finite difference.
3. The potential U.value is exactly −log p(q | x): the funnel posterior, the
   affine Gaussian, and the β-tempered funnel.
3b. The prior comes off the space: absent means flat, a supplied one enters U
   and ∇V, neither touches the metric, and a space whose blocks are coupled
   carries its whole chart Jacobian into W.
4. U.lik = ½‖ψ‖² is the temperature-free swap statistic.
5. The returned pieces are detached (no autograd graph pinned).
"""
import math

import torch
import pytest

from muMCMC.ChartRATTLE import ChartRATTLE
from muMCMC.spaces import (ConditionalLayer, ConditionalSpace,
                           LocationScaleLayer, NormalSpace,
                           UnnormalizedSpace)

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

def _log_det_B(layer, theta, x):
    """log|det B| at ``theta``, read off the latent Jacobian a layer returns
    with its inverse. It does not depend on the point, read here at ``x``."""
    theta = theta.detach().requires_grad_(True)   # a layer that differentiates
    B_inv = layer.inverse_with_jvp(theta, x.expand(theta.shape[0], -1))[2]
    return -torch.log(B_inv.diagonal(dim1=-2, dim2=-1).abs()).sum(-1).detach()


def _spd(n, scale):
    """A random SPD matrix, the source of a triangular B."""
    M = torch.eye(n) + scale * torch.randn(n, n)
    return M @ M.transpose(-2, -1)


def _eval(layer, x, eta, beta=1.0, grad=True):
    """evaluate_model through a minimal sampler at the given temperature."""
    n = eta.shape[-1]
    names = [f"v{i}" for i in range(n)]
    s = ChartRATTLE(layer, x, NormalSpace(names),
                    step_size=0.1, adapt_step_size=False)
    s.beta = beta
    return s.evaluate_model(eta, grad=grad)


def _funnel_pair(sigma=3.0, m=5, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(m)
    ls_mean = lambda eta: torch.zeros(eta.shape[0], m, dtype=eta.dtype)
    ls_cov = lambda eta: torch.exp(sigma * eta[:, 0])[:, None, None] * torch.eye(m, dtype=eta.dtype)
    return (FunnelLayer(sigma, m),
            LocationScaleLayer.from_covariance(ls_mean, ls_cov), x)


# ========================================================================== #
#  1. Metric and its Cholesky                                               #
# ========================================================================== #

def test_metric_is_identity_plus_gram_and_cholesky_matches():
    c, _, x = _funnel_pair()
    eta = torch.randn(6, 1)
    _, metric, _, W = _eval(c, x, eta, grad=False)
    n = eta.shape[-1]
    G = torch.eye(n) + W.transpose(-2, -1) @ W                # beta = 1
    assert torch.allclose(metric.value, G, atol=1e-10)
    assert torch.allclose(metric.L @ metric.L.transpose(-2, -1), G, atol=1e-10)


def test_metric_scales_with_beta():
    c, _, x = _funnel_pair()
    eta = torch.randn(4, 1)
    _, m1, _, W = _eval(c, x, eta, beta=1.0, grad=False)
    _, mb, _, _ = _eval(c, x, eta, beta=0.3, grad=False)
    n = eta.shape[-1]
    gram = W.transpose(-2, -1) @ W
    assert torch.allclose(m1.value, torch.eye(n) + gram, atol=1e-10)
    assert torch.allclose(mb.value, torch.eye(n) + 0.3 * gram, atol=1e-10)


# ========================================================================== #
#  2. Explicit Jacobians vs the autograd default                            #
# ========================================================================== #

def test_explicit_jacobian_matches_autograd_default():
    # Same funnel through explicit psi_with_jvp and through the classic-autograd
    # W (the location-scale layer). W, U and ∇V must agree.
    fun, ls, x = _funnel_pair()
    eta = torch.randn(8, 1)
    Uf, _, _, Wf, gf = _eval(fun, x, eta)
    Ul, _, _, Wl, gl = _eval(ls, x, eta)
    assert torch.allclose(Wf, Wl, atol=1e-9)
    assert torch.allclose(Uf.value, Ul.value, atol=1e-9)
    assert torch.allclose(gf, gl, atol=1e-8)


def test_grad_V_matches_finite_difference():
    torch.manual_seed(1)
    A = torch.randn(4, 2)
    B = torch.linalg.cholesky(_spd(4, 0.1))         # triangular, so det > 0
    x = torch.randn(4)
    layer = AffineLayer(A, B, torch.zeros(4))
    eta = torch.randn(3, 2)
    _, _, _, _, gV = _eval(layer, x, eta)

    def V(e):
        U, metric, _, _ = _eval(layer, x, e, grad=False)
        return U.value + 0.5 * metric.log_det_metric()
    h = 1e-6
    gfd = torch.zeros_like(eta)
    for j in range(eta.shape[-1]):
        ep = eta.clone(); ep[:, j] += h
        em = eta.clone(); em[:, j] -= h
        gfd[:, j] = (V(ep) - V(em)) / (2 * h)
    assert torch.allclose(gV, gfd, atol=1e-6)


# ========================================================================== #
#  3. The potential is exactly −log p(q | x)                                #
# ========================================================================== #

def test_potential_matches_funnel_posterior():
    fun, _, x = _funnel_pair(sigma=3.0, m=6, seed=2)
    eta = torch.randn(50, 1)
    U, _, _, _ = _eval(fun, x, eta, grad=False)
    e = eta[:, 0]
    neg_log_post = 0.5 * e * e + 0.5 * torch.exp(-fun.s * e) * (x * x).sum() \
        + 0.5 * x.shape[-1] * fun.s * e
    assert torch.allclose(U.value - U.value.mean(),
                          neg_log_post - neg_log_post.mean(), atol=1e-9)


def test_potential_is_the_exact_affine_gaussian():
    torch.manual_seed(3)
    A = torch.randn(5, 2)
    B = torch.linalg.cholesky(_spd(5, 0.2))
    x = torch.randn(5)
    layer = AffineLayer(A, B, torch.zeros(5))
    mu, Sigma = layer.posterior(x)
    prec = torch.linalg.inv(Sigma)

    eta = torch.randn(40, 2)
    U, _, _, _ = _eval(layer, x, eta, grad=False)
    d = eta - mu
    neg_log_gauss = 0.5 * torch.einsum("ni,ij,nj->n", d, prec, d)
    assert torch.allclose(U.value - U.value.mean(),
                          neg_log_gauss - neg_log_gauss.mean(), atol=1e-9)


def test_beta_tempers_the_data_fit_only():
    # U.value = base + β·½‖ψ‖²: the Mahalanobis is scaled by β, the base (prior +
    # log|det B|) is not, matching the β-tempered funnel posterior.
    fun, _, x = _funnel_pair(sigma=3.0, m=6, seed=2)
    eta = torch.randn(30, 1)
    U, _, _, _ = _eval(fun, x, eta, beta=0.4, grad=False)
    e = eta[:, 0]
    # Both densities are normalized, so the untempered part carries the prior's
    # ½ log 2π over one coordinate and the likelihood's over m.
    neg_log_post = 0.5 * e * e + 0.5 * math.log(2 * math.pi) \
        + 0.5 * x.shape[-1] * math.log(2 * math.pi) \
        + 0.5 * x.shape[-1] * fun.s * e \
        + 0.4 * 0.5 * torch.exp(-fun.s * e) * (x * x).sum()
    assert torch.allclose(U.value, neg_log_post, atol=1e-9)


# ========================================================================== #
#  3b. The space's prior replaces the standard-normal latent                #
# ========================================================================== #

def _prior_eval(layer, x, eta, mu, sigma, beta=1.0, grad=True):
    """evaluate_model through a sampler whose space has a Normal(mu, sigma)
    prior, so the chart is theta = mu + sigma q."""
    names = [f"v{i}" for i in range(eta.shape[-1])]
    s = ChartRATTLE(layer, x, NormalSpace(names, mu=mu, sigma=sigma),
                    step_size=0.1, adapt_step_size=False)
    s.beta = beta
    return s.evaluate_model(eta, grad=grad)


def _base_of(layer, x, eta, mu=0.0, sigma=1.0):
    """U.base at ``eta`` for a space with a Normal(mu, sigma) prior."""
    s = ChartRATTLE(layer, x, NormalSpace(["v0"], mu=mu, sigma=sigma),
                    step_size=0.1, adapt_step_size=False)
    return s.evaluate_model(eta, grad=False)[0].base


def test_a_space_without_a_prior_is_rejected():
    # ChartRATTLE reads M off the chart, and a space with no prior has none.
    c, _, x = _funnel_pair()
    with pytest.raises(ValueError, match="prior"):
        ChartRATTLE(c, x, UnnormalizedSpace(["v0"]), step_size=0.1,
                    adapt_step_size=False)


def test_standard_normal_prior_is_the_non_centered_latent():
    # The chart of Normal(0, 1) is the identity, so U.base is 1/2||q||^2 plus
    # the volume term and the two normalizers.
    c, _, x = _funnel_pair()
    eta = torch.randn(7, 1)
    expected = (0.5 * (eta * eta).sum(-1) + 0.5 * math.log(2 * math.pi)
                + 0.5 * x.shape[-1] * math.log(2 * math.pi)
                + _log_det_B(c, eta, x))
    assert torch.allclose(_base_of(c, x, eta), expected, atol=1e-12)


def test_space_prior_enters_U_as_its_chart():
    # A Normal(mu0, s0) prior is the chart theta = mu0 + s0 q, so U.base stays
    # the standard normal potential plus the volume term read at theta, and the
    # likelihood is the one evaluated at theta rather than at q.
    c, _, x = _funnel_pair(sigma=2.0, m=4, seed=3)
    eta = torch.randn(9, 1)
    mu0, s0 = 0.7, 1.6
    theta = mu0 + s0 * eta

    U_pri = _prior_eval(c, x, eta, mu0, s0)[0]

    expected_base = (0.5 * (eta * eta).sum(-1) + 0.5 * math.log(2 * math.pi)
                     + 0.5 * x.shape[-1] * math.log(2 * math.pi)
                     + _log_det_B(c, theta, x))
    assert torch.allclose(U_pri.base, expected_base, atol=1e-10)
    assert torch.allclose(U_pri.lik, _eval(c, x, theta)[0].lik, atol=1e-10)


def test_space_prior_enters_the_force():
    # grad V has to pick the prior up, or the integrator targets a different
    # density than U describes.
    torch.manual_seed(4)
    A = torch.randn(4, 2)
    B = torch.linalg.cholesky(_spd(4, 0.1))
    x = torch.randn(4)
    layer = AffineLayer(A, B, torch.zeros(4))
    eta = torch.randn(3, 2)
    mu, sd = [0.4, -0.3], [2.0, 0.7]
    gV = _prior_eval(layer, x, eta, mu, sd)[4]

    def V(e):
        U, metric, _, _ = _prior_eval(layer, x, e, mu, sd, grad=False)
        return U.value + 0.5 * metric.log_det_metric()
    h = 1e-6
    gfd = torch.zeros_like(eta)
    for j in range(eta.shape[-1]):
        ep = eta.clone(); ep[:, j] += h
        em = eta.clone(); em[:, j] -= h
        gfd[:, j] = (V(ep) - V(em)) / (2 * h)
    assert torch.allclose(gV, gfd, atol=1e-6)


def test_a_conditional_space_carries_its_coupling_into_W_and_U():
    # A space whose two blocks are coupled has a chart Jacobian that is not
    # diagonal, so W read in the chart is the constraint's W times the whole
    # Jacobian, and the volume terms of the chart follow the same route.
    torch.manual_seed(6)
    A = torch.randn(3, 2)
    B = torch.linalg.cholesky(_spd(3, 0.1))
    x = torch.randn(3)
    layer = AffineLayer(A, B, torch.zeros(3))

    def location_scale(theta_a):
        return 0.5 * theta_a, torch.exp(0.3 * theta_a).unsqueeze(-1)

    space = ConditionalSpace(NormalSpace(["v0"]), ["v1"],
                             LocationScaleLayer(location_scale))
    s = ChartRATTLE(layer, x, space, step_size=0.1, adapt_step_size=False)
    q = torch.randn(5, 2)
    U, _, psi, W = s.evaluate_model(q, grad=False)

    # W = -d psi(T(q))/dq, the constraint's own W pulled back along the chart.
    qg = q.clone().requires_grad_(True)
    theta = space.as_transform.forward_with_jvp(qg).mapped_point
    psi_of = lambda th: layer.inverse(th, x.expand(th.shape[0], -1))
    rows = [torch.autograd.grad(psi_of(theta)[:, i].sum(), qg, retain_graph=True)[0]
            for i in range(3)]
    assert torch.allclose(W, -torch.stack(rows, dim=1), atol=1e-10)

    # On the variables the potential is -log[p(theta) p(x | theta)], so the
    # chart's log determinant cancels against the one U carries.
    theta = theta.detach()
    expected = (-space.prior_log_prob_vector(theta) + _log_det_B(layer, theta, x)
                + 0.5 * 3 * math.log(2 * math.pi)
                + 0.5 * (psi_of(theta) ** 2).sum(-1))
    assert torch.allclose(s.potential(theta), expected, atol=1e-10)


def test_prior_block_of_the_metric_is_the_identity():
    # M is the prior block of G_M and the matrix the position solve
    # preconditions with. In the chart it is the identity whatever the prior,
    # which is what makes it constant.
    c, _, x = _funnel_pair()
    eta = torch.randn(5, 1)
    for mu, sd in [(0.0, 1.0), (2.0, 0.3)]:
        m = _prior_eval(c, x, eta, mu, sd)[1]
        assert torch.allclose(m.base, torch.eye(1).expand(5, 1, 1), atol=1e-12)


def test_space_prior_is_untempered():
    # beta multiplies the likelihood only, so the prior sits in base and the
    # swap statistic stays temperature-free.
    c, _, x = _funnel_pair()
    eta = torch.randn(6, 1)
    U1 = _prior_eval(c, x, eta, 0.2, 1.3, beta=1.0)[0]
    Ub = _prior_eval(c, x, eta, 0.2, 1.3, beta=0.25)[0]
    assert torch.allclose(U1.base, Ub.base, atol=1e-12)
    assert torch.allclose(U1.lik, Ub.lik, atol=1e-12)
    assert torch.allclose(Ub.value, Ub.base + 0.25 * Ub.lik, atol=1e-12)


# ========================================================================== #
#  4. U.lik: temperature-free swap statistic                                #
# ========================================================================== #

def test_u_lik_is_half_psi_squared_and_temperature_free():
    fun, _, x = _funnel_pair(sigma=3.0, m=5, seed=6)
    eta = torch.randn(8, 1)
    U1, _, psi, _ = _eval(fun, x, eta, beta=1.0, grad=False)
    Ub, _, _, _ = _eval(fun, x, eta, beta=0.2, grad=False)
    assert torch.allclose(U1.lik, 0.5 * (psi ** 2).sum(-1), atol=1e-10)
    assert torch.allclose(U1.lik, Ub.lik, atol=1e-10)     # β-free


def test_location_scale_log_abs_det_B_is_half_logdet_sigma():
    _, ls, x = _funnel_pair(sigma=2.0, m=4, seed=4)
    eta = torch.randn(6, 1)
    ldB = _log_det_B(ls, eta, x)
    Sigma = torch.exp(2.0 * eta[:, 0])[:, None, None] * torch.eye(4)
    assert torch.allclose(ldB, 0.5 * torch.logdet(Sigma), atol=1e-10)


# ========================================================================== #
#  5. Detached                                                              #
# ========================================================================== #

def test_evaluate_model_output_is_detached():
    _, ls, x = _funnel_pair()
    eta = torch.randn(4, 1)
    U, metric, psi, W, gV = _eval(ls, x, eta)
    for t in (U.value, U.lik, metric.value, metric.L, psi, W, gV):
        assert not t.requires_grad
