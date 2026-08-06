"""Contract tests for ``ConditionalSpace``.

The space is two blocks, ``theta_A`` with a prior of its own and ``theta_B``
normal given it, so its chart is triangular rather than elementwise. These tests
check the chart against autograd, the prior against its closed form, and that
it serves no marginal.

The reference throughout is the linear-Gaussian instance

    a ~ N(0, 1),  b | a ~ N(c a, s^2),

whose joint prior is the Gaussian with covariance [[1, c], [c, c^2 + s^2]], and
a nonlinear instance whose location and scale are neither affine nor constant.
"""
import math

import torch
import pytest

from muMCMC import HMC
from muMCMC.spaces import (ConditionalSpace, LocationScaleLayer, NormalSpace,
                           UnnormalizedSpace)

torch.set_default_dtype(torch.float64)

ATOL = 1e-10
Normal = torch.distributions.Normal
MVN = torch.distributions.MultivariateNormal

C, S = 0.8, 0.5


def _linear_location_scale(theta_a):
    """``b | a ~ N(c a, s^2)`` for the single leading name ``a``."""
    return C * theta_a, S * torch.ones_like(theta_a).unsqueeze(-1)


def _linear_space(**kw):
    return ConditionalSpace(NormalSpace(["a"], **kw), ["b"],
                            LocationScaleLayer(_linear_location_scale))


def _joint_prior_cov():
    return torch.tensor([[1.0, C], [C, C ** 2 + S ** 2]])


def _nonlinear_location_scale(theta_a):
    """A location and a scale that are neither affine nor constant in
    ``theta_a``, so the chart's coupling is nontrivial in every block."""
    a, b = theta_a[..., 0:1], theta_a[..., 1:2]
    mu = torch.cat([torch.tanh(a) + b, 0.5 * a * b], dim=-1)
    zero = torch.zeros_like(a)
    rows = [torch.cat([torch.exp(0.3 * a), zero], -1),
            torch.cat([0.4 * a, torch.exp(-0.2 * b)], -1)]
    return mu, torch.stack(rows, dim=-2)


def _nonlinear_space(**kw):
    return ConditionalSpace(
        NormalSpace(["a0", "a1"], mu=0.3, sigma=1.7, **kw), ["b0", "b1"],
        LocationScaleLayer(_nonlinear_location_scale))


# --------------------------------------------------------------------------- #
#  Construction and layout                                                    #
# --------------------------------------------------------------------------- #

def test_the_blocks_are_laid_out_in_order():
    s = _nonlinear_space()
    assert s.names == ["a0", "a1", "b0", "b1"]
    assert s.free_names == ["a0", "a1", "b0", "b1"]
    assert s.d == 4 and s.d_full == 4


def test_a_repeated_name_is_rejected():
    with pytest.raises(ValueError, match="must be new"):
        ConditionalSpace(NormalSpace(["a"]), ["a"],
                         LocationScaleLayer(_linear_location_scale))


def test_a_base_without_a_prior_is_rejected():
    # The joint prior is p(theta_A) p(theta_B | theta_A), so a base with no
    # prior has no chart for this one to stand on.
    with pytest.raises(ValueError, match="no chart"):
        ConditionalSpace(UnnormalizedSpace(["a"]), ["b"],
                         LocationScaleLayer(_linear_location_scale))


def test_a_layer_of_the_wrong_width_is_rejected():
    # The layer is read once at construction, so a width that disagrees with the
    # names is refused there rather than at the first draw.
    with pytest.raises(RuntimeError):
        ConditionalSpace(NormalSpace(["a"]), ["b0", "b1"],
                         LocationScaleLayer(_linear_location_scale))


def test_the_location_and_scale_read_the_full_leading_vector():
    # A fixed variable is part of theta_A, so it reaches the conditional at its
    # value rather than being dropped on the way.
    seen = []

    def location_scale(theta_a):
        seen.append(theta_a.clone())
        return C * theta_a[..., :1], S * torch.ones(theta_a.shape[:-1] + (1, 1))

    s = ConditionalSpace(NormalSpace(["a0", "a1"], fixed={"a1": 2.0}), ["b"],
                         LocationScaleLayer(location_scale))
    assert s.d == 2 and s.free_names == ["a0", "b"]
    s.prior_log_prob_vector(torch.zeros(3, 2))
    assert seen[-1].shape == (3, 2)
    assert torch.allclose(seen[-1][:, 1], torch.full((3,), 2.0), atol=ATOL)

    s.set_fixed({"a1": -4.0})
    s.prior_log_prob_vector(torch.zeros(3, 2))
    assert torch.allclose(seen[-1][:, 1], torch.full((3,), -4.0), atol=ATOL)


# --------------------------------------------------------------------------- #
#  The chart                                                                  #
# --------------------------------------------------------------------------- #

def _autograd_jacobian(fn, x):
    """``d fn / d x`` at each row of ``x``, shape ``(n, d, d)``."""
    J = torch.autograd.functional.jacobian(lambda v: fn(v).sum(0), x.clone())
    return J.permute(1, 0, 2)


def test_the_chart_round_trips():
    s = _nonlinear_space()
    z = torch.randn(5, 4)
    theta = s.as_transform.forward(z).mapped_point
    assert torch.allclose(s.as_transform.inverse(theta).mapped_point, z, atol=1e-9)


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_the_chart_jacobian_matches_autograd(direction):
    s = _nonlinear_space()
    x = torch.randn(5, 4)
    if direction == "inverse":
        x = s.as_transform.forward(x).mapped_point
    fn = getattr(s.as_transform, direction)
    m = fn(x)
    J = _autograd_jacobian(lambda v: fn(v).mapped_point, x)
    assert torch.allclose(m.dense_jacobian(), J, atol=1e-9)
    assert torch.allclose(m.jacobian_log_det, torch.linalg.slogdet(J)[1], atol=1e-9)
    # Triangular and not diagonal, so the map carries the matrix.
    assert m.is_dense


def test_the_interior_point_is_the_image_of_zero():
    s = _nonlinear_space()
    assert torch.allclose(s.as_transform.interior_point,
                          s.as_transform.forward(torch.zeros(1, 4)).mapped_point[0],
                          atol=ATOL)


def test_the_prior_metric_is_the_pullback_of_the_identity():
    # M = J^-T J^-1 on the variables, so its pushforward J^T M J to the chart is
    # the identity, which is what a chain running there reads.
    s = _nonlinear_space()
    z = torch.randn(5, 4)
    m = s.as_transform.forward(z)
    M = s.prior_metric(m.mapped_point)
    assert M.shape == (5, 4, 4)
    J = m.dense_jacobian()
    assert torch.allclose(J.mT @ M @ J, torch.eye(4).expand(5, 4, 4), atol=1e-8)


def test_the_chart_is_differentiable_through_the_coupling():
    # The force a metric scheme integrates differentiates the chart Jacobian, so
    # the coupling has to reach the input rather than arrive detached.
    s = _nonlinear_space()
    z = torch.randn(4, 4, requires_grad=True)
    (g,) = torch.autograd.grad(
        s.as_transform.forward(z).dense_jacobian().sum(), z)
    assert torch.isfinite(g).all() and float(g.abs().max()) > 0.0


# --------------------------------------------------------------------------- #
#  The prior                                                                  #
# --------------------------------------------------------------------------- #

def test_the_joint_prior_is_the_leading_prior_times_the_conditional():
    s = _nonlinear_space()
    theta = s.as_transform.forward(torch.randn(6, 4)).mapped_point
    mu, L = _nonlinear_location_scale(theta[:, :2])
    ref = (Normal(0.3, 1.7).log_prob(theta[:, :2]).sum(-1)
           + MVN(mu, scale_tril=L).log_prob(theta[:, 2:]))
    assert torch.allclose(s.prior_log_prob_vector(theta), ref, atol=1e-9)
    assert torch.allclose(s.prior_log_prob(s.from_free_vector(theta)), ref,
                          atol=1e-9)


def test_the_linear_instance_matches_its_joint_gaussian():
    # b | a normal with a mean linear in a makes (a, b) jointly normal, which is
    # the one case with a reference the space itself cannot express.
    s = _linear_space()
    theta = torch.randn(7, 2)
    ref = MVN(torch.zeros(2), covariance_matrix=_joint_prior_cov()).log_prob(theta)
    assert torch.allclose(s.prior_log_prob_vector(theta), ref, atol=1e-9)


def test_this_prior_serves_no_marginal():
    # The trailing block is conditional on the leading one, so a subset of the
    # names has no closed-form density here whichever subset it is. The leading
    # block's own marginals stay where they live, on the base space.
    s = _nonlinear_space()
    theta = torch.randn(6, 4)
    for subset in ({"a1": theta[:, 1]},
                   {"a0": theta[:, 0], "a1": theta[:, 1], "b1": theta[:, 3]},
                   {"a0": theta[:, 0], "b0": theta[:, 2]},
                   {"b0": theta[:, 2]}):
        with pytest.raises(ValueError, match="needs every free name"):
            s.prior_log_prob(subset)
    assert torch.allclose(s.base.prior_log_prob({"a1": theta[:, 1]}),
                          Normal(0.3, 1.7).log_prob(theta[:, 1]), atol=ATOL)


def test_the_prior_is_differentiable_in_both_blocks():
    s = _nonlinear_space()
    theta = torch.randn(5, 4, requires_grad=True)
    (g,) = torch.autograd.grad(s.prior_log_prob_vector(theta).sum(), theta)
    assert torch.isfinite(g).all() and float(g[:, :2].abs().min()) > 0.0


# --------------------------------------------------------------------------- #
#  Sampling                                                                   #
# --------------------------------------------------------------------------- #

def test_sample_recovers_the_joint_prior_moments():
    s = _linear_space()
    draw = s.sample(40000, generator=torch.Generator().manual_seed(3))
    x = torch.stack([draw["a"], draw["b"]], dim=-1)
    cov = torch.cov(x.T)
    assert torch.allclose(x.mean(0), torch.zeros(2), atol=0.03)
    assert torch.allclose(cov, _joint_prior_cov(), atol=0.03)


def test_sample_keeps_the_fixed_variables():
    s = ConditionalSpace(
        NormalSpace(["a0", "a1"], fixed={"a1": 2.0}), ["b"],
        LocationScaleLayer(lambda theta_a: (
            C * theta_a[..., :1], S * torch.ones(theta_a.shape[:-1] + (1, 1)))))
    draw = s.sample(16, generator=torch.Generator().manual_seed(0))
    assert set(draw) == {"a0", "a1", "b"}
    assert torch.allclose(draw["a1"], torch.full((16,), 2.0), atol=ATOL)


# --------------------------------------------------------------------------- #
#  Composition                                                                #
# --------------------------------------------------------------------------- #

def _third_block(theta):
    """``c | (a, b) ~ N(mu, L Lᵀ)`` over the whole leading vector."""
    s = theta.sum(-1, keepdim=True)
    return torch.tanh(s), torch.exp(0.2 * s).unsqueeze(-1)


def _nested_space():
    """Three blocks, the second conditional on the first and the third on both."""
    return ConditionalSpace(_nonlinear_space(), ["c"],
                            LocationScaleLayer(_third_block))


def test_a_space_composes_as_the_base_of_another():
    s = _nested_space()
    assert s.names == ["a0", "a1", "b0", "b1", "c"]
    assert s.d == 5
    z = torch.randn(4, 5)
    theta = s.as_transform.forward(z).mapped_point
    assert torch.allclose(s.as_transform.inverse(theta).mapped_point, z, atol=1e-9)


@pytest.mark.parametrize("direction", ["forward", "inverse"])
def test_the_nested_chart_jacobian_matches_autograd(direction):
    # The leading block is itself coupled, so its own off-diagonal entries have
    # to reach the assembled Jacobian and the coupling of the third block has to
    # be carried through them.
    s = _nested_space()
    x = torch.randn(4, 5)
    if direction == "inverse":
        x = s.as_transform.forward(x).mapped_point
    fn = getattr(s.as_transform, direction)
    J = _autograd_jacobian(lambda v: fn(v).mapped_point, x)
    assert torch.allclose(fn(x).dense_jacobian(), J, atol=1e-9)
    assert torch.allclose(fn(x).jacobian_log_det,
                          torch.linalg.slogdet(J)[1], atol=1e-9)


def test_the_nested_prior_is_the_chain_of_its_three_blocks():
    s = _nested_space()
    theta = s.as_transform.forward(torch.randn(6, 5)).mapped_point
    mu_b, L_b = _nonlinear_location_scale(theta[:, :2])
    mu_c, L_c = _third_block(theta[:, :4])
    ref = (Normal(0.3, 1.7).log_prob(theta[:, :2]).sum(-1)
           + MVN(mu_b, scale_tril=L_b).log_prob(theta[:, 2:4])
           + MVN(mu_c, scale_tril=L_c).log_prob(theta[:, 4:]))
    assert torch.allclose(s.prior_log_prob_vector(theta), ref, atol=1e-9)


def test_the_nested_prior_metric_is_the_pullback_of_the_identity():
    s = _nested_space()
    m = s.as_transform.forward(torch.randn(4, 5))
    M = s.prior_metric(m.mapped_point)
    J = m.dense_jacobian()
    assert torch.allclose(J.mT @ M @ J, torch.eye(5).expand(4, 5, 5), atol=1e-7)


# --------------------------------------------------------------------------- #
#  A sampler on the space                                                     #
# --------------------------------------------------------------------------- #

def test_hmc_recovers_the_linear_gaussian_posterior():
    # The chain reads the prior and its gradient through the conditional chart,
    # so this is the whole space driven by a sampler rather than by hand.
    sigma = 0.7
    x_obs = 1.3
    space = _linear_space()

    def potential_fn(theta):                   # U_lik = -log N(x; b, sigma^2)
        return (0.5 * (x_obs - theta[..., 1]) ** 2 / sigma ** 2
                + 0.5 * math.log(2 * math.pi * sigma ** 2))

    prec = torch.linalg.inv(_joint_prior_cov())
    prec = prec + torch.tensor([[0.0, 0.0], [0.0, 1.0 / sigma ** 2]])
    cov_post = torch.linalg.inv(prec)
    mean_post = cov_post @ torch.tensor([0.0, x_obs / sigma ** 2])

    torch.manual_seed(0)
    sampler = HMC(potential_fn, space, step_size=0.3, num_steps=8)
    out = sampler.run_mcmc({"a": torch.tensor(0.0), "b": torch.tensor(0.0)},
                           num_samples=2000, num_warmup_steps=500, num_chains=4,
                           disable_progbar=True)
    draw = torch.stack([out["a"].reshape(-1), out["b"].reshape(-1)], dim=-1)
    assert torch.allclose(draw.mean(0), mean_post, atol=0.05)
    assert torch.allclose(torch.cov(draw.T), cov_post, atol=0.05)
