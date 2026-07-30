"""Contract tests for ``NormalTransform`` and ``ElementwiseMap``.

The transform is the chart every space leans on: it exposes both directions,
each with its own diagonal Jacobian and log-determinant, and ChartRATTLE trusts
those to match the analytic Jacobian. These tests pin that contract down.

The transform is built from callables rather than by subclassing, so these
construct one directly instead of going through a space.
"""
import math

import torch
import pytest

from muMCMC.spaces import ElementwiseMap, NormalTransform
from muMCMC.spaces._transform import _std_normal_cdf, _std_normal_log_pdf

torch.set_default_dtype(torch.float64)

ATOL = 1e-10
MU = torch.tensor([0.5, -1.0, 2.0])
SIGMA = torch.tensor([1.0, 2.0, 0.3])


def _affine(*, log_prob=False):
    """``theta = mu + sigma z``, with ``log_prob`` switchable so the derivation
    from the inverse map can be checked against the closed form."""
    log_sigma = torch.log(SIGMA)
    kw = {}
    if log_prob:
        kw["log_prob"] = lambda th: _std_normal_log_pdf((th - MU) / SIGMA) - log_sigma
    return NormalTransform(
        lambda z: (MU + SIGMA * z, log_sigma.expand_as(z)),
        lambda th: ((th - MU) / SIGMA, (-log_sigma).expand_as(th)),
        reference=MU, **kw)


# --------------------------------------------------------------------------- #
#  ElementwiseMap                                                             #
# --------------------------------------------------------------------------- #

def test_jacobian_diag_is_the_exponential_of_the_stored_log():
    log_j = torch.tensor([[0.0, -2.0, 1.5]])
    m = ElementwiseMap(torch.zeros(1, 3), torch.ones(1, 3), log_j)
    assert torch.allclose(m.jacobian_diag, torch.exp(log_j), atol=ATOL)
    assert torch.allclose(m.jacobian_log_det, log_j.sum(-1), atol=ATOL)


def test_log_det_sums_over_the_coordinate_axis_only():
    log_j = torch.randn(2, 4, 3)
    m = ElementwiseMap(torch.zeros(2, 4, 3), torch.zeros(2, 4, 3), log_j)
    assert m.jacobian_log_det.shape == (2, 4)


def test_inv_swaps_the_endpoints_and_negates_the_log_jacobian():
    point, image, log_j = torch.randn(5, 3), torch.randn(5, 3), torch.randn(5, 3)
    m = ElementwiseMap(point, image, log_j).inv
    assert torch.allclose(m.point, image, atol=ATOL)
    assert torch.allclose(m.mapped_point, point, atol=ATOL)
    assert torch.allclose(m.jacobian_log_diag, -log_j, atol=ATOL)


def test_jvp_scales_elementwise():
    log_j, v = torch.randn(4, 3), torch.randn(4, 3)
    m = ElementwiseMap(torch.zeros(4, 3), torch.zeros(4, 3), log_j)
    assert torch.allclose(m.jvp(v), torch.exp(log_j) * v, atol=ATOL)


# --------------------------------------------------------------------------- #
#  The two directions                                                         #
# --------------------------------------------------------------------------- #

def test_forward_and_inverse_round_trip():
    T = _affine()
    z = torch.randn(7, 3)
    assert torch.allclose(T.inverse(T.forward(z).mapped_point).mapped_point, z,
                          atol=ATOL)


def test_the_two_jacobians_are_reciprocal():
    T = _affine()
    m = T.forward(torch.randn(7, 3))
    back = T.inverse(m.mapped_point)
    assert torch.allclose(back.jacobian_diag, 1.0 / m.jacobian_diag, atol=ATOL)


def test_forward_jacobian_matches_autograd():
    T = _affine()
    z = torch.randn(7, 3, requires_grad=True)
    (g,) = torch.autograd.grad(T.forward(z).mapped_point.sum(), z)
    assert torch.allclose(T.forward(z.detach()).jacobian_diag, g, atol=ATOL)


def test_metric_is_the_reciprocal_square_of_the_forward_jacobian():
    # M = J^-T J^-1, so its pushforward J^T M J is the identity, which is what
    # every sampler reads as the prior's metric in the chart.
    T = _affine()
    m = T.forward(torch.randn(6, 3))
    M = T.metric(m.mapped_point)
    assert torch.allclose(M * m.jacobian_diag ** 2, torch.ones_like(M), atol=ATOL)


def test_arbitrary_leading_batch_axes():
    m = _affine().forward(torch.randn(2, 5, 3))
    assert m.mapped_point.shape == (2, 5, 3)
    assert m.jacobian_log_det.shape == (2, 5)


def test_interior_point_is_the_image_of_zero():
    assert torch.allclose(_affine().interior_point, MU, atol=ATOL)


def test_reports_its_coordinate_count_dtype_and_device():
    T = _affine()
    assert T.d == 3 and T.dtype == MU.dtype and T.device == MU.device


# --------------------------------------------------------------------------- #
#  log_prob, given or derived                                                 #
# --------------------------------------------------------------------------- #

def test_derived_log_prob_matches_the_given_one():
    # Derived from the inverse map, which is an identity rather than an
    # approximation, so it must agree exactly with the closed form.
    theta = MU + SIGMA * torch.randn(9, 3)
    assert torch.allclose(_affine(log_prob=False).log_prob(theta),
                          _affine(log_prob=True).log_prob(theta), atol=ATOL)


def test_log_prob_is_the_density_of_the_pushed_forward_normal():
    theta = MU + SIGMA * torch.randn(9, 3)
    ref = torch.distributions.Normal(MU, SIGMA).log_prob(theta)
    assert torch.allclose(_affine().log_prob(theta), ref, atol=ATOL)


# --------------------------------------------------------------------------- #
#  Everything is supplied                                                     #
# --------------------------------------------------------------------------- #

def test_a_missing_log_jacobian_is_rejected():
    # Nothing is differentiated numerically, so a caller who leaves a Jacobian
    # out is told at construction rather than served an approximation.
    with pytest.raises(ValueError, match="both log"):
        NormalTransform(lambda z: (MU + SIGMA * z, None),
                        lambda th: ((th - MU) / SIGMA, torch.zeros_like(th)),
                        reference=MU)


def test_a_missing_inverse_log_jacobian_is_rejected():
    with pytest.raises(ValueError, match="both log"):
        NormalTransform(lambda z: (MU + SIGMA * z, torch.zeros_like(z)),
                        lambda th: ((th - MU) / SIGMA, None),
                        reference=MU)


def test_the_forward_map_is_differentiable_to_second_order():
    # Both directions are closed forms, so the graph reaches the input. The
    # force ChartRATTLE integrates differentiates the chart Jacobian, so it is
    # the second derivative that has to survive, on a chart that has one.
    T = NormalTransform(lambda z: (torch.exp(z), z),
                        lambda th: (torch.log(th), -torch.log(th)),
                        reference=torch.zeros(3))
    z = torch.randn(4, 3, requires_grad=True)
    (g,) = torch.autograd.grad(T.forward(z).mapped_point.sum(), z,
                               create_graph=True)
    (gg,) = torch.autograd.grad(g.sum(), z)
    assert torch.allclose(g, torch.exp(z).detach(), atol=ATOL)
    assert torch.allclose(gg, torch.exp(z).detach(), atol=ATOL)


def test_saturating_chart_keeps_the_log_jacobian_finite():
    # The log is the stored form precisely so that a saturating value, whose
    # Jacobian underflows, still reports a usable derivative.
    T = NormalTransform(
        lambda z: (_std_normal_cdf(z), _std_normal_log_pdf(z)),
        lambda th: (torch.zeros_like(th), torch.zeros_like(th)),
        reference=torch.zeros(2))
    m = T.forward(torch.full((1, 2), 40.0))
    assert torch.all(m.mapped_point == 1.0)
    assert torch.all(torch.isfinite(m.jacobian_log_diag))
    assert float(m.jacobian_log_diag.max()) < -700.0
    assert math.isclose(float(m.jacobian_diag.max()), 0.0, abs_tol=1e-300)
