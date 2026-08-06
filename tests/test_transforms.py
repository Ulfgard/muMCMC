"""Contract tests for ``NormalTransform`` and ``Map``.

The transform is the chart every space leans on: it exposes both directions,
each with its own Jacobian and log-determinant, and ChartRATTLE trusts those to
match the analytic Jacobian. These tests pin that contract down, for a map
holding its Jacobian diagonal and for one holding the Jacobian in full.

The transform is built from callables rather than by subclassing, so these
construct one directly instead of going through a space.
"""
import torch
import pytest

from muMCMC.spaces import Map, NormalTransform

torch.set_default_dtype(torch.float64)

ATOL = 1e-10

MU = torch.tensor([0.5, -1.0, 2.0])
SIGMA = torch.tensor([1.0, 2.0, 0.3])


def _affine():
    """``theta = mu + sigma z``."""
    return NormalTransform(
        lambda z: (MU + SIGMA * z, SIGMA.expand_as(z)),
        lambda th: ((th - MU) / SIGMA, (1.0 / SIGMA).expand_as(th)),
        reference=MU)


# --------------------------------------------------------------------------- #
#  Map carrying its Jacobian diagonal                                         #
# --------------------------------------------------------------------------- #

def test_the_diagonal_gives_the_determinant():
    diag = torch.tensor([[1.0, 0.25, 4.5]])
    m = Map(torch.zeros(1, 3), torch.ones(1, 3), diag)
    assert torch.allclose(m.jacobian_diag, diag, atol=ATOL)
    assert torch.allclose(m.jacobian_log_det, torch.log(diag).sum(-1), atol=ATOL)


def test_log_det_sums_over_the_coordinate_axis_only():
    diag = torch.rand(2, 4, 3) + 0.5
    m = Map(torch.zeros(2, 4, 3), torch.zeros(2, 4, 3), diag)
    assert m.jacobian_log_det.shape == (2, 4)


def test_inv_swaps_the_endpoints_and_inverts_the_diagonal():
    point, image = torch.randn(5, 3), torch.randn(5, 3)
    diag = torch.rand(5, 3) + 0.5
    m = Map(point, image, diag).inv
    assert torch.allclose(m.point, image, atol=ATOL)
    assert torch.allclose(m.mapped_point, point, atol=ATOL)
    assert torch.allclose(m.jacobian_diag, 1.0 / diag, atol=ATOL)


def test_a_map_on_its_diagonal_applies_itself():
    # The diagonal is the whole Jacobian, so it answers what a matrix answers,
    # each operation reading it in the form it is in.
    diag = torch.rand(4, 3) + 0.5
    m = Map(torch.zeros(4, 3), torch.zeros(4, 3), diag)
    J = torch.diag_embed(diag)
    v, W = torch.randn(4, 3), torch.randn(4, 2, 3)
    assert not m.is_dense
    assert torch.allclose(m.dense_jacobian(), J, atol=ATOL)
    assert torch.allclose(m.jvp(v), (J @ v[..., None])[..., 0], atol=ATOL)
    assert torch.allclose(m.pullback(W), W @ J, atol=ATOL)
    assert torch.allclose(m.gram(), J.mT @ J, atol=ATOL)


# --------------------------------------------------------------------------- #
#  Map carrying the Jacobian in full                                          #
# --------------------------------------------------------------------------- #

def _dense(n=4, d=5):
    """A map whose Jacobian is lower triangular and not diagonal."""
    J = torch.tril(torch.randn(n, d, d)) + 2.0 * torch.eye(d)
    return Map(torch.randn(n, d), torch.randn(n, d), J), J


def test_log_det_reads_the_diagonal_and_not_the_rest():
    # A triangular Jacobian has the determinant of its diagonal, which is why a
    # density costs the same whether or not the map is diagonal.
    m, J = _dense()
    assert torch.allclose(m.jacobian_log_det, torch.linalg.slogdet(J)[1], atol=1e-9)


def test_jvp_pullback_and_gram_take_the_dense_route():
    m, J = _dense()
    v, W = torch.randn(4, 5), torch.randn(4, 2, 5)
    assert torch.allclose(m.jvp(v), (J @ v[..., None])[..., 0], atol=ATOL)
    assert torch.allclose(m.pullback(W), W @ J, atol=ATOL)
    assert torch.allclose(m.gram(), J.mT @ J, atol=ATOL)


def test_inv_inverts_the_jacobian_it_holds():
    m, J = _dense()
    inv = m.inv
    assert torch.allclose(inv.point, m.mapped_point, atol=ATOL)
    assert torch.allclose(inv.dense_jacobian(), torch.linalg.inv(J), atol=1e-9)
    assert torch.allclose(inv.jacobian_log_det, -m.jacobian_log_det, atol=1e-9)


def test_the_diagonal_is_read_off_the_full_form():
    m, J = _dense()
    assert m.is_dense
    diag = J.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(m.jacobian_diag, diag, atol=ATOL)
    assert torch.allclose(m.jacobian_log_diag, torch.log(diag.abs()), atol=ATOL)


def test_an_elementwise_chart_carries_its_diagonal():
    # The map is elementwise, so the diagonal is the Jacobian rather than a
    # cheaper reading of it, and a caller wanting a matrix is handed one.
    T = _affine()
    m = T.forward(torch.randn(6, 3))
    assert not m.is_dense
    assert torch.allclose(m.dense_jacobian(), torch.diag_embed(SIGMA.expand(6, 3)),
                          atol=ATOL)


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


def test_the_metric_is_the_reciprocal_square_of_the_forward_jacobian():
    # M = J^-T J^-1 is the gram of the inverse map, so its pushforward J^T M J is
    # the identity, which is what every sampler reads as the prior's metric in
    # the chart.
    T = _affine()
    m = T.forward(torch.randn(6, 3))
    M = T.inverse(m.mapped_point).gram()
    assert M.shape == (6, 3, 3)
    J = torch.diag_embed(m.jacobian_diag)
    assert torch.allclose(J.mT @ M @ J, torch.eye(3).expand(6, 3, 3), atol=ATOL)


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
#  log_prob                                                                   #
# --------------------------------------------------------------------------- #

def test_log_prob_is_the_density_of_the_pushed_forward_normal():
    theta = MU + SIGMA * torch.randn(9, 3)
    ref = torch.distributions.Normal(MU, SIGMA).log_prob(theta)
    assert torch.allclose(_affine().log_prob(theta), ref, atol=ATOL)


# --------------------------------------------------------------------------- #
#  Everything is supplied                                                     #
# --------------------------------------------------------------------------- #

def test_a_missing_jacobian_is_rejected():
    # Nothing is differentiated numerically, so a caller who leaves a Jacobian
    # out is told at construction rather than served an approximation.
    with pytest.raises(ValueError, match="both Jacobians"):
        NormalTransform(lambda z: (MU + SIGMA * z, None),
                        lambda th: ((th - MU) / SIGMA, torch.ones_like(th)),
                        reference=MU)


def test_a_missing_inverse_jacobian_is_rejected():
    with pytest.raises(ValueError, match="both Jacobians"):
        NormalTransform(lambda z: (MU + SIGMA * z, torch.ones_like(z)),
                        lambda th: ((th - MU) / SIGMA, None),
                        reference=MU)


def test_the_forward_map_is_differentiable_to_second_order():
    # Both directions are closed forms, so the graph reaches the input. The
    # force ChartRATTLE integrates differentiates the chart Jacobian, so it is
    # the second derivative that has to survive, on a chart that has one.
    T = NormalTransform(lambda z: (torch.exp(z), torch.exp(z)),
                        lambda th: (torch.log(th), 1.0 / th),
                        reference=torch.zeros(3))
    z = torch.randn(4, 3, requires_grad=True)
    (g,) = torch.autograd.grad(T.forward(z).mapped_point.sum(), z,
                               create_graph=True)
    (gg,) = torch.autograd.grad(g.sum(), z)
    assert torch.allclose(g, torch.exp(z).detach(), atol=ATOL)
    assert torch.allclose(gg, torch.exp(z).detach(), atol=ATOL)
