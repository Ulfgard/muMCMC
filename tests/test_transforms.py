"""Contract tests for ``ElementwiseTransform`` and the ``transforms`` factory.

The transforms are the geometric core every space leans on: they expose the
mapped point, the diagonal Jacobian (``jvp`` / ``jacobian_diag``) and its
log-determinant, and the inverse transform, and the samplers and
``push_forward_metric`` trust those to match the actual analytic Jacobian.
These tests pin that contract down -- the Jacobian, its log-det, inverse round
trips, and batched vs. unbatched shapes.
"""
import math

import torch
import pytest

from muMCMC.spaces import ElementwiseTransform, transforms

torch.set_default_dtype(torch.float64)

ATOL = 1e-10


def _make(diag_J, *, p=None):
    """Build an ElementwiseTransform with a given diagonal Jacobian.

    ``p``/``p_prime`` are arbitrary here -- the Jacobian operations only depend
    on ``diag_J`` -- so we fill them with simple stand-ins and set
    ``log_abs_det_J`` to the value the factory helpers would produce.
    """
    if p is None:
        p = torch.zeros_like(diag_J)
    return ElementwiseTransform(
        p=p,
        p_prime=p + 1.0,
        diag_J=diag_J,
        log_abs_det_J=diag_J.abs().log().sum(-1),
    )


# --------------------------------------------------------------------------- #
#  ElementwiseTransform: Jacobian / inverse contract                          #
# --------------------------------------------------------------------------- #

def test_jvp_is_elementwise_scaling():
    diag = torch.tensor([2.0, 0.5, 3.0])
    t = _make(diag)
    v = torch.tensor([1.0, -4.0, 2.0])
    assert torch.allclose(t.jvp(v), diag * v, atol=ATOL)
    assert torch.allclose(t.jacobian_diag, diag, atol=ATOL)


def test_jacobian_log_det_property():
    diag = torch.tensor([2.0, 0.5, 3.0])
    t = _make(diag)
    assert torch.allclose(t.jacobian_log_det, diag.log().sum(-1), atol=ATOL)


def test_mapped_point_and_p_properties():
    p = torch.tensor([1.0, 2.0, 3.0])
    t = ElementwiseTransform(p=p, p_prime=p + 5.0, diag_J=torch.ones(3),
                             log_abs_det_J=torch.zeros(()))
    assert torch.equal(t.p, p)
    assert torch.equal(t.mapped_point, p + 5.0)


def test_inv_swaps_endpoints_and_negates_log_det():
    diag = torch.tensor([2.0, 0.5, 3.0])
    p = torch.tensor([0.1, 0.2, 0.3])
    t = _make(diag, p=p)
    ti = t.inv
    # endpoints swap
    assert torch.equal(ti.p, t.mapped_point)
    assert torch.equal(ti.mapped_point, t.p)
    # diagonal inverts, log-det negates
    assert torch.allclose(ti.jvp(torch.ones(3)), 1.0 / diag, atol=ATOL)
    assert torch.allclose(ti.jacobian_log_det, -t.jacobian_log_det, atol=ATOL)
    # inverse Jacobian is elementwise division
    v = torch.randn(3)
    assert torch.allclose(ti.jvp(v), v / diag, atol=ATOL)


def test_inv_of_inv_round_trips():
    diag = torch.tensor([2.0, 0.5, 3.0])
    t = _make(diag, p=torch.tensor([0.1, 0.2, 0.3]))
    tii = t.inv.inv
    v = torch.randn(3)
    assert torch.allclose(tii.jvp(v), t.jvp(v), atol=ATOL)
    assert torch.allclose(tii.jacobian_log_det, t.jacobian_log_det, atol=ATOL)


# --------------------------------------------------------------------------- #
#  transforms.identity                                                        #
# --------------------------------------------------------------------------- #

def test_identity_is_a_no_op():
    p = torch.tensor([1.0, -2.0, 3.0])
    t = transforms.identity(p)
    assert torch.equal(t.mapped_point, p)
    v = torch.randn(3)
    assert torch.allclose(t.jvp(v), v, atol=ATOL)


def test_identity_log_det_shapes():
    # 1d input -> scalar log-det
    t1 = transforms.identity(torch.randn(4))
    assert t1.jacobian_log_det.shape == ()
    assert torch.allclose(t1.jacobian_log_det, torch.zeros(()), atol=ATOL)
    # 2d (batched) input -> per-chain log-det
    t2 = transforms.identity(torch.randn(5, 4))
    assert t2.jacobian_log_det.shape == (5,)
    assert torch.allclose(t2.jacobian_log_det, torch.zeros(5), atol=ATOL)


# --------------------------------------------------------------------------- #
#  transforms.box / box_inv                                                   #
# --------------------------------------------------------------------------- #

def test_box_maps_into_open_interval():
    l = torch.tensor([-1.0, 0.0])
    u = torch.tensor([1.0, 10.0])
    z = torch.tensor([[0.0, 0.0], [5.0, -5.0], [-3.0, 2.0]])
    theta = transforms.box(z, l, u).mapped_point
    assert torch.all(theta > l)
    assert torch.all(theta < u)
    # z == 0 lands at the midpoint
    assert torch.allclose(theta[0], (l + u) / 2.0, atol=ATOL)


def test_box_round_trip_both_directions():
    l = torch.tensor([-2.0, 1.0, 0.0])
    u = torch.tensor([3.0, 4.0, 10.0])
    z = torch.tensor([[0.3, -1.2, 0.7], [1.5, 0.2, -0.4]])
    theta = transforms.box(z, l, u).mapped_point
    z_back = transforms.box_inv(theta, l, u).mapped_point
    assert torch.allclose(z_back, z, atol=1e-9)
    # and starting from a constrained point
    theta0 = torch.tensor([[0.0, 2.0, 5.0], [2.5, 3.0, 1.0]])
    z0 = transforms.box_inv(theta0, l, u).mapped_point
    theta_back = transforms.box(z0, l, u).mapped_point
    assert torch.allclose(theta_back, theta0, atol=1e-9)


def test_box_jacobian_matches_autograd():
    l = torch.tensor([-2.0, 1.0])
    u = torch.tensor([3.0, 4.0])
    z = torch.tensor([0.4, -0.7], requires_grad=True)
    theta = transforms.box(z, l, u).mapped_point
    # diagonal Jacobian: d theta_i / d z_i
    J = torch.autograd.functional.jacobian(
        lambda zz: transforms.box(zz, l, u).mapped_point, z.detach())
    diag_J_auto = J.diagonal()
    t = transforms.box(z.detach(), l, u)
    assert torch.allclose(t.jvp(torch.ones(2)), diag_J_auto, atol=1e-9)
    # log|det J| equals sum of log of the diagonal entries
    assert torch.allclose(t.jacobian_log_det,
                          diag_J_auto.abs().log().sum(), atol=1e-9)


def test_box_inv_is_inverse_transform_object():
    l = torch.tensor([0.0])
    u = torch.tensor([1.0])
    z = torch.tensor([[0.5], [-0.5]])
    fwd = transforms.box(z, l, u)
    inv = transforms.box_inv(fwd.mapped_point, l, u)
    # box_inv returns the z->theta... err theta->z map; its mapped_point is z
    assert torch.allclose(inv.mapped_point, z, atol=1e-9)
    # log-dets are negatives of each other (inverse transform)
    assert torch.allclose(inv.jacobian_log_det, -fwd.jacobian_log_det, atol=1e-9)


def test_box_accepts_scalar_limits():
    # atleast_1d in the factory should let scalar l/u broadcast.
    l = torch.tensor(-1.0)
    u = torch.tensor(1.0)
    z = torch.tensor([0.0, 0.6])
    theta = transforms.box(z, l, u).mapped_point
    assert theta.shape == (2,)
    assert torch.all((theta > -1.0) & (theta < 1.0))
