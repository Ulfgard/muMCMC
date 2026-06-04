"""Contract tests for ``TransformedMetric`` and the spaces' ``push_forward_metric``.

``TransformedMetric`` holds a position-dependent inverse metric in *decomposed*
form -- a Cholesky factor ``L`` plus a coordinate transform whose Jacobian ``J``
is applied only through jvp/vjp products -- and never materialises ``G_u`` or its
inverse.  The whole point is that the matrix-free products it exposes
(``metric_times_vec``, ``inv_metric_times_vec``, the square-root variants and
``log_det_metric``) agree with the dense linear algebra they stand in for.

We pin that down by building a dense reference for ``G_u = Jᵀ G_c J`` from a
diagonal Jacobian and a dense SPD ``G_c``, then checking every product against
it.  ``push_forward_metric`` is checked for the identity and box transforms,
including the fixed-coordinate projection.
"""
import torch
import pytest

from muMCMC.spaces import (
    ElementwiseTransform,
    TransformedMetric,
    UnconstrainedSpace,
    UniformBoxSpace,
    transforms,
)

torch.set_default_dtype(torch.float64)

ATOL = 1e-9


def _rand_spd(n, d, *, batch=True):
    """A (batched) SPD matrix and its lower-Cholesky factor."""
    A = torch.randn(n, d, d) if batch else torch.randn(d, d)
    G = A @ A.transpose(-2, -1) + d * torch.eye(d)
    L = torch.linalg.cholesky(G)
    return G, L


def _diag_transform(diag_J, p=None):
    if p is None:
        p = torch.zeros_like(diag_J)
    return ElementwiseTransform(
        p=p, p_prime=p,
        diag_J=diag_J,
        log_abs_det_J=diag_J.abs().log().sum(-1),
    )


def _dense_Gu(diag_J, G_c):
    """G_u = Jᵀ G_c J with J = diag(diag_J)."""
    J = torch.diag_embed(diag_J)
    return J.transpose(-2, -1) @ G_c @ J


def _matvec(M, v):
    return (M @ v[..., None])[..., 0]


# --------------------------------------------------------------------------- #
#  Matrix-free products vs. a dense Jᵀ G_c J reference                         #
# --------------------------------------------------------------------------- #

def test_inv_metric_times_vec_matches_dense():
    torch.manual_seed(0)
    n, d = 4, 3
    G_c, L = _rand_spd(n, d)
    diag_J = torch.rand(n, d) + 0.5
    tm = TransformedMetric(_diag_transform(diag_J), L)

    v = torch.randn(n, d)
    G_u = _dense_Gu(diag_J, G_c)
    G_u_inv = torch.linalg.inv(G_u)
    assert torch.allclose(tm.inv_metric_times_vec(v), _matvec(G_u_inv, v), atol=ATOL)


def test_metric_times_vec_matches_dense():
    torch.manual_seed(1)
    n, d = 4, 3
    G_c, L = _rand_spd(n, d)
    diag_J = torch.rand(n, d) + 0.5
    tm = TransformedMetric(_diag_transform(diag_J), L)

    v = torch.randn(n, d)
    G_u = _dense_Gu(diag_J, G_c)
    assert torch.allclose(tm.metric_times_vec(v), _matvec(G_u, v), atol=ATOL)


def test_metric_and_inverse_are_mutual_inverses():
    torch.manual_seed(2)
    n, d = 5, 4
    _, L = _rand_spd(n, d)
    diag_J = torch.rand(n, d) + 0.5
    tm = TransformedMetric(_diag_transform(diag_J), L)
    v = torch.randn(n, d)
    assert torch.allclose(tm.inv_metric_times_vec(tm.metric_times_vec(v)), v, atol=ATOL)
    assert torch.allclose(tm.metric_times_vec(tm.inv_metric_times_vec(v)), v, atol=ATOL)


def test_sqrt_metric_outer_product_is_metric():
    # (G_u^{1/2})(G_u^{1/2})^T == G_u, verified column-by-column.
    torch.manual_seed(3)
    n, d = 3, 4
    G_c, L = _rand_spd(n, d)
    diag_J = torch.rand(n, d) + 0.5
    tm = TransformedMetric(_diag_transform(diag_J), L)
    G_u = _dense_Gu(diag_J, G_c)

    # Build the dense sqrt by applying it to each basis vector.
    cols = []
    for i in range(d):
        e = torch.zeros(n, d)
        e[:, i] = 1.0
        cols.append(tm.sqrt_metric_times_vec(e))
    S = torch.stack(cols, dim=-1)            # (n, d, d), columns = sqrt @ e_i
    assert torch.allclose(S @ S.transpose(-2, -1), G_u, atol=ATOL)


def test_sqrt_and_inv_sqrt_round_trip():
    torch.manual_seed(4)
    n, d = 4, 3
    _, L = _rand_spd(n, d)
    diag_J = torch.rand(n, d) + 0.5
    tm = TransformedMetric(_diag_transform(diag_J), L)
    v = torch.randn(n, d)
    assert torch.allclose(tm.inv_sqrt_metric_times_vec(tm.sqrt_metric_times_vec(v)),
                          v, atol=ATOL)


def test_log_det_metric_matches_dense():
    torch.manual_seed(5)
    n, d = 4, 3
    G_c, L = _rand_spd(n, d)
    diag_J = torch.rand(n, d) + 0.5
    tm = TransformedMetric(_diag_transform(diag_J), L)
    G_u = _dense_Gu(diag_J, G_c)
    assert torch.allclose(tm.log_det_metric(),
                          torch.logdet(G_u), atol=ATOL)


def test_gc_inv_times_vec_matches_dense():
    torch.manual_seed(6)
    n, d = 4, 3
    G_c, L = _rand_spd(n, d)
    tm = TransformedMetric(_diag_transform(torch.ones(n, d)), L)
    v = torch.randn(n, d)
    G_c_inv = torch.linalg.inv(G_c)
    assert torch.allclose(tm.Gc_inv_times_vec(v), _matvec(G_c_inv, v), atol=ATOL)


def test_sample_momentum_shape():
    torch.manual_seed(7)
    n, d = 6, 3
    _, L = _rand_spd(n, d)
    tm = TransformedMetric(_diag_transform(torch.ones(n, d)), L)
    p = tm.sample_momentum()
    assert p.shape == (n, d)
    assert not p.requires_grad


# --------------------------------------------------------------------------- #
#  Per-chain select / reorder                                                 #
# --------------------------------------------------------------------------- #

def test_select_picks_per_chain_metric():
    torch.manual_seed(12)
    n, d = 2, 3
    _, La = _rand_spd(n, d)
    _, Lb = _rand_spd(n, d)
    ta = TransformedMetric(_diag_transform(torch.rand(n, d) + 0.5, p=torch.zeros(n, d)), La)
    tb = TransformedMetric(_diag_transform(torch.rand(n, d) + 0.5, p=torch.ones(n, d)), Lb)
    mask = torch.tensor([True, False])
    c = ta.select(mask, tb)
    v = torch.randn(n, d)
    out = c.inv_metric_times_vec(v)
    out_a = ta.inv_metric_times_vec(v)
    out_b = tb.inv_metric_times_vec(v)
    assert torch.allclose(out[0], out_a[0], atol=ATOL)
    assert torch.allclose(out[1], out_b[1], atol=ATOL)
    # log_det recomputed consistently from the mixed L
    assert torch.allclose(c.log_det_metric()[0], ta.log_det_metric()[0], atol=ATOL)
    assert torch.allclose(c.log_det_metric()[1], tb.log_det_metric()[1], atol=ATOL)


def test_reorder_permutes_chains():
    torch.manual_seed(13)
    n, d = 3, 3
    _, L = _rand_spd(n, d)
    tm = TransformedMetric(_diag_transform(torch.rand(n, d) + 0.5,
                                           p=torch.arange(n * d, dtype=torch.float64).reshape(n, d)), L)
    perm = torch.tensor([2, 0, 1])
    r = tm.reorder(perm)
    assert torch.allclose(r.log_det_metric(), tm.log_det_metric()[perm], atol=ATOL)
    # Row i of r uses original chain perm[i].  To compare against the original
    # metric we feed it an input aligned to the original chain ordering: with
    # v_orig[perm] = v, tm.inv(v_orig)[perm] reproduces r.inv(v) row by row.
    v = torch.randn(n, d)
    v_orig = torch.empty_like(v)
    v_orig[perm] = v
    ref = tm.inv_metric_times_vec(v_orig)[perm]
    assert torch.allclose(r.inv_metric_times_vec(v), ref, atol=ATOL)


# --------------------------------------------------------------------------- #
#  push_forward_metric on the spaces                                          #
# --------------------------------------------------------------------------- #

def test_push_forward_identity_space_recovers_G_inverse():
    # Identity transform: G_u == G_c, so inv_metric_times_vec == G^{-1} v.
    torch.manual_seed(14)
    n, d = 4, 3
    s = UnconstrainedSpace(["a", "b", "c"])
    G, _ = _rand_spd(n, d)
    theta = torch.randn(n, d)
    tm = s.push_forward_metric(theta, G)
    v = torch.randn(n, d)
    G_inv = torch.linalg.inv(G)
    assert torch.allclose(tm.inv_metric_times_vec(v), _matvec(G_inv, v), atol=ATOL)
    assert torch.allclose(tm.log_det_metric(), torch.logdet(G), atol=ATOL)


def test_push_forward_accepts_precomputed_cholesky():
    torch.manual_seed(15)
    n, d = 3, 3
    s = UnconstrainedSpace(["a", "b", "c"])
    G, L = _rand_spd(n, d)
    theta = torch.randn(n, d)
    tm_G = s.push_forward_metric(theta, G)
    tm_L = s.push_forward_metric(theta, L, G_is_lower_cholesky=True)
    v = torch.randn(n, d)
    assert torch.allclose(tm_G.inv_metric_times_vec(v),
                          tm_L.inv_metric_times_vec(v), atol=ATOL)


def test_push_forward_projects_out_trailing_fixed():
    # With c fixed (trailing), the pushed-forward metric must equal the
    # free-block metric G[:2, :2] under the identity transform.
    torch.manual_seed(16)
    n = 4
    s = UnconstrainedSpace(["a", "b", "c"], fixed={"c": 0.0})
    G, _ = _rand_spd(n, 3)
    theta_full = torch.randn(n, 3)
    tm = s.push_forward_metric(theta_full, G)
    v = torch.randn(n, 2)                     # free dimension is 2
    G_free = G[:, :2, :2]
    G_free_inv = torch.linalg.inv(G_free)
    assert torch.allclose(tm.inv_metric_times_vec(v), _matvec(G_free_inv, v), atol=ATOL)
    assert torch.allclose(tm.log_det_metric(), torch.logdet(G_free), atol=ATOL)


def test_push_forward_box_matches_dense_pullback():
    # Box transform: J is diagonal = box jacobian; check inv-metric vs dense.
    torch.manual_seed(17)
    n, d = 4, 2
    limits = {"x": (-1.0, 1.0), "y": (0.0, 4.0)}
    s = UniformBoxSpace(limits, ["x", "y"], device="cpu")
    theta = torch.stack([
        torch.empty(n).uniform_(-0.9, 0.9),
        torch.empty(n).uniform_(0.1, 3.9),
    ], dim=-1)
    G, _ = _rand_spd(n, d)
    tm = s.push_forward_metric(theta, G)

    # Dense reference: J = diag of the z->theta jacobian at these points.
    theta_map = s.map_to_unconstrained_vector(theta).inv   # z -> theta
    diag_J = theta_map.jvp(torch.ones(n, d))
    G_u = _dense_Gu(diag_J, G)
    v = torch.randn(n, d)
    G_u_inv = torch.linalg.inv(G_u)
    assert torch.allclose(tm.inv_metric_times_vec(v), _matvec(G_u_inv, v), atol=ATOL)
