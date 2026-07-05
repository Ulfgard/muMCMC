"""The implicit-midpoint residual Jacobian (building block for a Newton-type
inner solver) must equal the true Jacobian of the residual.  We check the
analytic block construction against a brute-force finite-difference Jacobian of
the exact residual, on a q-dependent metric (all blocks active) and a constant
metric (Hqq, Da vanish -- a clean floor)."""
import torch
import pytest

from muMCMC.RMHMC import (
    RMHMC, _midpoint_map, _implicit_midpoint_residual_jacobian,
)
from muMCMC.spaces import UnconstrainedSpace

torch.set_default_dtype(torch.float64)

D = 3


def model_qdep(theta):
    U = 0.5 * (theta ** 2).sum(-1)
    n = theta.shape[-1]
    G = torch.eye(n, dtype=theta.dtype) + 0.3 * theta[..., :, None] * theta[..., None, :]
    return U, G


def model_const(theta):
    U = 0.5 * (theta ** 2).sum(-1)
    n = theta.shape[-1]
    return U, torch.eye(n, dtype=theta.dtype).expand(*theta.shape[:-1], n, n)


def _evaluate_model(model_fn):
    space = UnconstrainedSpace([f"x{i}" for i in range(D)])
    s = RMHMC(model_fn, space, adapt_step_size=False)
    s.init(torch.zeros(1, D))
    return s.evaluate_model


def _residual(evaluate_model, q, p, eps, z):
    d = q.shape[-1]
    F_q, F_p = _midpoint_map(q, p, z[..., :d], z[..., d:], eps, evaluate_model)
    return z - torch.cat([F_q, F_p], dim=-1)


def _fd_jacobian(evaluate_model, q, p, eps, z, h=1e-6):
    twod = z.shape[-1]
    N = z.shape[0]
    J = torch.empty(N, twod, twod)
    for b in range(twod):
        e = torch.zeros(N, twod); e[:, b] = h
        J[:, :, b] = (_residual(evaluate_model, q, p, eps, z + e)
                      - _residual(evaluate_model, q, p, eps, z - e)) / (2 * h)
    return J


def _sample_inputs(N=4, seed=0):
    torch.manual_seed(seed)
    q = 0.5 * torch.randn(N, D)
    p = 0.5 * torch.randn(N, D)
    z = torch.cat([q + 0.1 * torch.randn(N, D), p + 0.1 * torch.randn(N, D)], dim=-1)
    eps = torch.rand(N) * 0.3 + 0.05          # per-chain step sizes
    return q, p, z, eps


@pytest.mark.parametrize("model_fn", [model_qdep, model_const])
def test_residual_jacobian_autodiff_matches_finite_difference(model_fn):
    """Exact second-order autodiff Jacobian must match the true (FD) Jacobian
    tightly -- the default path we want to use for the Newton corrector."""
    evaluate_model = _evaluate_model(model_fn)
    q, p, z, eps = _sample_inputs()

    J = _implicit_midpoint_residual_jacobian(
        q, p, eps, evaluate_model, z, force_hessian="autodiff")
    J_fd = _fd_jacobian(evaluate_model, q, p, eps, z)

    assert J.shape == (len(eps), 2 * D, 2 * D)
    assert torch.allclose(J, J_fd, atol=1e-7, rtol=1e-6)


@pytest.mark.parametrize("model_fn", [model_qdep, model_const])
@pytest.mark.parametrize("central", [False, True])
def test_residual_jacobian_fd_fallback_matches(model_fn, central):
    """The FD fallback for the force Hessian must also match the true Jacobian
    (looser, since one-sided FD is O(h))."""
    evaluate_model = _evaluate_model(model_fn)
    q, p, z, eps = _sample_inputs()

    J = _implicit_midpoint_residual_jacobian(
        q, p, eps, evaluate_model, z, force_hessian="fd", fd_central=central)
    J_fd = _fd_jacobian(evaluate_model, q, p, eps, z)

    atol = 1e-5 if not central else 1e-7
    assert torch.allclose(J, J_fd, atol=atol, rtol=1e-5)


def test_autodiff_and_fd_agree():
    """The exact and FD force-Hessian paths must produce the same Jacobian."""
    evaluate_model = _evaluate_model(model_qdep)
    q, p, z, eps = _sample_inputs()
    J_ad = _implicit_midpoint_residual_jacobian(q, p, eps, evaluate_model, z,
                                                force_hessian="autodiff")
    J_fd = _implicit_midpoint_residual_jacobian(q, p, eps, evaluate_model, z,
                                                force_hessian="fd", fd_central=True)
    assert torch.allclose(J_ad, J_fd, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("model_fn", [model_qdep, model_const])
@pytest.mark.parametrize("force_hessian", ["autodiff", "fd"])
def test_vectorized_matches_looped(model_fn, force_hessian):
    """The vectorized (is_grads_batched) Jacobian must equal the per-row looped
    one bit-for-bit up to float round-off."""
    evaluate_model = _evaluate_model(model_fn)
    q, p, z, eps = _sample_inputs()
    kw = dict(force_hessian=force_hessian, fd_central=True)
    J_vec = _implicit_midpoint_residual_jacobian(
        q, p, eps, evaluate_model, z, vectorized=True, **kw)
    J_loop = _implicit_midpoint_residual_jacobian(
        q, p, eps, evaluate_model, z, vectorized=False, **kw)
    assert torch.allclose(J_vec, J_loop, atol=1e-10, rtol=1e-8)


def test_residual_jacobian_is_detached():
    """No autograd graph escapes, even though the model's U/G carry one and Hqq
    uses a double backward internally."""
    evaluate_model = _evaluate_model(model_qdep)
    N = 2
    q = torch.randn(N, D); p = torch.randn(N, D)
    z = torch.cat([q, p], dim=-1)
    eps = torch.full((N,), 0.2)
    J = _implicit_midpoint_residual_jacobian(q, p, eps, evaluate_model, z)
    assert not J.requires_grad
