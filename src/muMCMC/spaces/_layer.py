from typing import Callable

import torch


class ConditionalLayer:
    """A bijection ``y = phi_a(eps)`` in its latent, conditioned on ``a``.

    Each direction comes in two forms. The plain one returns its value alone.
    The ``_with_jvp`` one returns its value with both Jacobians,

        A = d phi_a(eps) / da,        B = d phi_a(eps) / d eps,
        W = -d phi_a^-1(y) / da,      B^-1 = d phi_a^-1(y) / dy,

    each direction differentiating the value it returns. ``A = B W`` relates the
    two conditions, and the latent Jacobian is triangular, so a caller wanting
    ``log|det B|`` sums the log of its diagonal. Both forms give everything they
    return from one pass over the shared intermediates.

    The ``_with_jvp`` defaults take one reverse pass per latent coordinate for
    each Jacobian, so ``a`` and ``eps`` must require grad. Override them to
    return either from explicit or forward-mode Jacobians, and set
    :attr:`jvp_needs_grad` False when the override needs no grad mode.

    ``a`` is ``(N, n)``, ``y`` and ``eps`` are ``(N, m)``, ``A`` and ``W`` are
    ``(N, m, n)`` and ``B`` is ``(N, m, m)``. There is exactly one leading axis,
    which a chart reading a layer guarantees, so a network written for a batch
    of points is one.
    """

    # Whether the _with_jvp directions need grad mode enabled around them. True
    # for the defaults, which differentiate. A subclass returning its Jacobians
    # from explicit or forward-mode ones should set it False, which spares the
    # graph build on every call, and that is once per iteration under the newton
    # solver.
    jvp_needs_grad = True

    def forward(self, a: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        """``y`` at ``(a, eps)``, shape ``(N, m)``. Implemented by a
        subclass."""
        raise NotImplementedError

    def inverse(self, a: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``eps`` at ``(a, y)``, shape ``(N, m)``. Implemented by a
        subclass."""
        raise NotImplementedError

    def forward_with_jvp(self, a: torch.Tensor, eps: torch.Tensor):
        """``(y, A, B)`` at ``(a, eps)``."""
        y = self.forward(a, eps)
        return y, self._jacobian(y, a), self._jacobian(y, eps)

    def inverse_with_jvp(self, a: torch.Tensor, y: torch.Tensor):
        """``(eps, W, B⁻¹)`` at ``(a, y)``."""
        eps = self.inverse(a, y)
        return eps, -self._jacobian(eps, a), self._jacobian(eps, y)

    @staticmethod
    def _jacobian(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """``d out/dx``, one reverse pass per coordinate of ``out``, which must
        carry a graph reaching ``x``."""
        rows = [torch.autograd.grad(out[..., i].sum(), x, retain_graph=True,
                                    create_graph=True)[0]
                for i in range(out.shape[-1])]
        return torch.stack(rows, dim=-2)


class LocationScaleLayer(ConditionalLayer):
    """A conditionally-Gaussian layer ``y | a ~ N(mu(a), Sigma(a))``,

        phi_a(eps) = mu(a) + L(a) eps,   Sigma = L Lᵀ,

    so ``phi_a^-1(y) = L(a)^-1 (y - mu(a))`` and the latent Jacobian is ``L``.
    The two latent Jacobians are the factor and its inverse, so only the
    derivative in ``a`` is differentiated.

    Parameters
    ----------
    location_scale : callable
        ``a -> (mu, L)`` of shapes ``(N, n)`` to ``(N, m)`` and ``(N, m, m)``,
        with ``L`` lower triangular and positive on its diagonal. It is called
        once per evaluation, so one pass of a network may produce both.
        Differentiable in ``a``.
    """

    def __init__(self, location_scale: Callable):
        self.location_scale = location_scale

    @classmethod
    def from_covariance(cls, mean: Callable, cov: Callable) -> "LocationScaleLayer":
        """The layer from a mean ``a -> mu`` of shape ``(N, m)`` and a
        covariance ``a -> Sigma`` of shape ``(N, m, m)``, SPD, factorized on
        every evaluation. A ``cov`` that is not numerically SPD raises."""
        return cls(lambda a: (mean(a), torch.linalg.cholesky(cov(a))))

    def forward(self, a, eps):
        mu, L = self.location_scale(a)
        return mu + (L @ eps.unsqueeze(-1)).squeeze(-1)

    def inverse(self, a, y):
        mu, L = self.location_scale(a)
        return _solve(L, y - mu)

    def forward_with_jvp(self, a, eps):
        mu, L = self.location_scale(a)
        y = mu + (L @ eps.unsqueeze(-1)).squeeze(-1)
        return y, self._jacobian(y, a), L

    def inverse_with_jvp(self, a, y):
        mu, L = self.location_scale(a)
        eps = _solve(L, y - mu)
        eye = torch.eye(L.shape[-1], dtype=L.dtype, device=L.device).expand_as(L)
        return eps, -self._jacobian(eps, a), torch.linalg.solve_triangular(
            L, eye, upper=False)


def _solve(L: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.linalg.solve_triangular(L, v.unsqueeze(-1), upper=False).squeeze(-1)
