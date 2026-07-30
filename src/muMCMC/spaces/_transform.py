import math
from functools import cached_property

import torch

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _std_normal_log_pdf(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * z * z - _LOG_SQRT_2PI


def _std_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z * (0.5 ** 0.5)))


def _std_normal_icdf(u: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


class ElementwiseMap:
    """An elementwise map evaluated at a point, with its diagonal Jacobian.

    The log of the Jacobian diagonal is the stored form, because a saturating
    map has a diagonal that underflows while its logarithm stays finite.

    Parameters
    ----------
    point, mapped_point : Tensor, shape (..., d)
        The point the map was evaluated at and its image.
    log_jacobian_diag : Tensor, shape (..., d)
        Log of the Jacobian diagonal there, finite because the map is strictly
        increasing in each coordinate.
    """

    def __init__(
        self,
        point:              torch.Tensor,
        mapped_point:       torch.Tensor,
        log_jacobian_diag:  torch.Tensor,
    ):
        self._point             = point
        self._mapped_point      = mapped_point
        self._log_jacobian_diag = log_jacobian_diag

    @property
    def point(self) -> torch.Tensor:
        return self._point

    @property
    def mapped_point(self) -> torch.Tensor:
        return self._mapped_point

    @property
    def jacobian_log_diag(self) -> torch.Tensor:
        """Log of the diagonal of the Jacobian, shape ``(..., d)``."""
        return self._log_jacobian_diag

    @cached_property
    def jacobian_diag(self) -> torch.Tensor:
        """Diagonal of the Jacobian, shape ``(..., d)``."""
        return torch.exp(self._log_jacobian_diag)

    @cached_property
    def jacobian_log_det(self) -> torch.Tensor:
        """``log|det J|``, shape ``(...)``, summed over the coordinate axis."""
        return self._log_jacobian_diag.sum(dim=-1)

    @cached_property
    def inv(self) -> "ElementwiseMap":
        """The same map read backwards, so its Jacobian is the reciprocal."""
        return ElementwiseMap(self._mapped_point, self._point,
                              -self._log_jacobian_diag)

    def jvp(self, v: torch.Tensor) -> torch.Tensor:
        """``J v`` for a tangent ``v`` of shape ``(..., d)``."""
        return self.jacobian_diag * v


def _autograd_derivative(fn, x):
    """Elementwise ``d fn(x)_i / d x_i``, or None when the graph does not reach
    ``x``. Valid only because ``fn`` is elementwise, which makes one backward
    pass over the sum yield the whole diagonal."""
    try:
        with torch.enable_grad():
            x_leaf = x.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(fn(x_leaf).sum(), x_leaf,
                                       allow_unused=True)
    except RuntimeError:
        return None
    if g is None or not bool(torch.isfinite(g).all()):
        return None
    return g.detach()


def _fd_derivative(fn, x, order: int):
    """Elementwise ``d fn(x)_i / d x_i`` by a central difference of the given
    order, with the step scaled to the working precision."""
    h = torch.finfo(x.dtype).eps ** (1.0 / (order + 1))
    h = h * torch.clamp(x.abs(), min=1.0)
    with torch.no_grad():
        if order == 2:
            return (fn(x + h) - fn(x - h)) / (2.0 * h)
        return (-fn(x + 2.0 * h) + 8.0 * fn(x + h)
                - 8.0 * fn(x - h) + fn(x - 2.0 * h)) / (12.0 * h)


def _invert_increasing(fn, y, *, max_expand: int = 64, max_iter: int = 100):
    """Solve ``fn(x) = y`` elementwise for a strictly increasing ``fn``, by
    doubling a bracket around zero and then bisecting it."""
    lo = -torch.ones_like(y)
    hi = torch.ones_like(y)
    for _ in range(max_expand):
        low_hits = fn(lo) > y
        high_hits = fn(hi) < y
        if not bool((low_hits | high_hits).any()):
            break
        lo = torch.where(low_hits, 2.0 * lo, lo)
        hi = torch.where(high_hits, 2.0 * hi, hi)
    else:
        raise RuntimeError(
            "could not bracket the inverse of a transform. The value lies "
            "outside the image of the map, or the map is not increasing.")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        below = fn(mid) < y
        lo = torch.where(below, mid, lo)
        hi = torch.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _reattach(value, x, log_jac):
    """``value`` carrying the first derivative ``exp(log_jac)`` with respect to
    ``x``, for a value computed by a route autograd cannot follow. Returns it
    unchanged when the graph is already there or when none is wanted."""
    if not x.requires_grad or value.requires_grad:
        return value
    return value.detach() + (x - x.detach()) * torch.exp(log_jac).detach()


class NormalTransform:
    """``theta = T(z)``, elementwise and strictly increasing per coordinate,
    pushing ``z ~ N(0, I)`` forward to the prior on ``theta``.

    Each direction returns its value with the log of its Jacobian diagonal, so a
    caller supplies both from one pass over the shared intermediates. Anything
    left out is replaced: a missing log Jacobian by autograd, or by a central
    difference when the graph does not reach the input, a missing ``inverse`` by
    bisection, a missing ``log_prob`` by the inverse map. A replacement breaks
    the autograd graph, so the value carries its computed first derivative back
    to the input. Higher derivatives are then wrong, which is what
    :attr:`is_analytic` records.

    Parameters
    ----------
    forward : callable
        ``z -> (theta, log dtheta/dz)``, both of shape ``(..., d)``. The second
        may be None.
    inverse : callable, optional
        ``theta -> (z, log dz/dtheta)``, in the same shapes, second may be None.
    log_prob : callable, optional
        ``theta -> (..., d)``, the per-coordinate log-density of the prior.
    reference : Tensor, shape (d,)
        Parameter tensor fixing ``d``, the dtype and the device.
    fd_order : int
        Order of the central difference used when a derivative is neither given
        nor reachable by autograd.

    Attributes
    ----------
    is_analytic : bool
        Whether both directions and both Jacobians were given, so the map is
        differentiable to any order. Found by calling them once at ``z = 0``.
    interior_point : Tensor, shape (d,)
        ``T(0)``, a point in the support of every coordinate.
    """

    def __init__(self, forward, *, inverse=None, log_prob=None, reference,
                 fd_order: int = 4):
        self._forward_fn = forward
        self._inverse_fn = inverse
        self._log_prob_fn = log_prob
        self._reference = reference
        self.fd_order = fd_order

        theta, forward_jac = forward(torch.zeros_like(reference))
        self.interior_point = theta.detach()
        inverse_jac = None if inverse is None else inverse(theta)[1]
        self.is_analytic = forward_jac is not None and inverse_jac is not None

    @property
    def d(self) -> int:
        """Number of coordinates the transform spans."""
        return self._reference.shape[-1]

    @property
    def dtype(self) -> torch.dtype:
        return self._reference.dtype

    @property
    def device(self) -> torch.device:
        return self._reference.device

    def _log_derivative(self, fn, x):
        derivative = _autograd_derivative(fn, x)
        if derivative is None:
            derivative = _fd_derivative(fn, x, self.fd_order)
        return torch.log(derivative)

    def _invert(self, theta):
        if self._inverse_fn is not None:
            return self._inverse_fn(theta)
        return _invert_increasing(lambda z: self._forward_fn(z)[0], theta), None

    def forward(self, z: torch.Tensor) -> ElementwiseMap:
        """``theta = T(z)`` with ``dtheta/dz``, for ``z`` of shape ``(..., d)``."""
        theta, log_jac = self._forward_fn(z)
        if log_jac is None:
            log_jac = self._log_derivative(lambda t: self._forward_fn(t)[0], z)
        return ElementwiseMap(z, _reattach(theta, z, log_jac), log_jac)

    def inverse(self, theta: torch.Tensor) -> ElementwiseMap:
        """``z = T⁻¹(theta)`` with ``dz/dtheta``, for ``theta`` of shape
        ``(..., d)``."""
        z, log_jac = self._invert(theta)
        if log_jac is None:
            log_jac = self._log_derivative(lambda t: self._invert(t)[0], theta)
        return ElementwiseMap(theta, _reattach(z, theta, log_jac), log_jac)

    def metric(self, theta: torch.Tensor) -> torch.Tensor:
        """Diagonal of the prior's natural metric ``M = J⁻ᵀ J⁻¹`` at ``theta``,

            M_ii = (dz_i/dtheta_i)²,

        shape ``(..., d)``. It is the Jacobian of the inverse map that squares,
        so this is the reciprocal of ``(dtheta/dz)²``. Its pushforward to ``z``
        is the identity."""
        return self.inverse(theta).jacobian_diag ** 2

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-coordinate log-density of the prior at ``theta``,

            log p_i(theta_i) = log phi(z_i) + log(dz_i/dtheta_i),

        shape ``(..., d)``."""
        if self._log_prob_fn is not None:
            return self._log_prob_fn(theta)
        m = self.inverse(theta)
        return _std_normal_log_pdf(m.mapped_point) + m.jacobian_log_diag
