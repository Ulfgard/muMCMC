import math
from functools import cached_property

import torch

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


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


class NormalTransform:
    """``theta = T(z)``, elementwise and strictly increasing per coordinate,
    pushing ``z ~ N(0, I)`` forward to the prior on ``theta``.

    Both directions and both Jacobians are supplied by the caller, each
    direction returning its value with the log of its Jacobian diagonal so that
    one pass over the shared intermediates gives both. Nothing is differentiated
    numerically and nothing is inverted by a root find, so every quantity is
    exact, differentiable to any order, and costs one evaluation.

    Parameters
    ----------
    forward : callable
        ``z -> (theta, log dtheta/dz)``, both of shape ``(..., d)``.
    inverse : callable
        ``theta -> (z, log dz/dtheta)``, in the same shapes.
    reference : Tensor, shape (d,)
        Parameter tensor fixing ``d``, the dtype and the device.

    Attributes
    ----------
    interior_point : Tensor, shape (d,)
        ``T(0)``, a point in the support of every coordinate.

    Raises
    ------
    ValueError
        If either direction returns None in place of its log Jacobian, checked
        by calling both once at ``z = 0``.
    """

    def __init__(self, forward, inverse, *, reference):
        self._forward_fn = forward
        self._inverse_fn = inverse
        self._reference = reference

        theta, forward_jac = forward(torch.zeros_like(reference))
        self.interior_point = theta.detach()
        if forward_jac is None or inverse(theta)[1] is None:
            raise ValueError(
                "a NormalTransform needs both directions and both log "
                "Jacobians. Supply them in closed form.")

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

    def forward(self, z: torch.Tensor) -> ElementwiseMap:
        """``theta = T(z)`` with ``dtheta/dz``, for ``z`` of shape ``(..., d)``."""
        return ElementwiseMap(z, *self._forward_fn(z))

    def inverse(self, theta: torch.Tensor) -> ElementwiseMap:
        """``z = T⁻¹(theta)`` with ``dz/dtheta``, for ``theta`` of shape
        ``(..., d)``."""
        return ElementwiseMap(theta, *self._inverse_fn(theta))

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

        shape ``(..., d)``. The inverse map already carries both terms, so this
        is one call of it."""
        m = self.inverse(theta)
        z = m.mapped_point
        return -0.5 * z * z - _LOG_SQRT_2PI + m.jacobian_log_diag
