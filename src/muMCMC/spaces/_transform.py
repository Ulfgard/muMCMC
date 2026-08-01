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
        """The inverse map evaluated at :attr:`mapped_point`. The map is
        elementwise, so its Jacobian diagonal is the reciprocal of this one."""
        return ElementwiseMap(self._mapped_point, self._point,
                              -self._log_jacobian_diag)

    def jvp(self, v: torch.Tensor) -> torch.Tensor:
        """``J v`` for a tangent ``v`` of shape ``(..., d)``."""
        return self.jacobian_diag * v


class NormalTransform:
    """``theta = T(z)``, elementwise and strictly increasing per coordinate,
    pushing ``z ~ N(0, I)`` forward to the prior on ``theta``.

    The caller supplies both directions, each returning its value together with
    the log of its Jacobian diagonal. All four are required in closed form.
    ``inverse`` must be the exact inverse of ``forward``, and each log Jacobian
    must be that of the map it is returned beside. Neither condition is checked,
    and every result on this class is wrong if one of them fails.

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
        From the constructor, if either direction returns None in place of its
        log Jacobian.
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
        """Number of coordinates the map acts on."""
        return self._reference.shape[-1]

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the reference tensor, which the map is evaluated in."""
        return self._reference.dtype

    @property
    def device(self) -> torch.device:
        """Device of the reference tensor, which the map is evaluated on."""
        return self._reference.device

    def forward(self, z: torch.Tensor) -> ElementwiseMap:
        """``theta = T(z)`` with ``dtheta/dz``, for ``z`` of shape ``(..., d)``."""
        return ElementwiseMap(z, *self._forward_fn(z))

    def inverse(self, theta: torch.Tensor) -> ElementwiseMap:
        """``z = T⁻¹(theta)`` with ``dz/dtheta``, for ``theta`` of shape
        ``(..., d)``."""
        return ElementwiseMap(theta, *self._inverse_fn(theta))

    def metric(self, theta: torch.Tensor) -> torch.Tensor:
        """Diagonal of the metric the prior induces on ``theta``,
        ``M = J⁻ᵀ J⁻¹`` with ``J = dtheta/dz``, so

            M_ii = (dz_i/dtheta_i)²,

        of shape ``(..., d)`` for ``theta`` of shape ``(..., d)``. Its
        pushforward along the map is the identity."""
        return self.inverse(theta).jacobian_diag ** 2

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-coordinate log-density of the prior at ``theta``,

            log p_i(theta_i) = log phi(z_i) + log(dz_i/dtheta_i),

        where ``z = T⁻¹(theta)`` and ``phi`` is the standard normal density.
        Shape ``(..., d)`` for ``theta`` of shape ``(..., d)``."""
        m = self.inverse(theta)
        z = m.mapped_point
        return -0.5 * z * z - _LOG_SQRT_2PI + m.jacobian_log_diag
