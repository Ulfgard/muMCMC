import math
from functools import cached_property

import torch

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# =========================================================================== #
#  The two forms of a Jacobian                                                #
#                                                                             #
#  A map carries its Jacobian as a diagonal where it is an elementwise map,   #
#  and as a matrix otherwise. The form is a statement about the map, not      #
#  about what the caller asked for: a diagonal is never the diagonal of a     #
#  Jacobian that has entries beside it, so both forms answer everything a     #
#  Jacobian is asked, each at its own cost.                                   #
#                                                                             #
#  A triangular Jacobian is carried as the matrix it is. Written on its       #
#  blocks the operations here would save a constant factor on an O(d^3) step  #
#  the sampler's own Cholesky already matches, so they take the general       #
#  route.                                                                     #
# =========================================================================== #


class Map:
    """A map evaluated at a point, with its Jacobian there.

    Parameters
    ----------
    point, mapped_point : Tensor, shape (..., d)
        The point the map was evaluated at and its image.
    jacobian : Tensor, shape (..., d) or (..., d, d)
        The Jacobian, as its diagonal where the map is elementwise and as a
        matrix, lower triangular, otherwise, told apart by how many axes it has.
        A diagonal says the Jacobian is diagonal, so both forms answer
        everything asked of one here and neither is a cheaper reading of the
        other.
    """

    def __init__(
        self,
        point:          torch.Tensor,
        mapped_point:   torch.Tensor,
        jacobian:       torch.Tensor,
    ):
        self._point        = point
        self._mapped_point = mapped_point
        self._jacobian     = jacobian

    @property
    def point(self) -> torch.Tensor:
        return self._point

    @property
    def mapped_point(self) -> torch.Tensor:
        return self._mapped_point

    @property
    def is_dense(self) -> bool:
        """Whether the Jacobian is carried in full rather than as its diagonal,
        which is one axis more."""
        return self._jacobian.dim() > self._point.dim()

    def dense_jacobian(self) -> torch.Tensor:
        """The Jacobian as a matrix, shape ``(..., d, d)``, a diagonal one
        embedded in one."""
        return (self._jacobian if self.is_dense
                else torch.diag_embed(self._jacobian))

    @cached_property
    def jacobian_diag(self) -> torch.Tensor:
        """Diagonal of the Jacobian, shape ``(..., d)``."""
        if not self.is_dense:
            return self._jacobian
        return self._jacobian.diagonal(dim1=-2, dim2=-1)

    @cached_property
    def jacobian_log_diag(self) -> torch.Tensor:
        """Log of the absolute diagonal of the Jacobian, shape ``(..., d)``."""
        return torch.log(self.jacobian_diag.abs())

    @cached_property
    def jacobian_log_det(self) -> torch.Tensor:
        """``log|det J|``, shape ``(...)``, summed over the coordinate axis."""
        # Triangular, so the off-diagonal entries do not enter.
        return self.jacobian_log_diag.sum(dim=-1)

    @cached_property
    def inv(self) -> "Map":
        """The same map read backwards, so its Jacobian is ``J⁻¹``."""
        return Map(self._mapped_point, self._point,
                   torch.linalg.inv(self._jacobian) if self.is_dense
                   else 1.0 / self._jacobian)

    def reshaped(self, point: torch.Tensor) -> "Map":
        """This map, built on one leading axis, read at ``point`` and given that
        point's leading axes."""
        batch = point.shape[:-1]
        width = self._jacobian.shape[self._point.dim() - 1:]
        return Map(point,
                   self._mapped_point.reshape(batch + self._mapped_point.shape[-1:]),
                   self._jacobian.reshape(batch + width))

    def jvp(self, v: torch.Tensor) -> torch.Tensor:
        """``J v`` for a tangent ``v`` of shape ``(..., d)``."""
        if not self.is_dense:
            return self._jacobian * v
        return (self._jacobian @ v[..., None])[..., 0]

    def pullback(self, W: torch.Tensor) -> torch.Tensor:
        """``W J`` for the covector rows of ``W``, shape ``(..., k, d)``, giving
        ``(..., k, d)``. Row ``i`` of the result is row ``i`` of ``W`` read in
        the coordinates the map starts from."""
        if not self.is_dense:
            return W * self._jacobian[..., None, :]
        return W @ self._jacobian

    def gram(self) -> torch.Tensor:
        """``Jᵀ J``, shape ``(..., d, d)``. Read on the inverse map, where
        ``J = dz/dtheta``, this is the prior's metric ``M = J⁻ᵀ J⁻¹`` on the
        variables."""
        if not self.is_dense:
            return torch.diag_embed(self._jacobian ** 2)
        return self._jacobian.transpose(-2, -1) @ self._jacobian


def log_prob_coordinates(inverse_map: Map) -> torch.Tensor:
    """Per-coordinate log-density of the prior a chart carries, read at its
    inverse map,

        log p_i = log phi(z_i) + log(dz_i/dtheta_i),

    shape ``(..., d)``. The factors are the chain rule of the joint density
    along the coordinate order, so they sum to it. They are marginals one by one
    only for a chart whose Jacobian is diagonal.
    """
    z = inverse_map.mapped_point
    return -0.5 * z * z - _LOG_SQRT_2PI + inverse_map.jacobian_log_diag


class NormalTransform:
    """``theta = T(z)``, elementwise and strictly increasing per coordinate,
    pushing ``z ~ N(0, I)`` forward to the prior on ``theta``.

    Both directions and both Jacobians are supplied by the caller, each
    direction returning its value with its Jacobian diagonal so that one pass
    over the shared intermediates gives both. Being elementwise, the map has the
    diagonal for its whole Jacobian. Nothing is differentiated numerically and
    nothing is inverted by a root find, so every quantity is exact,
    differentiable to any order, and costs one evaluation.

    Parameters
    ----------
    forward : callable
        ``z -> (theta, dtheta/dz)``, both of shape ``(..., d)``.
    inverse : callable
        ``theta -> (z, dz/dtheta)``, in the same shapes.
    reference : Tensor, shape (d,)
        Parameter tensor fixing ``d``, the dtype and the device.

    Attributes
    ----------
    interior_point : Tensor, shape (d,)
        ``T(0)``, a point in the support of every coordinate.

    Raises
    ------
    ValueError
        If either direction returns None in place of its Jacobian, checked by
        calling both once at ``z = 0``.
    """

    def __init__(self, forward, inverse, *, reference):
        self._forward_fn = forward
        self._inverse_fn = inverse
        self._reference = reference

        theta, forward_jac = forward(torch.zeros_like(reference))
        self.interior_point = theta.detach()
        if forward_jac is None or inverse(theta)[1] is None:
            raise ValueError(
                "a NormalTransform needs both directions and both Jacobians. "
                "Supply them in closed form.")

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

    def forward(self, z: torch.Tensor) -> Map:
        """``theta = T(z)`` with ``dtheta/dz``, for ``z`` of shape ``(..., d)``,
        the map being elementwise and so carrying its diagonal."""
        theta, diag = self._forward_fn(z)
        return Map(z, theta, diag)

    def inverse(self, theta: torch.Tensor) -> Map:
        """``z = T⁻¹(theta)`` with ``dz/dtheta``, for ``theta`` of shape
        ``(..., d)``."""
        z, diag = self._inverse_fn(theta)
        return Map(theta, z, diag)

    def forward_point(self, z: torch.Tensor) -> torch.Tensor:
        """``T(z)`` alone, for a caller with no use for a Jacobian. Both
        directions here give their own from the same pass, so this drops it
        rather than saving it."""
        return self._forward_fn(z)[0]

    def inverse_point(self, theta: torch.Tensor) -> torch.Tensor:
        """``T⁻¹(theta)`` alone, as in :meth:`forward_point`."""
        return self._inverse_fn(theta)[0]

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-coordinate log-density of the prior at ``theta``,

            log p_i(theta_i) = log phi(z_i) + log(dz_i/dtheta_i),

        shape ``(..., d)``. The inverse map already carries both terms, so this
        is one call of it."""
        return log_prob_coordinates(self.inverse(theta))
