import torch

from ._space import Space
from ._transform import Map, log_prob_coordinates


class ConditionalTransform:
    """``theta = (theta_A, theta_B)`` at ``z = (z_A, eps)``,

        theta_A = T_A(z_A),
        theta_B = phi_{theta_A}(eps),

    pushing ``z ~ N(0, I)`` forward to ``p(theta_A) p(theta_B | theta_A)``, the
    trailing block's conditional being ``N(0, I)`` pushed through the layer.

    The leading coordinates are those of ``base`` in its own order and the
    trailing ones the layer's, so a free vector is the two blocks concatenated.

    Parameters
    ----------
    base : NormalTransform
        The chart ``T_A`` over the leading block.
    layer : ConditionalLayer
        The bijection ``phi_a``, read at the leading block.
    d_b : int
        Number of coordinates the layer spans.

    Attributes
    ----------
    interior_point : Tensor, shape (d,)
        ``T(0)``, the base chart's interior point followed by the layer's image
        of zero there.

    """

    def __init__(self, base, layer, d_b):
        self._base = base
        self._layer = layer
        self._d_b = d_b

        zero = torch.zeros(1, d_b, dtype=base.dtype, device=base.device)
        theta_b = layer.forward(base.interior_point.unsqueeze(0), zero)
        self.interior_point = torch.cat(
            [base.interior_point, theta_b[0].detach()], dim=-1)

    @property
    def d(self) -> int:
        """Number of coordinates the chart spans."""
        return self._base.d + self._d_b

    @property
    def dtype(self) -> torch.dtype:
        return self._base.dtype

    @property
    def device(self) -> torch.device:
        return self._base.device

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``theta = T(z)`` for ``z`` of shape ``(..., d)``.

        The layer is read for its value, which is one pass where its Jacobians
        are one more per coordinate of the trailing block, so this is the path a
        solve iterating on the map itself takes.
        """
        d_a = self._base.d
        z, shape = z.reshape(-1, z.shape[-1]), z.shape
        theta_a = self._base.forward(z[..., :d_a])
        theta_b = self._layer.forward(theta_a, z[..., d_a:])
        return torch.cat([theta_a, theta_b], dim=-1).reshape(shape)

    def inverse(self, theta: torch.Tensor) -> torch.Tensor:
        """``z = T⁻¹(theta)`` for ``theta`` of shape ``(..., d)``, as in
        :meth:`forward`."""
        d_a = self._base.d
        theta, shape = theta.reshape(-1, theta.shape[-1]), theta.shape
        z_a = self._base.inverse(theta[..., :d_a])
        eps = self._layer.inverse(theta[..., :d_a], theta[..., d_a:])
        return torch.cat([z_a, eps], dim=-1).reshape(shape)

    def forward_with_jvp(self, z: torch.Tensor) -> Map:
        """``theta = T(z)`` with ``dtheta/dz``.

        The Jacobian is triangular rather than diagonal, so the map carries it
        as a matrix and the layer is read for both of its own Jacobians.
        """
        return self._differentiated(z, self._forward_at).reshaped(z)

    def inverse_with_jvp(self, theta: torch.Tensor) -> Map:
        """``z = T⁻¹(theta)`` with ``dz/dtheta``, as a matrix for the reason in
        :meth:`forward_with_jvp`."""
        return self._differentiated(theta, self._inverse_at).reshaped(theta)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Per-coordinate log-density of the prior at ``theta``, shape
        ``(..., d)``, the chain rule of ``p(theta_A) p(theta_B | theta_A)`` along
        the coordinate order. The factors sum to the joint."""
        return log_prob_coordinates(self.inverse_with_jvp(theta))

    # ---- the blocks --------------------------------------------------------- #

    def _forward_at(self, z) -> Map:
        """The forward map at ``z``, carrying ``[[J_A, 0], [A J_A, B]]``.

        The trailing block reaches ``z_A`` only through ``theta_A``, so its
        coupling is the layer's derivative carried through the leading chart's
        own Jacobian, whatever that is.
        """
        d_a = self._base.d
        m_a = self._base.forward_with_jvp(z[..., :d_a])
        theta_b, A, B = self._layer.forward_with_jvp(m_a.mapped_point,
                                                     z[..., d_a:])
        return Map(z, torch.cat([m_a.mapped_point, theta_b], dim=-1),
                   _block(m_a.dense_jacobian(), m_a.pullback(A), B))

    def _inverse_at(self, theta) -> Map:
        """The inverse map at ``theta``, carrying ``[[J_A⁻¹, 0], [-W, B⁻¹]]``.

        The latent reaches ``theta_A`` directly rather than through the leading
        chart, so its coupling is the layer's own.
        """
        d_a = self._base.d
        m_a = self._base.inverse_with_jvp(theta[..., :d_a])
        eps, W, B_inv = self._layer.inverse_with_jvp(theta[..., :d_a],
                                                     theta[..., d_a:])
        return Map(theta, torch.cat([m_a.mapped_point, eps], dim=-1),
                   _block(m_a.dense_jacobian(), -W, B_inv))

    def _differentiated(self, x, build) -> Map:
        """``build(x)`` under the grad mode the layer's Jacobians need.

        The layer takes one leading batch axis, so the point is flattened to
        that first. A layer that differentiates its own value needs an input
        carrying a graph. Where the caller supplies one the result stays attached to it,
        which is what a scheme differentiating the chart Jacobian needs, and
        where it does not the map is built on a graph of its own and returned
        detached.
        """
        x = x.reshape(-1, x.shape[-1])
        if not self._layer.jvp_needs_grad or (
                x.requires_grad and torch.is_grad_enabled()):
            return build(x)
        with torch.enable_grad():
            m = build(x.detach().requires_grad_(True))
        return Map(x, m.mapped_point.detach(), m.dense_jacobian().detach())


def _block(leading, cross, trailing) -> torch.Tensor:
    """``[[J, 0], [C, S]]`` of shape ``(..., d, d)`` from the leading block
    ``J``, the coupling ``C`` and the trailing block ``S``."""
    d_a, d_b = leading.shape[-1], trailing.shape[-1]
    batch = torch.broadcast_shapes(leading.shape[:-2], cross.shape[:-2],
                                   trailing.shape[:-2])
    lead = torch.cat([leading.expand(batch + (d_a, d_a)),
                      torch.zeros(batch + (d_a, d_b), dtype=leading.dtype,
                                  device=leading.device)], dim=-1)
    trail = torch.cat([cross.expand(batch + (d_b, d_a)),
                       trailing.expand(batch + (d_b, d_b))], dim=-1)
    return torch.cat([lead, trail], dim=-2)


class ConditionalSpace(Space):
    """A space in two blocks, ``theta_A`` with a prior of its own and
    ``theta_B`` given by a layer read at ``theta_A``.

    The names of ``base`` lead the full layout and the names given here follow
    them, in the layer's own order. The prior is
    ``p(theta_A) p(theta_B | theta_A)``, which does not factorize over the two
    blocks, so it has no marginal over a subset of the names, where a prior over
    independent variables has one for every subset.

    Only ``base`` holds variables fixed. A coordinate of ``theta_B`` cannot be
    fixed, because holding one at a value conditions the rest of the block
    rather than leaving them alone.

    Parameters
    ----------
    base : Space
        The space over ``theta_A``, carrying its names, its prior and whatever
        it holds fixed.
    names : sequence of str
        The names of ``theta_B``, disjoint from the names of ``base``.
    layer : ConditionalLayer
        The bijection giving ``theta_B`` from a standard normal latent. It is
        read at the full leading vector, so a fixed variable reaches it at its
        value.

    Raises
    ------
    ValueError
        If a name repeats one of ``base``. A base with no prior has no chart to
        build on, and says so when this reads it.
    """

    def __init__(self, base, names, layer):
        names = list(names)
        repeated = [name for name in names if name in base.names]
        if repeated:
            raise ValueError(
                f"the names of the conditional block must be new, got {repeated}")
        super().__init__(list(base.names) + names, fixed=base.fixed)
        self._base = base
        self._transform = ConditionalTransform(
            base.as_transform,
            layer if not base.fixed else _AtFullVector(layer, base),
            len(names))

    @property
    def base(self):
        """The space over the leading block ``theta_A``."""
        return self._base

    @property
    def as_transform(self):
        return self._transform

    def set_fixed(self, fixed: dict) -> None:
        """Hold the same names fixed at new values, in place, here and in the
        base space, which is where they live.

        Raises
        ------
        ValueError
            If ``fixed`` does not name exactly the variables already fixed.
        """
        super().set_fixed(fixed)
        self._base.set_fixed(fixed)

    def prior_log_prob(self, theta: dict) -> torch.Tensor:
        """Joint log-prior at ``theta``, a dict of points on the variables of
        shape ``(...)`` holding every free name, giving ``(...)``. Fixed names
        are ignored.

        This prior serves no marginals, where one over a subset of the names is
        what :class:`Space` returns for a subset. The leading block's own
        marginals are :attr:`base`'s, and a marginal that ranges over the
        trailing block has no closed form at all.

        Raises
        ------
        ValueError
            If ``theta`` is missing a free name.
        """
        missing = [name for name in self.free_names if name not in theta]
        if missing:
            raise ValueError(
                f"the trailing block's prior is conditional on the leading one, "
                f"so this prior has no marginal to return and needs every free "
                f"name, missing {missing}. The leading block's marginals are "
                f"on base.")
        return self.prior_log_prob_vector(self.to_free_vector(theta))


class _AtFullVector:
    """A layer read at the full leading vector while the chart works in free
    coordinates.

    A fixed variable is part of ``theta_A``, so the layer is handed it at its
    value. Its derivative in the fixed coordinates is dropped, which is the
    chain rule through the widening, that being a selection.
    """

    def __init__(self, layer, base):
        self._layer = layer
        self._base = base
        self.jvp_needs_grad = layer.jvp_needs_grad

    def forward(self, a, eps):
        return self._layer.forward(self._base.to_full(a), eps)

    def inverse(self, a, y):
        return self._layer.inverse(self._base.to_full(a), y)

    def forward_with_jvp(self, a, eps):
        y, A, B = self._layer.forward_with_jvp(self._base.to_full(a), eps)
        return y, self._free_columns(A), B

    def inverse_with_jvp(self, a, y):
        eps, W, B_inv = self._layer.inverse_with_jvp(self._base.to_full(a), y)
        return eps, self._free_columns(W), B_inv

    def _free_columns(self, J):
        idx = torch.as_tensor(self._base.free_indices, device=J.device)
        return J.index_select(-1, idx)
