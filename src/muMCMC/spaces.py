import math
from functools import cached_property

import torch

# =========================================================================== #
#                                                                             #
#  The normal chart                                                           #
#                                                                             #
#  A space is a group of named variables carrying a prior. The prior is held  #
#  as a diffeomorphism of the standard normal,                                #
#                                                                             #
#      theta = T(z),   z ~ N(0, I),                                           #
#                                                                             #
#  elementwise and strictly increasing in each coordinate, and every sampler  #
#  runs in z. Bounds are not constraints here. A variable on (l, u) is a      #
#  Uniform prior whose T is the probit map, so the box lives in the prior     #
#  rather than in a separate transform.                                       #
#                                                                             #
#  Two identities follow, and they are why the chart is worth the indirection.#
#                                                                             #
#  The prior and the change of variables collapse. The temperature-free part  #
#  of the potential is U_base = -log p(theta) - log|det dtheta/dz|, and with  #
#  log p(theta) = log phi(z) - log|det dtheta/dz| that is                      #
#                                                                             #
#      U_base = -log phi(z) = ½‖z‖² + (d/2) log 2π,                           #
#                                                                             #
#  a closed form in z alone. The prior leaves the inner loop entirely.        #
#                                                                             #
#  The prior's metric is the identity. Its natural metric in theta is         #
#  M_theta = J^-T J^-1 with J = dtheta/dz, so its pushforward to z is         #
#  Jᵀ M_theta J = I. The same I is the exact Hessian of U_base, because the   #
#  prior read in its own chart is exactly N(0, I) rather than something       #
#  approximated by it. A scheme needing a constant prior metric therefore     #
#  needs no condition on the prior, which is what lets ChartRATTLE take one.  #
#                                                                             #
#  A prior is normalized by construction, being the pushforward of a          #
#  normalized density, so anything reading log p(x) as an evidence gets an    #
#  absolute value rather than one off by an unstated constant.                #
#                                                                             #
#  prior_log_prob is keyed on which free names appear in its argument. Every  #
#  free name gives the full log-prior, a subset gives the marginal over that  #
#  subset, which is the sum of just those factors because the prior           #
#  factorizes over the coordinate axis. A name left out by accident therefore #
#  returns a marginal instead of raising. That is the price of keeping the    #
#  interface a single method taking one dict.                                 #
#                                                                             #
#  UnnormalizedSpace is the one member with no chart. Its T is the identity   #
#  and its prior is absent, so z is theta, U_base is 0, and the prior metric  #
#  is None. An evidence or an entropy computed against it is not defined,     #
#  which is what its name records.                                            #
#                                                                             #
# =========================================================================== #

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def _std_normal_log_pdf(z: torch.Tensor) -> torch.Tensor:
    return -0.5 * z * z - _LOG_SQRT_2PI


def _std_normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z * (0.5 ** 0.5)))


def _std_normal_icdf(u: torch.Tensor) -> torch.Tensor:
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


class ElementwiseMap:
    """An elementwise map evaluated at a point, with its diagonal Jacobian.

    Carries the point, its image, and the log of the Jacobian diagonal. The log
    is the stored form because a saturating map has a diagonal that underflows
    while its logarithm stays finite.

    Parameters
    ----------
    point : Tensor, shape (..., d)
        Where the map was evaluated.
    mapped_point : Tensor, shape (..., d)
        The image of ``point``.
    log_jacobian_diag : Tensor, shape (..., d)
        Log of the Jacobian diagonal at ``point``, which is positive because the
        map is strictly increasing in each coordinate.
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


def _implements(dist, method: str) -> bool:
    """Whether ``dist`` gives ``method`` a body rather than inheriting the one
    that raises."""
    try:
        getattr(dist, method)(torch.zeros((), dtype=torch.get_default_dtype()))
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


def _support_chart(support):
    """An increasing bijection from the line onto ``support``, so that a search
    over the support runs as an unbounded one. The identity when the support is
    the whole line."""
    support = getattr(support, "base_constraint", support)
    lo = getattr(support, "lower_bound", None)
    hi = getattr(support, "upper_bound", None)
    if lo is None and hi is None:
        return lambda t: t
    if hi is None:
        return lambda t: lo + torch.exp(t)
    if lo is None:
        return lambda t: hi - torch.exp(-t)
    return lambda t: lo + (hi - lo) * torch.sigmoid(t)


class NormalTransform:
    """``theta = T(z)``, elementwise and strictly increasing per coordinate,
    pushing ``z ~ N(0, I)`` forward to the prior on ``theta``.

    The map is given as callables rather than by subclassing, so a space builds
    one by handing over the mathematics of its own prior. Each direction returns
    its value together with the log of its Jacobian diagonal, which is how a
    caller supplies both from one pass over the shared intermediates.

    Anything left out is replaced: a missing log Jacobian is differentiated by
    autograd, or by a central difference when the graph does not reach the
    input, a missing ``inverse`` is found by bisection, and a missing
    ``log_prob`` is assembled from the inverse map. A replaced route breaks the
    autograd graph, so the value carries the computed first derivative back to
    its input. First derivatives of the target are then right and higher ones
    are not, which is what :attr:`is_analytic` records.

    Parameters
    ----------
    forward : callable
        ``z -> (theta, log dtheta/dz)`` for ``z`` of shape ``(..., d)``, both
        entries of shape ``(..., d)``. The second may be None.
    inverse : callable, optional
        ``theta -> (z, log dz/dtheta)``, in the same shapes. The second entry may
        be None.
    log_prob : callable, optional
        ``theta -> (..., d)``, the per-coordinate log-density of the prior. Give
        it when a closed form beats going through ``inverse``.
    reference : Tensor, shape (d,)
        Parameter tensor fixing the coordinate count, the dtype and the device.
    fd_order : int
        Order of the central difference used when a derivative is neither given
        nor reachable by autograd.

    Attributes
    ----------
    is_analytic : bool
        Whether both directions and both Jacobians were given, so the map is
        differentiable to any order. Determined by evaluating the callables once
        at ``z = 0``.
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


def _distribution_transform(dist) -> NormalTransform:
    """The normal chart ``theta = F⁻¹(Phi(z))`` of a batched distribution.

    The Jacobian is analytic from the density, ``dtheta/dz = phi(z)/f(theta)``,
    so a distribution offering ``icdf``, ``cdf`` and ``log_prob`` needs no
    replacement. A missing ``icdf`` is replaced by bisection on ``cdf`` over the
    support, and a missing ``cdf`` by bisection on the forward map.
    """
    # Argument validation runs on a copy, so a bisection iterate sitting on the
    # boundary of the support does not raise on the way to the root.
    dist = dist.expand(dist.batch_shape)
    dist._validate_args = False
    chart = _support_chart(dist.support)
    has_icdf, has_cdf = _implements(dist, "icdf"), _implements(dist, "cdf")

    def forward(z):
        u = _std_normal_cdf(z)
        theta = (dist.icdf(u) if has_icdf
                 else chart(_invert_increasing(lambda t: dist.cdf(chart(t)), u)))
        return theta, _std_normal_log_pdf(z) - dist.log_prob(theta)

    def inverse(theta):
        z = _std_normal_icdf(dist.cdf(theta))
        return z, dist.log_prob(theta) - _std_normal_log_pdf(z)

    params = [getattr(dist, name) for name in dist.arg_constraints
              if torch.is_tensor(getattr(dist, name, None))]
    reference = (params[0].expand(dist.batch_shape) if params
                 else torch.zeros(dist.batch_shape))
    return NormalTransform(forward, inverse=inverse if has_cdf else None,
                           log_prob=dist.log_prob, reference=reference)


# =========================================================================== #
#  Tempered evaluation objects: metric and potential, affine in beta          #
# =========================================================================== #

def _solve_triangular_vec(triag_mat: torch.Tensor, vec: torch.Tensor, upper: bool):
    # triag_mat is (..., d, d) and vec is (..., d).
    return torch.linalg.solve_triangular(triag_mat, vec[..., None], upper=upper)[..., 0]


def broadcast_beta(beta, n_trailing: int):
    """``beta`` reshaped to broadcast over ``n_trailing`` trailing axes.

    A per-batch-element ``beta`` is a ``(N,)`` tensor that has to line up with a
    ``(N, *feat)`` quantity. A scalar (Python float or 0-d tensor) already
    broadcasts and is returned unchanged.
    """
    if torch.is_tensor(beta) and beta.ndim > 0:
        return beta.reshape((-1,) + (1,) * n_trailing)
    return beta


class TemperedAffine:
    """
    Quantity assembled affinely in an inverse temperature:

        value = beta * lik + base

    ``lik`` is the temperature-scaled (likelihood) part and ``base`` the
    temperature-free part (``None`` when absent).  ``lik`` and ``base`` share a
    leading batch axis and may carry further trailing feature axes, over which
    ``beta`` broadcasts.  ``beta`` is slot-bound: :meth:`select` and
    :meth:`reorder` mix or permute ``lik``/``base`` along the batch axis while
    leaving ``beta`` in place, so a moved configuration is retempered to its
    slot's temperature.

    Parameters
    ----------
    lik : Tensor, shape (N, *feat)
        Temperature-scaled part.
    base : Tensor, shape (N, *feat), or None
        Temperature-free part.
    beta : float or Tensor
        Inverse temperature scaling ``lik``.
    """

    def __init__(self, lik: torch.Tensor, base, beta):
        self._lik = lik
        self._base = base
        self._beta = beta

    # Read-only: ``value`` (and ``TemperedMetric.L``) are cached_property, so a
    # post-construction mutation of these inputs would silently return a stale
    # result. Retempering/mixing goes through reorder/select, which build fresh
    # objects instead of mutating in place.
    @property
    def lik(self):
        return self._lik

    @property
    def base(self):
        return self._base

    @property
    def beta(self):
        return self._beta

    def _beta_bcast(self):
        """``beta`` reshaped to broadcast over ``lik``'s trailing feature axes."""
        return broadcast_beta(self.beta, self.lik.dim() - 1)

    @cached_property
    def value(self) -> torch.Tensor:
        v = self._beta_bcast() * self.lik
        return v if self.base is None else v + self.base

    def select(self, mask: torch.Tensor, other: "TemperedAffine") -> "TemperedAffine":
        """This quantity where ``mask`` is True, ``other`` where False, per batch
        element.  Both share the same temperature."""
        m = mask.reshape(mask.shape + (1,) * (self.lik.dim() - mask.dim()))
        return type(self)(
            torch.where(m, self.lik, other.lik),
            None if self.base is None else torch.where(m, self.base, other.base),
            self.beta,
        )

    def reorder(self, perm: torch.Tensor) -> "TemperedAffine":
        """Permute the batch axis: row ``i`` of the result is row ``perm[i]``.
        ``beta`` is slot-bound and stays in place."""
        return type(self)(
            self.lik[perm],
            None if self.base is None else self.base[perm],
            self.beta,
        )


class TemperedMetric(TemperedAffine):
    """
    Metric ``G = beta * A_lik + A_prior`` in the normal chart, an ``(N, d, d)``
    SPD :attr:`value` whose operations are built from its Cholesky factor
    ``G = L Lᵀ``.

    ``A_lik`` (``lik``) is the likelihood metric pushed forward to the chart (see
    ``Space.push_forward_metric``). ``A_prior`` (``base``) is the prior's metric
    there, which is the identity for a space carrying a prior and ``None`` for
    one that does not.
    """

    @cached_property
    def L(self) -> torch.Tensor:
        """Lower-triangular Cholesky factor of :attr:`value`, positive diagonal."""
        return torch.linalg.cholesky(self.value)

    def inv_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """G⁻¹ v = L⁻ᵀ L⁻¹ v via two triangular solves."""
        return _solve_triangular_vec(
            self.L.transpose(-2, -1),
            _solve_triangular_vec(self.L, v, upper=False),
            upper=True,
        )

    def log_det_metric(self) -> torch.Tensor:
        """log det G = 2 Σ log|diag L|."""
        return 2.0 * self.L.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)

    def sample_momentum(self) -> torch.Tensor:
        """Sample p ~ N(0, G) via p = L ξ, ξ ~ N(0, I)."""
        xi = torch.randn(self.lik.shape[:-1], dtype=self.lik.dtype, device=self.lik.device)
        return (self.L @ xi[..., None])[..., 0].detach()


# =========================================================================== #
#  Spaces                                                                     #
# =========================================================================== #

class Space:
    """Named variables with a prior, read in the prior's normal chart.

    A subclass supplies :attr:`as_transform`, the map ``theta = T(z)`` over the
    free variables. Everything else here is the naming and the free/fixed
    bookkeeping, which is independent of which prior the group carries.

    Vectors come in two layouts. A full vector is ``(..., d_full)`` over
    :attr:`names` in order, which is what a model potential is handed. A free
    vector is ``(..., d)`` over :attr:`free_names` in the same relative order,
    which is what the transform and the prior act on. :meth:`to_full` and
    :meth:`to_free` move between them.

    Parameters
    ----------
    names : sequence of str
        Variable names, defining the full vector layout.
    fixed : dict[str, float] or None
        Names held at the given value. They keep their place in the full layout
        and are absent from the free one, so a fixed name may sit between two
        free ones. None or empty means nothing is fixed.

    Raises
    ------
    ValueError
        If a fixed name does not appear in ``names``.
    """

    def __init__(self, names, *, fixed=None):
        self.names = list(names)
        self.fixed = dict(fixed) if fixed else {}

        unknown = [name for name in self.fixed if name not in self.names]
        if unknown:
            raise ValueError(
                f"every fixed name must appear in names, got {unknown}")

        self._free_names = [name for name in self.names if name not in self.fixed]
        index = {name: i for i, name in enumerate(self.names)}
        self.free_indices = [index[name] for name in self._free_names]
        self.fixed_indices = [index[name] for name in self.fixed]
        self._template_cache = {}

    # ---- naming ------------------------------------------------------------ #

    @property
    def d(self) -> int:
        """Number of free variables."""
        return len(self._free_names)

    @property
    def d_full(self) -> int:
        """Number of variables, free and fixed."""
        return len(self.names)

    @property
    def free_names(self):
        return self._free_names

    @property
    def as_transform(self) -> NormalTransform:
        """The map ``theta = T(z)`` over the free variables."""
        raise NotImplementedError

    @property
    def is_proper(self) -> bool:
        """Whether the prior is a normalized density, and so has a normal chart.
        False for a space carrying no prior, whose evidence and entropy are then
        not defined."""
        return True

    # ---- vector layouts ---------------------------------------------------- #

    def _free_index(self, ref: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(self.free_indices, device=ref.device)

    def _template(self, ref: torch.Tensor) -> torch.Tensor:
        """``(d_full,)`` holding each fixed value at its place and zero
        elsewhere, cached per dtype and device."""
        key = (ref.dtype, ref.device)
        if key not in self._template_cache:
            t = torch.zeros(self.d_full, dtype=ref.dtype, device=ref.device)
            for name, value in self.fixed.items():
                t[self.names.index(name)] = value
            self._template_cache[key] = t
        return self._template_cache[key]

    def to_free(self, theta_full: torch.Tensor) -> torch.Tensor:
        """Full vector ``(..., d_full)`` to free vector ``(..., d)``."""
        if not self.fixed:
            return theta_full
        return theta_full.index_select(-1, self._free_index(theta_full))

    def to_full(self, theta_free: torch.Tensor) -> torch.Tensor:
        """Free vector ``(..., d)`` to full vector ``(..., d_full)``, with each
        fixed variable at its value."""
        if not self.fixed:
            return theta_free
        shape = theta_free.shape[:-1] + (self.d_full,)
        out = self._template(theta_free).expand(shape).clone()
        out[..., self._free_index(theta_free)] = theta_free
        return out

    def to_vector(self, samples: dict) -> torch.Tensor:
        """Dict keyed by name to a full vector ``(..., d_full)``. Fixed names
        absent from ``samples`` are filled in."""
        samples = self.add_fixed(samples)
        return torch.stack([samples[name] for name in self.names], dim=-1)

    def to_free_vector(self, samples: dict) -> torch.Tensor:
        """Dict keyed by name to a free vector ``(..., d)``."""
        return torch.stack([samples[name] for name in self._free_names], dim=-1)

    def from_full_vector(self, theta_full: torch.Tensor) -> dict:
        """Full vector ``(..., d_full)`` to a dict over every name."""
        self._check_width(theta_full, self.d_full, "full")
        return {name: theta_full[..., i] for i, name in enumerate(self.names)}

    def from_free_vector(self, theta_free: torch.Tensor) -> dict:
        """Free vector ``(..., d)`` to a dict over the free names."""
        self._check_width(theta_free, self.d, "free")
        return {name: theta_free[..., i]
                for i, name in enumerate(self._free_names)}

    def _check_width(self, vec, width, layout):
        if vec.shape[-1] != width:
            raise ValueError(
                f"expected a {layout} vector of size {width}, got "
                f"{vec.shape[-1]}.")

    # ---- fixed variables --------------------------------------------------- #

    def add_fixed(self, samples: dict) -> dict:
        """``samples`` with each fixed name added at its value, broadcast to the
        shape the other entries carry."""
        if not self.fixed:
            return samples
        samples = samples.copy()
        ref = next(iter(samples.values()))
        for name, value in self.fixed.items():
            samples[name] = torch.full(ref.shape, value,
                                       device=ref.device, dtype=ref.dtype)
        return samples

    def remove_fixed(self, samples: dict) -> dict:
        """``samples`` with every fixed name dropped."""
        if not self.fixed:
            return samples
        samples = samples.copy()
        for name in self.fixed:
            samples.pop(name, None)
        return samples

    # ---- prior ------------------------------------------------------------- #

    def prior_log_prob(self, y: dict) -> torch.Tensor:
        """Factorized log-prior over the free names present in ``y``.

        Passing every free name gives the full log-prior. Passing a subset gives
        the marginal log-prior over that subset, which is valid because the prior
        factorizes over the coordinate axis. A name left out of ``y`` therefore
        returns a marginal rather than raising. Fixed names in ``y`` are ignored.

        Parameters
        ----------
        y : dict[str, Tensor]
            Constrained points keyed by name, each of shape ``(...)``.

        Returns
        -------
        Tensor, shape (...)

        Raises
        ------
        ValueError
            If ``y`` holds none of the free names.
        """
        present = [i for i, name in enumerate(self._free_names) if name in y]
        if not present:
            raise ValueError("y contains none of the free parameter names")

        # A variable the marginal does not range over is held at T(0), which is
        # in the support and so keeps its factor finite before it is dropped.
        ref = y[self._free_names[present[0]]]
        interior = self.as_transform.interior_point.to(ref.dtype)
        theta = interior.expand(ref.shape + (self.d,)).clone()
        for i in present:
            theta[..., i] = y[self._free_names[i]]
        idx = torch.as_tensor(present, device=theta.device)
        return self.as_transform.log_prob(theta).index_select(-1, idx).sum(-1)

    def prior_log_prob_vector(self, theta_free: torch.Tensor) -> torch.Tensor:
        """Log-prior on a free vector ``(..., d)``, shape ``(...)``."""
        return self.as_transform.log_prob(theta_free).sum(-1)

    def prior_log_prob_normal(self, z_free: torch.Tensor) -> torch.Tensor:
        """Log-prior read in the normal chart, ``log N(z; 0, I)``, for ``z`` of
        shape ``(..., d)``. It equals ``prior_log_prob_vector(T(z))`` plus
        ``log|det dtheta/dz|``, so a caller needing both the prior and the change
        of variables takes this one instead."""
        return _std_normal_log_pdf(z_free).sum(-1)

    def prior_metric(self, theta_free: torch.Tensor) -> torch.Tensor:
        """Diagonal of the prior's natural metric at ``theta_free``, shape
        ``(..., d)``. Its pushforward to the normal chart is the identity, which
        is what :meth:`prior_metric_normal` returns."""
        return self.as_transform.metric(theta_free)

    def prior_metric_normal(self, z_free: torch.Tensor):
        """The prior's metric in the normal chart, the identity of shape
        ``(..., d, d)``, or None when the space carries no prior."""
        eye = torch.eye(self.d, dtype=z_free.dtype, device=z_free.device)
        return eye.expand(z_free.shape[:-1] + (self.d, self.d))

    def push_forward_metric(self, G: torch.Tensor,
                            jacobian_diag: torch.Tensor) -> torch.Tensor:
        """The constrained-space metric ``G`` read in the normal chart,

            G_z = diag(J) G_ff diag(J),   J = dtheta/dz,

        where ``G_ff`` is the free block of ``G``.

        Parameters
        ----------
        G : Tensor, shape (..., d_full, d_full)
            Metric in constrained coordinates over every variable.
        jacobian_diag : Tensor, shape (..., d)
            Jacobian diagonal of the ``z`` to ``theta`` map on the free block,
            from ``as_transform.forward(z).jacobian_diag``.

        Returns
        -------
        Tensor, shape (..., d, d)
        """
        idx = self._free_index(G)
        G_ff = G.index_select(-2, idx).index_select(-1, idx)
        return jacobian_diag[..., :, None] * G_ff * jacobian_diag[..., None, :]

    def sample(self, n_samples: int, *, generator=None) -> dict:
        """``n_samples`` draws from the prior, keyed by name and including the
        fixed variables, each of shape ``(n_samples,)``.

        Parameters
        ----------
        n_samples : int
            Number of draws.
        generator : torch.Generator, optional
            RNG driving the draw. The global RNG when omitted.
        """
        T = self.as_transform
        z = torch.randn(n_samples, self.d, dtype=T.dtype, device=T.device,
                        generator=generator)
        return self.add_fixed(self.from_free_vector(T.forward(z).mapped_point))


def _as_parameter(value, free_names, name, dtype, device):
    """A per-variable parameter as a ``(d,)`` tensor. A scalar is shared across
    the free variables and a dict is read in free-name order."""
    if isinstance(value, dict):
        missing = [n for n in free_names if n not in value]
        if missing:
            raise ValueError(f"{name} is missing an entry for {missing}")
        value = [value[n] for n in free_names]
    out = torch.as_tensor(value, dtype=dtype, device=device)
    if out.dim() == 0:
        out = out.expand(len(free_names))
    if out.shape != (len(free_names),):
        raise ValueError(
            f"{name} must be a scalar, a dict over the free names, or a tensor "
            f"of shape ({len(free_names)},), got shape {tuple(out.shape)}")
    return out.contiguous()


class NormalSpace(Space):
    """Variables with independent normal priors, ``theta_i ~ Normal(mu_i,
    sigma_i)``, whose normal chart is ``theta = mu + sigma z``.

    Parameters
    ----------
    names : sequence of str
        Variable names, defining the full vector layout.
    mu, sigma : float, dict[str, float] or Tensor
        Location and scale per free variable. A scalar is shared across them, a
        dict is keyed by name, and a tensor is read in free-name order.
        ``sigma`` is positive.
    fixed : dict[str, float] or None
        Names held at the given value.
    dtype, device : optional
        Working dtype and device of the prior parameters.

    Raises
    ------
    ValueError
        If a parameter is missing a free name or has the wrong shape.
    """

    def __init__(self, names, mu=0.0, sigma=1.0, *, fixed=None,
                 dtype=None, device=None):
        super().__init__(names, fixed=fixed)
        mu = _as_parameter(mu, self._free_names, "mu", dtype, device)
        sigma = _as_parameter(sigma, self._free_names, "sigma", dtype, device)
        log_sigma = torch.log(sigma)
        self._transform = NormalTransform(
            lambda z: (mu + sigma * z, log_sigma.expand_as(z)),
            inverse=lambda th: ((th - mu) / sigma, (-log_sigma).expand_as(th)),
            log_prob=lambda th: _std_normal_log_pdf((th - mu) / sigma) - log_sigma,
            reference=mu)

    @property
    def as_transform(self):
        return self._transform


class LogNormalSpace(Space):
    """Variables with independent log-normal priors, whose normal chart is
    ``theta = exp(mu + sigma z)``, so every variable is positive.

    Parameters
    ----------
    names : sequence of str
        Variable names, defining the full vector layout.
    mu, sigma : float, dict[str, float] or Tensor
        Location and scale of the underlying normal per free variable, in the
        forms ``NormalSpace`` takes. ``sigma`` is positive.
    fixed : dict[str, float] or None
        Names held at the given value.
    dtype, device : optional
        Working dtype and device of the prior parameters.
    """

    def __init__(self, names, mu=0.0, sigma=1.0, *, fixed=None,
                 dtype=None, device=None):
        super().__init__(names, fixed=fixed)
        mu = _as_parameter(mu, self._free_names, "mu", dtype, device)
        sigma = _as_parameter(sigma, self._free_names, "sigma", dtype, device)
        log_sigma = torch.log(sigma)

        def forward(z):
            w = mu + sigma * z
            return torch.exp(w), log_sigma + w

        def inverse(theta):
            log_theta = torch.log(theta)
            return (log_theta - mu) / sigma, -log_sigma - log_theta

        self._transform = NormalTransform(
            forward, inverse=inverse,
            log_prob=lambda th: (_std_normal_log_pdf((torch.log(th) - mu) / sigma)
                                 - log_sigma - torch.log(th)),
            reference=mu)

    @property
    def as_transform(self):
        return self._transform


class UniformSpace(Space):
    """Variables with independent uniform priors on ``(low, high)``, whose normal
    chart is the probit map ``theta = low + (high - low) Phi(z)``.

    A variable whose interval is degenerate, meaning its endpoints are equal, is
    fixed at that value instead of being given a prior.

    ``Phi`` saturates in floating point, so a ``z`` far from the origin maps to
    an endpoint exactly rather than to a point strictly inside. The log Jacobian
    stays exact there and goes to minus infinity, and the potential in ``z``
    grows as ``½‖z‖²``, so the region is unreachable rather than mishandled.

    Parameters
    ----------
    names : sequence of str
        Variable names, defining the full vector layout.
    low, high : float, dict[str, float] or Tensor
        Interval endpoints per free variable, in the forms ``NormalSpace`` takes.
    limits : dict[str, tuple[float, float]] or None
        Endpoints given as one dict of pairs, an alternative to ``low`` and
        ``high``. A name whose pair has equal entries becomes fixed.
    fixed : dict[str, float] or None
        Names held at the given value.
    dtype, device : optional
        Working dtype and device of the prior parameters.

    Raises
    ------
    ValueError
        If both ``limits`` and either endpoint argument are given, or if
        ``limits`` misses a name.
    """

    def __init__(self, names, low=None, high=None, *, limits=None, fixed=None,
                 dtype=None, device=None):
        names = list(names)
        if limits is not None:
            if low is not None or high is not None:
                raise ValueError("give either limits or low and high, not both")
            missing = [n for n in names if n not in limits]
            if missing:
                raise ValueError(f"limits is missing an entry for {missing}")
            fixed = dict(fixed) if fixed else {}
            for name in names:
                lo, hi = limits[name]
                if lo == hi:
                    fixed.setdefault(name, lo)
            low = {n: limits[n][0] for n in names if n not in fixed}
            high = {n: limits[n][1] for n in names if n not in fixed}

        super().__init__(names, fixed=fixed)
        low = _as_parameter(low, self._free_names, "low", dtype, device)
        high = _as_parameter(high, self._free_names, "high", dtype, device)
        log_width = torch.log(high - low)

        def inverse(theta):
            z = _std_normal_icdf((theta - low) / (high - low))
            return z, -log_width - _std_normal_log_pdf(z)

        self._transform = NormalTransform(
            lambda z: (low + (high - low) * _std_normal_cdf(z),
                       log_width + _std_normal_log_pdf(z)),
            inverse=inverse,
            log_prob=lambda th: (-log_width).expand_as(th),
            reference=low)

    @property
    def as_transform(self):
        return self._transform


class DistributionSpace(Space):
    """Variables with independent priors from one batched distribution, whose
    normal chart is ``theta = F⁻¹(Phi(z))``.

    Parameters
    ----------
    names : sequence of str
        Variable names, defining the full vector layout.
    dist : torch.distributions.Distribution
        Univariate distribution whose batch shape is ``(d,)``, one entry per free
        variable in free-name order. A batch shape of ``()`` is expanded to all
        of them.
    fixed : dict[str, float] or None
        Names held at the given value.

    Raises
    ------
    ValueError
        If the batch shape matches neither the free variables nor a single
        variable.
    """

    def __init__(self, names, dist, *, fixed=None):
        super().__init__(names, fixed=fixed)
        shape = tuple(dist.batch_shape)
        if shape == ():
            dist = dist.expand((self.d,))
        elif shape != (self.d,):
            raise ValueError(
                f"dist must have batch shape ({self.d},) or (), got {shape}")
        self._transform = _distribution_transform(dist)

    @property
    def as_transform(self):
        return self._transform


class UnnormalizedSpace(Space):
    """Variables with no prior, whose chart is the identity so that ``z`` is
    ``theta``.

    The target is whatever the model potential defines, with nothing added. An
    evidence, an entropy or a prior draw is not defined against it, and the
    methods computing those raise. A scheme requiring the prior's metric is not
    available either, since :meth:`prior_metric_normal` is None.

    Parameters
    ----------
    names : sequence of str
        Variable names, defining the full vector layout.
    fixed : dict[str, float] or None
        Names held at the given value.
    dtype, device : optional
        Working dtype and device.

    Raises
    ------
    ValueError
        From :meth:`prior_log_prob` and :meth:`sample`, which have no meaning
        without a prior.
    """

    def __init__(self, names, *, fixed=None, dtype=None, device=None):
        super().__init__(names, fixed=fixed)
        zero = torch.zeros(self.d, dtype=dtype, device=device)
        self._transform = NormalTransform(
            lambda z: (z, torch.zeros_like(z)),
            inverse=lambda th: (th, torch.zeros_like(th)),
            log_prob=torch.zeros_like,
            reference=zero)

    @property
    def as_transform(self):
        return self._transform

    @property
    def is_proper(self) -> bool:
        return False

    def prior_log_prob(self, y: dict) -> torch.Tensor:
        raise ValueError(
            "UnnormalizedSpace carries no prior, so prior_log_prob is not "
            "defined. Use a space with a prior if you need one.")

    def prior_log_prob_vector(self, theta_free: torch.Tensor) -> torch.Tensor:
        return torch.zeros(theta_free.shape[:-1], dtype=theta_free.dtype,
                           device=theta_free.device)

    def prior_log_prob_normal(self, z_free: torch.Tensor) -> torch.Tensor:
        return torch.zeros(z_free.shape[:-1], dtype=z_free.dtype,
                           device=z_free.device)

    def prior_metric_normal(self, z_free: torch.Tensor):
        return None

    def sample(self, n_samples: int, *, generator=None) -> dict:
        raise ValueError(
            "UnnormalizedSpace carries no prior, so it cannot be sampled from.")
