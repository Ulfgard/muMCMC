"""
Parameter spaces and their transforms.

Each space exposes ``prior_log_prob_vector``, operating on flat free vectors,
and ``prior_metric``, returning the constrained-space metric contribution of
the prior as a ``(d_full, d_full)`` SPD tensor, or None when there is no
contribution.

``push_forward_metric`` pushes a constrained-space metric forward to the free
unconstrained coordinates.  ``TemperedAffine`` holds a quantity affinely in an
inverse temperature.

Prior contract
--------------
The prior ``p(y)`` is assumed to (a) **factorize** over the parameter names and
(b) be a **normalized** density over the free coordinates. Both are relied on by
anything that reads ``log p(x) = log ∫ p(x|y) p(y) dy`` as an evidence.
Factorization lets a marginal drop the integrated-out names cleanly. A missing
normalizer shifts the evidence by exactly that constant.

Factorization also fixes the marginal interface. ``prior_log_prob`` is keyed on
*which* free names are present in its argument. Passing every free name returns
the full log-prior. Passing a subset returns the marginal log-prior over that
subset, the sum of just those factors. The footgun is that a name accidentally
dropped from the argument silently yields a marginal rather than an error, an
accepted cost of keeping the interface a single dict-in method.

A space normalizes any prior it *defines itself*. For example the implicit
uniform of a bounded box contributes ``-log(u_i - l_i)`` per coordinate.
User-supplied per-name priors are taken as given. The caller is responsible for
their normalization, and in particular an explicit prior is **not** renormalized
for truncation to a box. Detecting an unnormalized prior is out of scope for the
current interface, so it is a documented precondition rather than a checked one.
"""

from functools import cached_property

import torch


class ElementwiseTransform:
    """
    Elementwise transform p' = T(p) with diagonal Jacobian.

    Exposes the mapped point, the (diagonal) Jacobian and its log-determinant,
    and the inverse transform.
    """

    def __init__(
        self,
        p:              torch.Tensor,
        p_prime:        torch.Tensor,
        diag_J:         torch.Tensor,
        log_abs_det_J:  torch.Tensor,
    ):
        self._p             = p
        self._p_prime       = p_prime
        self._diag_J        = diag_J
        self._log_abs_det_J = log_abs_det_J

    @property
    def mapped_point(self) -> torch.Tensor:
        return self._p_prime

    @property
    def p(self) -> torch.Tensor:
        return self._p

    @cached_property
    def inv(self) -> "ElementwiseTransform":
        return ElementwiseTransform(
            p             = self._p_prime,
            p_prime       = self._p,
            diag_J        = 1.0 / self._diag_J,
            log_abs_det_J = -self._log_abs_det_J,
        )

    @property
    def jacobian_log_det(self) -> torch.Tensor:
        return self._log_abs_det_J

    @property
    def jacobian_diag(self) -> torch.Tensor:
        """Diagonal of the Jacobian dp'/dp."""
        return self._diag_J

    def jvp(self, v: torch.Tensor) -> torch.Tensor:
        return self._diag_J * v


class transforms:

    @staticmethod
    def identity(p: torch.Tensor) -> ElementwiseTransform:
        shape = (p.shape[0],) if p.dim() == 2 else ()
        return ElementwiseTransform(
            p             = p,
            p_prime       = p,
            diag_J        = torch.ones_like(p),
            log_abs_det_J = torch.zeros(shape, device=p.device, dtype=p.dtype),
        )

    @staticmethod
    def _box(p, p_prime, l, u):
        """Box <-> unconstrained tanh transform.

            p'            = (u+l)/2 + (u-l)/2 * tanh(p)
            p             = atanh( 2 (p' - (u+l)/2) / (u-l) )
            d p'/d p      = (u-l)/2 * sech^2(p)
            log|d p'/d p| = log((u-l)/2) - 2 log|cosh(p)|
        """
        scale = (u - l) / 2.0
        log_diag_J = torch.log(scale) - 2.0 * torch.log(torch.cosh(p))
        return ElementwiseTransform(
            p             = p,
            p_prime       = p_prime,
            diag_J        = torch.exp(log_diag_J),
            log_abs_det_J = log_diag_J.sum(dim=-1),
        )

    @staticmethod
    def box(p: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> ElementwiseTransform:
        """Unconstrained p -> constrained p' = (u+l)/2 + (u-l)/2 * tanh(p)."""
        l, u = torch.atleast_1d(l), torch.atleast_1d(u)
        center = (u + l) / 2.0
        half_range = (u - l) / 2.0
        p_prime = center + half_range * torch.tanh(p)
        return transforms._box(p, p_prime, l, u)

    @staticmethod
    def box_inv(p_prime: torch.Tensor, l: torch.Tensor, u: torch.Tensor) -> ElementwiseTransform:
        """Constrained p' in (l, u) -> unconstrained p = atanh(2(p'-c)/(u-l))."""
        l, u = torch.atleast_1d(l), torch.atleast_1d(u)
        center = (u + l) / 2.0
        half_range = (u - l) / 2.0
        p = torch.atanh((p_prime - center) / half_range)
        return transforms._box(p, p_prime, l, u).inv


# ====================================================================== #
#  Tempered evaluation objects: metric and potential, affine in beta     #
# ====================================================================== #

def _solve_triangular_vec(triag_mat: torch.Tensor, vec: torch.Tensor, upper: bool):
    # triag_mat is (..., d, d) and vec is (..., d).
    return torch.linalg.solve_triangular(triag_mat, vec[..., None], upper=upper)[..., 0]


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
        beta = self.beta
        if torch.is_tensor(beta) and beta.ndim > 0:
            beta = beta.reshape((-1,) + (1,) * (self.lik.dim() - 1))
        return beta

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
    Free-space metric ``G = beta * A_lik + A_prior``, an ``(N, d, d)`` SPD
    :attr:`value` whose operations are built from its Cholesky factor ``G = L Lᵀ``.

    ``A_lik`` and ``A_prior`` (``lik`` and ``base``) are the likelihood and prior
    metrics pushed forward to free unconstrained coordinates (see
    ``space.push_forward_metric``). ``A_prior`` is ``None`` when the prior
    contributes no metric.
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


# ====================================================================== #
#  Spaces                                                                #
# ====================================================================== #
 


class UnconstrainedSpace:
    def __init__(self, names, priors=None, *, prior_metric_fn=None, fixed=None):
        """
        Parameters
        ----------
        names : sequence of str
            Parameter names (full / ambient ordering).
        priors : dict[str, distribution] or None
            Per-name priors.  When None, prior_log_prob is unavailable and
            prior_log_prob_vector returns zeros.
        prior_metric_fn : callable or None
            Optional function returning the prior's metric contribution in
            constrained coords as a (d_full, d_full) SPD tensor, accessed via
            ``prior_metric``.  Defaults to None (no contribution).
        fixed : dict[str, float] or None
            Names to hold fixed at the given value.  Removed from the free
            space.  None or empty means nothing fixed.
        """
        self.names = list(names)
        self.priors = priors
        self.prior_metric_fn = prior_metric_fn
        self.fixed = dict(fixed) if fixed else {}

        if self.priors is not None:
            if not all(name in priors for name in names):
                raise ValueError("priors must either be None or have one element for each name in names")
        if not all(name in self.names for name in self.fixed):
            raise ValueError("every fixed name must appear in names")

        self._free_names = [yi for yi in self.names if yi not in self.fixed]
        name_to_idx = {yi: i for i, yi in enumerate(self.names)}
        self.free_indices = [name_to_idx[yi] for yi in self._free_names]
        self.fixed_indices = [name_to_idx[yi] for yi in self.fixed]

    @property
    def d(self) -> int:
        return len(self._free_names)

    @property
    def d_full(self) -> int:
        return len(self.names)

    @property
    def free_names(self):
        return self._free_names

    def to_free_vector(self, samples):
        return torch.stack([samples[yi] for yi in self._free_names], dim=-1)

    def from_vector(self, vec):
        n = vec.shape[-1]
        if n == self.d:
            return {yi: vec[..., i] for i, yi in enumerate(self._free_names)}
        elif n == self.d_full:
            return {yi: vec[..., idx] for yi, idx in zip(self._free_names, self.free_indices)}
        else:
            raise ValueError(
                f"Expected vector of size {self.d} (free) or "
                f"{self.d_full} (full), got {n}."
            )

    def map_to_unconstrained_vector(self, theta_vec):
        if theta_vec.shape[-1] > self.d:
            theta_vec = theta_vec[..., self.free_indices]
        return transforms.identity(theta_vec)

    def map_to_constrained_vector(self, z_vec):
        return transforms.identity(z_vec)

    def prior_log_prob(self, y):
        """Factorized log-prior over the free names present in ``y``.

        Passing every free name gives the full log-prior. Passing a subset gives
        the marginal log-prior over that subset, valid because the prior
        factorizes over names. Footgun: a name accidentally dropped from ``y``
        silently yields a marginal instead of raising."""
        if self.priors is None:
            raise ValueError("Unconstrained space without priors does not allow for prior_log_prob to be computed")
        names = [yi for yi in self._free_names if yi in y]
        if not names:
            raise ValueError("y contains none of the free parameter names")
        result = 0
        for yi in names:
            result = result + self.priors[yi].log_prob(y[yi]).squeeze(-1)
        return result

    def prior_log_prob_vector(self, theta_free):
        """Prior log-density on a free vector, zero if no prior is configured."""
        if self.priors is None:
            return torch.zeros(theta_free.shape[:-1], device=theta_free.device, dtype=theta_free.dtype)
        return self.prior_log_prob(self.from_vector(theta_free))

    def prior_metric(self, theta_full):
        """Constrained-space prior metric, or None when not configured."""
        if self.prior_metric_fn is None:
            return None
        return self.prior_metric_fn(theta_full)
        
    def push_forward_metric(self, G, theta_map):
        """G_free = dJ · G_ff · dJ, diagonal Jacobian ``dJ = dθ/dz`` on the free
        block ``G_ff``.

        Parameters
        ----------
        G : Tensor, shape (N, d_full, d_full)
            Constrained-space metric.
        theta_map : the z->θ map (``map_to_constrained_vector``).

        Returns
        -------
        Tensor, shape (N, d, d)
        """
        dJ = theta_map.jacobian_diag                        # (N, d) = dθ/dz, free coords
        fi = torch.as_tensor(self.free_indices, device=G.device)
        G_ff = G.index_select(-2, fi).index_select(-1, fi)  # (N, d, d)
        return dJ[..., :, None] * G_ff * dJ[..., None, :]

    def sample(self, n_samples):
        if self.priors is None:
            raise ValueError("Unconstrained space without priors cannot be sampled from")
        samples = {}
        for yi in self._free_names:
            # Per-name prior is univariate. reshape normalises a trailing
            # singleton and rejects a multivariate prior.
            samples[yi] = self.priors[yi].sample([n_samples]).reshape(n_samples)
        return self.add_fixed(samples)

    def remove_fixed(self, samples):
        if not self.fixed:
            return samples
        samples = samples.copy()
        for yi in self.fixed.keys():
            samples.pop(yi, None)
        return samples

    def add_fixed(self, samples):
        if not self.fixed:
            return samples
        samples = samples.copy()
        ref = next(iter(samples.values()))
        for yi, val in self.fixed.items():
            samples[yi] = val * torch.ones(ref.shape, device=ref.device, dtype=ref.dtype)
        return samples

    def point_inside(self, y):
        return True

    def to_vector(self, samples):
        samples = self.add_fixed(samples)
        point = []
        for yi in self.names:
            point.append(samples[yi])
        return torch.stack(point, axis=-1)


class UniformBoxSpace:
    # Maximum resampling rounds for the rejection sampler in ``sample`` when
    # per-name priors are supplied.
    _MAX_REJECTION_ROUNDS = 100

    def __init__(self, limits, names, device, priors=None, *, prior_metric_fn=None):
        self.names = names
        self.priors = priors if priors is not None else {}
        self.prior_metric_fn = prior_metric_fn
        self.fixed = {}

        self.l = []
        self.u = []
        for yi in self.names:
            min_val = limits[yi][0]
            max_val = limits[yi][1]

            if abs(min_val - max_val) < 1.e-15:
                self.fixed[yi] = min_val
                continue
            self.l.append(min_val)
            self.u.append(max_val)

        self.l = torch.tensor(self.l, device=device)
        self.u = torch.tensor(self.u, device=device)
        self.free_names = [yi for yi in self.names if yi not in self.fixed]
        self.d = len(self.free_names)
        self.d_full = len(self.names)

        name_to_idx = {yi: i for i, yi in enumerate(self.names)}
        self.free_indices = [name_to_idx[yi] for yi in self.free_names]
        self.fixed_indices = [name_to_idx[yi] for yi in self.fixed]

        # Per-coordinate normalizer of the box prior. A free coordinate with no
        # explicit prior is uniform on [l_i, u_i] and contributes -log(u_i - l_i)
        # to its normalized log-density. Kept per name so a marginal prior over a
        # subset of names sums only the provided coordinates' constants, and so
        # the full prior integrates to 1 over the box (a well-defined evidence
        # needs this). A constant offset in the potential does not affect
        # sampling. Explicit priors are taken as given and carry no entry.
        self._uniform_log_norm = {
            yi: float(-torch.log(self.u[i] - self.l[i]))
            for i, yi in enumerate(self.free_names)
            if yi not in self.priors
        }

    def to_free_vector(self, samples):
        return torch.stack([samples[yi] for yi in self.free_names], dim=-1)

    def from_vector(self, vec):
        n = vec.shape[-1]
        if n == self.d:
            return {yi: vec[..., i] for i, yi in enumerate(self.free_names)}
        elif n == self.d_full:
            return {yi: vec[..., idx] for yi, idx in zip(self.free_names, self.free_indices)}
        else:
            raise ValueError(
                f"Expected vector of size {self.d} (free) or "
                f"{self.d_full} (full), got {n}."
            )

    def map_to_unconstrained_vector(self, theta_vec):
        if theta_vec.shape[-1] > self.d:
            theta_vec = theta_vec[...,self.free_indices]
        return transforms.box_inv(theta_vec, self.l, self.u)

    def map_to_constrained_vector(self, z_vec):
        return transforms.box(z_vec, self.l, self.u)

    def prior_log_prob(self, y):
        """Factorized, box-normalized log-prior over the free names present in
        ``y``.

        Coordinates without an explicit prior are uniform on ``[l_i, u_i]`` and
        contribute ``-log(u_i - l_i)``. Explicit per-coordinate priors are added
        as given (not renormalized for truncation to the box). Passing every free
        name gives the full log-prior. Passing a subset gives the marginal
        log-prior over that subset, since the prior factorizes over names.
        Footgun: a name accidentally dropped from ``y`` silently yields a
        marginal."""
        first = next(iter(y.values()))
        names = [yi for yi in self.free_names if yi in y]
        if not names:
            raise ValueError("y contains none of the free parameter names")
        log_norm = sum(self._uniform_log_norm[yi]
                       for yi in names if yi not in self.priors)
        result = torch.full(first.shape, float(log_norm),
                            device=first.device, dtype=first.dtype)
        for yi in names:
            if yi in self.priors:
                result = result + self.priors[yi].log_prob(y[yi]).squeeze(-1)
        return result

    def prior_log_prob_vector(self, theta_free):
        return self.prior_log_prob(self.from_vector(theta_free))

    def prior_metric(self, theta_full):
        """Constrained-space prior metric, or None when not configured."""
        if self.prior_metric_fn is None:
            return None
        return self.prior_metric_fn(theta_full)
        
    def push_forward_metric(self, G, theta_map):
        """G_free = dJ · G_ff · dJ, diagonal Jacobian ``dJ = dθ/dz`` on the free
        block ``G_ff``.

        Parameters
        ----------
        G : Tensor, shape (N, d_full, d_full)
            Constrained-space metric.
        theta_map : the z->θ map (``map_to_constrained_vector``).

        Returns
        -------
        Tensor, shape (N, d, d)
        """
        dJ = theta_map.jacobian_diag                        # (N, d) = dθ/dz, free coords
        fi = torch.as_tensor(self.free_indices, device=G.device)
        G_ff = G.index_select(-2, fi).index_select(-1, fi)  # (N, d, d)
        return dJ[..., :, None] * G_ff * dJ[..., None, :]

    def sample(self, n_samples):
        if not self.priors:
            # Uniform sample within the box.
            u = torch.rand(n_samples, self.d, device=self.l.device, dtype=self.l.dtype)
            theta = self.l + u * (self.u - self.l)
            samples = {yi: theta[..., i] for i, yi in enumerate(self.free_names)}
            return self.add_fixed(samples)

        # Per-coord rejection sampling: draw from the prior, resample draws
        # outside [l, u]. Coords are independent.
        dev, dt = self.l.device, self.l.dtype
        samples = {}
        for i, yi in enumerate(self.free_names):
            l_i, u_i = self.l[i], self.u[i]
            prior = self.priors.get(yi)
            if prior is None:                       # no prior -> uniform on its interval
                samples[yi] = l_i + torch.rand(n_samples, device=dev, dtype=dt) * (u_i - l_i)
                continue
            out    = torch.empty(n_samples, device=dev, dtype=dt)
            filled = torch.zeros(n_samples, dtype=torch.bool, device=dev)
            for _ in range(self._MAX_REJECTION_ROUNDS):
                idx = torch.nonzero(~filled, as_tuple=False).squeeze(-1)
                if idx.numel() == 0:
                    break
                cand = prior.sample([idx.numel()]).reshape(-1).to(device=dev, dtype=dt)
                ok = (cand > l_i) & (cand < u_i)
                out[idx[ok]] = cand[ok]
                filled[idx[ok]] = True
            if not bool(filled.all()):
                raise RuntimeError(
                    f"rejection sampling for '{yi}' did not fill all draws; the "
                    f"prior places too little mass inside [{float(l_i)}, {float(u_i)}]."
                )
            samples[yi] = out
        return self.add_fixed(samples)

    def remove_fixed(self, samples):
        samples = samples.copy()
        for yi in self.fixed.keys():
            del samples[yi]
        return samples

    def add_fixed(self, samples):
        samples = samples.copy()
        ref = next(iter(samples.values()))
        for yi in self.fixed.keys():
            samples[yi] = self.fixed[yi] * torch.ones(
                ref.shape, device=ref.device, dtype=ref.dtype)
        return samples

    def point_inside(self, y):
        for i, yi in enumerate(self.free_names):
            if torch.any(y[yi] <= self.l[i]) or torch.any(y[yi] >= self.u[i]):
                return False
        return True

    def to_vector(self, samples):
        samples = self.add_fixed(samples)
        point = []
        for yi in self.names:
            point.append(samples[yi])
        return torch.stack(point, axis=-1)