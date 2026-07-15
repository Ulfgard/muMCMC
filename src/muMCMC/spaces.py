"""
Parameter spaces and their transforms.

Each space exposes `prior_log_prob_vector`, operating on flat free vectors.

Each space exposes `prior_metric`, returning the constrained-space metric
contribution of the prior as a (d_full, d_full) SPD tensor, or None when there
is no contribution (uniform priors, or unconstrained spaces without an explicit
prior metric).

`TransformedMetric` encapsulates a position-dependent inverse metric without
forming dense matrices, working through the Jacobian-vector-product interface
that the space's transforms expose.
"""

from functools import cached_property

import torch


class ElementwiseTransform:
    """
    Elementwise transform p' = T(p) with diagonal Jacobian.

    Carries enough information for cheap jvp/vjp/log-det operations.
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

    def jvp(self, v: torch.Tensor) -> torch.Tensor:
        return self._diag_J * v

    def vjp(self, v: torch.Tensor) -> torch.Tensor:
        return v * self._diag_J

    def jinvvp(self, v: torch.Tensor) -> torch.Tensor:
        return v / self._diag_J

    def vjinvp(self, v: torch.Tensor) -> torch.Tensor:
        return v / self._diag_J

    def where(self, mask: torch.Tensor, other: "ElementwiseTransform") -> "ElementwiseTransform":
        """Per-chain select: take this transform's entries where ``mask`` is
        True, ``other``'s where False.  ``mask`` is an ``(N,)`` bool over the
        leading batch axis.  Returns a new transform.
        """
        def sel(a, b):
            m = mask.reshape(mask.shape + (1,) * (a.dim() - mask.dim()))
            return torch.where(m, a, b)
        return ElementwiseTransform(
            p             = sel(self._p,             other._p),
            p_prime       = sel(self._p_prime,       other._p_prime),
            diag_J        = sel(self._diag_J,        other._diag_J),
            log_abs_det_J = sel(self._log_abs_det_J, other._log_abs_det_J),
        )

    def reorder(self, perm: torch.Tensor) -> "ElementwiseTransform":
        """Permute chains along the leading batch axis: row ``i`` of the result
        is row ``perm[i]`` of this transform.  ``perm`` is an ``(N,)`` long
        index tensor.  Returns a new transform.
        """
        return ElementwiseTransform(
            p             = self._p[perm],
            p_prime       = self._p_prime[perm],
            diag_J        = self._diag_J[perm],
            log_abs_det_J = self._log_abs_det_J[perm],
        )

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
        """Helper for box <-> unconstrained transform.  Uses tanh:
            p' = (u+l)/2 + (u-l)/2 * tanh(p)
            p  = atanh( 2 (p' - (u+l)/2) / (u-l) )
        Jacobian (in the unconstrained-to-constrained direction):
            d p' / d p = (u-l)/2 * sech^2(p)
        so log|d p'/d p| = log((u-l)/2) - 2 log|cosh(p)|.
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
#  TransformedMetric: position-dependent inverse metric, decomposed      #
# ====================================================================== #

def _solve_triangular_vec(triag_mat: torch.Tensor, vec: torch.Tensor, upper: bool):
    # triag_mat is (..., d, d) and vec is (..., d).
    return torch.linalg.solve_triangular(triag_mat, vec[..., None], upper=upper)[..., 0]

class TransformedMetric:
    """
    Holds a decomposed position-dependent inverse metric and provides
    efficient operations without ever forming G or G⁻¹ as dense matrices.
 
    The pull-back of the constrained-space metric to unconstrained space is:
        G_u⁻¹ = J⁻¹ G_c⁻¹ J⁻ᵀ
 
    where J = ∂θ/∂z is the Jacobian of the map from unconstrained to
    constrained coordinates. L is the lower-triangular Cholesky factor of
    the constrained-space metric G_c:

        G_c = L Lᵀ  ⟹  G_c⁻¹ = L⁻ᵀ L⁻¹
            G_u⁻¹ = J⁻¹ L⁻ᵀ L⁻¹ J⁻ᵀ

    The Jacobian J is never formed explicitly. All operations are expressed
    through the jacobian-vector product interface of z_transform, which may
    implement these efficiently (e.g. diagonally, via QR, etc.).

    Parameters
    ----------
    z_transform : any object implementing jvp, vjp, jinvvp, vjinvp,
                  jacobian_log_det
        Represents the map from unconstrained z to constrained θ.
    L : Tensor [d, d]
        Lower-triangular Cholesky factor of G_c, with positive diagonal.
    """
    def __init__(self, z_transform, L: torch.Tensor):
        self.z_transform     = z_transform
        self.L               = L
        # L is (..., d, d); reduce the diagonal over the last axis only.
        self.log_det_L       = L.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
 
    def Gc_inv_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute G_c⁻¹ v = L⁻ᵀ L⁻¹ v via two triangular solves."""
        return _solve_triangular_vec(
            self.L.transpose(-2, -1),
            _solve_triangular_vec(self.L, v, upper=False),
            upper=True,
        )

    def sqrt_Gc_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute G_c^{½} v = L v."""
        return (self.L @ v[..., None])[..., 0]

    def inv_sqrt_Gc_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute G_c^{-½} v = L⁻¹ v via one triangular solve."""
        return _solve_triangular_vec(self.L, v, upper=False)
 
    def inv_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute G_u⁻¹ v = J⁻¹ G_c⁻¹ J⁻ᵀ v."""
        w = self.z_transform.vjinvp(v)
        w = self.Gc_inv_times_vec(w)
        return self.z_transform.jinvvp(w)
 
    def metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute G_u v = Jᵀ G_c J v."""
        w = self.z_transform.jvp(v)
        # G_c w = L Lᵀ w
        w = (self.L @ (self.L.transpose(-2, -1) @ w[..., None]))[..., 0]
        return self.z_transform.vjp(w)
 
    def sqrt_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute G_u^{½} v = Jᵀ G_c^{½} v."""
        w = self.sqrt_Gc_times_vec(v)
        return self.z_transform.vjp(w)
 
    def inv_sqrt_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """Compute G_u^{-½} v = G_c^{-½} J^{-ᵀ} v."""
        w = self.z_transform.vjinvp(v)
        return self.inv_sqrt_Gc_times_vec(w)
 
    def sample_momentum(self) -> torch.Tensor:
        """Sample p ~ N(0, G_u) via p = Jᵀ G_c^{½} ξ, ξ ~ N(0, I)."""
        xi = torch.randn_like(self.z_transform.p)
        return self.sqrt_metric_times_vec(xi).detach()
 
    def log_det_metric(self) -> torch.Tensor:
        """log det G_u = 2 log|det J| + log det G_c,  with log det G_c = 2 log|det L|."""
        return 2.0 * self.z_transform.jacobian_log_det + 2.0 * self.log_det_L

    def select(self, mask: torch.Tensor, other: "TransformedMetric") -> "TransformedMetric":
        """Per-chain select between two batched metrics: take this metric's
        chains where ``mask`` is True, ``other``'s where False.  ``mask`` is an
        ``(N,)`` bool over the leading batch axis.  Returns a new
        ``TransformedMetric``; equivalent per chain to the metric evaluated at
        the selected points.
        """
        m = mask.reshape(mask.shape + (1,) * (self.L.dim() - mask.dim()))
        L = torch.where(m, self.L, other.L)
        z = self.z_transform.where(mask, other.z_transform)
        return TransformedMetric(z, L)

    def reorder(self, perm: torch.Tensor) -> "TransformedMetric":
        """Permute chains along the leading batch axis: row ``i`` of the result
        is row ``perm[i]`` of this metric.  ``perm`` is an ``(N,)`` long index
        tensor.  Returns a new ``TransformedMetric``; equivalent per chain to
        the metric evaluated at the permuted points.
        """
        z = self.z_transform.reorder(perm)
        return TransformedMetric(z, self.L[perm])
 
 
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
            Names to hold fixed (pinned to the given value in this
            parameterization).  Fixed names are removed from the sampled
            (free) space but spliced back for model evaluation, and their
            row/column is projected out of the metric.  None / empty means
            nothing fixed.
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
        # Whether the fixed coordinates are the trailing ones: then the metric
        # projection is the leading Cholesky block; otherwise a QR is used.
        self._fixed_are_trailing = (
            len(self.fixed_indices) == 0
            or self.fixed_indices == list(range(self.d, self.d_full))
        )

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
        if self.priors is None:
            raise ValueError("Unconstrained space without priors does not allow for prior_log_prob to be computed")
        result = 0
        # Sum over free names only: a fixed coordinate contributes a constant
        # to the log-prior (irrelevant for sampling) and is absent from y.
        for yi in self._free_names:
            result += self.priors[yi].log_prob(y[yi]).squeeze(-1)
        return result

    def prior_log_prob_vector(self, theta_free):
        """Vector form of prior_log_prob; zero if no prior is configured."""
        if self.priors is None:
            return torch.zeros(theta_free.shape[:-1], device=theta_free.device, dtype=theta_free.dtype)
        return self.prior_log_prob(self.from_vector(theta_free))

    def prior_metric(self, theta_full):
        """Constrained-space prior metric, or None when not configured."""
        if self.prior_metric_fn is None:
            return None
        return self.prior_metric_fn(theta_full)
        
    def push_forward_metric(self, theta, G, theta_map=None, G_is_lower_cholesky=False):
        """Push forward a metric from constrained coordinates at ``theta`` to
        free unconstrained space, returning a TransformedMetric.

            Arguments:
            theta: base point in (full or free) constrained coordinates where the metric is computed
            G: the metric computed at theta
            theta_map: optional transform encapsulating the map z->theta, i.e. theta_map.mapped_point = free_variables(theta)
            G_is_lower_cholesky: whether G is provided as a lower Cholesky factor. Default: False
        """

        # If the map is not provided, invert theta->z to get z->theta.
        if theta_map is None:
            theta_map = self.map_to_unconstrained_vector(theta).inv
        # torch.linalg.cholesky requires every chain's G to be PD; a non-PD
        # chain aborts the whole batch.
        L = G if G_is_lower_cholesky else torch.linalg.cholesky(G)

        # Project out fixed coordinates: the leading Cholesky block when fixed
        # are trailing, else a QR onto the free indices.  L is (..., d_full, d_full).
        if self._fixed_are_trailing:
            L = L[..., :self.d, :self.d]
        elif self.d < self.d_full:
            fi = self.free_indices
            L_sub = L[..., fi, :]
            Q, R = torch.linalg.qr(L_sub.transpose(-2, -1), mode="reduced")
            signs = R.diagonal(dim1=-2, dim2=-1).sign()
            L = (R * signs.unsqueeze(-2)).transpose(-2, -1)

        metric = TransformedMetric(theta_map, L)
        return metric

    def sample(self, n_samples):
        if self.priors is None:
            raise ValueError("Unconstrained space without priors cannot be sampled from")
        samples = {}
        for yi in self._free_names:
            # Each name is a single scalar coordinate, so a per-name prior is
            # univariate.  reshape normalises a trailing singleton (e.g. a prior
            # built as Normal(zeros(1), ones(1))) and rejects a multivariate
            # prior, which the space cannot represent.
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

        self._fixed_are_trailing = (
            len(self.fixed_indices) == 0
            or self.fixed_indices == list(range(self.d, self.d_full))
        )
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
        # Returns the unnormalized log probability.  With no prior given,
        # returns zero -- the uniform prior on the box.
        first = next(iter(y.values()))
        result = torch.zeros(first.shape, device=first.device, dtype=first.dtype)
        for yi in self.free_names:
            if yi in self.priors:
                result = result + self.priors[yi].log_prob(y[yi]).squeeze(-1)
        return result

    def prior_log_prob_vector(self, theta_free):
        if not self.priors:
            return torch.zeros(theta_free.shape[:-1], device=theta_free.device, dtype=theta_free.dtype)
        return self.prior_log_prob(self.from_vector(theta_free))

    def prior_metric(self, theta_full):
        """Constrained-space prior metric, or None when not configured."""
        if self.prior_metric_fn is None:
            return None
        return self.prior_metric_fn(theta_full)
        
    def push_forward_metric(self, theta, G, theta_map=None, G_is_lower_cholesky=False):
        """Push forward a metric from constrained coordinates at ``theta`` to
        free unconstrained space, returning a TransformedMetric.

            Arguments:
            theta: base point in (full or free) constrained coordinates where the metric is computed
            G: the metric computed at theta
            theta_map: optional transform encapsulating the map z->theta, i.e. theta_map.mapped_point = free_variables(theta)
            G_is_lower_cholesky: whether G is provided as a lower Cholesky factor. Default: False
        """

        # If the map is not provided, invert theta->z to get z->theta.
        if theta_map is None:
            theta_map = self.map_to_unconstrained_vector(theta).inv
        # torch.linalg.cholesky requires every chain's G to be PD; a non-PD
        # chain aborts the whole batch.
        L = G if G_is_lower_cholesky else torch.linalg.cholesky(G)

        # Handle fixed variables.  L is (..., d_full, d_full).
        if self._fixed_are_trailing:
            L = L[..., :self.d, :self.d]
        else:
            fi = self.free_indices
            L_sub = L[..., fi, :]
            Q, R = torch.linalg.qr(L_sub.transpose(-2, -1), mode="reduced")
            signs = R.diagonal(dim1=-2, dim2=-1).sign()
            L = (R * signs.unsqueeze(-2)).transpose(-2, -1)
        
        metric = TransformedMetric(theta_map, L)
        return metric

    def sample(self, n_samples):
        if not self.priors:
            # Uniform sample within the box.
            u = torch.rand(n_samples, self.d, device=self.l.device, dtype=self.l.dtype)
            theta = self.l + u * (self.u - self.l)
            samples = {yi: theta[..., i] for i, yi in enumerate(self.free_names)}
            return self.add_fixed(samples)

        # Per-coord rejection sampling: draw from the prior and resample
        # anything outside its [l, u] so every draw lies in the box.  Coords
        # are independent (one scalar column per name).
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
        batch_shape = (next(iter(samples.values()))).shape
        device = (next(iter(samples.values()))).device
        for yi in self.fixed.keys():
            samples[yi] = self.fixed[yi] * torch.ones(batch_shape, device=device)
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