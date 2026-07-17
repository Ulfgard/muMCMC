"""
Parameter spaces and their transforms.

Each space exposes `prior_log_prob_vector`, operating on flat free vectors.

Each space exposes `prior_metric`, returning the constrained-space metric
contribution of the prior as a (d_full, d_full) SPD tensor, or None when there
is no contribution (uniform priors, or unconstrained spaces without an explicit
prior metric).

`push_forward_metric` pushes a constrained-space metric forward to the free
unconstrained coordinates; `TemperedMetric` and `TemperedPotential` hold the
pushed-forward metric and the potential affinely in an inverse temperature, so a
driver can retemper a moved configuration by reordering alone.
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
#  Tempered evaluation objects: metric and potential, affine in beta     #
# ====================================================================== #

def _solve_triangular_vec(triag_mat: torch.Tensor, vec: torch.Tensor, upper: bool):
    # triag_mat is (..., d, d) and vec is (..., d).
    return torch.linalg.solve_triangular(triag_mat, vec[..., None], upper=upper)[..., 0]


class TemperedMetric:
    """
    Free-space metric assembled affinely in an inverse temperature:

        G_u(beta) = beta * A_lik + A_prior

    where ``A_lik`` and ``A_prior`` are the likelihood and prior metrics already
    pushed forward to free unconstrained coordinates (see
    ``space.push_forward_metric``), dense ``(N, d, d)`` SPD tensors.  ``beta`` is
    slot-bound: :meth:`reorder` and :meth:`select` permute or mix the ``A``
    pieces while leaving ``beta`` in place, so a moved configuration is
    retempered to its slot's temperature.

    The Cholesky factor ``L`` of ``G_u(beta)`` is computed on first use, so a
    swap that only reorders costs no factorization.  Operations follow from
    ``G_u = L Lᵀ``.

    Parameters
    ----------
    A_lik : Tensor (N, d, d)
        Pushed-forward likelihood metric, scaled by ``beta``.
    A_prior : Tensor (N, d, d) or None
        Pushed-forward prior metric, untempered.
    beta : float or Tensor
        Inverse temperature scaling ``A_lik``.
    """

    def __init__(self, A_lik: torch.Tensor, A_prior, beta):
        self.A_lik = A_lik
        self.A_prior = A_prior
        self.beta = beta

    @cached_property
    def L(self) -> torch.Tensor:
        """Lower-triangular Cholesky factor of ``G_u(beta)``, positive diagonal."""
        beta = self.beta
        if torch.is_tensor(beta) and beta.ndim > 0:     # per-chain: broadcast over (N, d, d)
            beta = beta.reshape(-1, 1, 1)
        G = beta * self.A_lik if self.A_prior is None else beta * self.A_lik + self.A_prior
        return torch.linalg.cholesky(G)

    def inv_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """G_u⁻¹ v = L⁻ᵀ L⁻¹ v via two triangular solves."""
        return _solve_triangular_vec(
            self.L.transpose(-2, -1),
            _solve_triangular_vec(self.L, v, upper=False),
            upper=True,
        )

    def log_det_metric(self) -> torch.Tensor:
        """log det G_u = 2 Σ log|diag L|."""
        return 2.0 * self.L.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)

    def sample_momentum(self) -> torch.Tensor:
        """Sample p ~ N(0, G_u) via p = L ξ, ξ ~ N(0, I)."""
        xi = torch.randn(self.A_lik.shape[:-1], dtype=self.A_lik.dtype, device=self.A_lik.device)
        return (self.L @ xi[..., None])[..., 0].detach()

    def select(self, mask: torch.Tensor, other: "TemperedMetric") -> "TemperedMetric":
        """Per-chain select: this metric's chains where ``mask`` is True,
        ``other``'s where False.  Both share the same temperature."""
        m = mask.reshape(mask.shape + (1,) * (self.A_lik.dim() - mask.dim()))
        return TemperedMetric(
            torch.where(m, self.A_lik, other.A_lik),
            None if self.A_prior is None else torch.where(m, self.A_prior, other.A_prior),
            self.beta,
        )

    def reorder(self, perm: torch.Tensor) -> "TemperedMetric":
        """Permute chains: row ``i`` of the result is row ``perm[i]``.  ``beta``
        is slot-bound and stays in place, retempering the moved configuration."""
        return TemperedMetric(
            self.A_lik[perm],
            None if self.A_prior is None else self.A_prior[perm],
            self.beta,
        )


class TemperedPotential:
    """
    Potential assembled affinely in an inverse temperature:

        U = beta * U_lik + U_base

    with ``U_lik = -log p(data | theta)`` and ``U_base = U_prior - log|det J|``.
    ``beta`` is slot-bound: :meth:`reorder` and :meth:`select` permute or mix
    ``U_lik``/``U_base`` while leaving ``beta`` in place, so a moved configuration
    is retempered to its slot's temperature.  ``value`` is the assembled ``(N,)``
    potential.

    Parameters
    ----------
    U_lik, U_base : Tensor (N,)
        The temperature-scaled and temperature-free pieces of the potential.
    beta : float or Tensor
        Inverse temperature scaling ``U_lik``.
    """

    def __init__(self, U_lik: torch.Tensor, U_base: torch.Tensor, beta):
        self.U_lik = U_lik
        self.U_base = U_base
        self.beta = beta

    @cached_property
    def value(self) -> torch.Tensor:
        return self.beta * self.U_lik + self.U_base

    def select(self, mask: torch.Tensor, other: "TemperedPotential") -> "TemperedPotential":
        """Per-chain select: this potential's chains where ``mask`` is True,
        ``other``'s where False.  Both share the same temperature."""
        return TemperedPotential(
            torch.where(mask, self.U_lik, other.U_lik),
            torch.where(mask, self.U_base, other.U_base),
            self.beta,
        )

    def reorder(self, perm: torch.Tensor) -> "TemperedPotential":
        """Permute chains: row ``i`` of the result is row ``perm[i]``.  ``beta``
        is slot-bound and stays in place, retempering the moved configuration."""
        return TemperedPotential(self.U_lik[perm], self.U_base[perm], self.beta)


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
        
    def push_forward_metric(self, G, theta_map):
        """Push a constrained-space metric ``G`` (``(N, d_full, d_full)``) forward
        to the free unconstrained coordinates: restrict to the free block and
        scale by the diagonal Jacobian ``dθ/dz``.  ``theta_map`` is the z->θ map
        (``map_to_constrained_vector``).  Returns the dense ``(N, d, d)`` metric.

        The transform is elementwise (diagonal ``J``), so the free block of the
        push-forward is the push-forward of the free block -- fixed coordinates
        do not couple in.
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
        
    def push_forward_metric(self, G, theta_map):
        """Push a constrained-space metric ``G`` (``(N, d_full, d_full)``) forward
        to the free unconstrained coordinates: restrict to the free block and
        scale by the diagonal Jacobian ``dθ/dz``.  ``theta_map`` is the z->θ map
        (``map_to_constrained_vector``).  Returns the dense ``(N, d, d)`` metric.

        The transform is elementwise (diagonal ``J``), so the free block of the
        push-forward is the push-forward of the free block -- fixed coordinates
        do not couple in.
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