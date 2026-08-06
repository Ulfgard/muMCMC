import torch


# =========================================================================== #
#  The two identities of the normal chart                                     #
#                                                                             #
#  A chain runs on the variables, where the prior contributes -log p(theta)   #
#  to the potential and M_theta = J^-T J^-1, with J = dtheta/dz, to the        #
#  metric. That metric is the pullback of the identity along the chart, so it  #
#  is the one the prior itself induces, and it varies with position.          #
#                                                                             #
#  Read instead in the chart, the prior is exactly N(0, I) and its metric is   #
#  J^T M_theta J = I, constant. A scheme needing a fixed prior block therefore #
#  puts no condition on the prior, only on where it is read, and it can        #
#  compute both from the chart without asking the space for them.             #
# =========================================================================== #


class Space:
    """Named variables with a prior, read in the prior's normal chart.

    A point is a dict keyed by name, which is the only form a caller deals in.
    Inside, it is a free vector ``(..., d)`` over :attr:`free_names`, which the
    chain and the prior act on, and :meth:`to_full` widens that to the
    ``(..., d_full)`` layout over :attr:`names` that a model potential is
    handed.

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
    def as_transform(self):
        """The chart ``theta = T(z)`` over the free variables."""
        raise NotImplementedError

    # ---- vector layouts ---------------------------------------------------- #

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

    def to_full(self, theta_free: torch.Tensor) -> torch.Tensor:
        """Free vector ``(..., d)`` to full vector ``(..., d_full)``, with each
        fixed variable at its value."""
        if not self.fixed:
            return theta_free
        shape = theta_free.shape[:-1] + (self.d_full,)
        out = self._template(theta_free).expand(shape).clone()
        idx = torch.as_tensor(self.free_indices, device=theta_free.device)
        out[..., idx] = theta_free
        return out

    def to_free_vector(self, samples: dict) -> torch.Tensor:
        """Dict keyed by name to a free vector ``(..., d)``. Fixed names in
        ``samples`` are ignored, since they are not sampled.

        Raises
        ------
        ValueError
            If ``samples`` is missing a free name.
        """
        missing = [name for name in self._free_names if name not in samples]
        if missing:
            raise ValueError(f"samples is missing {missing}")
        return torch.stack([samples[name] for name in self._free_names], dim=-1)

    def from_free_vector(self, theta_free: torch.Tensor) -> dict:
        """Free vector ``(..., d)`` to a dict over the free names.

        Raises
        ------
        ValueError
            If the vector is not ``d`` wide.
        """
        if theta_free.shape[-1] != self.d:
            raise ValueError(
                f"expected a free vector of size {self.d}, got "
                f"{theta_free.shape[-1]}.")
        return {name: theta_free[..., i]
                for i, name in enumerate(self._free_names)}

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

    def set_fixed(self, fixed: dict) -> None:
        """Hold the same names fixed at new values, in place.

        The free/fixed split does not move, so everything derived from it stays
        valid and nothing is rebuilt. The values are read by :meth:`to_full` and
        by :meth:`add_fixed`, so this changes what a model potential is handed
        and what a run reports, and nothing else. It mutates rather than
        returning a copy, so a sampler already holding the space picks the new
        values up, which is the point when one space serves a sequence of
        problems differing only in what they pin.

        Raises
        ------
        ValueError
            If ``fixed`` does not name exactly the variables already fixed.
        """
        if set(fixed) != set(self.fixed):
            raise ValueError(
                f"set_fixed holds the same names fixed, so that what is derived "
                f"from the free/fixed split stays valid: got {sorted(fixed)} "
                f"against {sorted(self.fixed)}")
        self.fixed = dict(fixed)
        self._template_cache.clear()

    def remove_fixed(self, samples: dict) -> dict:
        """``samples`` with every fixed name dropped."""
        if not self.fixed:
            return samples
        samples = samples.copy()
        for name in self.fixed:
            samples.pop(name, None)
        return samples

    # ---- prior ------------------------------------------------------------- #

    def prior_log_prob(self, theta: dict) -> torch.Tensor:
        """Factorized log-prior over the free names present in ``theta``, a dict
        of points on the variables of shape ``(...)``, giving ``(...)``.

        The prior factorizes over the coordinate axis, so a subset of the free
        names gives the marginal over that subset. A name left out by accident
        therefore returns a marginal rather than raising. Fixed names are
        ignored.

        Raises
        ------
        ValueError
            If ``theta`` holds none of the free names.
        """
        present = [i for i, name in enumerate(self._free_names) if name in theta]
        if not present:
            raise ValueError("theta contains none of the free parameter names")

        # A variable the marginal does not range over is held at T(0), which is
        # in the support and so keeps its factor finite before it is dropped.
        ref = theta[self._free_names[present[0]]]
        interior = self.as_transform.interior_point.to(ref.dtype)
        theta_free = interior.expand(ref.shape + (self.d,)).clone()
        for i in present:
            theta_free[..., i] = theta[self._free_names[i]]
        idx = torch.as_tensor(present, device=theta_free.device)
        return self.as_transform.log_prob(theta_free).index_select(-1, idx).sum(-1)

    def prior_log_prob_vector(self, theta_free: torch.Tensor) -> torch.Tensor:
        """Log-prior on a free vector ``(..., d)``, shape ``(...)``."""
        return self.as_transform.log_prob(theta_free).sum(-1)

    def prior_metric(self, theta_free: torch.Tensor):
        """The prior's natural metric ``M = J⁻ᵀ J⁻¹`` at ``theta_free``, shape
        ``(..., d, d)``, with ``J = dtheta/dz``.

        It is the pullback of the identity along the chart, so it is the metric
        the prior itself induces on the variables, and it varies with position.
        """
        return self.as_transform.inverse(theta_free).gram()

    def free_block(self, G: torch.Tensor) -> torch.Tensor:
        """The free block of a metric ``G`` of shape ``(..., d_full, d_full)``
        over every variable, giving ``(..., d, d)``. The fixed variables are not
        sampled, so their rows and columns are dropped."""
        idx = torch.as_tensor(self.free_indices, device=G.device)
        return G.index_select(-2, idx).index_select(-1, idx)

    def sample(self, n_samples: int, *, generator=None) -> dict:
        """``n_samples`` draws from the prior, keyed by name and including the
        fixed variables, each of shape ``(n_samples,)``. ``generator`` drives the
        draw, the global RNG when omitted.
        """
        T = self.as_transform
        z = torch.randn(n_samples, self.d, dtype=T.dtype, device=T.device,
                        generator=generator)
        return self.add_fixed(self.from_free_vector(T.forward_point(z)))
