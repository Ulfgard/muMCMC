import torch

from ._space import Space
from ._transform import (
    NormalTransform, _std_normal_cdf, _std_normal_icdf, _std_normal_log_pdf,
)


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
    """Independent normal priors, chart ``theta = mu + sigma z``.

    Parameters
    ----------
    mu, sigma : float, dict[str, float] or Tensor
        Location and scale per free variable, ``sigma`` positive.
    """

    def __init__(self, names, mu=0.0, sigma=1.0, *, fixed=None,
                 dtype=None, device=None):
        super().__init__(names, fixed=fixed)
        mu = _as_parameter(mu, self._free_names, "mu", dtype, device)
        sigma = _as_parameter(sigma, self._free_names, "sigma", dtype, device)
        log_sigma = torch.log(sigma)
        self._transform = NormalTransform(
            lambda z: (mu + sigma * z, log_sigma.expand_as(z)),
            lambda th: ((th - mu) / sigma, (-log_sigma).expand_as(th)),
            log_prob=lambda th: _std_normal_log_pdf((th - mu) / sigma) - log_sigma,
            reference=mu)

    @property
    def as_transform(self):
        return self._transform


class LogNormalSpace(Space):
    """Independent log-normal priors, chart ``theta = exp(mu + sigma z)``, so
    every free variable is positive.

    Parameters
    ----------
    mu, sigma : float, dict[str, float] or Tensor
        Location and scale of the underlying normal, ``sigma`` positive.
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
            forward, inverse,
            log_prob=lambda th: (_std_normal_log_pdf((torch.log(th) - mu) / sigma)
                                 - log_sigma - torch.log(th)),
            reference=mu)

    @property
    def as_transform(self):
        return self._transform


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


def _distribution_transform(dist) -> NormalTransform:
    """The normal chart ``theta = F⁻¹(Phi(z))`` of a batched distribution.

    The Jacobian is analytic from the density, ``dtheta/dz = phi(z)/f(theta)``,
    so ``icdf``, ``cdf`` and ``log_prob`` are all it takes.

    Raises
    ------
    ValueError
        If ``dist`` implements neither ``icdf`` nor ``cdf``, since the chart
        would then have no closed form in that direction.
    """
    missing = [name for name in ("icdf", "cdf") if not _implements(dist, name)]
    if missing:
        raise ValueError(
            f"{type(dist).__name__} does not implement {' and '.join(missing)}, "
            f"so its normal chart has no closed form. Build the chart directly "
            f"as a NormalTransform, or pick a distribution that does.")
    dist = dist.expand(dist.batch_shape)

    def forward(z):
        theta = dist.icdf(_std_normal_cdf(z))
        return theta, _std_normal_log_pdf(z) - dist.log_prob(theta)

    def inverse(theta):
        z = _std_normal_icdf(dist.cdf(theta))
        return z, dist.log_prob(theta) - _std_normal_log_pdf(z)

    params = [getattr(dist, name) for name in dist.arg_constraints
              if torch.is_tensor(getattr(dist, name, None))]
    reference = (params[0].expand(dist.batch_shape) if params
                 else torch.zeros(dist.batch_shape))
    return NormalTransform(forward, inverse, log_prob=dist.log_prob,
                           reference=reference)


class DistributionSpace(Space):
    """Independent priors from one batched distribution, chart
    ``theta = F⁻¹(Phi(z))``.

    Parameters
    ----------
    dist : torch.distributions.Distribution
        Univariate distribution of batch shape ``(d,)``, one entry per free
        variable in free-name order. A batch shape of ``()`` is expanded.

    Raises
    ------
    ValueError
        If the batch shape is neither ``(d,)`` nor ``()``.
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
    """No prior, so the chart is the identity and ``z`` is ``theta``.

    The target is whatever the model potential defines, with nothing added.
    Having no normalized density, it has no evidence and no entropy, and
    :meth:`prior_metric_normal` is None so a scheme needing the prior's metric
    is not available.

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
            lambda th: (th, torch.zeros_like(th)),
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
