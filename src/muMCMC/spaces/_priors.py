import torch

from ._space import Space
from ._transform import NormalTransform


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

    The chart is affine, so the metric it induces is the constant diagonal
    ``sigma⁻²``.

    Parameters
    ----------
    names, fixed
        As for :class:`Space`.
    mu, sigma : float, dict[str, float] or Tensor
        Location and scale per free variable, ``sigma`` positive. A scalar is
        shared across the free variables, a dict is read by free name and a
        tensor is read in :attr:`Space.free_names` order.
    dtype, device
        Dtype and device the chart and its draws are built in. None takes the
        torch defaults.

    Raises
    ------
    ValueError
        If ``mu`` or ``sigma`` is a dict missing a free name, or a tensor that
        is not of shape ``(d,)``.
    """

    def __init__(self, names, mu=0.0, sigma=1.0, *, fixed=None,
                 dtype=None, device=None):
        super().__init__(names, fixed=fixed)
        mu = _as_parameter(mu, self._free_names, "mu", dtype, device)
        sigma = _as_parameter(sigma, self._free_names, "sigma", dtype, device)
        log_sigma = torch.log(sigma)
        self._transform = NormalTransform(
            lambda z: (mu + sigma * z, log_sigma.expand_as(z)),
            lambda theta: ((theta - mu) / sigma, (-log_sigma).expand_as(theta)),
            reference=mu)

    @property
    def as_transform(self):
        return self._transform


class UnnormalizedSpace(Space):
    """No prior, so the chart is the identity and ``z`` is ``theta``.

    The target is the model potential alone, with no prior term added, and it
    need not be normalizable. This departs from :class:`Space` in three places.
    :meth:`prior_log_prob_vector` is zero, :meth:`prior_metric` returns None, so
    a metric-based sampler contributes only the model's own metric, and
    :meth:`prior_log_prob` and :meth:`sample` raise.

    Parameters
    ----------
    names, fixed
        As for :class:`Space`.
    dtype, device
        Dtype and device the chart is built in. None takes the torch defaults.

    Raises
    ------
    ValueError
        From :meth:`prior_log_prob` and :meth:`sample`, which are not defined
        without a prior.
    """

    def __init__(self, names, *, fixed=None, dtype=None, device=None):
        super().__init__(names, fixed=fixed)
        zero = torch.zeros(self.d, dtype=dtype, device=device)
        self._transform = NormalTransform(
            lambda z: (z, torch.zeros_like(z)),
            lambda theta: (theta, torch.zeros_like(theta)),
            reference=zero)

    @property
    def as_transform(self):
        return self._transform

    def prior_log_prob(self, theta: dict) -> torch.Tensor:
        raise ValueError(
            "UnnormalizedSpace carries no prior, so prior_log_prob is not "
            "defined. Use a space with a prior if you need one.")

    def prior_log_prob_vector(self, theta_free: torch.Tensor) -> torch.Tensor:
        """Zero of shape ``(...)`` for ``theta_free`` of shape ``(..., d)``, so
        the potential is the model's alone."""
        return torch.zeros(theta_free.shape[:-1], dtype=theta_free.dtype,
                           device=theta_free.device)

    def prior_metric(self, theta_free: torch.Tensor):
        """None, there being no prior to induce a metric."""
        return None

    def sample(self, n_samples: int, *, generator=None) -> dict:
        raise ValueError(
            "UnnormalizedSpace carries no prior, so it cannot be sampled from.")
