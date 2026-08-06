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
        inv_sigma = 1.0 / sigma
        self._transform = NormalTransform(
            lambda z: (mu + sigma * z, sigma.expand_as(z)),
            lambda theta: ((theta - mu) / sigma, inv_sigma.expand_as(theta)),
            reference=mu)

    @property
    def as_transform(self):
        return self._transform


class UnnormalizedSpace(Space):
    """No prior, so no chart either. The target is whatever the model potential
    defines, with nothing added.

    A chart is the map a prior is standard normal in, so a space without one has
    none to hand out, and a scheme that runs in the chart is not available here.
    Having no normalized density, it has no evidence and no entropy, and
    :meth:`prior_metric` is None so a scheme needing the prior's metric is not
    available either.

    Raises
    ------
    ValueError
        From :meth:`as_transform`, :meth:`prior_log_prob` and :meth:`sample`,
        which have no meaning without a prior.
    """

    @property
    def as_transform(self):
        raise ValueError(
            "UnnormalizedSpace carries no prior, so it has no chart. A scheme "
            "reading the prior in its chart needs a space with one.")

    def prior_log_prob(self, theta: dict) -> torch.Tensor:
        raise ValueError(
            "UnnormalizedSpace carries no prior, so prior_log_prob is not "
            "defined. Use a space with a prior if you need one.")

    def prior_log_prob_vector(self, theta_free: torch.Tensor) -> torch.Tensor:
        return torch.zeros(theta_free.shape[:-1], dtype=theta_free.dtype,
                           device=theta_free.device)

    def prior_metric(self, theta_free: torch.Tensor):
        return None

    def sample(self, n_samples: int, *, generator=None) -> dict:
        raise ValueError(
            "UnnormalizedSpace carries no prior, so it cannot be sampled from.")
