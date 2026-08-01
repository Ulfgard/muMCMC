from functools import cached_property

import torch


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
    temperature-free part, ``None`` when absent. ``lik`` and ``base`` share a
    leading batch axis and may carry further trailing feature axes, over which
    ``beta`` broadcasts.

    ``beta`` is bound to the batch position rather than to the parts held there.
    :meth:`select` and :meth:`reorder` mix or permute ``lik`` and ``base`` along
    the batch axis and leave ``beta`` in place, so a configuration moved to
    another position is evaluated at that position's temperature.

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

    # Read-only because ``value`` and ``TemperedMetric.L`` are cached, so
    # mutating these inputs after construction would return a stale result.
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
        """This quantity where ``mask`` is True and ``other`` where it is False,
        per batch element. Both must carry the same ``beta``, which the result
        keeps."""
        m = mask.reshape(mask.shape + (1,) * (self.lik.dim() - mask.dim()))
        return type(self)(
            torch.where(m, self.lik, other.lik),
            None if self.base is None else torch.where(m, self.base, other.base),
            self.beta,
        )

    def reorder(self, perm: torch.Tensor) -> "TemperedAffine":
        """Permute the batch axis, so row ``i`` of the result is row ``perm[i]``
        of this one. ``beta`` is bound to the batch position and stays in
        place."""
        return type(self)(
            self.lik[perm],
            None if self.base is None else self.base[perm],
            self.beta,
        )


class TemperedMetric(TemperedAffine):
    """
    Metric ``G = beta * A_lik + A_prior``, an ``(N, d, d)`` SPD :attr:`value`.

    ``A_lik`` (``lik``) is the free block of the model's metric and ``A_prior``
    (``base``) the metric the prior induces, ``None`` for a space carrying no
    prior. Both are read in the coordinates the chain runs in.

    Raises
    ------
    RuntimeError
        From :attr:`L`, and from every method that uses it, when ``G`` is not
        positive definite.
    """

    @cached_property
    def L(self) -> torch.Tensor:
        """Lower-triangular Cholesky factor ``G = L Lᵀ`` of :attr:`value`, with
        positive diagonal, of shape ``(N, d, d)``."""
        return torch.linalg.cholesky(self.value)

    def inv_metric_times_vec(self, v: torch.Tensor) -> torch.Tensor:
        """``G⁻¹ v = L⁻ᵀ L⁻¹ v`` for ``v`` of shape ``(N, d)``, giving
        ``(N, d)``."""
        return _solve_triangular_vec(
            self.L.transpose(-2, -1),
            _solve_triangular_vec(self.L, v, upper=False),
            upper=True,
        )

    def log_det_metric(self) -> torch.Tensor:
        """``log det G = 2 Σ_i log L_ii``, of shape ``(N,)``."""
        return 2.0 * self.L.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)

    def sample_momentum(self) -> torch.Tensor:
        """A draw ``p ~ N(0, G)`` of shape ``(N, d)``, detached and taken from
        the global RNG."""
        xi = torch.randn(self.lik.shape[:-1], dtype=self.lik.dtype, device=self.lik.device)
        return (self.L @ xi[..., None])[..., 0].detach()
