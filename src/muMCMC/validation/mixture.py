"""Full-covariance Gaussian mixture: fit, density, sampling, and conditioning.

Fitted to a point cloud by EM (fixed number of components), it serves as the
BAR reference ``q̂`` and, through :meth:`GaussianMixture.conditional`, supplies
the proposal ``q(z_b | z_a)`` for marginalization. The conditional of a Gaussian
mixture is again a mixture, with the same per-component conditional Gaussians and
component weights reweighted by each component's responsibility for ``z_a``.
"""
from __future__ import annotations

from typing import Optional

import torch

MVN = torch.distributions.MultivariateNormal


def _kmeanspp_init(z: torch.Tensor, k: int,
                   generator: Optional[torch.Generator]) -> torch.Tensor:
    """k-means++ seeding: ``k`` component means chosen from ``z`` far apart."""
    n = z.shape[0]
    first = int(torch.randint(n, (1,), generator=generator))
    centers = [z[first:first + 1]]
    d2 = ((z - centers[0]) ** 2).sum(-1)
    for _ in range(1, k):
        idx = int(torch.multinomial(d2 / d2.sum(), 1, generator=generator))
        centers.append(z[idx:idx + 1])
        d2 = torch.minimum(d2, ((z - centers[-1]) ** 2).sum(-1))
    return torch.cat(centers, 0)


class GaussianMixture:
    """A ``K``-component full-covariance Gaussian mixture in ``d`` dimensions.

    Parameters
    ----------
    weights : Tensor, shape (K,)
        Mixture weights, summing to one.
    means : Tensor, shape (K, d)
        Component means.
    scale_tril : Tensor, shape (K, d, d)
        Lower Cholesky factor of each component covariance.
    """

    def __init__(self, weights: torch.Tensor, means: torch.Tensor,
                 scale_tril: torch.Tensor):
        self.weights = weights
        self.means = means
        self.scale_tril = scale_tril
        self.covs = scale_tril @ scale_tril.mT
        self._log_weights = torch.log(weights)

    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    @property
    def dim(self) -> int:
        return self.means.shape[-1]

    @classmethod
    def fit(cls, z: torch.Tensor, n_components: int, *, jitter: float = 1e-6,
            max_iter: int = 200, tol: float = 1e-5,
            generator: Optional[torch.Generator] = None) -> "GaussianMixture":
        """EM fit of a ``n_components`` mixture to ``z`` (shape ``(n, d)``).

        ``jitter`` loads every component covariance diagonal so the Choleskys stay
        stable. ``K = 1`` is the plain sample mean and covariance. EM stops on a
        relative log-likelihood change below ``tol`` or after ``max_iter`` steps.
        """
        z = z.detach()
        n, d = z.shape
        k = int(n_components)
        eye = jitter * torch.eye(d, dtype=z.dtype, device=z.device)
        if k == 1:
            cov = torch.cov(z.T).reshape(d, d) + eye
            return cls(z.new_ones(1), z.mean(0, keepdim=True),
                       torch.linalg.cholesky(cov).unsqueeze(0))

        means = _kmeanspp_init(z, k, generator)
        covs = (torch.cov(z.T).reshape(d, d) + eye).expand(k, d, d).clone()
        weights = z.new_full((k,), 1.0 / k)
        prev = None
        for _ in range(max_iter):
            log_r = torch.log(weights) + MVN(means, covariance_matrix=covs).log_prob(z.unsqueeze(1))
            ll = torch.logsumexp(log_r, dim=1)
            total = float(ll.sum())
            r = torch.exp(log_r - ll[:, None])                       # (n, K)
            nk = r.sum(0).clamp_min(1e-8)
            weights = nk / n
            means = (r.T @ z) / nk[:, None]
            for j in range(k):
                diff = z - means[j]
                cov = (r[:, j, None] * diff).T @ diff / nk[j] + eye
                covs[j] = 0.5 * (cov + cov.T)
            if prev is not None and abs(total - prev) <= tol * abs(prev):
                break
            prev = total
        return cls(weights, means, torch.linalg.cholesky(covs))

    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """Mixture log-density at ``z`` (shape ``(..., d)`` -> ``(...)``)."""
        lp = MVN(self.means, scale_tril=self.scale_tril).log_prob(z.unsqueeze(-2))
        return torch.logsumexp(self._log_weights + lp, dim=-1)

    def sample(self, n: int, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draw ``n`` points (shape ``(n, d)``)."""
        if self.n_components == 1:
            comp = torch.zeros(n, dtype=torch.long, device=self.means.device)
        else:
            comp = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        eps = torch.randn(n, self.dim, dtype=self.means.dtype,
                          device=self.means.device, generator=generator)
        return self.means[comp] + torch.einsum("nij,nj->ni", self.scale_tril[comp], eps)

    def conditional(self, a_idx, b_idx, z_a: torch.Tensor, *,
                    jitter: float = 1e-6) -> "ConditionalGaussianMixture":
        """Conditional mixture ``q(z_b | z_a)`` for query points ``z_a`` (shape
        ``(M, |a|)``), with blocks ``a_idx`` (given) and ``b_idx`` (free).

        Each component keeps its Gaussian conditional ``N(mu_b + Sigma_ba
        Sigma_aa^{-1}(z_a - mu_a), Sigma_bb - Sigma_ba Sigma_aa^{-1} Sigma_ab)``,
        and its mixing weight is scaled by the component's responsibility for
        ``z_a`` under the ``a``-marginal.
        """
        device = self.means.device
        a_t = torch.as_tensor(a_idx, device=device)
        b_t = torch.as_tensor(b_idx, device=device)
        nb = b_t.numel()

        mu_a, mu_b = self.means[:, a_t], self.means[:, b_t]           # (K, |a|), (K, |b|)
        Saa = self.covs[:, a_t[:, None], a_t]                         # (K, |a|, |a|)
        Sab = self.covs[:, a_t[:, None], b_t]                         # (K, |a|, |b|)
        Sbb = self.covs[:, b_t[:, None], b_t]                         # (K, |b|, |b|)
        A = torch.linalg.solve(Saa, Sab)                             # (K, |a|, |b|)
        S_cond = Sbb - Sab.mT @ A
        eye = jitter * torch.eye(nb, dtype=z_a.dtype, device=device)
        L_cond = torch.linalg.cholesky(S_cond + eye)                 # (K, |b|, |b|)

        diff = z_a[:, None, :] - mu_a[None]                          # (M, K, |a|)
        mu_cond = mu_b[None] + torch.einsum("mka,kab->mkb", diff, A)  # (M, K, |b|)
        resp = MVN(mu_a, covariance_matrix=Saa).log_prob(z_a[:, None, :])  # (M, K)
        log_w = torch.log_softmax(self._log_weights + resp, dim=1)
        return ConditionalGaussianMixture(mu_cond, L_cond, log_w)


class ConditionalGaussianMixture:
    """Per-query-point conditional mixture from :meth:`GaussianMixture.conditional`.

    Parameters
    ----------
    means : Tensor, shape (M, K, |b|)
        Component conditional means, one set per query point.
    scale_tril : Tensor, shape (K, |b|, |b|)
        Component conditional Cholesky factors, shared across query points.
    log_weights : Tensor, shape (M, K)
        Per-query-point log mixing weights (the responsibilities).
    """

    def __init__(self, means: torch.Tensor, scale_tril: torch.Tensor,
                 log_weights: torch.Tensor):
        self.means = means
        self.scale_tril = scale_tril
        self.log_weights = log_weights

    def log_prob(self, z_b: torch.Tensor) -> torch.Tensor:
        """Conditional log-density at ``z_b`` (shape ``(M, n, |b|)`` -> ``(M, n)``)."""
        parts = []
        for k in range(self.means.shape[1]):
            comp = MVN(self.means[:, k, None, :], scale_tril=self.scale_tril[k])
            parts.append(self.log_weights[:, k, None] + comp.log_prob(z_b))
        return torch.logsumexp(torch.stack(parts), dim=0)

    def sample(self, n: int, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        """Draw ``n`` points per query point (shape ``(M, n, |b|)``)."""
        M, K, nb = self.means.shape
        if K == 1:
            comp = torch.zeros(M, n, dtype=torch.long, device=self.means.device)
        else:
            comp = torch.multinomial(torch.exp(self.log_weights), n,
                                     replacement=True, generator=generator)   # (M, n)
        m = torch.arange(M, device=self.means.device)[:, None].expand(M, n)
        mean_sel = self.means[m, comp]                                        # (M, n, |b|)
        tril_sel = self.scale_tril[comp]                                      # (M, n, |b|, |b|)
        eps = torch.randn(M, n, nb, dtype=self.means.dtype,
                          device=self.means.device, generator=generator)
        return mean_sel + torch.einsum("mnij,mnj->mni", tril_sel, eps)
