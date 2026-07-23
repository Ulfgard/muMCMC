"""Posterior evaluation from MCMC draws: evidence and posterior density.

Given posterior draws ``y ~ p(y|x)``, the likelihood ``p(x|y)`` bound in the
sampler, and the space's prior ``p(y)``, this module estimates

    log p(x) = log ∫ p(x|y) p(y) dy          (the evidence, ``log_evidence``)
    log p(y|x) = loglik(y) + log p(y) − log p(x)   (``log_posterior``)

The evidence uses BAR (Bennett acceptance ratio, as reverse logistic
regression).  Work happens in the sampler's unconstrained coordinates ``z``,
where ``sampler.evaluate_model(z).value`` is exactly ``−log f(z)`` for the
unnormalized posterior density ``f(z) = p(x|y(z)) p(y(z)) |dy/dz|`` whose
integral over ``z`` is the evidence.  A Gaussian reference ``q̂`` is fitted to
the ``z``-draws; with the log-ratio

    W(z) = log f(z) − log q̂(z) = −evaluate_model(z).value − log q̂(z)

evaluated on the posterior draws (``n1`` of them) and on ``n0`` draws from
``q̂``, the scalar ``b`` solving ``Σ_pooled σ(W + b) = n1`` (σ = expit) gives

    log p(x) = log(n1 / n0) − b .

``g(b) = n1 − Σ σ(W + b)`` is strictly decreasing from ``+n1`` to ``−n0``, so
the root is unique and always brackets.

The evidence is reparameterization-invariant, so the ``z``-space value equals
the ``y``-space integral.  Everything uses the sampler's current ``beta``; a
tempered sampler yields the corresponding tempered evidence, which is the
caller's choice.

The prior is assumed proper and normalized (see the prior contract in
``spaces``); an unnormalized prior shifts ``log p(x)`` by its missing constant.
"""
from __future__ import annotations

import math
from functools import cached_property
from typing import Optional

import torch


def _bar_root(W_post: torch.Tensor, W_q: torch.Tensor,
              *, max_expand: int = 64, max_iter: int = 100) -> float:
    """Solve ``Σ_pooled σ(W + b) = n1`` for ``b`` and return ``log(n1/n0) − b``.

    Pure core over arrays, directly testable.

    Parameters
    ----------
    W_post : Tensor, shape (n1,)
        Log-ratio ``W = log f − log q̂`` on the posterior draws.
    W_q : Tensor, shape (n0,)
        The same log-ratio on the ``q̂`` draws.

    Returns
    -------
    float
        ``log p(x)`` under the pooled BAR estimating equation.
    """
    n1 = W_post.numel()
    n0 = W_q.numel()
    W = torch.cat([W_post.reshape(-1), W_q.reshape(-1)]).double()

    def g(b: float) -> float:                       # strictly decreasing in b
        return n1 - torch.sigmoid(W + b).sum().item()

    # Bracket [lo, hi] with g(lo) > 0 > g(hi); g -> +n1 as b -> -inf, -> -n0 as
    # b -> +inf, so expansion always terminates.
    lo, hi = -1.0, 1.0
    for _ in range(max_expand):
        if g(lo) > 0.0:
            break
        lo *= 2.0
    for _ in range(max_expand):
        if g(hi) < 0.0:
            break
        hi *= 2.0

    # Safeguarded Newton inside the bracket: Newton for speed (g is smooth and
    # monotone), bisection fallback whenever a step leaves the bracket.
    b = 0.5 * (lo + hi)
    for _ in range(max_iter):
        s = torch.sigmoid(W + b)
        gb = n1 - s.sum().item()
        if gb > 0.0:
            lo = b
        else:
            hi = b
        if abs(gb) < 1e-10:
            break
        gp = -(s * (1.0 - s)).sum().item()          # g'(b) < 0
        b_newton = b - gb / gp if gp != 0.0 else 0.5 * (lo + hi)
        b = b_newton if lo < b_newton < hi else 0.5 * (lo + hi)
    return math.log(n1 / n0) - b


class PosteriorEvaluation:
    """Evidence and posterior density from posterior draws.

    Precomputes and caches the quantities shared by the estimators -- the
    unconstrained draws, the fitted reference ``q̂``, and the BAR log-ratios --
    so that :attr:`log_evidence`, :meth:`log_posterior`, and :attr:`diagnostics`
    reuse them without recomputation.

    Parameters
    ----------
    sampler : MCMCSampler
        The sampler that drew ``samples``.  Supplies ``evaluate_model`` and
        ``space``; its current ``beta`` sets the temperature of the evidence.
    samples : dict[str, Tensor]
        Constrained draws keyed by free parameter name, as returned by
        ``run_mcmc`` (grouped by chain, shape ``(num_chains, num_samples)``; a
        single ungrouped axis ``(num_samples,)`` is also accepted).
    n_q : int, optional
        Number of ``q̂`` draws ``n0``.  Default: the number of posterior draws.
    jitter : float
        Diagonal loading added to the fitted covariance for a stable Cholesky.
    generator : torch.Generator, optional
        RNG for the ``q̂`` draws, for reproducibility.

    Notes
    -----
    v1 fits a full-covariance Gaussian ``q̂`` to the pooled draws and reuses all
    draws for both the fit and the estimate; the reuse bias is ``≈ p / (2
    n_eff)`` for a ``p``-parameter ``q̂``.  ``q̂`` quality affects variance, not
    consistency.
    """

    def __init__(
        self,
        sampler,
        samples: dict,
        *,
        n_q: Optional[int] = None,
        jitter: float = 1e-6,
        generator: Optional[torch.Generator] = None,
    ):
        self.sampler = sampler
        self.space = sampler.space

        # Constrained free vector grouped by chain: (K, n, d).
        theta_free = self.space.to_free_vector(samples)
        if theta_free.dim() == 2:                   # (n, d) -> single chain
            theta_free = theta_free.unsqueeze(0)
        K, n, d = theta_free.shape
        self._n_chains, self._n_per_chain, self._d = K, n, d

        # Unconstrained draws, flattened over chains for the shared computations.
        z = self.space.map_to_unconstrained_vector(theta_free).mapped_point
        z_flat = z.reshape(K * n, d)

        # Fit q̂: a full-covariance Gaussian on the pooled draws.
        mean = z_flat.mean(dim=0)
        cov = torch.cov(z_flat.T).reshape(d, d)
        cov = cov + jitter * torch.eye(d, dtype=cov.dtype, device=cov.device)
        L = torch.linalg.cholesky(cov)
        self._q = torch.distributions.MultivariateNormal(mean, scale_tril=L)

        n0 = K * n if n_q is None else int(n_q)
        eps = torch.randn(n0, d, dtype=z_flat.dtype, device=z_flat.device,
                          generator=generator)
        z_q = mean + eps @ L.T

        # W = log f − log q̂ = −evaluate_model(z).value − log q̂(z), on both sets.
        self._W_post = self._log_ratio(z_flat).reshape(K, n)
        self._W_q = self._log_ratio(z_q)
        self._n1 = K * n
        self._n0 = n0

    def _log_ratio(self, z: torch.Tensor) -> torch.Tensor:
        """``W(z) = −evaluate_model(z).value − log q̂(z)`` for a batch ``(N, d)``."""
        neg_log_f = self.sampler.evaluate_model(z)[0].value
        return -neg_log_f - self._q.log_prob(z)

    @cached_property
    def log_evidence(self) -> float:
        """BAR estimate of ``log p(x)`` (cached)."""
        return _bar_root(self._W_post, self._W_q)

    def log_posterior(self, y: dict) -> torch.Tensor:
        """``log p(y|x)`` at constrained points ``y`` (a dict keyed by free name).

        Vectorized over the batch axis of ``y``.  Returns the density with
        respect to the constrained measure ``dy``.  v1 requires all free names;
        the marginal over a name subset is planned for v2.
        """
        missing = [name for name in self.space.free_names if name not in y]
        if missing:
            raise NotImplementedError(
                f"log_posterior needs all free names; missing {missing}. "
                "Marginalizing over a name subset is planned for v2."
            )
        theta_free = self.space.to_free_vector(y)
        z = self.space.map_to_unconstrained_vector(theta_free).mapped_point
        value = self.sampler.evaluate_model(z)[0].value
        jac = self.space.map_to_constrained_vector(z).jacobian_log_det
        # log p(y|x) = loglik + log_prior − logZ = −value − log|dθ/dz| − logZ.
        return -value - jac - self.log_evidence

    @cached_property
    def _per_chain_log_evidence(self) -> torch.Tensor:
        """Per-chain BAR estimate against the shared ``q̂`` draws, shape (K,)."""
        return torch.tensor(
            [_bar_root(self._W_post[k], self._W_q) for k in range(self._n_chains)]
        )

    @cached_property
    def diagnostics(self) -> dict:
        """Scalar diagnostics for automated gating.

        - ``W_percentiles``: percentiles of ``W`` on the posterior draws; a heavy
          upper tail means ``q̂`` misses posterior mass.
        - ``per_chain_log_evidence``: per-chain ``logZ`` (shape ``(K,)``).
        - ``log_evidence_se``: standard error of the pooled estimate from the
          per-chain spread (``NaN`` for a single chain); absorbs autocorrelation.
        - ``n1``, ``n0``: raw posterior and ``q̂`` draw counts.
        """
        probs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                             dtype=torch.float64)
        Wp = self._W_post.reshape(-1).double()
        percentiles = {float(p): float(v)
                       for p, v in zip(probs, torch.quantile(Wp, probs))}
        per_chain = self._per_chain_log_evidence
        K = self._n_chains
        se = (float(per_chain.std(unbiased=True) / math.sqrt(K))
              if K > 1 else float("nan"))
        return {
            "W_percentiles": percentiles,
            "per_chain_log_evidence": per_chain,
            "log_evidence_se": se,
            "n1": self._n1,
            "n0": self._n0,
        }
