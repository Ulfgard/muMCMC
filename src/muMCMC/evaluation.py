"""Posterior evaluation from MCMC draws: evidence and posterior density.

Given posterior draws ``y ~ p(y|x)``, the likelihood ``p(x|y)`` bound in the
sampler, and the space's prior ``p(y)``, this module estimates

    log p(x) = log ∫ p(x|y) p(y) dy          (the evidence, ``log_evidence``)
    log p(y|x) = loglik(y) + log p(y) − log p(x)   (``log_posterior``)

The evidence uses BAR (Bennett acceptance ratio, as reverse logistic
regression). Work happens in the sampler's unconstrained coordinates ``z``,
where ``sampler.evaluate_model(z).value`` is exactly ``−log f(z)`` for the
unnormalized posterior density ``f(z) = p(x|y(z)) p(y(z)) |dy/dz|`` whose
integral over ``z`` is the evidence. A Gaussian reference ``q̂`` is fitted to
the ``z``-draws. With the log-ratio

    W(z) = log f(z) − log q̂(z) = −evaluate_model(z).value − log q̂(z)

evaluated on the posterior draws (``n1`` of them) and on ``n0`` draws from
``q̂``, the scalar ``b`` solving ``Σ_pooled σ(W + b) = n1`` (σ = expit) gives

    log p(x) = log(n1 / n0) − b .

``g(b) = n1 − Σ σ(W + b)`` is strictly decreasing from ``+n1`` to ``−n0``, so
the root is unique and always brackets.

The evidence is reparameterization-invariant, so the ``z``-space value equals
the ``y``-space integral. Everything uses the sampler's current ``beta``. A
tempered sampler yields the corresponding tempered evidence, which is the
caller's choice.

The prior is assumed proper and normalized (see the prior contract in
``spaces``). An unnormalized prior shifts ``log p(x)`` by its missing constant.
"""
from __future__ import annotations

import math
from functools import cached_property
from typing import Optional

import torch
from scipy.optimize import brentq
from scipy.special import expit


def _bar_root(W_post: torch.Tensor, W_q: torch.Tensor, *, pad: float = 1.0) -> float:
    """Solve ``Σ_pooled σ(W + b) = n1`` for ``b`` and return ``log(n1/n0) − b``.

    Pure core over arrays, directly testable.  ``g(b) = n1 − Σ σ(W + b)`` is
    strictly decreasing in ``b``.  Since ``σ`` is monotone, the pooled sum is
    squeezed by its extreme samples, ``N σ(W_min + b) ≤ Σ σ ≤ N σ(W_max + b)``
    with ``N = n1 + n0``, which pins the root in closed form:

        b* ∈ [ log(n1/n0) − W_max ,  log(n1/n0) − W_min ]

    (equivalently ``log p(x) ∈ [W_min, W_max]``). A small ``pad`` makes the sign
    change strict and absorbs the degenerate case where every ``W`` is equal. The
    root is then refined by ``scipy.optimize.brentq``.

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
    W = torch.cat([W_post.reshape(-1), W_q.reshape(-1)]).detach().double().cpu().numpy()

    def g(b: float) -> float:                       # strictly decreasing in b
        return n1 - expit(W + b).sum()

    offset = math.log(n1 / n0)
    lo = offset - float(W.max()) - pad              # g(lo) > 0
    hi = offset - float(W.min()) + pad              # g(hi) < 0
    return offset - brentq(g, lo, hi)


class PosteriorEvaluation:
    """Evidence and posterior density from posterior draws.

    Estimates the log-evidence ``log Z`` of the density the sampler targets and
    the posterior log-density ``log p(y|x)`` that follows from it. When that
    density is a valid posterior ``p(x|y) p(y)``, ``log Z`` is exactly the model
    evidence ``log p(x)``. Otherwise it is the log normalizer of whatever
    unnormalized density the sampler was run on.

    The estimator is BAR (Bennett acceptance ratio, Bennett 1976), cast as
    reverse logistic regression (Geyer 1994). A reference ``q̂`` is fitted to the
    draws as a full-covariance normal in the sampler's unconstrained coordinates,
    and ``log Z`` is the intercept discriminating the draws from samples of
    ``q̂``. Fitting ``q̂`` on the same draws leaves the estimator consistent. The
    quality of the normal fit sets the variance, not the limit.

    Parameters
    ----------
    sampler : MCMCSampler
        The sampler that drew ``samples``. Supplies ``evaluate_model`` and
        ``space``. Its current ``beta`` sets the temperature of the evidence.
    samples : dict[str, Tensor]
        Constrained draws keyed by free parameter name, as returned by
        ``run_mcmc`` (grouped by chain, shape ``(num_chains, num_samples)``). A
        single ungrouped axis ``(num_samples,)`` is also accepted.
    n_q : int, optional
        Number of ``q̂`` draws ``n0``. Default is the number of posterior draws.
    jitter : float
        Diagonal loading added to the fitted covariance for a stable Cholesky.
    generator : torch.Generator, optional
        RNG for the ``q̂`` draws, for reproducibility.
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

        Vectorized over the batch axis of ``y``. Returns the density with
        respect to the constrained measure ``dy``. v1 requires all free names.
        The marginal over a name subset is planned for v2.
        """
        missing = [name for name in self.space.free_names if name not in y]
        if missing:
            raise NotImplementedError(
                f"log_posterior needs all free names, missing {missing}. "
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

        - ``W_percentiles``: percentiles of ``W`` on the posterior draws. A heavy
          upper tail means ``q̂`` misses posterior mass.
        - ``per_chain_log_evidence``: per-chain ``logZ`` (shape ``(K,)``).
        - ``log_evidence_se``: standard error of the pooled estimate, taken from
          the spread of the per-chain estimates as independent replicates. Each
          chain estimate carries its own within-chain autocorrelation, so their
          spread already reflects it and no effective-sample-size correction is
          needed. Present only for more than one chain.
        - ``n1``, ``n0``: raw posterior and ``q̂`` draw counts.
        """
        probs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                             dtype=torch.float64)
        Wp = self._W_post.reshape(-1).double()
        percentiles = {float(p): float(v)
                       for p, v in zip(probs, torch.quantile(Wp, probs))}
        per_chain = self._per_chain_log_evidence
        K = self._n_chains
        out = {
            "W_percentiles": percentiles,
            "per_chain_log_evidence": per_chain,
            "n1": self._n1,
            "n0": self._n0,
        }
        if K > 1:
            out["log_evidence_se"] = float(per_chain.std(unbiased=True) / math.sqrt(K))
        return out
