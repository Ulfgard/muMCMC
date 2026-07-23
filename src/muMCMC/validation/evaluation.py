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


def _fit_gaussian(z: torch.Tensor, jitter: float) -> torch.distributions.MultivariateNormal:
    """Full-covariance normal fitted to ``z`` (shape ``(n, d)``).

    ``jitter`` loads the diagonal so the Cholesky is stable even when the sample
    covariance is rank-deficient.
    """
    d = z.shape[-1]
    mean = z.mean(dim=0)
    cov = torch.cov(z.T).reshape(d, d)
    cov = cov + jitter * torch.eye(d, dtype=cov.dtype, device=cov.device)
    return torch.distributions.MultivariateNormal(
        mean, scale_tril=torch.linalg.cholesky(cov))


def _bar_gaussian(z: torch.Tensor, log_target, *, n_q: Optional[int] = None,
                  jitter: float = 1e-6, generator: Optional[torch.Generator] = None,
                  log_target_z: Optional[torch.Tensor] = None,
                  q: Optional[torch.distributions.MultivariateNormal] = None) -> float:
    """BAR log-evidence of ``exp(log_target)`` from its draws ``z`` (shape ``(n, d)``).

    Fits a full-covariance normal ``q̂`` to ``z``, draws ``n_q`` points from it,
    and solves the BAR root over ``W = log_target − log q̂`` on both sets.

    Parameters
    ----------
    z : Tensor, shape (n, d)
        Draws from the normalized target, in the coordinates of ``log_target``.
    log_target : callable
        Maps ``(N, d) -> (N,)``, the unnormalized target log-density ``log f``.
    n_q : int, optional
        Number of ``q̂`` draws. Default is ``n``.
    jitter : float
        Diagonal loading for the covariance Cholesky.
    generator : torch.Generator, optional
        RNG for the ``q̂`` draws.
    log_target_z : Tensor, optional
        Precomputed ``log_target(z)``. Pass it to avoid re-evaluating an
        expensive target on the input draws.
    q : MultivariateNormal, optional
        A ``q̂`` already fitted to ``z``. Pass it to avoid refitting.

    Returns
    -------
    float
        BAR estimate of ``log ∫ f``.
    """
    n, d = z.shape
    if q is None:
        q = _fit_gaussian(z, jitter)
    n0 = n if n_q is None else int(n_q)
    eps = torch.randn(n0, d, dtype=z.dtype, device=z.device, generator=generator)
    z_q = q.loc + eps @ q.scale_tril.mT
    lf_z = log_target(z) if log_target_z is None else log_target_z
    W_post = lf_z - q.log_prob(z)
    W_q = log_target(z_q) - q.log_prob(z_q)
    return _bar_root(W_post, W_q)


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
        Number of ``q̂`` draws ``n0`` for the pooled estimate. Default is the
        number of posterior draws. Per-chain estimates use their own chain size.
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
        self._jitter = jitter
        self._generator = generator

        # Constrained free vector grouped by chain: (K, n, d).
        theta_free = self.space.to_free_vector(samples)
        if theta_free.dim() == 2:                   # (n, d) -> single chain
            theta_free = theta_free.unsqueeze(0)
        K, n, d = theta_free.shape
        self._n_chains, self._n_per_chain, self._d = K, n, d

        # Unconstrained draws grouped by chain. Detached so the cached draws (and
        # everything derived from them: log f, q̂) never carry an autograd graph.
        self._z = self.space.map_to_unconstrained_vector(theta_free).mapped_point.detach()
        self._n1 = K * n
        self._n0 = K * n if n_q is None else int(n_q)

        # log f on the posterior draws, the one expensive evaluation. Reused for
        # the pooled estimate, the per-chain estimates, and the W diagnostic.
        self._log_f_post = self._log_target(self._z.reshape(K * n, d))

        # Joint q̂ over all coordinates, fitted once. Drives the pooled estimate,
        # the W diagnostic, and the conditional proposal for marginals.
        self._q_pool = _fit_gaussian(self._z.reshape(K * n, d), jitter)

        # Main estimate: BAR on the pooled draws over all chains.
        self._log_evidence = _bar_gaussian(
            self._z.reshape(K * n, d), self._log_target, n_q=self._n0,
            jitter=jitter, generator=generator, log_target_z=self._log_f_post,
            q=self._q_pool)

    def _log_target(self, z: torch.Tensor) -> torch.Tensor:
        """``log f(z) = −evaluate_model(z).value`` for a batch ``(N, d)``."""
        return -self.sampler.evaluate_model(z)[0].value

    @property
    def log_evidence(self) -> float:
        """BAR estimate of ``log p(x)``, pooled over all chains."""
        return self._log_evidence

    @cached_property
    def entropy(self) -> float:
        """Posterior entropy ``H[p(y|x)] = logZ − E_post[loglik] − E_post[log_prior]``,
        a plain Monte Carlo average over the draws (w.r.t. ``dy``)."""
        z = self._z.reshape(self._n1, self._d)
        theta_free = self.space.map_to_constrained_vector(z).mapped_point
        loglik = self._tempered_loglik(z)
        log_prior = self.space.prior_log_prob_vector(theta_free)
        return self.log_evidence - float(loglik.mean()) - float(log_prior.mean())

    def information_gain(self, y_star: dict, *, target_ess: Optional[float] = None,
                         max_marginal: Optional[int] = None, prior_weight: float = 0.5,
                         generator: Optional[torch.Generator] = None,
                         return_ess: bool = False):
        """``log[ p(y*|x) / p(y*) ]`` at constrained points ``y*``.

        With every free name present this is ``loglik(y*) − logZ`` (the prior
        cancels). With a subset present the rest is marginalized out, giving the
        marginal information gain ``log ∫ p(x|y*_a,y_b) p(y_b) dy_b − logZ`` by the
        same importance sampler as :meth:`log_posterior`.
        """
        excluded = [name for name in self.space.free_names if name not in y_star]
        if not excluded:
            z = self.space.map_to_unconstrained_vector(
                self.space.to_free_vector(y_star)).mapped_point
            ig = self._tempered_loglik(z) - self.log_evidence
            return (ig, None) if return_ess else ig

        log_post, ess = self._log_marginal_posterior(
            y_star, excluded, target_ess, max_marginal, prior_weight, generator)
        ig = log_post - self.space.prior_log_prob(y_star)
        return (ig, ess) if return_ess else ig

    def log_posterior(self, y: dict, *, target_ess: Optional[float] = None,
                      max_marginal: Optional[int] = None, prior_weight: float = 0.5,
                      generator: Optional[torch.Generator] = None,
                      return_ess: bool = False):
        """``log p(y|x)`` at constrained points ``y``, a density w.r.t. ``dy``.

        The free names present in ``y`` select the marginal. All names gives the
        exact full density ``loglik(y) + log_prior(y) − logZ``. A subset gives the
        marginal over those names, integrating the rest out by importance
        sampling from a mixture of the prior and the joint ``q̂`` conditioned on
        ``y_a`` (``prior_weight`` is the mixture weight on the prior). Draws are
        added until every query point reaches ``target_ess`` or ``max_marginal``
        draws are spent.

        Parameters
        ----------
        y : dict[str, Tensor]
            Constrained query points keyed by free name. All free names gives the
            full density, a subset the marginal over those names.
        target_ess : float, optional
            Per-query-point weight ESS to reach before stopping. Default draws
            ``max_marginal`` in one shot.
        max_marginal : int, optional
            Cap on importance draws. Default is the posterior draw count.
        prior_weight : float
            Mixture weight on the prior component, in ``[0, 1]``. ``0`` is the pure
            conditional proposal, ``1`` plain prior sampling.
        generator : torch.Generator, optional
            RNG for the marginal draws.
        return_ess : bool
            If True, also return the per-query-point weight ESS (``None`` for the
            exact full density).
        """
        excluded = [name for name in self.space.free_names if name not in y]
        if not excluded:
            theta_free = self.space.to_free_vector(y)
            z = self.space.map_to_unconstrained_vector(theta_free).mapped_point
            value = self.sampler.evaluate_model(z)[0].value
            jac = self.space.map_to_constrained_vector(z).jacobian_log_det
            # log p(y|x) = loglik + log_prior − logZ = −value − log|dθ/dz| − logZ.
            log_post = -value - jac - self.log_evidence
            return (log_post, None) if return_ess else log_post

        log_post, ess = self._log_marginal_posterior(
            y, excluded, target_ess, max_marginal, prior_weight, generator)
        return (log_post, ess) if return_ess else log_post

    def _log_marginal_posterior(self, y: dict, excluded: list,
                                target_ess: Optional[float], max_marginal: Optional[int],
                                alpha: float, generator: Optional[torch.Generator]):
        """Marginal ``log p(y_a|x)`` with the ``excluded`` block integrated out by
        mixture importance sampling, drawn adaptively. Returns ``(log_post, ess)``."""
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"prior_weight must be in [0, 1], got {alpha}")
        free_names = self.space.free_names
        a_idx = [i for i, name in enumerate(free_names) if name in y]
        b_idx = [i for i, name in enumerate(free_names) if name not in y]
        if not a_idx:
            raise ValueError("log_posterior needs at least one free name in y")
        a_names = [free_names[i] for i in a_idx]
        b_names = [free_names[i] for i in b_idx]
        na, nb = len(a_idx), len(b_idx)

        M = y[a_names[0]].shape[0]
        d = self._d
        dtype, device = self._z.dtype, self._z.device
        a_t = torch.tensor(a_idx, device=device)
        b_t = torch.tensor(b_idx, device=device)

        # Query y_a -> unconstrained z_a. The transform is elementwise, so fill
        # the b block with any interior placeholder (the image of z = 0) and keep
        # only the a coordinates of the result.
        y0 = self.space.from_vector(
            self.space.map_to_constrained_vector(torch.zeros(d, dtype=dtype, device=device)).mapped_point)
        full = {name: y[name] for name in a_names}
        for name in b_names:
            full[name] = y0[name].expand(M)
        z_a = self.space.map_to_unconstrained_vector(
            self.space.to_free_vector(full)).mapped_point[:, a_t]        # (M, |a|)

        # Conditional Gaussian q(z_b | z_a) from the pooled joint fit.
        mu, Sigma = self._q_pool.loc, self._q_pool.covariance_matrix
        A = torch.linalg.solve(Sigma[a_t[:, None], a_t], Sigma[a_t[:, None], b_t])
        mu_cond = mu[b_t] + (z_a - mu[a_t]) @ A                          # (M, |b|)
        S_cond = Sigma[b_t[:, None], b_t] - Sigma[a_t[:, None], b_t].mT @ A
        L_cond = torch.linalg.cholesky(S_cond + self._jitter * torch.eye(nb, dtype=dtype, device=device))
        q_b = torch.distributions.MultivariateNormal(mu_cond.unsqueeze(1), scale_tril=L_cond)

        def draw(n_prior, n_cond):
            """One mixture batch -> (loglik, log_prior_z, log_qcond), each (M, n)."""
            blocks = []
            if n_cond > 0:
                eps = torch.randn(n_cond, nb, dtype=dtype, device=device, generator=generator)
                blocks.append(mu_cond.unsqueeze(1) + eps @ L_cond.mT)
            if n_prior > 0:
                prior = self.space.sample(n_prior, generator=generator)  # shared over query points
                full_p = {name: y0[name].expand(n_prior) for name in a_names}
                for name in b_names:
                    full_p[name] = prior[name]
                z_bp = self.space.map_to_unconstrained_vector(
                    self.space.to_free_vector(full_p)).mapped_point[:, b_t]
                blocks.append(z_bp[None].expand(M, n_prior, nb))
            z_b = torch.cat(blocks, dim=1)                               # (M, n, |b|)
            n = z_b.shape[1]
            z_full = torch.empty(M, n, d, dtype=dtype, device=device)
            z_full[..., a_t] = z_a[:, None, :].expand(M, n, na)
            z_full[..., b_t] = z_b
            z_flat = z_full.reshape(M * n, d)
            tmap = self.space.map_to_constrained_vector(z_flat)
            jac_b = torch.log(tmap.jacobian_diag[:, b_t]).sum(-1).reshape(M, n)
            prior_b = self.space.prior_log_prob(
                {name: tmap.mapped_point[:, i] for name, i in zip(b_names, b_idx)}).reshape(M, n)
            loglik = self._tempered_loglik(z_flat).reshape(M, n)
            return loglik, prior_b + jac_b, q_b.log_prob(z_b)

        # Accumulate mixture draws until every point reaches target_ess or the
        # cap is spent. log_qmix uses the running sampling fractions, so weights
        # stay comparable as draws pool across rounds.
        budget = self._n1 if max_marginal is None else int(max_marginal)
        target = float("inf") if target_ess is None else float(target_ess)
        loglik = log_pi = log_qc = None
        n_prior_tot = n_cond_tot = drawn = 0
        step = budget if math.isinf(target) else min(budget, 8192)
        while drawn < budget:
            n = min(step, budget - drawn)
            n_prior = int(round(alpha * n))
            ll, lp, lq = draw(n_prior, n - n_prior)
            loglik = ll if loglik is None else torch.cat([loglik, ll], 1)
            log_pi = lp if log_pi is None else torch.cat([log_pi, lp], 1)
            log_qc = lq if log_qc is None else torch.cat([log_qc, lq], 1)
            n_prior_tot += n_prior
            n_cond_tot += n - n_prior
            drawn += n

            terms = []
            if n_prior_tot > 0:
                terms.append(math.log(n_prior_tot / drawn) + log_pi)
            if n_cond_tot > 0:
                terms.append(math.log(n_cond_tot / drawn) + log_qc)
            log_qmix = terms[0] if len(terms) == 1 else torch.logaddexp(terms[0], terms[1])
            log_w = loglik + log_pi - log_qmix
            ess = torch.exp(2 * torch.logsumexp(log_w, 1) - torch.logsumexp(2 * log_w, 1))
            if float(ess.min()) >= target:
                break
            step *= 2

        log_integral = torch.logsumexp(log_w, 1) - math.log(drawn)
        log_post = self.space.prior_log_prob(y) + log_integral - self.log_evidence
        return log_post, ess

    def _tempered_loglik(self, z: torch.Tensor) -> torch.Tensor:
        """``beta·loglik(z) = −beta·U_lik``, the tempered log-likelihood."""
        pot = self.sampler.evaluate_model(z)[0]
        return -pot.value if pot.base is None else pot.base - pot.value

    @cached_property
    def _per_chain_log_evidence(self) -> torch.Tensor:
        """Independent per-chain BAR estimate, shape (K,).

        Each chain runs the whole estimator on its own draws, fitting its own
        ``q̂`` and drawing its own ``q̂`` points. The estimates share no Monte
        Carlo data, so they are independent replicates of the pooled estimate.
        The precomputed ``log f`` on the posterior draws is sliced per chain, so
        no chain re-evaluates the target on its own draws.
        """
        K, n = self._n_chains, self._n_per_chain
        return torch.tensor([
            _bar_gaussian(self._z[k], self._log_target, jitter=self._jitter,
                          generator=self._generator,
                          log_target_z=self._log_f_post[k * n:(k + 1) * n])
            for k in range(K)
        ])

    @cached_property
    def diagnostics(self) -> dict:
        """Scalar diagnostics for automated gating.

        - ``W_percentiles``: percentiles of ``W = log f − log q̂`` on the pooled
          posterior draws. A heavy upper tail means ``q̂`` misses posterior mass.
        - ``per_chain_log_evidence``: an independent per-chain estimate
          (shape ``(K,)``), each chain fitting its own ``q̂``.
        - ``log_evidence_se``: standard error of the pooled estimate from the
          spread of the per-chain replicates. The replicates are independent, so
          their spread captures both the posterior and the ``q̂`` Monte Carlo
          variance with no effective-sample-size correction. Present for more
          than one chain.
        - ``n1``, ``n0``: raw posterior and ``q̂`` draw counts of the pooled
          estimate.
        """
        z_flat = self._z.reshape(self._n1, self._d)
        Wp = (self._log_f_post - self._q_pool.log_prob(z_flat)).double()
        probs = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                             dtype=torch.float64)
        percentiles = {float(p): float(v)
                       for p, v in zip(probs, torch.quantile(Wp, probs))}
        K = self._n_chains
        per_chain = self._per_chain_log_evidence
        out = {
            "W_percentiles": percentiles,
            "per_chain_log_evidence": per_chain,
            "n1": self._n1,
            "n0": self._n0,
        }
        if K > 1:
            out["log_evidence_se"] = float(
                per_chain.std(unbiased=True) / math.sqrt(K))
        return out
