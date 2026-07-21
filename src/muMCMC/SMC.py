from __future__ import annotations

import math
import sys
from typing import Dict

import torch
from tqdm.auto import tqdm

from .BaseSampler import BaseSampler


def _systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    """Systematic resampling along the last axis of the normalized ``weights``
    (``(..., M)``) to ``(..., M)`` ancestor indices, drawn with replacement in
    proportion to them.
    """
    M = weights.shape[-1]
    batch = weights.shape[:-1]
    u0 = torch.rand(batch + (1,), device=weights.device, dtype=weights.dtype)
    positions = (torch.arange(M, device=weights.device, dtype=weights.dtype) + u0) / M
    cumsum = weights.cumsum(dim=-1)
    cumsum[..., -1] = 1.0                              # guard rounding at the top
    return torch.searchsorted(cumsum, positions).clamp_(max=M - 1)


def _rhat(x: torch.Tensor) -> torch.Tensor:
    """Gelman-Rubin R-hat across the C populations of ``x`` (``(C, M)``): the
    ratio of the pooled variance estimate to the within-population variance.
    """
    C, M = x.shape
    chain_mean = x.mean(dim=1)
    W = x.var(dim=1, unbiased=True).mean()
    B = M * chain_mean.var(unbiased=True)
    var_plus = (M - 1) / M * W + B / M
    return torch.sqrt(var_plus / W)


class SMC:
    """
    Adaptive tempered Sequential Monte Carlo on top of a batched sampler.

    Transports a particle population from the prior (beta=0) to the posterior
    (beta=1) along ``prior * likelihood**beta``.  Each stage is reweight ->
    systematic resample -> mutate.  The wrapped ``sampler`` supplies the
    mutation kernel via its ``init`` / ``step`` / ``beta`` interface; the
    likelihood potential for reweighting is read from the kernel state's
    tempered potential (``state.U.lik``).

    ``num_chains`` independent populations are run in parallel over the kernel's
    batch axis: each has its own particles, resampling, and adaptively-bisected
    schedule (``beta_{k+1}`` chosen so its post-reweighting ESS equals
    ``ess_target * num_particles``).  Independent populations give a between-run
    estimate of Monte Carlo error -- the spread of the log-evidence estimates
    and R-hat across populations (see :meth:`diagnostics`).  The mutation kernel
    runs at a fixed step size, its adaptation frozen via ``end_warmup``.

    Parameters
    ----------
    sampler
        The mutation kernel (a :class:`BaseSampler`).
    ess_target : float
        Post-reweighting ESS as a fraction of the particle count, in (0, 1).
    num_mcmc_steps : int
        Mutation transitions applied per temperature.
    min_dbeta : float
        Smallest temperature increment, to guarantee progress on peaked targets.
    """

    def __init__(
        self,
        sampler: BaseSampler,
        *,
        ess_target: float = 0.5,
        num_mcmc_steps: int = 5,
        min_dbeta: float = 1e-4,
    ):
        if not 0.0 < ess_target < 1.0:
            raise ValueError(f"ess_target must be in (0, 1), got {ess_target}")

        self.sampler = sampler
        self.space = sampler.space
        self.ess_target = ess_target
        self.num_mcmc_steps = num_mcmc_steps
        self.min_dbeta = min_dbeta

        # standing diagnostics, filled by run_smc
        self._betas = []
        self._ess = []
        self._accept = []
        self._log_evidence = None
        self._r_hat = {}

    def _next_dbeta(self, u_lik: torch.Tensor, max_dbeta: torch.Tensor,
                    max_iter: int = 60) -> torch.Tensor:
        """Per-chain temperature increment whose post-reweighting ESS equals
        ``ess_target * M``.  ``u_lik`` is ``(C, M)`` and ``max_dbeta`` ``(C,)``.
        ESS(d) = exp(2*LSE(-d*u) - LSE(-2*d*u)) is monotone decreasing in ``d``;
        each chain takes ``max_dbeta`` if it already meets the target, else
        bisects, floored at ``min_dbeta`` to guarantee progress.
        """
        M = u_lik.shape[-1]
        max_dbeta = torch.as_tensor(max_dbeta, dtype=u_lik.dtype, device=u_lik.device)
        log_target = math.log(self.ess_target * M)

        def log_ess(d):
            a = torch.logsumexp(-d.unsqueeze(-1) * u_lik, dim=-1)
            b = torch.logsumexp(-2.0 * d.unsqueeze(-1) * u_lik, dim=-1)
            return 2.0 * a - b

        full_ok = log_ess(max_dbeta) >= log_target
        lo = torch.zeros_like(max_dbeta)
        hi = max_dbeta.clone()
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            below = log_ess(mid) < log_target
            hi = torch.where(below, mid, hi)
            lo = torch.where(below, lo, mid)
        d = torch.minimum(torch.clamp(lo, min=self.min_dbeta), max_dbeta)
        return torch.where(full_ok, max_dbeta, d)

    def run_smc(
        self,
        num_particles: int,
        *,
        num_chains: int = 1,
        disable_progbar: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Transport ``num_chains`` independent populations of ``num_particles``
        each from the prior to the posterior (beta = 1) and return the final
        populations in constrained space, keyed by parameter name, each of
        shape ``(num_chains, num_particles)``.  Post-run schedule / ESS /
        evidence / R-hat are available via :meth:`diagnostics`.
        """
        sampler, space = self.sampler, self.space
        C, M, N = num_chains, num_particles, num_chains * num_particles

        # initial populations ~ prior, in unconstrained free coordinates
        theta0 = space.to_vector(space.sample(N))                 # (N, d_full)
        z = sampler._init_z_free(theta0)                          # (N, d)
        d = z.shape[-1]

        beta = torch.zeros(C, dtype=z.dtype, device=z.device)     # per-chain
        self._betas = [beta.clone()]
        self._ess = []
        self._accept = []
        self._log_evidence = torch.zeros(C, dtype=z.dtype, device=z.device)

        # Evaluate the prior population once; the kernel state carries U_lik,
        # so reweighting reads it (grad-free) instead of recomputing.
        sampler.beta = beta.unsqueeze(-1).expand(C, M).reshape(N)
        s = sampler.init(z)

        bar_format = "{l_bar}{bar}| {n:.3f}/{total:.3f} [{elapsed}{postfix}]"
        with tqdm(total=1.0, file=sys.stderr, disable=disable_progbar,
                  bar_format=bar_format, desc="SMC") as bar:
            progressed = 0.0
            while bool((beta < 1.0).any()):
                # reweight: per-chain schedule + incremental weights from U_lik
                u_lik = s.U.lik.reshape(C, M)                     # (C, M), from the state
                dbeta = self._next_dbeta(u_lik, 1.0 - beta)       # (C,)
                log_w = -dbeta.unsqueeze(-1) * u_lik              # (C, M)
                self._log_evidence += torch.logsumexp(log_w, dim=-1) - math.log(M)

                # resample: systematic, within each chain
                W = torch.softmax(log_w, dim=-1)                  # (C, M)
                ess = 1.0 / (W * W).sum(dim=-1)                   # (C,)
                idx = _systematic_resample(W)                     # (C, M)
                z = s.q.reshape(C, M, d).gather(
                    1, idx.unsqueeze(-1).expand(C, M, d)).reshape(N, d)
                beta = beta + dbeta

                # mutate: fixed kernel at each chain's temperature, adaptation frozen
                sampler.beta = beta.unsqueeze(-1).expand(C, M).reshape(N)
                s = sampler.init(z)
                sampler.end_warmup()
                for _ in range(self.num_mcmc_steps):
                    s = sampler.step(s)

                self._betas.append(beta.clone())
                self._ess.append(ess)
                self._accept.append(
                    sampler.diagnostics()["accept_rate"].reshape(C, M).mean(dim=-1))
                new = float(beta.min())
                bar.update(new - progressed)
                progressed = new
                bar.set_postfix(beta=f"{new:.3f}", ess=f"{float(ess.mean()):.0f}",
                                logZ=f"{float(self._log_evidence.mean()):.2f}",
                                refresh=False)

        sampler.beta = 1.0                                        # restore kernel default
        z = s.q                                                   # final mutated population

        theta_free = space.map_to_constrained_vector(z).mapped_point
        free = space.from_vector(theta_free)
        if C >= 2:
            self._r_hat = {name: _rhat(v.reshape(C, M)) for name, v in free.items()}
        return {k: v.reshape(C, M) for k, v in space.add_fixed(free).items()}

    def diagnostics(self) -> dict:
        """Post-run schedule and population diagnostics (empty before
        :meth:`run_smc`).

          ``betas``       per-chain temperature schedule, ``(stages+1, num_chains)``
          ``ess``         per-stage per-chain ESS after reweighting
          ``accept_rate`` per-stage per-chain mean mutation acceptance
          ``log_evidence``          per-chain log marginal likelihood, ``(num_chains,)``
          ``log_evidence_estimate`` combined estimate, log-mean of the per-chain values
          ``log_evidence_se``       between-chain standard error of the estimate
          ``r_hat``       per free parameter, Gelman-Rubin across populations
                          (only for num_chains >= 2)
        """
        if self._log_evidence is None:
            return {}
        logZ = self._log_evidence
        C = logZ.shape[0]
        return {
            "betas": torch.stack(self._betas),
            "ess": torch.stack(self._ess),
            "accept_rate": torch.stack(self._accept),
            "log_evidence": logZ,
            "log_evidence_estimate": torch.logsumexp(logZ, dim=0) - math.log(C),
            "log_evidence_se": logZ.std(unbiased=True) / math.sqrt(C) if C >= 2
                               else torch.zeros((), dtype=logZ.dtype),
            "r_hat": self._r_hat,
        }
