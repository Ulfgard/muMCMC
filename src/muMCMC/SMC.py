from __future__ import annotations

import math
import sys
from typing import Dict

import torch
from tqdm.auto import tqdm

from .BaseSampler import BaseSampler


def _systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    """Systematic resampling of the normalized ``weights`` (``(N,)``) to
    ``(N,)`` ancestor indices, drawn with replacement in proportion to them.
    """
    N = weights.shape[0]
    u0 = torch.rand((), device=weights.device, dtype=weights.dtype)
    positions = (torch.arange(N, device=weights.device, dtype=weights.dtype) + u0) / N
    cumsum = torch.cumsum(weights, dim=0)
    cumsum[-1] = 1.0                                   # guard rounding at the top
    return torch.searchsorted(cumsum, positions).clamp_(max=N - 1)


class SMC:
    """
    Adaptive tempered Sequential Monte Carlo on top of a batched sampler.

    Transports a particle population from the prior (beta=0) to the posterior
    (beta=1) along ``prior * likelihood**beta``.  Each stage is reweight ->
    systematic resample -> mutate.  The wrapped ``sampler`` supplies the
    mutation kernel via its ``init`` / ``step`` / ``beta`` interface and the
    likelihood potential via ``potential_likelihood``.

    Only the schedule is adaptive: each ``beta_{k+1}`` is bisected so the
    post-reweighting ESS equals ``ess_target * num_particles``.  The mutation
    kernel runs at a fixed step size, its adaptation frozen via ``end_warmup``.

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
        self._log_evidence = 0.0

    def _next_dbeta(self, u_lik: torch.Tensor, max_dbeta: float,
                    max_iter: int = 60) -> float:
        """Temperature increment whose post-reweighting ESS equals
        ``ess_target * N``.  ESS(d) = exp(2*LSE(-d*U_lik) - LSE(-2*d*U_lik)) is
        monotone decreasing in ``d``; take ``max_dbeta`` if it already meets the
        target, else bisect, floored at ``min_dbeta`` to guarantee progress.
        """
        N = u_lik.shape[0]
        log_target = math.log(self.ess_target * N)

        def log_ess(d: float) -> float:
            a = torch.logsumexp(-d * u_lik, dim=0)
            b = torch.logsumexp(-2.0 * d * u_lik, dim=0)
            return float(2.0 * a - b)

        if log_ess(max_dbeta) >= log_target:
            return max_dbeta

        lo, hi = 0.0, max_dbeta                        # ESS(lo) >= target > ESS(hi)
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            if log_ess(mid) < log_target:
                hi = mid
            else:
                lo = mid
        return min(max(lo, self.min_dbeta), max_dbeta)

    def run_smc(
        self,
        num_particles: int,
        *,
        disable_progbar: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Transport ``num_particles`` from the prior to the posterior (beta = 1)
        and return the final population in constrained space, keyed by
        parameter name, each of shape ``(num_particles,)``.  Post-run schedule /
        ESS / evidence are available via :meth:`diagnostics`.
        """
        sampler, space = self.sampler, self.space

        # initial population ~ prior, in unconstrained free coordinates
        theta0 = space.to_vector(space.sample(num_particles))     # (N, d_full)
        z = sampler._init_z_free(theta0)                          # (N, d)

        beta = 0.0
        self._betas = [0.0]
        self._ess = []
        self._accept = []
        self._log_evidence = 0.0

        bar_format = "{l_bar}{bar}| {n:.3f}/{total:.3f} [{elapsed}{postfix}]"
        with tqdm(total=1.0, file=sys.stderr, disable=disable_progbar,
                  bar_format=bar_format, desc="SMC") as bar:
            while beta < 1.0:
                # reweight: schedule + incremental weights from U_lik
                u_lik = sampler.potential_likelihood(z)
                dbeta = self._next_dbeta(u_lik, max_dbeta=1.0 - beta)
                log_w = -dbeta * u_lik
                self._log_evidence += float(
                    torch.logsumexp(log_w, dim=0) - math.log(num_particles))

                # resample: systematic, to equal weights
                W = torch.softmax(log_w, dim=0)
                ess = float(1.0 / (W * W).sum())
                z = z[_systematic_resample(W)]
                beta += dbeta

                # mutate: re-ready at the new temperature, freeze adaptation
                sampler.beta = beta
                s = sampler.init(z)
                sampler.end_warmup()
                for _ in range(self.num_mcmc_steps):
                    s = sampler.step(s)
                z = s.q

                self._betas.append(beta)
                self._ess.append(ess)
                self._accept.append(float(sampler.diagnostics()["accept_rate"].mean()))
                bar.update(dbeta)
                bar.set_postfix(beta=f"{beta:.3f}", ess=f"{ess:.0f}",
                                logZ=f"{self._log_evidence:.2f}",
                                acc=f"{self._accept[-1]:.2f}", refresh=False)

        theta_free = space.map_to_constrained_vector(z).mapped_point
        return space.add_fixed(space.from_vector(theta_free))

    def diagnostics(self) -> dict:
        """Post-run schedule and population diagnostics (empty before
        :meth:`run_smc`).

          ``betas``         the temperature schedule, ``0 = beta_0 < ... = 1``
          ``ess``           per-stage effective sample size after reweighting
          ``accept_rate``   per-stage mean mutation acceptance rate
          ``log_evidence``  log marginal likelihood of the posterior, summed
                            from the per-stage log mean incremental weight
        """
        return {
            "betas": list(self._betas),
            "ess": list(self._ess),
            "accept_rate": list(self._accept),
            "log_evidence": self._log_evidence,
        }
