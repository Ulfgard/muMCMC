from __future__ import annotations

import math
import sys
from typing import Dict

import torch
from tqdm.auto import tqdm

from .MCMCSampler import MCMCSampler


def _systematic_resample(weights: torch.Tensor) -> torch.Tensor:
    """Systematic resampling of normalized ``weights`` ``(..., M)`` to ``(..., M)``
    ancestor indices, with replacement in proportion to the weights.

    Parameters
    ----------
    weights : torch.Tensor
        Normalized weights along the last axis, ``(..., M)``.
    """
    M = weights.shape[-1]
    batch = weights.shape[:-1]
    u0 = torch.rand(batch + (1,), device=weights.device, dtype=weights.dtype)
    positions = (torch.arange(M, device=weights.device, dtype=weights.dtype) + u0) / M
    cumsum = weights.cumsum(dim=-1)
    cumsum[..., -1] = 1.0                              # guard rounding at the top
    return torch.searchsorted(cumsum, positions).clamp_(max=M - 1)


def _rhat(x: torch.Tensor) -> torch.Tensor:
    """Gelman-Rubin R-hat across the ``C`` populations of ``x``,

        R-hat = sqrt(var_plus / W),  var_plus = (M-1)/M W + B/M,

    with ``W`` the mean within-population variance and ``B`` the between-
    population variance scaled by ``M``. A scalar.

    Parameters
    ----------
    x : Tensor, shape (C, M)
        Draws, ``C`` populations of ``M`` each.
    """
    C, M = x.shape
    chain_mean = x.mean(dim=1)
    W = x.var(dim=1, unbiased=True).mean()
    B = M * chain_mean.var(unbiased=True)
    var_plus = (M - 1) / M * W + B / M
    return torch.sqrt(var_plus / W)


# =========================================================================== #
#  Why the populations are independent                                        #
#                                                                             #
#  num_chains populations share the kernel's batch axis but nothing else.     #
#  Each carries its own particles, its own resampling and its own schedule.   #
#  That independence is what turns the spread of their log-evidence           #
#  estimates and their R-hat into an estimate of the Monte Carlo error, which #
#  a single population cannot give.                                           #
#                                                                             #
#  The likelihood potential the reweighting needs is already on the kernel    #
#  state as state.U.lik, held apart from the prior part and carrying no       #
#  graph, so a stage reads it rather than evaluating the model again.         #
# =========================================================================== #


class SMC:
    """
    Adaptive tempered Sequential Monte Carlo around a batched sampler.

    Transports a particle population from the prior at ``beta = 0`` to the
    posterior at ``beta = 1`` along ``prior * likelihood**beta``. A stage
    reweights, resamples systematically, and then mutates. The next
    temperature is bisected per population so that the post-reweighting ESS is
    ``ess_target * num_particles``, the incremental weights are
    ``exp(-dbeta * U_lik)``, and the log-evidence accumulates
    ``LSE(log_w) - log M``.

    The mutation kernel is ``sampler``, driven through its ``init``, ``step``
    and ``beta``. This class takes over that ``beta`` for the run and freezes
    the kernel's warmup adaptation, so the kernel mutates at a fixed step size.

    Parameters
    ----------
    sampler : MCMCSampler
        The mutation kernel, used for every particle of every population at
        once.
    ess_target : float
        Post-reweighting ESS as a fraction of the particle count, in (0, 1).
    num_mcmc_steps : int
        Mutation transitions applied at each temperature.
    min_dbeta : float
        Floor on the temperature increment, which keeps a stage from making no
        progress. The last stage takes ``1 - beta`` even where that is smaller.

    Raises
    ------
    ValueError
        If ``ess_target`` is not in ``(0, 1)``.
    """

    def __init__(
        self,
        sampler: MCMCSampler,
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
        """Per-chain temperature increment ``d`` solving
        ESS(d) = exp(2 LSE(-d u) - LSE(-2 d u)) = ``ess_target * M``.

        ESS is monotone decreasing in ``d``, so each chain takes ``max_dbeta`` if
        it already meets the target, else bisects, floored at ``min_dbeta``.

        Parameters
        ----------
        u_lik : Tensor, shape (C, M)
            Likelihood potentials of the current particles.
        max_dbeta : Tensor, shape (C,)
            Per-population upper bound ``1 - beta``.
        max_iter : int
            Bisection iterations.
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
        """Transport ``num_chains`` independent populations of
        ``num_particles`` each from the prior to the posterior at ``beta = 1``.

        The initial particles are drawn from the prior, so the space must carry
        one. The kernel's ``beta`` is left at 1.0 afterwards, and its warmup
        adaptation stays frozen.

        Parameters
        ----------
        num_particles : int
            Particles per population.
        num_chains : int
            Independent populations, advanced together in one batch.
        disable_progbar : bool
            Suppress the progress bar.

        Returns
        -------
        dict[str, Tensor]
            The final populations on the variables, keyed by name, each of
            shape ``(num_chains, num_particles)``. The fixed names are present
            too, each held at its value. The schedule, the ESS, the evidence
            and R-hat are read from :meth:`diagnostics`.
        """
        sampler, space = self.sampler, self.space
        C, M, N = num_chains, num_particles, num_chains * num_particles

        # initial populations ~ prior
        q = sampler._init_position(space.sample(N))               # (N, d)
        d = q.shape[-1]

        beta = torch.zeros(C, dtype=q.dtype, device=q.device)     # per-chain
        self._betas = [beta.clone()]
        self._ess = []
        self._accept = []
        self._log_evidence = torch.zeros(C, dtype=q.dtype, device=q.device)

        # Evaluate the prior population once. The kernel state carries U_lik,
        # so reweighting reads it (grad-free) instead of recomputing.
        sampler.beta = beta.unsqueeze(-1).expand(C, M).reshape(N)
        s = sampler.init(q)

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
                q = s.q.reshape(C, M, d).gather(
                    1, idx.unsqueeze(-1).expand(C, M, d)).reshape(N, d)
                beta = beta + dbeta

                # mutate: fixed kernel at each chain's temperature, adaptation frozen
                sampler.beta = beta.unsqueeze(-1).expand(C, M).reshape(N)
                s = sampler.init(q)
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
        theta_free = sampler.to_variables(s.q)                    # final population
        free = space.from_free_vector(theta_free)
        if C >= 2:
            self._r_hat = {name: _rhat(v.reshape(C, M)) for name, v in free.items()}
        return {k: v.reshape(C, M) for k, v in space.add_fixed(free).items()}

    def diagnostics(self) -> dict:
        """Schedule and population diagnostics of the last run, empty before
        :meth:`run_smc`.

          betas                  temperature schedule, (stages+1, num_chains)
          ess                    ESS after reweighting, (stages, num_chains)
          accept_rate            mean mutation acceptance, (stages, num_chains)
          log_evidence           per-population log marginal likelihood,
                                 (num_chains,)
          log_evidence_estimate  the combined estimate, the log mean of the
                                 per-population values
          log_evidence_se        standard error of that estimate across
                                 populations, zero for a single population
          r_hat                  Gelman-Rubin across populations, per free
                                 name, and empty below two populations
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
