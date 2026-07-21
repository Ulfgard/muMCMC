from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Callable, Dict, Optional, Tuple

import torch
from tqdm.auto import tqdm
import pyro
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel

from .spaces import TemperedMetric, TemperedAffine


class BaseSampler(ABC):
    """
    Base class for MCMC samplers.

    A sampler exposes a small operator interface that a driver composes into
    a chain::

        s = init(q)            # initial chain state
        repeat: s = step(s)    # one transition each
        end_warmup()           # warmup -> sampling, when the driver decides

    plus the user-facing :meth:`run_mcmc`, which drives that interface and
    returns constrained-space samples.  ``init`` may also arm a sampler's
    warmup/adaptation.  Everything else (integrator, acceptance rule,
    adaptation) is the sampler's own implementation detail.

    Two optional hooks complete the interface, both defaulting to an empty
    dict: :meth:`logging` (per-step stats surfaced on the progress bar) and
    :meth:`diagnostics` (post-run, per-chain summaries).  Samplers override
    them to expose live/standing statistics.

    Posterior evaluation is shared.  The user supplies ``potential_fn`` in
    **constrained** coordinates while the sampler works in unconstrained
    space; :meth:`evaluate_model` performs the pull-back and assembles the
    unconstrained-space potential (and, for metric-based samplers, the
    metric).  A sampler with different needs may override it.

    Parameters
    ----------
    potential_fn : callable
        The model's contribution to the potential, ``U = -log p``, in
        constrained coordinates.  Its exact signature is method-dependent
        (see :meth:`evaluate_model`).
    space
        Parameter space: transforms, free/fixed split, vector<->dict
        conversions, the prior, and (for metric-based use) the prior metric.
    requires_metric : bool
        Whether this sampler needs a position-dependent metric.  Subclasses
        pass it explicitly.
    """

    def __init__(
        self,
        potential_fn: Callable,
        space,
        *,
        requires_metric: bool,
    ):
        self.potential_fn = potential_fn
        self.space = space
        self.requires_metric = requires_metric
        self.beta = 1.0   # inverse temperature

    def evaluate_model(
        self, z_free: torch.Tensor, beta: Optional[float] = None,
        grad: bool = False,
    ):
        """
        Posterior evaluation at the unconstrained free vector ``z`` as
        tempering-aware objects.

        Depending on ``requires_metric`` the user's ``potential_fn`` is:

          requires_metric=False:  potential_fn(theta) -> scalar U_lik
          requires_metric=True:   potential_fn(theta) -> (U_lik, G_lik), with
              G_lik a (d_full, d_full) SPD metric in constrained coordinates.

        Returns ``(potential, metric)``, or ``(potential, metric, gradient)``
        when ``grad`` is True:

          - potential: :class:`TemperedAffine` with ``value = beta * U_lik +
            U_base``, ``U_base = U_prior - log|det dtheta/dz|``.
          - :class:`TemperedMetric` with ``G_u(beta) = beta * A_lik + A_prior``,
            the likelihood/prior metrics pushed forward to free unconstrained
            coordinates (``None`` when ``requires_metric`` is False).
          - gradient: :class:`TemperedAffine` with ``value = ∂U/∂z``, returned
            only when ``grad`` is True.  ``grad`` detaches all returned objects.

        Both own their inverse temperature and a ``reorder`` that retempers a
        moved configuration, so a caller keeping them across a temperature
        change needs no knowledge of ``beta`` (default ``self.beta``, 1.0 =
        untempered).  Callers wanting the scalar potential use ``.value``.

        Batched over the leading axis: ``(N, d)`` -> ``(N,)`` potential
        (plus a batched ``TemperedMetric`` in the metric branch).
        """
        if beta is None:
            beta = self.beta
        if grad:
            z_free = z_free.detach().requires_grad_(True)

        with torch.enable_grad() if grad else nullcontext():
            theta_map = self.space.map_to_constrained_vector(z_free)
            theta_free = theta_map.mapped_point
            theta_full = self._free_to_full(theta_free)

            result = self.potential_fn(theta_full)
            if self.requires_metric:
                u_likelihood, G_lik = result
            else:
                u_likelihood = result

            U_base = -self.space.prior_log_prob_vector(theta_free) - theta_map.jacobian_log_det

        metric = None
        if self.requires_metric:
            G_prior = self.space.prior_metric(theta_full)
            A_lik = self.space.push_forward_metric(G_lik, theta_map)
            A_prior = None if G_prior is None else self.space.push_forward_metric(G_prior, theta_map)
            metric = TemperedMetric(A_lik, A_prior, beta)

        if not grad:
            return TemperedAffine(u_likelihood, U_base, beta), metric

        def grad_of(out):
            # U_base is constant in z with no prior and a volume-preserving
            # transform, so guard the backward.
            if not out.requires_grad:
                return torch.zeros_like(z_free)
            g, = torch.autograd.grad(out.sum(), z_free, retain_graph=True,
                                     allow_unused=True)
            return torch.zeros_like(z_free) if g is None else g

        gradient = TemperedAffine(grad_of(u_likelihood).detach(),
                                  grad_of(U_base).detach(), beta)
        potential = TemperedAffine(u_likelihood.detach(), U_base.detach(), beta)
        return potential, metric, gradient

    def _free_to_full(self, theta_free: torch.Tensor) -> torch.Tensor:
        """Free constrained vector -> full constrained vector (with fixed)."""
        return self.space.to_vector(
            self.space.add_fixed(self.space.from_vector(theta_free))
        )

    def _init_z_free(self, initial_params: torch.Tensor) -> torch.Tensor:
        """Full or free constrained vector -> unconstrained free vector."""
        theta_free = self.space.to_free_vector(
            self.space.from_vector(initial_params)
        )
        return self.space.map_to_unconstrained_vector(theta_free).mapped_point

    def logging(self) -> dict:
        """Per-step statistics for the progress bar, as a dict of short
        preformatted strings (e.g. ``{"eps": "1.6e-01", "acc. prob": "0.99"}``).
        Default: empty (no postfix).  Samplers override to surface live stats."""
        return {}

    def diagnostics(self) -> dict:
        """Post-run, per-chain diagnostics (acceptance rate, divergences, ...).
        Default: empty.  Samplers override to expose standing statistics."""
        return {}

    def run_mcmc(
        self,
        initial_params: torch.Tensor,
        num_samples: int,
        num_warmup_steps: int,
        *,
        num_chains: int = 1,
        disable_progbar: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Run MCMC via the own batched driver and return constrained samples.

        Drives the sampler's operator interface directly -- ``init`` then
        repeated ``step`` -- holding all ``num_chains`` chains in a single
        batched state (no process spawning).  ``end_warmup`` is called once
        warmup is done, so ``num_warmup_steps == 0`` is a clean no-op.  Live
        :meth:`logging` stats are shown on the progress bar.

        This driver is single-threaded and batched; multiprocessing knobs do
        not apply.  Extra keyword arguments are accepted for call-site
        compatibility and ignored.

        Returns
        -------
        dict[str, Tensor]
            Samples in constrained space, keyed by free parameter name,
            grouped by chain (shape ``(num_chains, num_samples, ...)``) --
            the same contract as the Pyro path.
        """
        # constrained point -> unconstrained free vector, batched over chains
        z_free_init = self._init_z_free(initial_params)
        if z_free_init.dim() == 1:
            z_free_init = z_free_init.unsqueeze(0).expand(num_chains, -1).contiguous()

        s = self.init(z_free_init)
        collected = []
        total = num_warmup_steps + num_samples

        # Single tqdm bar; the postfix carries whatever logging() returns.
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
        with tqdm(total=total, file=sys.stderr, disable=disable_progbar,
                  bar_format=bar_format,
                  desc="Warmup" if num_warmup_steps else "Sample") as bar:
            if getattr(bar, "ncols", None) is not None:
                bar.ncols = min(120, max(80, bar.ncols))   # clamp width
            for it in range(total):
                if it == num_warmup_steps:          # warmup done -> freeze/finalize
                    self.end_warmup()
                    bar.set_description("Sample")
                s = self.step(s)
                if it >= num_warmup_steps:
                    collected.append(s.q.clone())   # (num_chains, d)
                post = self.logging()
                if post:
                    bar.set_postfix(post, refresh=False)
                bar.update()

        # (num_samples, K, d) -> (K, num_samples, d) to match group_by_chain.
        samples_unc = torch.stack(collected, dim=0).transpose(0, 1)
        theta_free_all = self.space.map_to_constrained_vector(samples_unc).mapped_point
        return self.space.add_fixed(self.space.from_vector(theta_free_all))


class PyroSampler(BaseSampler):
    """
    BaseSampler specialization that runs through Pyro's ``MCMC`` driver.

    For kernels that are Pyro ``MCMCKernel`` s driven by Pyro's own
    multi-chain machinery (e.g. NUTS).  Holds the Pyro-specific logic: the
    scalar ``_pyro_potential`` bridge and a ``run_mcmc`` that builds and runs
    a ``pyro.infer.mcmc.MCMC``.
    """

    @property
    @abstractmethod
    def kernel(self) -> MCMCKernel:
        """The Pyro ``MCMCKernel`` driven by Pyro's ``MCMC``."""
        ...

    def _pyro_potential(self, params_dict: dict) -> torch.Tensor:
        """Pyro-compatible scalar potential wrapper.

        Pyro's HMC/NUTS kernel calls ``potential_fn(params_dict)`` with a single
        ``(d,)`` state and expects a scalar back; this lifts it to ``(1, d)``,
        evaluates, and squeezes back.  A bound method (not a closure) so Pyro
        can pickle it for multi-chain spawning.  Only valid when
        ``requires_metric=False``.
        """
        z = params_dict["params"]                  # (d,)
        potential, _ = self.evaluate_model(z.unsqueeze(0))
        return potential.value.squeeze(0)          # (1,d)->(1,)->()

    def diagnostics(self) -> dict:
        """Per-chain diagnostics in the common schema -- ``accept_rate``,
        ``num_divergences``, ``step_size`` (each a ``(num_chains,)`` tensor).
        Empty before :meth:`run_mcmc` has run.  Full Pyro detail (r_hat, n_eff,
        inverse mass matrix, divergence indices, ...) is available via
        ``self.mcmc.diagnostics()``."""
        mcmc = getattr(self, "mcmc", None)
        if mcmc is None:
            return {}
        d = mcmc.diagnostics()
        chains = sorted(d["acceptance rate"], key=lambda k: int(k.split()[-1]))
        return {
            "accept_rate":     torch.tensor([d["acceptance rate"][c] for c in chains]),
            "num_divergences": torch.tensor([len(d["divergences"][c]) for c in chains],
                                            dtype=torch.long),
            "step_size":       torch.tensor([d["step_size"][c] for c in chains]),
        }

    def run_mcmc(
        self,
        initial_params: torch.Tensor,
        num_samples: int,
        num_warmup_steps: int,
        *,
        num_chains: int = 1,
        mp_context: str = "spawn",
        disable_progbar: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run MCMC through Pyro's ``MCMC`` driver and return constrained samples.

        Parameters
        ----------
        initial_params : Tensor
            Full constrained flat vector (including fixed parameters).
        num_samples : int
            Number of post-warmup samples.
        num_warmup_steps : int
            Warmup / burn-in iterations.
        num_chains : int
            Number of parallel chains (Pyro spawns one worker each).
        mp_context : str
            Multiprocessing context for multi-chain.
        disable_progbar : bool
            Disable the progress bar.

        Returns
        -------
        dict[str, Tensor]
            Samples in constrained space, keyed by free parameter name,
            grouped by chain.
        """
        pyro.clear_param_store()

        # transform constrained point to unconstrained parameters
        z_free_init = self._init_z_free(initial_params)
        # Pyro expects initial_params of shape (num_chains, d) for
        # num_chains > 1; replicate the single anchor across chains.
        if num_chains > 1 and z_free_init.dim() == 1:
            z_free_init = z_free_init.unsqueeze(0).expand(num_chains, -1).contiguous()

        mcmc = MCMC(
            self.kernel,
            initial_params={"params": z_free_init},
            num_samples=num_samples,
            warmup_steps=num_warmup_steps,
            num_chains=num_chains,
            disable_progbar=disable_progbar,
            mp_context=mp_context,
        )
        mcmc.run()
        # Stash the MCMC object so callers can read per-chain diagnostics via
        # self.mcmc.diagnostics().
        self.mcmc = mcmc

        # Transform back to constrained space.
        samples_unc = mcmc.get_samples(group_by_chain=True)["params"]
        theta_free_all = self.space.map_to_constrained_vector(samples_unc).mapped_point
        return self.space.add_fixed(self.space.from_vector(theta_free_all))