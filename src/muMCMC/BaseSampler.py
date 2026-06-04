from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import torch
from tqdm.auto import tqdm
import pyro
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel

from .spaces import TransformedMetric


class BaseSampler(ABC):
    """
    Base class for MCMC samplers.

    A sampler exposes a small operator interface that a driver composes into
    a chain::

        s = init(q)            # initial chain state
        repeat: s = step(s)    # one transition each
        end_warmup()           # warmup -> sampling, when the driver decides

    plus the user-facing :meth:`run_mcmc`, which drives that interface and
    returns constrained-space samples.  ``init`` / ``step`` / ``end_warmup``
    are the interface (duck-typed; concrete samplers such as RMHMC implement
    them).  Note ``init`` need not be stateless -- it may also arm a
    sampler's warmup/adaptation machinery.  Everything else (integrator,
    acceptance rule, adaptation) is a sampler's own implementation detail and
    is deliberately *not* part of this base contract.

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

    def evaluate_model(
        self, z_free: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, TransformedMetric]]:
        """
        Posterior evaluation at the unconstrained free vector ``z`` -- the
        overridable default a sampler relies on (override for needs beyond
        the assembly below).

        Depending on ``requires_metric`` the user's ``potential_fn`` is:

          requires_metric=False:  potential_fn(theta) -> scalar U_lik
          requires_metric=True:   potential_fn(theta) -> (U_lik, G_lik), with
              G_lik a (d_full, d_full) SPD metric in constrained coordinates.

        This method assembles the full unconstrained-space potential

            U(z) = U_lik(theta(z)) + U_prior(theta(z)) - log|det dtheta/dz|

        (prior from ``space.prior_log_prob_vector``, Jacobian log-det from the
        transform), and -- when ``requires_metric=True`` -- the pulled-back
        metric ``G_u(z) = J^T (G_lik + G_prior) J`` (summing the space's prior
        metric, Cholesky-factoring, restricting to free coordinates, wrapped
        in a ``TransformedMetric``).

        Returns
        -------
        Tensor U of shape (N,)                          if requires_metric is False
        (Tensor U of shape (N,), TransformedMetric)     if requires_metric is True

        Batching note
        -------------
        Batched-pure: takes ``(N, d)`` and returns ``U`` of shape ``(N,)``
        (plus, in the metric branch, a batched ``TransformedMetric``); it does
        no squeeze/unsqueeze.  The own batched driver calls this directly.
        The single-element ``(d,)`` contract Pyro's NUTS path needs is
        provided by ``PyroSampler._pyro_potential``.
        """
        theta_map = self.space.map_to_constrained_vector(z_free)
        theta_free = theta_map.mapped_point
        theta_full = self._free_to_full(theta_free)

        result = self.potential_fn(theta_full)
        if self.requires_metric:
            u_likelihood, G_lik = result
        else:
            u_likelihood = result

        u_prior = -self.space.prior_log_prob_vector(theta_free)
        U = u_likelihood + u_prior - theta_map.jacobian_log_det

        if not self.requires_metric:
            return U

        G_prior = self.space.prior_metric(theta_full)
        G_full = G_lik if G_prior is None else G_lik + G_prior
        metric = self.space.push_forward_metric(theta_full, G_full, theta_map=theta_map)

        return U, metric

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
        repeated ``step`` (one transition each) -- holding all ``num_chains``
        chains in a single batched state (no process spawning).
        ``end_warmup`` is called once warmup is done (just before the first
        sampling transition, which also makes ``num_warmup_steps == 0`` a
        clean no-op).  A future PT driver inserts a ``state.reorder`` swap
        between transitions here.  Live :meth:`logging` stats are shown on the
        progress bar.

        This driver is single-threaded and batched; multiprocessing knobs do
        not apply.  Extra keyword arguments (e.g. the Pyro path's
        ``mp_context``) are accepted for call-site compatibility and ignored.

        Returns
        -------
        dict[str, Tensor]
            Samples in constrained space, keyed by free parameter name,
            grouped by chain (shape ``(num_chains, num_samples, ...)``) --
            the same contract as the Pyro path.
        """
        # transform constrained point to unconstrained free vector, batched
        # over chains (the sampler batches over the leading axis).
        z_free_init = self._init_z_free(initial_params)
        if z_free_init.dim() == 1:
            z_free_init = z_free_init.unsqueeze(0).expand(num_chains, -1).contiguous()

        s = self.init(z_free_init)
        collected = []
        total = num_warmup_steps + num_samples

        # Single tqdm bar, Pyro-style: the postfix carries whatever logging()
        # returns (eps / |dH| / acc. prob / ...), desc switches Warmup->Sample.
        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}{postfix}]"
        with tqdm(total=total, file=sys.stderr, disable=disable_progbar,
                  bar_format=bar_format,
                  desc="Warmup" if num_warmup_steps else "Sample") as bar:
            if getattr(bar, "ncols", None) is not None:
                bar.ncols = min(120, max(80, bar.ncols))   # clamp width, like Pyro
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
    multi-chain machinery (e.g. NUTS).  All Pyro-specific logic lives here:
    the scalar ``_pyro_potential`` bridge and a ``run_mcmc`` that builds and
    runs a ``pyro.infer.mcmc.MCMC``.  The own batched driver in the base
    class is thus kept free of Pyro.
    """

    @property
    @abstractmethod
    def kernel(self) -> MCMCKernel:
        """The Pyro ``MCMCKernel`` driven by Pyro's ``MCMC``."""
        ...

    def _pyro_potential(self, params_dict: dict) -> torch.Tensor:
        """Pyro-compatible scalar potential wrapper.

        Pyro's HMC/NUTS kernel calls ``potential_fn(params_dict)`` with a
        single ``(d,)`` state and expects a scalar back.  This is the
        single-element bridge to the batched ``evaluate_model``: it lifts
        the state to ``(1, d)``, evaluates, and squeezes the ``(1,)``
        potential back to a scalar.  Implemented as a bound method (not a
        closure) so that Pyro's multi-chain spawning can pickle it.  Only
        valid when ``requires_metric=False``.
        """
        z = params_dict["params"]                  # (d,)
        return self.evaluate_model(z.unsqueeze(0)).squeeze(0)   # (1,d)->(1,)->()

    def diagnostics(self) -> dict:
        """Per-chain diagnostics in the common schema -- ``accept_rate``,
        ``num_divergences``, ``step_size`` (each a ``(num_chains,)`` tensor) --
        translated from Pyro's ``MCMC.diagnostics()`` (which keys per chain as
        ``'chain i'``).  Empty before :meth:`run_mcmc` has run.  The full Pyro
        detail (r_hat, n_eff, inverse mass matrix, divergence indices, ...)
        remains available via ``self.mcmc.diagnostics()``."""
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
        # Pyro's MCMC expects initial_params of shape (num_chains, d) when
        # num_chains > 1.  Replicate the single anchor across chains so
        # that NUTS's per-chain randomization (momentum, slice variable)
        # is the only source of inter-chain variation at the start.
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
        # Stash the MCMC object so callers can read per-chain diagnostics
        # via `sampler.mcmc.diagnostics()` after run_mcmc returns.  Pyro
        # populates this from each worker's kernel.diagnostics() call,
        # so it reflects the true post-warmup state per chain (the
        # parent's self.kernel never adapted; only the pickled copies in
        # workers did).
        self.mcmc = mcmc

        # Transform back to constrained space.
        samples_unc = mcmc.get_samples(group_by_chain=True)["params"]
        theta_free_all = self.space.map_to_constrained_vector(samples_unc).mapped_point
        return self.space.add_fixed(self.space.from_vector(theta_free_all))