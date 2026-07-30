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


class MCMCSampler(ABC):
    """Base class for MCMC samplers.

    Operator interface (abstract): ``init(q)`` returns the initial state,
    ``step(s)`` performs one transition, ``end_warmup()`` switches from warmup
    to sampling. ``run_mcmc`` is the batched driver that composes them and
    returns samples on the variables. Optional hooks ``logging`` (per-step
    progress-bar stats) and ``diagnostics`` (post-run per-chain summaries)
    default to ``{}``.

    Parameters
    ----------
    potential_fn : callable
        Model potential ``U = -log p`` on the variables. Signature is
        method-dependent (see ``evaluate_model``).
    space
        Parameter space: the prior's normal chart, the free/fixed split, and the
        vector and dict conversions.
    requires_metric : bool
        Whether the sampler needs a position-dependent metric.
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
        self, theta_free: torch.Tensor, beta: Optional[float] = None,
        grad: bool = False,
    ):
        """``value = beta * U_lik + U_base``, ``U_base = -log p(theta)``.

        Posterior evaluation at the free variable vector ``theta``, which is
        where the chain runs and where the model is written, so nothing is
        transformed. The user's ``potential_fn`` is:

          requires_metric=False:  potential_fn(theta) -> scalar U_lik
          requires_metric=True:   potential_fn(theta) -> (U_lik, G_lik), with
              G_lik a (d_full, d_full) SPD metric on the same vector.

        Batched over the leading axis: ``(N, d)`` -> ``(N,)`` potential.

        Parameters
        ----------
        theta_free : Tensor
            Free variable vector.
        beta : float, optional
            Inverse temperature. Default ``self.beta`` (1.0 = untempered).
        grad : bool
            If True, also return the gradient and detach all returned objects.

        Returns
        -------
        potential
            ``value = beta * U_lik + U_base``.
        metric
            ``G(beta) = beta * A_lik + M``, the free block of the likelihood
            metric plus the prior's own metric on the variables, ``None`` for a
            space with no prior. ``None`` when ``requires_metric`` is False.
        gradient
            ``value = ∂U/∂theta``. Returned only when ``grad`` is True.
        """
        if beta is None:
            beta = self.beta
        if grad:
            theta_free = theta_free.detach().requires_grad_(True)

        with torch.enable_grad() if grad else nullcontext():
            theta_full = self.space.to_full(theta_free)

            result = self.potential_fn(theta_full)
            if self.requires_metric:
                u_likelihood, G_lik = result
            else:
                u_likelihood = result

            U_base = -self.space.prior_log_prob_vector(theta_free)

        metric = None
        if self.requires_metric:
            metric = TemperedMetric(self.space.free_block(G_lik),
                                    self.space.prior_metric(theta_free), beta)

        if not grad:
            return TemperedAffine(u_likelihood, U_base, beta), metric

        def grad_of(out):
            # guard backward: U_base has no grad for a space with no prior
            if not out.requires_grad:
                return torch.zeros_like(theta_free)
            g, = torch.autograd.grad(out.sum(), theta_free, retain_graph=True,
                                     allow_unused=True)
            return torch.zeros_like(theta_free) if g is None else g

        gradient = TemperedAffine(grad_of(u_likelihood).detach(),
                                  grad_of(U_base).detach(), beta)
        potential = TemperedAffine(u_likelihood.detach(), U_base.detach(), beta)
        return potential, metric, gradient

    def to_position(self, theta_free: torch.Tensor) -> torch.Tensor:
        """The chain's position at the free variables ``theta_free``. The chain
        runs on the variables, so this is the identity. A sampler running in
        other coordinates overrides it together with :meth:`to_variables`."""
        return theta_free

    def to_variables(self, q_free: torch.Tensor) -> torch.Tensor:
        """The free variables at the chain's position ``q_free``, the inverse of
        :meth:`to_position`."""
        return q_free

    def _init_position(self, initial_params: dict) -> torch.Tensor:
        """Starting point keyed by name to the chain's starting position."""
        return self.to_position(self.space.to_free_vector(initial_params))

    def logging(self) -> dict:
        """Per-step statistics for the progress bar, as a dict of short
        preformatted strings (e.g. ``{"eps": "1.6e-01", "acc. prob": "0.99"}``).
        Default ``{}``."""
        return {}

    def diagnostics(self) -> dict:
        """Post-run per-chain diagnostics (acceptance rate, divergences, ...).
        Default ``{}``."""
        return {}

    # ---- operator interface (composed by the batched run_mcmc) -------------- #

    @abstractmethod
    def init(self, q_free: torch.Tensor):
        """Return the initial batched chain state at the positions ``q_free``
        (shape ``(num_chains, d)``)."""
        ...

    @abstractmethod
    def step(self, state):
        """Advance the batched chain state by one transition and return it."""
        ...

    @abstractmethod
    def end_warmup(self) -> None:
        """Switch from warmup to sampling, freezing any warmup adaptation."""
        ...

    def run_mcmc(
        self,
        initial_params: dict,
        num_samples: int,
        num_warmup_steps: int,
        *,
        num_chains: int = 1,
        disable_progbar: bool = False,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run MCMC via the batched driver and return samples on the variables.

        Holds all ``num_chains`` chains in one batched state: ``init``, then
        repeated ``step``, then ``end_warmup`` once warmup is done. Extra
        keyword arguments are ignored.

        Parameters
        ----------
        initial_params : dict[str, Tensor]
            Starting point keyed by name, in the form a run returns. Fixed names
            are ignored.
        num_samples : int
            Number of post-warmup samples.
        num_warmup_steps : int
            Warmup iterations.
        num_chains : int
            Number of parallel chains.
        disable_progbar : bool
            Disable the progress bar.

        Returns
        -------
        dict[str, Tensor]
            Samples on the variables, keyed by free parameter name, grouped by
            chain (shape ``(num_chains, num_samples, ...)``).
        """
        # variable point -> the chain's starting position, over chains
        q_init = self._init_position(initial_params)
        if q_init.dim() == 1:
            q_init = q_init.unsqueeze(0).expand(num_chains, -1).contiguous()

        s = self.init(q_init)
        collected = []
        total = num_warmup_steps + num_samples

        # single tqdm bar, postfix carries logging() output
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
        positions = torch.stack(collected, dim=0).transpose(0, 1)
        theta_free_all = self.to_variables(positions)
        return self.space.add_fixed(self.space.from_free_vector(theta_free_all))


class PyroSampler(MCMCSampler):
    """MCMCSampler specialization running through Pyro's ``MCMC`` driver.

    For kernels that are Pyro ``MCMCKernel`` s (e.g. NUTS). Provides the scalar
    ``_pyro_potential`` bridge and a ``run_mcmc`` built on
    ``pyro.infer.mcmc.MCMC``.
    """

    @property
    @abstractmethod
    def kernel(self) -> MCMCKernel:
        """The Pyro ``MCMCKernel`` driven by Pyro's ``MCMC``."""
        ...

    # Sampling runs through Pyro's ``MCMC`` (see ``run_mcmc``), not the batched
    # init/step/end_warmup interface, so these satisfy the abstract contract only
    # to keep the Pyro-backed samplers instantiable.
    _NO_BATCHED_IFACE = (
        "PyroSampler samples through Pyro's MCMC driver. The batched operator "
        "interface (init/step/end_warmup) is not used."
    )

    def init(self, q_free: torch.Tensor):
        raise NotImplementedError(self._NO_BATCHED_IFACE)

    def step(self, state):
        raise NotImplementedError(self._NO_BATCHED_IFACE)

    def end_warmup(self) -> None:
        raise NotImplementedError(self._NO_BATCHED_IFACE)

    # ===================================================================== #
    # A bound method, not a closure, so Pyro can pickle it when spawning
    # multi-chain workers.
    # ===================================================================== #
    def _pyro_potential(self, params_dict: dict) -> torch.Tensor:
        """Pyro-compatible scalar potential wrapper.

        Pyro's HMC/NUTS kernel calls ``potential_fn(params_dict)`` with a single
        ``(d,)`` state and expects a scalar. Only valid when
        ``requires_metric=False``.
        """
        z = params_dict["params"]                  # (d,)
        potential, _ = self.evaluate_model(z.unsqueeze(0))
        return potential.value.squeeze(0)          # (1,d)->(1,)->()

    def diagnostics(self) -> dict:
        """Per-chain diagnostics in the common schema: ``accept_rate``,
        ``num_divergences``, ``step_size`` (each a ``(num_chains,)`` tensor).
        Empty before ``run_mcmc`` has run. Full Pyro detail (r_hat, n_eff,
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
        initial_params: dict,
        num_samples: int,
        num_warmup_steps: int,
        *,
        num_chains: int = 1,
        mp_context: str = "spawn",
        disable_progbar: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Run MCMC through Pyro's ``MCMC`` driver and return samples on the
        variables.

        Parameters
        ----------
        initial_params : Tensor
            Starting point keyed by name, in the form a run returns. Fixed
            names are ignored.
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
            Samples on the variables, keyed by free parameter name, grouped by
            chain.
        """
        pyro.clear_param_store()

        # variable point -> the chain's starting position
        q_init = self._init_position(initial_params)
        # Pyro expects initial_params of shape (num_chains, d) for num_chains > 1.
        # replicate the single anchor across chains.
        if num_chains > 1 and q_init.dim() == 1:
            q_init = q_init.unsqueeze(0).expand(num_chains, -1).contiguous()

        mcmc = MCMC(
            self.kernel,
            initial_params={"params": q_init},
            num_samples=num_samples,
            warmup_steps=num_warmup_steps,
            num_chains=num_chains,
            disable_progbar=disable_progbar,
            mp_context=mp_context,
        )
        mcmc.run()
        # stash MCMC object for per-chain diagnostics via self.mcmc.diagnostics()
        self.mcmc = mcmc

        # Read the draws back on the variables.
        theta_free_all = self.to_variables(
            mcmc.get_samples(group_by_chain=True)["params"])
        return self.space.add_fixed(self.space.from_free_vector(theta_free_all))