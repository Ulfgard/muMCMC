from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Callable, Dict, Optional

import torch
from tqdm.auto import tqdm
import pyro
from pyro.infer.mcmc import MCMC
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel

from .spaces import TemperedMetric, TemperedAffine


class MCMCSampler(ABC):
    """Base class for MCMC samplers.

    A subclass implements three abstract methods, :meth:`init`, :meth:`step`
    and :meth:`end_warmup`, which act on a batched state holding every chain.
    :meth:`run_mcmc` composes them into a run and returns samples on the
    variables. :meth:`logging` and :meth:`diagnostics` are optional and return
    ``{}`` here.

    The model is reached through :meth:`evaluate_model`, which adds the prior
    and applies the temperature, so a subclass does not call ``potential_fn``
    itself.

    Parameters
    ----------
    potential_fn : callable
        Model potential on the full variable vector, the part of ``U`` that
        carries the inverse temperature. It is ``-log p(data | theta)`` when
        the space carries a prior and the whole target when it does not. Its
        signature depends on ``requires_metric``, see :meth:`evaluate_model`.
    space : Space
        Parameter space, giving the prior's normal chart, the free/fixed split
        and the conversions between a dict of named variables and a free
        vector.
    requires_metric : bool
        Whether ``potential_fn`` also returns a position-dependent metric, which
        a Riemannian sampler needs and the others do not.

    Attributes
    ----------
    beta : float or Tensor, shape (num_chains,)
        Inverse temperature applied to the model potential, 1.0 at
        construction. 1.0 is the untempered posterior and 0.0 leaves the prior
        alone. A tempering driver such as :class:`~muMCMC.PT.PT` or
        :class:`~muMCMC.SMC.SMC` rebinds it to a per-chain tensor for the
        duration of its run.
    """

    def __init__(
        self,
        potential_fn: Callable,
        space,
        *,
        requires_metric: bool,
    ):
        self.potential_fn = potential_fn
        self._space = space
        self.requires_metric = requires_metric
        self.beta = 1.0   # inverse temperature

    @property
    def space(self):
        """The sampler's parameter space."""
        return self._space

    def evaluate_model(
        self, theta_free: torch.Tensor, beta: Optional[float] = None,
        grad: bool = False,
    ):
        """The tempered potential ``U = beta * U_lik + U_base`` at
        ``theta_free``, with its metric and, on request, its gradient.

        The chain runs on the variables and the model is written on them, so
        nothing is transformed here. ``potential_fn`` is handed the full vector
        ``space.to_full(theta_free)`` of shape ``(N, d_full)`` over
        ``space.names``, carrying each fixed variable at its value, and returns

          requires_metric=False:  U_lik of shape (N,)
          requires_metric=True:   (U_lik, G_lik), with G_lik an SPD metric of
              shape (N, d_full, d_full) read on that same vector.

        Parameters
        ----------
        theta_free : Tensor, shape (N, d)
            Free variable vector.
        beta : float or Tensor, optional
            Inverse temperature. Default :attr:`beta`.
        grad : bool
            Whether to also return the gradient.

        Returns
        -------
        potential : TemperedAffine
            ``U = beta * U_lik + U_base``, of shape ``(N,)``, where
            ``U_base = -log p(theta_free)`` is the prior potential. Detached
            when ``grad`` is True.
        metric : TemperedMetric or None
            ``G = beta * A_lik + M``, of shape ``(N, d, d)``, where ``A_lik``
            is the free block of ``G_lik`` and ``M`` the metric the prior
            induces on the free variables. ``M`` is None for a space with no
            prior, and the metric itself is None when ``requires_metric`` is
            False. It carries its autograd graph whether or not ``grad`` is
            set.
        gradient : TemperedAffine
            ``∂U/∂theta_free``, of shape ``(N, d)`` and split in the same two
            parts as ``potential``. Returned only when ``grad`` is True, and
            detached.
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

    def potential(self, theta_free: torch.Tensor,
                  beta: Optional[float] = None) -> torch.Tensor:
        """``U(theta, beta)`` on the free variables, of shape ``(N,)`` for
        ``theta_free`` of shape ``(N, d)``.

        It holds ``p(theta | beta) = exp(-U(theta, beta)) / Z_beta``, a density
        with respect to ``dtheta``. The chain runs on the variables, so this is
        :meth:`evaluate_model`'s value. A sampler running in other coordinates
        overrides it to pull its own potential back to the variables.

        Parameters
        ----------
        theta_free : Tensor, shape (N, d)
            Free variable vector.
        beta : float or Tensor, optional
            Inverse temperature. Default :attr:`beta`.
        """
        return self.evaluate_model(theta_free, beta)[0].value

    def to_position(self, theta_free: torch.Tensor) -> torch.Tensor:
        """The chain's position at the free variables ``theta_free`` of shape
        ``(..., d)``. The chain runs on the variables, so this is the identity.
        A sampler running in other coordinates overrides it together with
        :meth:`to_variables`."""
        return theta_free

    def to_variables(self, q_free: torch.Tensor) -> torch.Tensor:
        """The free variables at the chain's position ``q_free`` of shape
        ``(..., d)``, the inverse of :meth:`to_position`."""
        return q_free

    def _init_position(self, initial_params: dict) -> torch.Tensor:
        """Starting point keyed by name to the chain's starting position."""
        return self.to_position(self.space.to_free_vector(initial_params))

    def logging(self) -> dict:
        """Per-step statistics for the progress bar, keyed by column label and
        already formatted as short strings, for example
        ``{"eps": "1.6e-01", "acc. prob": "0.99"}``. Empty here."""
        return {}

    def diagnostics(self) -> dict:
        """Per-chain summaries of the last run, each a ``(num_chains,)`` tensor
        keyed by name, such as ``accept_rate`` and ``num_divergences``. Empty
        here."""
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
        """Run MCMC and return samples on the variables.

        All ``num_chains`` chains are held in one batched state. :meth:`init`
        builds it, :meth:`step` advances it ``num_warmup_steps + num_samples``
        times, and :meth:`end_warmup` runs once between the last warmup step
        and the first sampling step. Only the sampling steps are collected.
        Extra keyword arguments are ignored, so a caller can pass the union of
        the arguments the samplers take.

        Parameters
        ----------
        initial_params : dict[str, Tensor]
            Starting point keyed by name, in the form a run returns. Fixed
            names are ignored. An entry of shape ``()`` starts every chain at
            the same point, one of shape ``(num_chains,)`` starts each chain at
            its own.
        num_samples : int
            Number of post-warmup samples per chain.
        num_warmup_steps : int
            Number of warmup iterations, during which adaptation is live and
            nothing is collected.
        num_chains : int
            Number of chains, advanced together in one batch.
        disable_progbar : bool
            Disable the progress bar.

        Returns
        -------
        dict[str, Tensor]
            Samples on the variables, keyed by name and grouped by chain, each
            of shape ``(num_chains, num_samples)``. The fixed names are present
            too, each held at its value.
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
    """Base class for samplers whose transition is a Pyro ``MCMCKernel``.

    A subclass supplies :attr:`kernel` instead of :meth:`init`, :meth:`step`
    and :meth:`end_warmup`, and :meth:`run_mcmc` drives it through
    ``pyro.infer.mcmc.MCMC``. The three batched methods raise here, so a
    subclass of this class is used through :meth:`run_mcmc` alone.

    The kernel is given ``self._pyro_potential`` as its potential, which
    restricts these samplers to ``requires_metric=False``.

    Raises
    ------
    NotImplementedError
        From :meth:`init`, :meth:`step` and :meth:`end_warmup`, which the Pyro
        driver does not use.
    """

    @property
    @abstractmethod
    def kernel(self) -> MCMCKernel:
        """The Pyro ``MCMCKernel`` that :meth:`run_mcmc` drives."""
        ...

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

    # A bound method, not a closure, so Pyro can pickle it when spawning
    # multi-chain workers.
    def _pyro_potential(self, params_dict: dict) -> torch.Tensor:
        """The potential in the form a Pyro kernel calls it.

        A Pyro kernel passes ``{"params": q}`` with a single unbatched ``(d,)``
        state and expects a scalar back, so this is :meth:`evaluate_model` on a
        batch of one. Only defined when ``requires_metric`` is False.
        """
        theta_free = params_dict["params"]         # (d,)
        potential, _ = self.evaluate_model(theta_free.unsqueeze(0))
        return potential.value.squeeze(0)          # (1,d)->(1,)->()

    def diagnostics(self) -> dict:
        """Per-chain ``accept_rate``, ``num_divergences`` and ``step_size``,
        each a ``(num_chains,)`` tensor ordered by chain index. Empty before
        :meth:`run_mcmc` has run.

        These are the entries the other samplers report as well. What Pyro
        records beyond them, such as ``r_hat``, ``n_eff`` and the divergence
        indices, is reached through ``self.mcmc.diagnostics()``."""
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
        """Run :attr:`kernel` through Pyro's ``MCMC`` and return samples on the
        variables.

        The completed ``MCMC`` object is kept as ``self.mcmc``, which
        :meth:`diagnostics` reads.

        Parameters
        ----------
        initial_params : dict[str, Tensor]
            Starting point keyed by name, in the form a run returns. Fixed
            names are ignored. An entry of shape ``()`` starts every chain at
            the same point, one of shape ``(num_chains,)`` starts each chain at
            its own.
        num_samples : int
            Number of post-warmup samples per chain.
        num_warmup_steps : int
            Number of warmup iterations, during which Pyro adapts the step size
            and the mass matrix and nothing is collected.
        num_chains : int
            Number of chains. Pyro runs each in its own worker process.
        mp_context : str
            Multiprocessing start method Pyro spawns those workers with.
        disable_progbar : bool
            Disable the progress bar.

        Returns
        -------
        dict[str, Tensor]
            Samples on the variables, keyed by name and grouped by chain, each
            of shape ``(num_chains, num_samples)``. The fixed names are present
            too, each held at its value.
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