from abc import abstractmethod
from typing import Callable
from collections import OrderedDict
import math

import torch

from .MCMCSampler import MCMCSampler

# =========================================================================== #
#  The chain state a subclass supplies                                        #
#                                                                             #
#  The base class runs the transition loop and the Metropolis accept/reject   #
#  over a state object it never inspects beyond three members.                #
#                                                                             #
#      q                            positions, (N, d)                         #
#      reorder(perm)                the state with its batch axis permuted    #
#      select_accepted(mask, other) this state where mask is True and other   #
#                                   where it is False, per chain              #
#                                                                             #
#  Momentum, energy, metric and chart are the subclass's own and are read     #
#  only by the subclass hooks, which is what lets one loop serve HMC, LMC and #
#  RMHMC, whose states share little beyond q. reorder is used by the          #
#  tempering drivers, which relabel configurations across temperature slots   #
#  between transitions.                                                       #
#                                                                             #
#  The step size is held by the adapter rather than by the sampler and is     #
#  passed into integrate as an argument. A subclass therefore integrates at   #
#  whatever step size it is handed, and the trajectory-length normalization   #
#  can revise that step size between transitions on its own.                  #
# =========================================================================== #


class HamiltonianSampler(MCMCSampler):
    """Base class for the samplers whose transition is an explicit integration.

    One transition draws a momentum, takes ``num_steps`` integrator substeps at
    the current step size, and accepts or rejects the endpoint by Metropolis.
    The integrator, the momentum and the energy come from the subclass hooks at
    the bottom of this class.

    Diagnostics and progress-bar entries are registries. A subclass adds its own
    through :meth:`register_diagnostic` and :meth:`register_logging` from its
    ``__init__`` rather than overriding :meth:`diagnostics` or :meth:`logging`.

    Parameters
    ----------
    model_fn : callable
        Model potential on the full variable vector, as for
        :class:`~muMCMC.MCMCSampler.MCMCSampler`.
    space : Space
        Parameter space, giving the prior's normal chart and the free/fixed
        split.
    requires_metric : bool
        Whether ``model_fn`` also returns a position-dependent metric.
    num_steps : int
        Integrator substeps per transition. Revised each transition under
        ``step_normalization="max"``.
    adapter
        Step-size adapter, holding the per-chain step size and its warmup
        adaptation. Built by the subclass from its own arguments, so it is not
        something a caller supplies.
    divergence_threshold : float
        A transition counts as a divergence when ``|delta_H|`` exceeds this or
        is not finite.
    trajectory_length : float, optional
        Target for ``num_steps * step_size``, held by ``step_normalization``.
        Required when that is set and unused otherwise.
    step_normalization : {None, "fixed", "max"}, optional
        How the trajectory length is held at ``trajectory_length``. None leaves
        the step size to the adapter alone. "fixed" keeps ``num_steps`` and caps
        every step size at ``trajectory_length / num_steps``. "max" additionally
        re-derives ``num_steps`` each transition from the chain with the largest
        step size, so no chain's trajectory exceeds the target.

    Raises
    ------
    ValueError
        If ``step_normalization`` is not one of the three values above, or is
        set without a ``trajectory_length``.
    """

    def __init__(
        self,
        model_fn: Callable,
        space,
        *,
        requires_metric: bool,
        num_steps: int,
        adapter,
        divergence_threshold: float,
        trajectory_length: float = None,
        step_normalization: str = None,
    ):
        super().__init__(potential_fn=model_fn, space=space,
                         requires_metric=requires_metric)
        self.num_steps             = num_steps
        self._step_size_adapter    = adapter
        self._divergence_threshold = divergence_threshold

        if step_normalization not in (None, "fixed", "max"):
            raise ValueError(
                f"step_normalization must be None, 'fixed' or 'max', got "
                f"{step_normalization!r}")
        if step_normalization is not None and trajectory_length is None:
            raise ValueError("step_normalization requires a trajectory_length")
        self._trajectory_length = trajectory_length
        self._step_normalization = step_normalization

        # Evaluated on each diagnostics() call, so an entry reads current state.
        self._diagnostics = {}
        self.register_diagnostic("accept_rate",      lambda: self._accepted / max(self._step, 1))
        self.register_diagnostic("num_divergences",  lambda: self._num_divergences)
        self.register_diagnostic("step_size",        lambda: self.step_size)
        self.register_diagnostic("delta_H_abs_mean", lambda: self._delta_H_abs_sum / max(self._step, 1))
        self.register_diagnostic("delta_H_abs_max",  lambda: self._delta_H_abs_max)

        # Kept to the two entries every subclass reports the same way.
        self._logging = {}
        self.register_logging("eps",       lambda: "{:.2e}".format(float(self.step_size.mean())))
        self.register_logging("acc. prob", lambda: "{:.3f}".format(float((self._accepted / max(self._step, 1)).mean())))

    @property
    def step_size(self):
        """The per-chain step size, of shape ``(num_chains,)``. It is the
        exponential of the adapter's log step size, which :meth:`end_warmup`
        freezes at its warmup average."""
        return torch.exp(self._step_size_adapter.get_state()[0])

    # ---- operator interface (composed by run_mcmc) -------------------------- #

    def init(self, q):
        """The initial chain state at the positions ``q`` of shape
        ``(num_chains, d)``, from :meth:`build_initial_state`. The number of
        chains is taken from ``q`` here, so the adapter and the per-chain
        counters are sized and zeroed first."""
        N = q.shape[0]
        self._step_size_adapter.reset(N, q.dtype, q.device)
        self._step = 0
        self._accepted = torch.zeros(N, dtype=torch.long, device=q.device)
        self._num_divergences = torch.zeros(N, dtype=torch.long, device=q.device)
        self._reset_diagnostics()
        return self.build_initial_state(q)

    def step(self, state):
        """One transition per chain. A fresh momentum, then ``num_steps``
        integrator substeps at the current :attr:`step_size`, then Metropolis
        accept/reject of the endpoint against ``state``."""
        state = self.sample_momentum(state)
        self._normalize_trajectory()
        step_size = self.step_size
        proposal = state
        for _ in range(self.num_steps):
            proposal = self.integrate(proposal, step_size)
        return self.accept(proposal, state)

    def _normalize_trajectory(self):
        """Hold ``num_steps * step_size`` at ``trajectory_length``, doing
        nothing when ``step_normalization`` is None. In "max" mode ``num_steps``
        is re-derived from the largest smoothed step size, rounded up so the cap
        below binds it exactly and floored at 1. Either mode then caps every
        step size at ``trajectory_length / num_steps``."""
        if self._step_normalization is None:
            return
        if self._step_normalization == "max":
            h = torch.exp(self._step_size_adapter.get_state()[1])   # smoothed (N,)
            self.num_steps = max(1, math.ceil(self._trajectory_length / float(h.max())))
        cap = self._trajectory_length / self.num_steps
        self._step_size_adapter.set_upper_bound(math.log(cap))

    def accept(self, new, old):
        """The state kept by a per-chain Metropolis accept/reject between the
        endpoint ``new`` and the start ``old``, accepting with probability
        ``min(1, exp(-delta))`` for the ``delta`` of :meth:`acceptance_delta`.

        A chain counts as divergent when its ``delta`` is not finite or exceeds
        ``divergence_threshold`` in absolute value. A non-finite ``delta`` is
        replaced by 300 before the test, so that proposal is rejected. The
        ``accept_prob`` passed to :meth:`adapt` is zero for a divergent chain
        whether or not its proposal was kept, so a divergence never drives the
        step size upwards.

        Every transition is folded into the counters :meth:`diagnostics`
        reports."""
        delta_raw = self.acceptance_delta(new, old)             # (N,)

        # A subclass also uses non-finite to signal a proposal it could not
        # build at all. The clamp is Metropolis-ratio safety only.
        is_divergent = (~torch.isfinite(delta_raw)) \
            | (delta_raw.abs() > self._divergence_threshold)
        delta = torch.where(torch.isfinite(delta_raw), delta_raw,
                            delta_raw.new_full((), 300.0)).clamp(-300.0, 300.0)

        N = new.q.shape[0]
        accepted = torch.log(torch.rand(N, device=new.q.device, dtype=new.q.dtype)) < -delta

        # accept_prob = min(1, exp(-delta)), forced to 0 on divergence.
        accept_prob = torch.exp(torch.clamp(-delta, max=0.0))
        accept_prob = torch.where(is_divergent, torch.zeros_like(accept_prob), accept_prob)

        self._bookkeep(accepted, delta, is_divergent, accept_prob)
        return new.select_accepted(accepted, old)

    def end_warmup(self):
        """Freeze the step size at its warmup average and zero the counters, so
        the reported diagnostics cover the sampling phase alone."""
        self._step_size_adapter.finalize()
        self._accepted = torch.zeros_like(self._accepted)
        self._num_divergences = torch.zeros_like(self._num_divergences)
        self._step = 0
        self._reset_diagnostics()

    # ---- diagnostics / logging registries ----------------------------------- #

    def register_diagnostic(self, key, fn):
        """Add a per-chain diagnostic under ``key``, where ``fn()`` returns a
        ``(num_chains,)`` tensor and is called afresh on each
        :meth:`diagnostics` call. Call this from a subclass ``__init__``."""
        self._diagnostics[key] = fn

    def register_logging(self, key, fn):
        """Add a progress-bar entry under ``key``, where ``fn()`` returns an
        already formatted string and is called once per step. Call this from a
        subclass ``__init__``."""
        self._logging[key] = fn

    def diagnostics(self):
        """The registered diagnostics, each a ``(num_chains,)`` tensor. Always
        present are ``accept_rate``, ``num_divergences``, ``step_size``,
        ``delta_H_abs_mean`` and ``delta_H_abs_max``, all covering the steps
        since the last :meth:`init` or :meth:`end_warmup`."""
        return {key: fn() for key, fn in self._diagnostics.items()}

    def logging(self):
        """The registered progress-bar entries, empty before the first
        step."""
        if self._step == 0:
            return {}
        return OrderedDict((key, fn()) for key, fn in self._logging.items())

    # ---- internal ----------------------------------------------------------- #

    def _bookkeep(self, accepted, delta, is_divergent, accept_prob):
        """Fold one transition into the per-chain counters and the delta_H
        summaries, then run the adapter update."""
        # Detached, so no per-step model graph is held alive by a counter.
        dH = delta.detach()
        self._delta_H_last    = dH
        self._delta_H_abs_sum = self._delta_H_abs_sum + dH.abs()
        self._delta_H_abs_max = torch.maximum(self._delta_H_abs_max, dH.abs())
        self._accepted = self._accepted + accepted
        self._num_divergences = self._num_divergences + is_divergent.long()
        self._step += 1
        self.adapt(accept_prob.detach(), dH)

    def _reset_diagnostics(self):
        """Zero the delta_H summaries (and the subclass ones) for a new phase."""
        N = self.step_size.shape[0]
        zeros = torch.zeros(N, dtype=self.step_size.dtype, device=self.step_size.device)
        self._delta_H_last    = zeros.clone()
        self._delta_H_abs_sum = zeros.clone()
        self._delta_H_abs_max = zeros.clone()
        self.reset_extra_diagnostics()

    # ---- hooks the subclass overrides --------------------------------------- #

    @abstractmethod
    def build_initial_state(self, q):
        """Hook, called by :meth:`init`. Return the initial chain state at the
        positions ``q`` of shape ``(num_chains, d)``. The momentum is drawn by
        :meth:`sample_momentum` at the start of the first transition, so it need
        not be set here."""
        ...

    @abstractmethod
    def sample_momentum(self, state):
        """Hook, called at the start of each :meth:`step`. Draw a fresh momentum
        on ``state``, reset whatever the subclass keeps for the length of one
        transition, and return ``state``."""
        ...

    @abstractmethod
    def integrate(self, state, step_size):
        """Hook, called ``num_steps`` times per :meth:`step`. Advance ``state``
        by one integrator substep at the per-chain ``step_size`` of shape
        ``(num_chains,)`` and return the new state."""
        ...

    @abstractmethod
    def acceptance_delta(self, new, old):
        """Hook, called by :meth:`accept`. Return the per-chain Metropolis
        exponent ``delta`` of shape ``(num_chains,)``, which the proposal is
        accepted with probability ``min(1, exp(-delta))``. It carries the energy
        difference together with any Jacobian correction the proposal needs to
        satisfy detailed balance. A non-finite value rejects the proposal and
        counts as a divergence, which is how a proposal that could not be built
        at all is reported.

        It must also populate ``new.U``, and ``new.metric`` where the sampler
        has one, so that the state :meth:`accept` keeps carries them into the
        next transition."""
        ...

    @abstractmethod
    def adapt(self, accept_prob, delta_H):
        """Hook, called by :meth:`accept` once per transition with this
        transition's ``accept_prob`` and ``delta_H``, both of shape
        ``(num_chains,)``. Pass them to the step-size adapter and to any other
        warmup adaptation the subclass runs. A finalized adapter ignores the
        update, so this does nothing after :meth:`end_warmup`."""
        ...

    def reset_extra_diagnostics(self):
        """Hook, called by :meth:`init` and :meth:`end_warmup`. Override to zero
        the running diagnostic state a subclass keeps. Does nothing here."""
        pass
