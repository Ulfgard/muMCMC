from typing import Callable
from collections import OrderedDict

import torch

from .BaseSampler import BaseSampler

# =========================================================================== #
#                                                                             #
#  HamiltonianSampler: shared driver for the explicit-integrator family       #
#  (HMC, LMC, RMHMC).                                                          #
#                                                                             #
#  This class owns the transition loop, the Metropolis accept/reject, the     #
#  warmup step-size freeze, and the per-chain diagnostics.  A subclass         #
#  supplies its integrator and energy by overriding these hooks:              #
#                                                                             #
#      build_initial_state(q)      -> initial chain state at q                 #
#      sample_momentum(state)      -> state with a fresh momentum / velocity   #
#      integrate(state, step_size) -> one integrator substep                   #
#      acceptance_delta(new, old)  -> Metropolis exponent (energy + Jacobian)  #
#      adapt(accept_prob, delta_H) -> feed the step-size (etc.) adapter        #
#      reset_extra_diagnostics()   -> reset a subclass's running diagnostics   #
#                                                                             #
#  and a state exposing q, reorder(perm) and select_accepted(accepted, other).#
#  The step size lives in a step-size adapter (subclass-provided), so the      #
#  integrator receives it as an argument rather than reading it off self.      #
#  Diagnostics and progress-bar entries are registries a subclass extends from #
#  its __init__ (register_*), so it never overrides diagnostics()/logging().   #
#                                                                             #
# =========================================================================== #


class HamiltonianSampler(BaseSampler):
    """Base class for the explicit-integrator HMC-family samplers.

    Parameters
    ----------
    model_fn : callable
        Model potential in constrained coordinates (see ``BaseSampler``).
    space : object
        Parameter space (priors, transform, free/fixed split).
    requires_metric : bool
        Whether the sampler needs a position-dependent metric.
    num_steps : int
        Integrator substeps per transition.
    adapter : NoAdaptation | DualAveraging | Reinforce
        Step-size adapter: owns the per-chain step size and its warmup
        adaptation.
    divergence_threshold : float
        Value of ``|delta_H|`` above which, or non-finite for which, a step is
        a divergence.
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
    ):
        super().__init__(potential_fn=model_fn, space=space,
                         requires_metric=requires_metric)
        self.num_steps             = num_steps
        self._step_size_adapter    = adapter
        self._divergence_threshold = divergence_threshold

        # Diagnostics returned by diagnostics(): key -> callable giving an (N,)
        # tensor, evaluated on each call. Subclasses add entries in __init__.
        self._diagnostics = {}
        self.register_diagnostic("accept_rate",      lambda: self._accepted / max(self._step, 1))
        self.register_diagnostic("num_divergences",  lambda: self._num_divergences)
        self.register_diagnostic("step_size",        lambda: self.step_size)
        self.register_diagnostic("delta_H_abs_mean", lambda: self._delta_H_abs_sum / max(self._step, 1))
        self.register_diagnostic("delta_H_abs_max",  lambda: self._delta_H_abs_max)

        # Progress-bar entries: key -> callable giving a preformatted string.
        # Kept minimal; subclasses add the ones that matter for them.
        self._logging = {}
        self.register_logging("eps",       lambda: "{:.2e}".format(float(self.step_size.mean())))
        self.register_logging("acc. prob", lambda: "{:.3f}".format(float((self._accepted / max(self._step, 1)).mean())))

    @property
    def step_size(self):
        """The current per-chain step size, ``exp(x)`` of the adapter's log-step
        state (frozen to the warmup average after ``end_warmup``)."""
        return torch.exp(self._step_size_adapter.get_state()[0])

    # ---- operator interface (composed by run_mcmc) -------------------------- #

    def init(self, q):
        """Size the step-size adapter and per-chain counters from ``q`` and
        return the initial chain state (via :meth:`build_initial_state`)."""
        N = q.shape[0]
        self._step_size_adapter.reset(N, q.dtype, q.device)
        self._step = 0
        self._accepted = torch.zeros(N, dtype=torch.long, device=q.device)
        self._num_divergences = torch.zeros(N, dtype=torch.long, device=q.device)
        self._reset_diagnostics()
        return self.build_initial_state(q)

    def step(self, state):
        """One chain transition: fresh momentum, ``num_steps`` integrator
        substeps at the adapter's step size, then Metropolis accept/reject."""
        state = self.sample_momentum(state)
        step_size = self.step_size
        proposal = state
        for _ in range(self.num_steps):
            proposal = self.integrate(proposal, step_size)
        return self.accept(proposal, state)

    def accept(self, new, old):
        """Per-chain Metropolis accept/reject between the endpoint ``new`` and
        the start ``old``. Records the transition and, while the adapter is
        adapting, updates the step size. Returns the chosen state."""
        delta_raw = self.acceptance_delta(new, old)             # (N,)

        # Divergence: non-finite or |delta_H| over threshold (a non-finite
        # delta also carries a failed-proposal signal, e.g. RMHMC's unconverged
        # solve). The clamp below is Metropolis-ratio safety only.
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
        """Freeze the step size for the sampling phase and reset the counters."""
        self._step_size_adapter.finalize()
        self._accepted = torch.zeros_like(self._accepted)
        self._num_divergences = torch.zeros_like(self._num_divergences)
        self._step = 0
        self._reset_diagnostics()

    # ---- diagnostics / logging registries ----------------------------------- #

    def register_diagnostic(self, key, fn):
        """Add a per-chain diagnostic: ``key`` -> ``fn()`` returning an ``(N,)``
        tensor, evaluated on each :meth:`diagnostics` call. Call from a
        subclass ``__init__``."""
        self._diagnostics[key] = fn

    def register_logging(self, key, fn):
        """Add a progress-bar entry: ``key`` -> ``fn()`` returning a preformatted
        string, evaluated each step. Call from a subclass ``__init__``."""
        self._logging[key] = fn

    def diagnostics(self):
        """Per-chain ``(num_chains,)`` diagnostics from the registry."""
        return {key: fn() for key, fn in self._diagnostics.items()}

    def logging(self):
        """Progress-bar entries from the registry (empty before the first
        step)."""
        if self._step == 0:
            return {}
        return OrderedDict((key, fn()) for key, fn in self._logging.items())

    # ---- internal ----------------------------------------------------------- #

    def _bookkeep(self, accepted, delta, is_divergent, accept_prob):
        """Fold one transition into the per-chain counters and delta_H
        summaries (detached, so no per-step model graph is pinned), then run
        the adapter update."""
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
        z = torch.zeros(N, dtype=self.step_size.dtype, device=self.step_size.device)
        self._delta_H_last    = z.clone()
        self._delta_H_abs_sum = z.clone()
        self._delta_H_abs_max = z.clone()
        self.reset_extra_diagnostics()

    # ---- hooks the subclass overrides --------------------------------------- #

    def build_initial_state(self, q):
        """Hook, called by :meth:`init`. Return the initial chain state at
        positions ``q`` (model evaluated; momentum drawn later by
        :meth:`step`)."""
        raise NotImplementedError

    def sample_momentum(self, state):
        """Hook, called at the start of each :meth:`step`. Draw a fresh momentum
        on ``state`` (and reset any per-transition scratch); return ``state``."""
        raise NotImplementedError

    def integrate(self, state, step_size):
        """Hook, called ``num_steps`` times per :meth:`step`. Advance ``state``
        by one integrator substep at the per-chain ``step_size``; return the new
        state."""
        raise NotImplementedError

    def acceptance_delta(self, new, old):
        """Hook, called by :meth:`accept`. Return the per-chain Metropolis
        exponent (accept with probability ``min(1, exp(-delta))``), including
        any Jacobian correction; a non-finite value forces rejection. Must also
        populate ``new.U`` (and ``new.metric``) so the selected state carries
        them."""
        raise NotImplementedError

    def adapt(self, accept_prob, delta_H):
        """Hook, called by :meth:`accept` each transition. Feed the step-size
        adapter (and any other warmup adaptation) with this transition's
        ``accept_prob`` / ``delta_H``. A finalized adapter ignores the update,
        so this is a no-op after warmup."""
        raise NotImplementedError

    def reset_extra_diagnostics(self):
        """Hook, called by :meth:`init` and :meth:`end_warmup`. Override to
        (re)initialise the running diagnostic state a subclass keeps. Default:
        nothing."""
        pass
