from typing import Callable, Optional
from collections import OrderedDict

import torch

from .BaseSampler import BaseSampler
from .adapters import DualAveraging

# =========================================================================== #
#                                                                             #
#  Euclidean HMC  (explicit leapfrog, constant mass matrix)                   #
#                                                                             #
#  Standard HMC with a position-independent mass matrix M.  The leapfrog is   #
#  explicit and symplectic, so acceptance is the plain energy difference with #
#  no Jacobian correction.  This is the constant-metric limit of RMHMC and    #
#  shares the BaseSampler operator interface.                                 #
#                                                                             #
#  The chain state carries q with the momentum p and the potential and its    #
#  gradient as tempered objects.  A trajectory ends where the next one starts, #
#  so the endpoint evaluation is stored and no gradient is recomputed at the   #
#  start of a step.  reorder retempers the tempered objects, so PT can permute #
#  a swapped configuration across temperature slots without a model eval.      #
#                                                                             #
# =========================================================================== #


class HMCState:
    """Batched HMC state over ``(N,)`` chains.

    Parameters
    ----------
    q : Tensor, shape (N, d)
        Position in free unconstrained coordinates.
    U : TemperedAffine
        Potential at ``q``.
    grad : TemperedAffine
        Gradient ``∂U/∂q`` at ``q``.
    p : Tensor, shape (N, d), or None
        Momentum, set by ``init_momentum`` and unset between transitions.
    """

    def __init__(self, q, U, grad, p=None):
        self.q = q
        self.U = U
        self.grad = grad
        self.p = p

    def reorder(self, perm: torch.Tensor) -> "HMCState":
        """Reorder the batch elements by ``perm``."""
        return HMCState(self.q[perm], self.U.reorder(perm), self.grad.reorder(perm),
                        None if self.p is None else self.p[perm])


# =========================================================================== #
#                                                                             #
#  HMC sampler                                                                #
#                                                                             #
# =========================================================================== #

class HMC(BaseSampler):
    """Euclidean Hamiltonian Monte Carlo with an explicit leapfrog integrator.

    Samples ``q`` under the Hamiltonian

        H(q, p) = U(q) + 1/2 pᵀ M⁻¹ p,

    with ``U`` the full unconstrained potential assembled by ``BaseSampler``
    and ``M`` a constant mass matrix.  Momentum is drawn ``p ~ N(0, M)``.  The
    model is given in constrained coordinates and evaluated through the space
    pull-back.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(theta_full) -> U_lik``, the scalar likelihood potential
        ``-log p(data | theta)`` in constrained coordinates.
    space : object
        Parameter space (priors, transform, free/fixed split).
    step_size : float
        Initial leapfrog step size.
    num_steps : int
        Leapfrog steps per transition.
    mass_matrix : Tensor or None
        Constant mass matrix ``M`` over the ``d`` free coordinates.  ``None``
        (default) is the identity, otherwise a ``(d, d)`` SPD tensor.
    adapt_step_size : bool
        Adapt ``step_size`` during warmup by dual averaging toward
        ``target_accept_prob``.
    target_accept_prob : float
        Target Metropolis acceptance probability for the adapter.
    da_gamma : float
        Dual-averaging step scale.
    divergence_threshold : float
        Raw ``|delta_H|`` above which, or non-finite for which, a step is a
        divergence.  Default 100.
    """

    def __init__(
        self,
        model_fn: Callable,
        space,
        *,
        step_size: float = 0.1,
        num_steps: int = 10,
        mass_matrix: Optional[torch.Tensor] = None,
        adapt_step_size: bool = True,
        target_accept_prob: float = 0.65,
        da_gamma: float = 0.05,
        divergence_threshold: float = 100.0,
    ):
        super().__init__(potential_fn=model_fn, space=space, requires_metric=False)

        if not 0.0 < target_accept_prob < 1.0:
            raise ValueError(
                f"target_accept_prob must be in (0, 1), got {target_accept_prob}")

        self._step_size_init       = step_size
        self.step_size             = step_size
        self.num_steps             = num_steps
        self._mass_matrix          = mass_matrix
        self._adapt_step_size      = adapt_step_size
        self._target_accept        = target_accept_prob
        self._da_gamma             = da_gamma
        self._divergence_threshold = divergence_threshold

    @property
    def trajectory_length(self):
        """Mean step size times ``num_steps``."""
        eps = self.step_size
        eps = float(eps.mean()) if torch.is_tensor(eps) else eps
        return eps * self.num_steps

    # ---- Mass matrix -------------------------------------------------------- #

    def _setup_mass(self, d, dtype, device):
        """Cholesky-factor the mass matrix (identity when unspecified) for a
        ``d``-dim free space.  Called once per run from ``init``.
        """
        if self._mass_matrix is None:
            M = torch.eye(d, dtype=dtype, device=device)
        else:
            M = torch.as_tensor(self._mass_matrix, dtype=dtype, device=device)
            if M.shape != (d, d):
                raise ValueError(
                    f"mass_matrix must have shape ({d}, {d}), got {tuple(M.shape)}")
        self._mass_chol = torch.linalg.cholesky(M)          # M = L Lᵀ

    def _sample_momentum(self, N, d, dtype, device):
        """Draw ``p ~ N(0, M)``, shape ``(N, d)``."""
        xi = torch.randn(N, d, dtype=dtype, device=device)
        return (self._mass_chol @ xi[..., None])[..., 0]     # p = L ξ

    def _inv_mass_times(self, p):
        """Return ``M⁻¹ p``, shape ``(N, d)``."""
        return torch.cholesky_solve(p[..., None], self._mass_chol)[..., 0]

    def _kinetic(self, p):
        """Return ``1/2 pᵀ M⁻¹ p``, shape ``(N,)``."""
        return 0.5 * (p * self._inv_mass_times(p)).sum(-1)

    # ---- Operator interface (composed by run_mcmc) -------------------------- #

    def init(self, q):
        """Size the per-chain ``step_size``, mass matrix, adapter, and counters
        from ``q``, arm adaptation, and return the initial :class:`HMCState`.
        """
        N, d = q.shape
        dtype, device = q.dtype, q.device

        self.step_size = torch.full((N,), float(self._step_size_init),
                                    dtype=dtype, device=device)
        self._setup_mass(d, dtype, device)

        self._step = 0
        self._accepted = torch.zeros(N, dtype=torch.long, device=device)
        self._num_divergences = torch.zeros(N, dtype=torch.long, device=device)
        self._reset_diagnostics()

        self._adapting = self._adapt_step_size
        if self._adapt_step_size:
            self._adapter = DualAveraging(gamma=self._da_gamma)
            self._adapter.prox_center = torch.log(self.step_size)
            self._adapter.reset()

        U, _, grad = self.evaluate_model(q, grad=True)
        return HMCState(q, U, grad)

    def step(self, s):
        """One chain transition: sample momentum at ``s``, integrate
        ``num_steps`` steps, then Metropolis accept/reject.  The returned state
        has momentum unset (the next ``step`` samples it)."""
        s = self.init_momentum(s)
        new = s
        for _ in range(self.num_steps):
            new = self.integration_step(new)
        return self.accept(new, s)

    def init_momentum(self, s):
        """Resample the momentum ``p ~ N(0, M)`` on ``s`` and return it."""
        N, d = s.q.shape
        s.p = self._sample_momentum(N, d, s.q.dtype, s.q.device)
        return s

    def integration_step(self, s):
        """One leapfrog step.  Returns a new state carrying the endpoint
        position, momentum, and the tempered potential / gradient there."""
        eps = self.step_size.unsqueeze(-1)          # (N, 1)
        p = s.p - 0.5 * eps * s.grad.value
        q = s.q + eps * self._inv_mass_times(p)
        U, _, grad = self.evaluate_model(q, grad=True)
        p = p - 0.5 * eps * grad.value
        return HMCState(q, U, grad, p)

    def accept(self, new, old):
        """Per-chain Metropolis accept/reject between the trajectory endpoint
        ``new`` and its start ``old``, plus bookkeeping and (while adapting) the
        step-size update.  Returns the chosen state with potential / gradient
        mixed per chain via their ``select`` and momentum unset -- the next
        ``step`` samples it."""
        H_new = new.U.value + self._kinetic(new.p)
        H_old = old.U.value + self._kinetic(old.p)
        delta_H_raw = H_new - H_old

        is_divergent = (~torch.isfinite(delta_H_raw)) \
            | (delta_H_raw > self._divergence_threshold)
        # Clamp is for Metropolis-ratio safety only, not accounting.
        delta_H = torch.where(torch.isfinite(delta_H_raw),
                              delta_H_raw, delta_H_raw.new_full((), 300.0))
        delta_H = delta_H.clamp(-300.0, 300.0)

        N = new.q.shape[0]
        accepted = torch.log(torch.rand(N, device=new.q.device, dtype=new.q.dtype)) < -delta_H
        chosen_q = torch.where(accepted.unsqueeze(-1), new.q, old.q)
        chosen_U = new.U.select(accepted, old.U)
        chosen_grad = new.grad.select(accepted, old.grad)

        # accept_prob = min(1, exp(-delta_H)), forced to 0 on divergence.
        accept_prob = torch.exp(torch.clamp(-delta_H, max=0.0))
        accept_prob = torch.where(is_divergent, torch.zeros_like(accept_prob), accept_prob)

        self._bookkeep(accepted, delta_H, is_divergent, accept_prob)
        return HMCState(chosen_q, chosen_U, chosen_grad)

    def _reset_diagnostics(self):
        """Zero the running per-chain energy summaries."""
        N = self.step_size.shape[0]
        dtype, device = self.step_size.dtype, self.step_size.device
        z = torch.zeros(N, dtype=dtype, device=device)
        self._delta_H_last    = z.clone()
        self._delta_H_abs_sum = z.clone()
        self._delta_H_abs_max = z.clone()

    def _bookkeep(self, accepted, delta_H, is_divergent, accept_prob):
        """Fold one transition into the per-chain counters and, while adapting,
        step the dual-averaging step-size update.
        """
        dH = delta_H.detach()
        self._delta_H_last     = dH
        self._delta_H_abs_sum += dH.abs()
        self._delta_H_abs_max  = torch.maximum(self._delta_H_abs_max, dH.abs())

        self._accepted += accepted
        self._num_divergences += is_divergent.long()
        self._step += 1

        if self._adapting:
            # g = target - accept_prob feeds dual averaging on log(step_size).
            g = self._target_accept - accept_prob
            self._adapter.step(g)
            self.step_size = torch.exp(self._adapter.get_state()[0])

    def end_warmup(self):
        """Freeze ``step_size`` to the dual-averaging running average, stop
        adapting, and reset the counters for the sampling phase.
        """
        if self._adapt_step_size:
            self.step_size = torch.exp(self._adapter.get_state()[1])
        self._adapting = False
        self._accepted = torch.zeros_like(self._accepted)
        self._num_divergences = torch.zeros_like(self._num_divergences)
        self._step = 0
        self._reset_diagnostics()

    def logging(self):
        """Per-step ``eps`` / ``|dH|`` / ``acc. prob`` strings for the progress
        bar.
        """
        if self._step == 0:
            return {}
        eps   = float(self.step_size.mean())
        dH    = float(self._delta_H_last.abs().max())
        accpr = float((self._accepted / self._step).mean())
        return OrderedDict(
            [
                ("eps", "{:.2e}".format(eps)),
                ("|dH|", "{:.2e}".format(dH)),
                ("acc. prob", "{:.3f}".format(accpr)),
            ]
        )

    def diagnostics(self):
        """Per-chain ``(num_chains,)`` diagnostics: ``accept_rate``,
        ``num_divergences``, ``step_size``, ``delta_H_abs_mean``,
        ``delta_H_abs_max``.
        """
        steps = max(self._step, 1)
        return {
            "accept_rate": self._accepted / steps,
            "num_divergences": self._num_divergences,
            "step_size": self.step_size,
            "delta_H_abs_mean": self._delta_H_abs_sum / steps,
            "delta_H_abs_max":  self._delta_H_abs_max,
        }
