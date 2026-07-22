from typing import Callable, Optional
import math

import torch

from .HamiltonianSampler import HamiltonianSampler
from .adapters import DualAveraging, NoAdaptation

# =========================================================================== #
#                                                                             #
#  Euclidean HMC  (explicit leapfrog, constant mass matrix)                   #
#                                                                             #
#  Standard HMC with a position-independent mass matrix M.  The leapfrog is   #
#  explicit and symplectic, so acceptance is the plain energy difference with #
#  no Jacobian correction.  This is the constant-metric limit of RMHMC and    #
#  shares the HamiltonianSampler transition machinery.                        #
#                                                                             #
#  The chain state carries q with the momentum p and the potential and its    #
#  gradient as tempered objects.  A trajectory ends where the next one starts, #
#  so the endpoint evaluation is stored and no gradient is recomputed at the   #
#  start of a step.  reorder retempers the tempered objects, so PT can permute #
#  a swapped configuration across temperature slots without a model eval.      #
#                                                                             #
# =========================================================================== #


class HMCState:
    """Batched HMC state over ``(N,)`` chains. Every field is config-bound, so
    ``reorder`` permutes all of them.

    Parameters
    ----------
    q : Tensor, shape (N, d)
        Position in free unconstrained coordinates.
    U : TemperedAffine
        Potential at ``q``.
    grad : TemperedAffine
        Gradient ``dU/dq`` at ``q``.
    p : Tensor, shape (N, d), or None
        Momentum. Drawn by ``sample_momentum``; ``None`` only on the initial
        state before the first step.
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

    def select_accepted(self, accepted: torch.Tensor, other: "HMCState") -> "HMCState":
        """Per-chain choice between this endpoint (where ``accepted``) and the
        start ``other``."""
        pick = accepted.unsqueeze(-1)
        return HMCState(
            torch.where(pick, self.q, other.q),
            self.U.select(accepted, other.U),
            self.grad.select(accepted, other.grad),
            torch.where(pick, self.p, other.p),
        )


# =========================================================================== #
#                                                                             #
#  HMC sampler                                                                #
#                                                                             #
# =========================================================================== #

class HMC(HamiltonianSampler):
    """Euclidean Hamiltonian Monte Carlo with an explicit leapfrog integrator.

    Samples ``q`` under the Hamiltonian

        H(q, p) = U(q) + 1/2 pT M^-1 p,

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
        if not 0.0 < target_accept_prob < 1.0:
            raise ValueError(
                f"target_accept_prob must be in (0, 1), got {target_accept_prob}")

        # The adapters work on the log step size; step_size = exp(adapter value).
        log_eps = math.log(step_size)
        if adapt_step_size:
            adapter = DualAveraging(init=log_eps, gamma=da_gamma)
        else:
            adapter = NoAdaptation(init=log_eps)
        super().__init__(model_fn, space, requires_metric=False, num_steps=num_steps,
                         adapter=adapter, divergence_threshold=divergence_threshold)

        self._mass_matrix   = mass_matrix
        self._target_accept = target_accept_prob

    # ---- Mass matrix -------------------------------------------------------- #

    def _setup_mass(self, d, dtype, device):
        """Cholesky-factor the mass matrix (identity when unspecified) for a
        ``d``-dim free space."""
        if self._mass_matrix is None:
            M = torch.eye(d, dtype=dtype, device=device)
        else:
            M = torch.as_tensor(self._mass_matrix, dtype=dtype, device=device)
            if M.shape != (d, d):
                raise ValueError(
                    f"mass_matrix must have shape ({d}, {d}), got {tuple(M.shape)}")
        self._mass_chol = torch.linalg.cholesky(M)          # M = L LT

    def _sample_momentum(self, N, d, dtype, device):
        """Draw ``p ~ N(0, M)``, shape ``(N, d)``."""
        xi = torch.randn(N, d, dtype=dtype, device=device)
        return (self._mass_chol @ xi[..., None])[..., 0]     # p = L xi

    def _inv_mass_times(self, p):
        """Return ``M^-1 p``, shape ``(N, d)``."""
        return torch.cholesky_solve(p[..., None], self._mass_chol)[..., 0]

    def _kinetic(self, p):
        """Return ``1/2 pT M^-1 p``, shape ``(N,)``."""
        return 0.5 * (p * self._inv_mass_times(p)).sum(-1)

    # ---- Hooks -------------------------------------------------------------- #

    def build_initial_state(self, q):
        """Set up the mass matrix and return the initial :class:`HMCState`."""
        self._setup_mass(q.shape[1], q.dtype, q.device)
        U, _, grad = self.evaluate_model(q, grad=True)
        return HMCState(q, U, grad)

    def sample_momentum(self, state):
        """Draw the momentum ``p ~ N(0, M)`` on ``state``."""
        N, d = state.q.shape
        state.p = self._sample_momentum(N, d, state.q.dtype, state.q.device)
        return state

    def integrate(self, state, step_size):
        """One leapfrog step at ``step_size``. Returns a new state carrying the
        endpoint position, momentum, and the tempered potential / gradient."""
        eps = step_size.unsqueeze(-1)               # (N, 1)
        p = state.p - 0.5 * eps * state.grad.value
        q = state.q + eps * self._inv_mass_times(p)
        U, _, grad = self.evaluate_model(q, grad=True)
        p = p - 0.5 * eps * grad.value
        return HMCState(q, U, grad, p)

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) - H(old)``; the endpoint potential is already on
        ``new`` (evaluated by the last leapfrog step)."""
        H_new = new.U.value + self._kinetic(new.p)
        H_old = old.U.value + self._kinetic(old.p)
        return H_new - H_old

    def adapt(self, accept_prob, delta_H):
        """Dual averaging toward ``target_accept_prob``."""
        self._step_size_adapter.update(self._target_accept - accept_prob)
