from typing import Callable, Optional
import math

import torch

from .HamiltonianSampler import HamiltonianSampler
from ._adapters import DualAveraging, NoAdaptation

# =========================================================================== #
#  What the HMC state carries and why                                         #
#                                                                             #
#  The state holds the momentum p beside q, and the potential and its         #
#  gradient as tempered objects. Carrying the evaluation is what lets a       #
#  trajectory end where the next one starts, since the last leapfrog step     #
#  already produced the gradient the next transition starts from.             #
#                                                                             #
#  Holding U and grad as TemperedAffine rather than as their values is what   #
#  lets reorder move a configuration to another temperature slot. The two     #
#  parts are kept apart, so the slot's beta recombines them and no model      #
#  evaluation is needed to retemper.                                          #
# =========================================================================== #


class HMCState:
    """Batched HMC state over ``N`` chains. Every field is a property of the
    configuration and not of the batch position it occupies, so ``reorder``
    permutes all of them.

    Parameters
    ----------
    q : Tensor, shape (N, d)
        Position, which is the free variables ``theta`` in free-name order.
    U : TemperedAffine
        Potential at ``q``.
    grad : TemperedAffine
        Gradient ``dU/dq`` at ``q``.
    p : Tensor, shape (N, d), or None
        Momentum. Drawn by ``sample_momentum`` and ``None`` only on the initial
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


class HMC(HamiltonianSampler):
    """Euclidean Hamiltonian Monte Carlo with an explicit leapfrog integrator.

    Samples ``q`` under the Hamiltonian

        H(q, p) = U(q) + 1/2 pᵀ M⁻¹ p,

    with ``U`` the potential assembled by
    :class:`~muMCMC.MCMCSampler.MCMCSampler` and ``M`` a constant mass matrix.
    The momentum is drawn ``p ~ N(0, M)``. The chain runs on the free
    variables, so ``q`` is ``theta`` and nothing is transformed.

    The leapfrog is symplectic and reversible, so the Metropolis exponent is
    the energy difference ``H(new) - H(old)`` alone.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(theta_full) -> U_lik``, the likelihood potential
        ``-log p(data | theta)`` on the full variable vector.
    space : Space
        Parameter space, giving the prior's normal chart and the free/fixed
        split.
    step_size : float
        Leapfrog step size at the start of warmup, and for the whole run when
        ``adapt_step_size`` is False.
    num_steps : int
        Leapfrog steps per transition.
    mass_matrix : Tensor or None
        Constant mass matrix ``M``, an SPD tensor of shape ``(d, d)`` over the
        free coordinates. None is the identity.
    adapt_step_size : bool
        Adapt the step size during warmup by dual averaging toward
        ``target_accept_prob``.
    target_accept_prob : float
        Target Metropolis acceptance probability for the adaptation.
    da_gamma : float
        Dual-averaging gain. The log step size is displaced from its initial
        value by ``sqrt(t)/da_gamma`` times the averaged acceptance error, so a
        smaller value adapts faster. Unused when ``adapt_step_size`` is False.
    divergence_threshold : float
        A transition counts as a divergence when ``|delta_H|`` exceeds this or
        is not finite.

    Raises
    ------
    ValueError
        From the constructor, if ``target_accept_prob`` is not in ``(0, 1)``.
        From the first :meth:`init`, where ``d`` becomes known, if
        ``mass_matrix`` is not of shape ``(d, d)``.
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

        # The adapters work on the log step size, so step_size is its exponential.
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
        """Cholesky-factor the mass matrix, the identity when none was given,
        for a free space of dimension ``d``."""
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
        """Return ``M⁻¹ p``, shape ``(N, d)``."""
        return torch.cholesky_solve(p[..., None], self._mass_chol)[..., 0]

    def _kinetic(self, p):
        """Return ``1/2 pᵀ M⁻¹ p``, shape ``(N,)``."""
        return 0.5 * (p * self._inv_mass_times(p)).sum(-1)

    # ---- Hooks -------------------------------------------------------------- #

    def build_initial_state(self, q):
        """The initial :class:`HMCState` at ``q``, with the potential and its
        gradient evaluated. The free dimension is known here for the first
        time, so the mass matrix is checked and factored now."""
        self._setup_mass(q.shape[1], q.dtype, q.device)
        U, _, grad = self.evaluate_model(q, grad=True)
        return HMCState(q, U, grad)

    def sample_momentum(self, state):
        """Draw the momentum ``p ~ N(0, M)`` on ``state``."""
        N, d = state.q.shape
        state.p = self._sample_momentum(N, d, state.q.dtype, state.q.device)
        return state

    def integrate(self, state, step_size):
        """One leapfrog step at ``step_size``, returning a new state with the
        potential and its gradient evaluated at the endpoint."""
        eps = step_size.unsqueeze(-1)               # (N, 1)
        p = state.p - 0.5 * eps * state.grad.value
        q = state.q + eps * self._inv_mass_times(p)
        U, _, grad = self.evaluate_model(q, grad=True)
        p = p - 0.5 * eps * grad.value
        return HMCState(q, U, grad, p)

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) - H(old)``, with no Jacobian correction because
        the leapfrog preserves volume. ``new.U`` is already populated by the
        last leapfrog step."""
        H_new = new.U.value + self._kinetic(new.p)
        H_old = old.U.value + self._kinetic(old.p)
        return H_new - H_old

    def adapt(self, accept_prob, delta_H):
        """Update the step size by dual averaging on the acceptance error
        ``target_accept_prob - accept_prob``. ``delta_H`` is unused."""
        self._step_size_adapter.update(self._target_accept - accept_prob)
