from typing import Callable
import math

import torch

from .HamiltonianSampler import HamiltonianSampler
from .adapters import DualAveraging, NoAdaptation

# ============================================================================ #
#  Lagrangian Monte Carlo  (explicit geodesic integrator)                      #
# ============================================================================ #
#  Riemannian sampling in the velocity v = G(q)^-1 p (Lan, Stathopoulos,       #
#  Shahbaba & Girolami 2015).  The substitution p -> v turns RMHMC's implicit  #
#  generalized leapfrog into a fully explicit integrator (no fixed-point       #
#  solve), at the price of a non-unit Jacobian in the Metropolis test.         #
#                                                                              #
#  Energy.  p ~ N(0, G) and p = G v give v | q ~ N(0, G^-1) and                #
#                                                                              #
#      E(q, v) = U(q) - 1/2 log det G(q) + 1/2 v^T G(q) v,                     #
#                                                                              #
#  the log-det sign opposite RMHMC's Hamiltonian (p -> v contributes +det G).  #
#                                                                              #
#  Explicit integrator (Lan et al. Algorithm 2, e-RMLMC).  With the matrix     #
#  Omega(q, v)_ij = sum_k v_k Gamma^i_kj and phi = U + 1/2 log det G, one step #
#  is two matrix-solve half-kicks around an explicit drift:                    #
#                                                                              #
#      v_half = [I + (eps/2) Omega(q, v)]^-1 (v - (eps/2) G^-1 grad phi(q))    #
#      q'     = q + eps v_half                                                 #
#      v'     = [I + (eps/2) Omega(q', v_half)]^-1 (v_half - (eps/2) ...)      #
#                                                                              #
#  The velocity update is a d-by-d linear solve (explicit, no fixed point).    #
#                                                                              #
#  The map is reversible but not volume-preserving.  Each half-kick has        #
#  log-Jacobian  log det(I - (eps/2) Omega(v_out)) - log det(I + (eps/2)       #
#  Omega(v_in)), summed over the trajectory into                               #
#                                                                              #
#      alpha = min(1, exp(E_old - E_new) * det J).                             #
#                                                                              #
#  phi = U + 1/2 log det G, so grad phi is one backward pass.  Omega is built  #
#  directly, without the rank-3 Christoffel tensor: with w = G v and v fixed,  #
#  J = dw/dq is one batched reverse pass, D = (v . grad) G a double-backward,  #
#  and Omega(v) = 1/2 G^-1 (D + J - J^T).                                      #
#                                                                              #
#  With G constant Omega and the Jacobian vanish and LMC reduces to HMC with   #
#  mass matrix G.                                                              #
# ============================================================================ #


class LMCState:
    """Batched LMC state over ``(N,)`` chains. Every field is config-bound, so
    ``reorder`` permutes all of them (``log_jac`` is the current config's
    trajectory Jacobian, reset each transition).

    Parameters
    ----------
    q : Tensor, shape (N, d)
        Position in free unconstrained coordinates.
    v : Tensor, shape (N, d), or None
        Velocity. Drawn by ``sample_momentum``; ``None`` only on the initial
        state before the first step.
    U : TemperedAffine or None
        Potential at ``q``.
    metric : TemperedMetric or None
        Metric at ``q``.
    log_jac : Tensor, shape (N,), or None
        Accumulated ``log det J`` over the current trajectory.
    """

    def __init__(self, q, v=None, U=None, metric=None, log_jac=None):
        self.q = q
        self.v = v
        self.U = U
        self.metric = metric
        self.log_jac = log_jac

    def reorder(self, perm: torch.Tensor) -> "LMCState":
        """Reorder the batch elements by ``perm``. ``U`` and ``metric``
        retemper."""
        return LMCState(
            self.q[perm],
            None if self.v is None else self.v[perm],
            None if self.U is None else self.U.reorder(perm),
            None if self.metric is None else self.metric.reorder(perm),
            None if self.log_jac is None else self.log_jac[perm],
        )

    def select_accepted(self, accepted: torch.Tensor, other: "LMCState") -> "LMCState":
        """Per-chain choice between this endpoint (where ``accepted``) and the
        start ``other``; the Jacobian accumulator resets for the next step."""
        pick = accepted.unsqueeze(-1)
        z = torch.zeros(self.q.shape[0], dtype=self.q.dtype, device=self.q.device)
        return LMCState(
            torch.where(pick, self.q, other.q),
            torch.where(pick, self.v, other.v),
            self.U.select(accepted, other.U),
            self.metric.select(accepted, other.metric),
            z,
        )


# =========================================================================== #
#                                                                              #
#  LMC sampler                                                                 #
#                                                                              #
# =========================================================================== #

class LMC(HamiltonianSampler):
    """Lagrangian Monte Carlo with an explicit geodesic integrator.

    Samples ``q`` in the velocity variable ``v = G(q)^-1 p`` under the energy

        E(q, v) = U(q) - 1/2 log det G(q) + 1/2 v^T G(q) v,

    with ``U`` the full unconstrained potential and ``G`` the metric assembled
    by ``BaseSampler``. The integrator is explicit (a linear solve per
    half-kick); acceptance carries the trajectory Jacobian ``det J``.

    User contract
    -------------
    ``model_fn(theta_full) -> (U_lik, G_lik)`` as for RMHMC: the scalar
    likelihood potential and the ``(d_full, d_full)`` SPD metric in constrained
    coordinates.

    Parameters
    ----------
    model_fn : callable
        See above.
    space : object
        Parameter space (priors, transform, free/fixed split).
    step_size : float
        Initial integration step size.
    num_steps : int
        Integration steps per transition.
    adapt_step_size : bool
        Adapt ``step_size`` during warmup by dual averaging toward
        ``target_accept_prob``.
    target_accept_prob : float
        Target Metropolis acceptance probability for the adapter.
    da_gamma : float
        Dual-averaging step scale.
    divergence_threshold : float
        Raw ``|delta_H|`` above which a step is recorded as a divergence.
        Default 100.
    """

    def __init__(
        self,
        model_fn: Callable,
        space,
        *,
        step_size: float = 0.1,
        num_steps: int = 10,
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
        super().__init__(model_fn, space, requires_metric=True, num_steps=num_steps,
                         adapter=adapter, divergence_threshold=divergence_threshold)

        self._target_accept = target_accept_prob

    # ---- Geometry ----------------------------------------------------------- #

    def _geometry(self, q):
        """Force term and velocity operator at ``q``. Returns
        ``(ginv_grad_phi, omega)`` with ``ginv_grad_phi = G^-1 grad phi``,
        ``phi = U + 1/2 log det G``, and a closure ``omega(v)`` returning the
        matrix ``Omega(v) = 1/2 G^-1 (D + J - J^T)``, shape ``(N, d, d)``, where
        ``J = d(G v)/dq`` and ``D = (v . grad) G``. The closure differentiates
        through ``G`` on a retained graph, so each ``Omega`` is one batched
        reverse pass plus one double-backward, with no rank-3 tensor.
        """
        N, d = q.shape
        q = q.detach().requires_grad_(True)
        with torch.enable_grad():
            potential, metric = self.evaluate_model(q)
            U = potential.value                                   # (N,)
            G = metric.value                                      # (N, d, d)

            # grad phi = grad(U + 1/2 log det G) in one reverse pass.
            phi = U + 0.5 * torch.logdet(G)
            (grad_phi,) = torch.autograd.grad(phi.sum(), q, retain_graph=True)

            # D = (v . grad) G via double-backward: vjp_c(w) = sum_ab w_ab dG_abc
            # is linear in the seed w, so its derivative contracted with v gives
            # sum_c v_c dG_abc, without materialising dG_abc.
            if G.requires_grad:
                seed = torch.zeros_like(G, requires_grad=True)
                (vjp,) = torch.autograd.grad(G, q, grad_outputs=seed,
                                             create_graph=True, retain_graph=True)
            else:
                seed = vjp = None

        Ginv = torch.linalg.inv(G.detach())                       # (N, d, d)
        ginv_grad_phi = torch.einsum("nkl,nl->nk", Ginv, grad_phi.detach())

        units = torch.eye(d, dtype=q.dtype, device=q.device)[:, None, :].expand(d, N, d)

        def omega(v):
            if vjp is None:                                       # G constant: Omega = 0
                return torch.zeros(N, d, d, dtype=q.dtype, device=q.device)
            Gv = (G @ v[..., None])[..., 0]                       # (N, d)
            (jac,) = torch.autograd.grad(Gv, q, grad_outputs=units,
                                         is_grads_batched=True, retain_graph=True)
            J = jac.permute(1, 0, 2)                              # J_lj = d(Gv)_l/dq_j
            (D,) = torch.autograd.grad(vjp, seed, grad_outputs=v, retain_graph=True)
            S = D + J - J.transpose(-1, -2)
            return 0.5 * torch.einsum("nkl,nlj->nkj", Ginv, S)

        return ginv_grad_phi, omega

    def _half_kick(self, omega, ginv_grad_phi, v, step_size):
        """Explicit half-kick (Lan et al. eq 12/14) at fixed geometry and
        per-chain ``step_size``.

        Returns ``(v_out, dlogdet)`` with
        ``v_out = [I + (eps/2) Omega(v)]^-1 (v - (eps/2) G^-1 grad phi)`` and
        ``dlogdet = log det(I - (eps/2) Omega(v_out)) - log det(I + (eps/2) Omega(v))``
        (nan if either determinant is non-positive, tripping a divergence).
        """
        d = v.shape[-1]
        eye = torch.eye(d, dtype=v.dtype, device=v.device)
        half = 0.5 * step_size.view(-1, 1, 1)
        A = eye + half * omega(v)
        rhs = v - 0.5 * step_size.unsqueeze(-1) * ginv_grad_phi
        v_out = torch.linalg.solve(A, rhs[..., None])[..., 0]
        dlogdet = (torch.log(torch.linalg.det(eye - half * omega(v_out)))
                   - torch.log(torch.linalg.det(A)))
        return v_out, dlogdet

    def _energy(self, U, metric, v):
        """Return ``E = U - 1/2 log det G + 1/2 v^T G v``, shape ``(N,)``."""
        Gv = (metric.value @ v[..., None])[..., 0]
        return U + 0.5 * (v * Gv).sum(-1) - 0.5 * metric.log_det_metric()

    # ---- Hooks -------------------------------------------------------------- #

    def build_initial_state(self, q):
        """Evaluate the model at ``q`` and return the initial :class:`LMCState`
        (velocity drawn later by :meth:`sample_momentum`)."""
        with torch.no_grad():
            U, metric = self.evaluate_model(q)
        return LMCState(q, None, U, metric)

    def sample_momentum(self, state):
        """Draw the velocity ``v ~ N(0, G(q)^-1)`` on ``state`` and reset its
        trajectory Jacobian accumulator."""
        N = state.q.shape[0]
        state.v = state.metric.inv_metric_times_vec(state.metric.sample_momentum())
        state.log_jac = torch.zeros(N, dtype=state.q.dtype, device=state.q.device)
        return state

    def integrate(self, state, step_size):
        """One explicit geodesic leapfrog step at ``step_size``. Returns a new
        state carrying the endpoint position, velocity, and accumulated
        Jacobian."""
        eps = step_size.unsqueeze(-1)                         # (N, 1)
        gphi0, omega0 = self._geometry(state.q)
        v_half, ld0 = self._half_kick(omega0, gphi0, state.v, step_size)   # eq (12)
        q_new = state.q + eps * v_half                        # eq (13)
        gphi1, omega1 = self._geometry(q_new)
        v_new, ld1 = self._half_kick(omega1, gphi1, v_half, step_size)     # eq (14)
        return LMCState(q_new, v_new, None, None, state.log_jac + ld0 + ld1)

    def acceptance_delta(self, new, old):
        """``delta = E(new) - E(old) - log det J``, evaluating the endpoint
        potential/metric (which the integrator left unset). A non-positive
        Jacobian factor makes ``log_jac`` (hence ``delta``) nan, so an unstable
        step is caught by ``accept``'s non-finite branch."""
        with torch.no_grad():
            new.U, new.metric = self.evaluate_model(new.q)
        E_new = self._energy(new.U.value, new.metric, new.v)
        E_old = self._energy(old.U.value, old.metric, old.v)
        return E_new - E_old - new.log_jac

    def adapt(self, accept_prob, delta_H):
        """Dual averaging toward ``target_accept_prob``."""
        self._step_size_adapter.update(self._target_accept - accept_prob)
