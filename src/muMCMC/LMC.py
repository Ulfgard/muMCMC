from typing import Callable, Optional
from collections import OrderedDict

import torch

from .BaseSampler import BaseSampler
from .adapters import DualAveraging

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
#  Gamma and Omega come from grad U and dG_abc = d G_ab / d q_c (first-order   #
#  autodiff through G): Gamma^k_ij = 1/2 sum_l G^kl (dG_jli + dG_ilj - dG_ijl).#
#                                                                              #
#  With G constant Omega and the Jacobian vanish and LMC reduces to HMC with   #
#  mass matrix G.                                                              #
# ============================================================================ #


class LMCState:
    """Batched LMC state over ``(N,)`` chains.

    Parameters
    ----------
    q : Tensor, shape (N, d)
        Position in free unconstrained coordinates.
    v : Tensor, shape (N, d), or None
        Velocity, set by ``init_momentum`` and unset between transitions.
    U : TemperedAffine or None
        Potential at ``q``.
    metric : TemperedMetric or None
        Metric at ``q``.
    log_jac : Tensor, shape (N,)
        Accumulated ``log det J`` over the current trajectory (``nan`` once a
        half-kick determinant is non-positive).
    """

    def __init__(self, q, v=None, U=None, metric=None, log_jac=None):
        self.q = q
        self.v = v
        self.U = U
        self.metric = metric
        self.log_jac = log_jac

    def reorder(self, perm: torch.Tensor) -> "LMCState":
        """Reorder the batch elements by ``perm``. ``U`` and ``metric`` retemper;
        ``log_jac`` is slot-bound and stays in place."""
        return LMCState(
            self.q[perm],
            None if self.v is None else self.v[perm],
            None if self.U is None else self.U.reorder(perm),
            None if self.metric is None else self.metric.reorder(perm),
            self.log_jac,
        )


# =========================================================================== #
#                                                                              #
#  LMC sampler                                                                 #
#                                                                              #
# =========================================================================== #

class LMC(BaseSampler):
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
        super().__init__(potential_fn=model_fn, space=space, requires_metric=True)

        if not 0.0 < target_accept_prob < 1.0:
            raise ValueError(
                f"target_accept_prob must be in (0, 1), got {target_accept_prob}")

        self._step_size_init       = step_size
        self.step_size             = step_size
        self.num_steps             = num_steps
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

    # ---- Geometry ----------------------------------------------------------- #

    def _geometry(self, q):
        """Christoffel tensor and force term at ``q``. Returns
        ``(gamma, ginv_grad_phi)`` with ``gamma[n,k,i,j] = Gamma^k_ij`` and
        ``ginv_grad_phi = G^-1 grad phi``, ``phi = U + 1/2 log det G``. Both
        detached; derivatives come from first-order autodiff through ``G``.
        """
        N, d = q.shape
        q = q.detach().requires_grad_(True)
        with torch.enable_grad():
            potential, metric = self.evaluate_model(q)
            U = potential.value                                   # (N,)
            G = metric.value                                      # (N, d, d)

            def grad_or_zero(out):
                # a constant G (or U) has no grad path in q
                if not out.requires_grad:
                    return torch.zeros_like(q)
                (g,) = torch.autograd.grad(out.sum(), q, retain_graph=True,
                                           allow_unused=True)
                return torch.zeros_like(q) if g is None else g

            gU = grad_or_zero(U)
            # dG[n, a, b, c] = d G_ab / d q_c
            cols = [grad_or_zero(G[..., a, b]) for a in range(d) for b in range(d)]
            dG = torch.stack(cols, dim=1).reshape(N, d, d, d)
        G, gU, dG = G.detach(), gU.detach(), dG.detach()

        Ginv = torch.linalg.inv(G)                                # (N, d, d)
        # grad(U + 1/2 log det G): grad log det G_c = sum_ab Ginv_ab dG_abc
        grad_phi = gU + 0.5 * torch.einsum("nab,nabc->nc", Ginv, dG)

        # Gamma^k_ij = 1/2 sum_l Ginv_kl (dG_jli + dG_ilj - dG_ijl)
        T = (dG.permute(0, 3, 1, 2)                               # [n,i,j,l]=dG_jli
             + dG.permute(0, 1, 3, 2)                             # [n,i,j,l]=dG_ilj
             - dG)                                                # [n,i,j,l]=dG_ijl
        gamma = 0.5 * torch.einsum("nkl,nijl->nkij", Ginv, T)     # (N, d, d, d)
        return gamma, torch.einsum("nkl,nl->nk", Ginv, grad_phi)

    @staticmethod
    def _omega(gamma, v):
        """Return the matrix ``Omega(v)_ij = sum_k v_k Gamma^i_kj``, ``(N, d, d)``."""
        return torch.einsum("nkij,ni->nkj", gamma, v)

    def _half_kick(self, gamma, ginv_grad_phi, v):
        """Explicit half-kick (Lan et al. eq 12/14) at fixed geometry.

        Returns ``(v_out, dlogdet)`` with
        ``v_out = [I + (eps/2) Omega(v)]^-1 (v - (eps/2) G^-1 grad phi)`` and
        ``dlogdet = log det(I - (eps/2) Omega(v_out)) - log det(I + (eps/2) Omega(v))``
        (nan if either determinant is non-positive, tripping a divergence).
        """
        d = v.shape[-1]
        eye = torch.eye(d, dtype=v.dtype, device=v.device)
        half = 0.5 * self.step_size.view(-1, 1, 1)
        A = eye + half * self._omega(gamma, v)
        rhs = v - 0.5 * self.step_size.unsqueeze(-1) * ginv_grad_phi
        v_out = torch.linalg.solve(A, rhs[..., None])[..., 0]
        dlogdet = (torch.log(torch.linalg.det(eye - half * self._omega(gamma, v_out)))
                   - torch.log(torch.linalg.det(A)))
        return v_out, dlogdet

    def _energy(self, U, metric, v):
        """Return ``E = U - 1/2 log det G + 1/2 v^T G v``, shape ``(N,)``."""
        Gv = (metric.value @ v[..., None])[..., 0]
        return U + 0.5 * (v * Gv).sum(-1) - 0.5 * metric.log_det_metric()

    # ---- Operator interface (composed by run_mcmc) -------------------------- #

    def init(self, q):
        """Size the per-chain ``step_size``, adapter, and counters from ``q``,
        arm adaptation, and return the initial :class:`LMCState`."""
        N, d = q.shape
        dtype, device = q.dtype, q.device

        self.step_size = torch.full((N,), float(self._step_size_init),
                                    dtype=dtype, device=device)
        self._step = 0
        self._accepted = torch.zeros(N, dtype=torch.long, device=device)
        self._num_divergences = torch.zeros(N, dtype=torch.long, device=device)
        self._reset_diagnostics()

        self._adapting = self._adapt_step_size
        if self._adapt_step_size:
            self._adapter = DualAveraging(gamma=self._da_gamma)
            self._adapter.prox_center = torch.log(self.step_size)
            self._adapter.reset()

        with torch.no_grad():
            U, metric = self.evaluate_model(q)
        return LMCState(q, None, U, metric,
                        torch.zeros(N, dtype=dtype, device=device))

    def step(self, s):
        """One chain transition: sample velocity at ``s``, integrate
        ``num_steps`` steps, then Metropolis accept/reject. The returned state
        has velocity unset (the next ``step`` samples it)."""
        s = self.init_momentum(s)
        new = s
        for _ in range(self.num_steps):
            new = self.integration_step(new)
        return self.accept(new, s)

    def init_momentum(self, s):
        """Resample the velocity ``v ~ N(0, G(q)^-1)`` on ``s`` and reset the
        trajectory Jacobian accumulator; return ``s``."""
        N = s.q.shape[0]
        s.v = s.metric.inv_metric_times_vec(s.metric.sample_momentum())
        s.log_jac = torch.zeros(N, dtype=s.q.dtype, device=s.q.device)
        return s

    def integration_step(self, s):
        """One explicit geodesic leapfrog step. Returns a new state carrying the
        endpoint position, velocity, and the accumulated Jacobian."""
        eps = self.step_size.unsqueeze(-1)                    # (N, 1)
        gamma0, gphi0 = self._geometry(s.q)
        v_half, ld0 = self._half_kick(gamma0, gphi0, s.v)     # eq (12)
        q_new = s.q + eps * v_half                            # eq (13)
        gamma1, gphi1 = self._geometry(q_new)
        v_new, ld1 = self._half_kick(gamma1, gphi1, v_half)   # eq (14)
        return LMCState(q_new, v_new, None, None, s.log_jac + ld0 + ld1)

    def accept(self, new, old):
        """Per-chain Metropolis accept/reject between the trajectory endpoint
        ``new`` and its start ``old``, with the trajectory Jacobian folded in,
        plus bookkeeping and (while adapting) the step-size update."""
        with torch.no_grad():
            new.U, new.metric = self.evaluate_model(new.q)

        E_new = self._energy(new.U.value, new.metric, new.v)
        E_old = self._energy(old.U.value, old.metric, old.v)
        delta_H_raw = E_new - E_old - new.log_jac                 # includes Jacobian

        # A non-positive Jacobian factor makes log_jac (hence delta_H) nan, so an
        # unstable step is caught by the non-finite branch.
        is_divergent = (~torch.isfinite(delta_H_raw)) \
            | (delta_H_raw > self._divergence_threshold)
        delta_H = torch.where(torch.isfinite(delta_H_raw),
                              delta_H_raw, delta_H_raw.new_full((), 300.0))
        delta_H = delta_H.clamp(-300.0, 300.0)

        N = new.q.shape[0]
        accepted = torch.log(torch.rand(N, device=new.q.device, dtype=new.q.dtype)) < -delta_H
        chosen_q = torch.where(accepted.unsqueeze(-1), new.q, old.q)
        chosen_U = new.U.select(accepted, old.U)
        chosen_metric = new.metric.select(accepted, old.metric)

        accept_prob = torch.exp(torch.clamp(-delta_H, max=0.0))
        accept_prob = torch.where(is_divergent, torch.zeros_like(accept_prob), accept_prob)

        self._bookkeep(accepted, delta_H, is_divergent, accept_prob)
        z = torch.zeros(N, dtype=chosen_q.dtype, device=chosen_q.device)
        return LMCState(chosen_q, None, chosen_U, chosen_metric, z)

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
        step the dual-averaging step-size update."""
        dH = delta_H.detach()
        self._delta_H_last     = dH
        self._delta_H_abs_sum += dH.abs()
        self._delta_H_abs_max  = torch.maximum(self._delta_H_abs_max, dH.abs())

        self._accepted += accepted
        self._num_divergences += is_divergent.long()
        self._step += 1

        if self._adapting:
            # g = target - accept_prob feeds dual averaging on log(step_size)
            g = self._target_accept - accept_prob
            self._adapter.step(g)
            self.step_size = torch.exp(self._adapter.get_state()[0])

    def end_warmup(self):
        """Freeze ``step_size`` to the dual-averaging running average, stop
        adapting, and reset the counters for the sampling phase."""
        if self._adapt_step_size:
            self.step_size = torch.exp(self._adapter.get_state()[1])
        self._adapting = False
        self._accepted = torch.zeros_like(self._accepted)
        self._num_divergences = torch.zeros_like(self._num_divergences)
        self._step = 0
        self._reset_diagnostics()

    def logging(self):
        """Per-step ``eps`` / ``|dH|`` / ``acc. prob`` strings for the progress
        bar."""
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
        ``delta_H_abs_max``."""
        steps = max(self._step, 1)
        return {
            "accept_rate": self._accepted / steps,
            "num_divergences": self._num_divergences,
            "step_size": self.step_size,
            "delta_H_abs_mean": self._delta_H_abs_sum / steps,
            "delta_H_abs_max":  self._delta_H_abs_max,
        }
