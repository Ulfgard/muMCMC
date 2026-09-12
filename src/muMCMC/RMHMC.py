from typing import Callable, Tuple

import torch
import math

from .HamiltonianSampler import HamiltonianSampler
from .spaces import TemperedMetric
from ._adapters import Reinforce, NoAdaptation
from ._solvers import FixedPointSolver

# =========================================================================== #
#                                                                             #
#  RMHMC helpers  (implicit midpoint integrator)                              #
#                                                                             #
#  The implicit midpoint rule on H = U + ½ pᵀG⁻¹p + ½ log det G reads, with   #
#  qb = ½(q0 + q1), pb = ½(p0 + p1) and h the step,                           #
#                                                                             #
#      q1 = q0 + h G(qb)⁻¹ pb,      p1 = p0 − h ∂H/∂q(qb, pb).                #
#                                                                             #
#  The first equation gives pb = G(qb) dq / h with dq = q1 − q0, and at that  #
#  pb the kinetic gradient is −Γ(qb, dq)/h² with Γ(q, v) = ∇_q ½ vᵀG(q)v, so  #
#  p drops out and the step is one root find in d unknowns,                   #
#                                                                             #
#      F(q1) = G(qb) dq − ½ Γ(qb, dq) + (h²/2) ∇V(qb) − h p0 = 0,             #
#      p1    = [G(qb) dq + ½ Γ(qb, dq)] / h − (h/2) ∇V(qb),                   #
#                                                                             #
#  with V = U + ½ log det G. DF = G(q0) + O(h²) + O(dq), so preconditioning   #
#  Picard by G(q0)⁻¹ makes it a frozen-Jacobian Newton, and the residual is   #
#  read in the G(q0)⁻¹ norm, the implied position error in the metric of the  #
#  substep's start. The exact DF carries ∂Γ/∂q, second derivatives of the     #
#  metric the model interface does not expose, which rules the newton rule    #
#  out here.                                                                  #
#                                                                             #
#  Picard and Anderson drive the same F, so the endpoint is solver- and       #
#  damping-independent. Only the iteration count and the stability differ.    #
#                                                                             #
# =========================================================================== #

#  ---- Hamiltonian --------------------------------------------------------- #

def _hamiltonian(
    q: torch.Tensor,
    p: torch.Tensor,
    U: torch.Tensor,
    metric: TemperedMetric,
) -> torch.Tensor:
    """
    H(q, p) = U + ½ pᵀ G⁻¹(q) p + ½ log det G(q).

    Parameters
    ----------
    q : torch.Tensor
        Position. Unused, present to mirror H(q, p).
    p : torch.Tensor
        Momentum.
    U : torch.Tensor
        Potential pre-evaluated at q.
    metric : TemperedMetric
        Metric pre-evaluated at q.
    """
    Ginv_p = metric.inv_metric_times_vec(p)
    kinetic = 0.5 * (p * Ginv_p).sum(-1)
    return U + kinetic + 0.5 * metric.log_det_metric()


#  ---- Midpoint terms ------------------------------------------------------ #

def _midpoint_terms(
    q0: torch.Tensor,
    q1: torch.Tensor,
    eps: torch.Tensor,
    evaluate_model: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The two terms of the eliminated midpoint equation at qb = ½(q0 + q1),
    dq = q1 − q0: ``G(qb) dq`` and the gradient

        ∇S(qb),   S(qb) = (ε²/2) V(qb) − ¼ dqᵀ G(qb) dq,

    which is (ε²/2) ∇V − ½ Γ(qb, dq), so the residual is ``G dq + ∇S − ε p0``
    and the endpoint momentum ``(G dq − ∇S) / ε``. One model evaluation and
    one backward pass, dq held fixed.

    Parameters
    ----------
    q0, q1 : (N, d)
        Start and endpoint iterate.
    eps : (N,)
        Per-chain step size.
    evaluate_model : Callable
        Maps q to (potential, metric).
    """
    dq = (q1 - q0).detach()
    qb = (0.5 * (q0 + q1)).detach().requires_grad_(True)     # fresh leaf
    with torch.enable_grad():
        potential, metric = evaluate_model(qb)
        G_dq = (metric.value @ dq.unsqueeze(-1)).squeeze(-1)
        V = potential.value + 0.5 * metric.log_det_metric()
        S = 0.5 * eps * eps * V - 0.25 * (dq * G_dq).sum(-1)
        # S has shape (N,) with no cross-chain coupling, so grad of the sum
        # is the per-chain gradient.
        (grad_S,) = torch.autograd.grad(S.sum(), qb)
    return G_dq.detach(), grad_S


#  ---- Implicit midpoint step ---------------------------------------------  #

def _implicit_midpoint_step(q, p, eps, evaluate_model, solver, metric=None,
                            q_init=None):
    """Solve the implicit-midpoint equation for the step endpoint q1, then
    read p1 off it (see the module header). The Picard iteration is
    preconditioned by G(q)⁻¹, DF at dq = 0, and the residual is measured in
    the G(q)⁻¹ norm ``max|L⁻¹ F|`` with L the Cholesky factor of G(q).

    Parameters
    ----------
    q, p : (N, d)
        Start-of-step position and momentum.
    eps : (N,)
        Per-chain step size.
    evaluate_model : callable
        Maps q to (potential, metric).
    solver : FixedPointSolver
        Owns the update rule, the tolerance and the fallback ladder.
    metric : TemperedMetric or None
        The metric at ``q``. None evaluates it.
    q_init : (N, d) or None
        Warm start. None starts from ``q``, which is also where the fallback
        ladder restarts.

    Returns
    -------
    tuple
        ``(q_out, p_out, iters, residual)``, the last two per chain, the
        residual in the G(q)⁻¹ norm.
    """
    if metric is None:
        with torch.no_grad():
            _, metric = evaluate_model(q)
    L = metric.L.detach()
    e = eps.unsqueeze(-1)

    def residual_fn(q1):
        G_dq, grad_S = _midpoint_terms(q, q1, eps, evaluate_model)
        return G_dq + grad_S - e * p

    def precond(F):                                       # G(q)⁻¹ F
        return torch.cholesky_solve(F.unsqueeze(-1), L).squeeze(-1)

    def norm_fn(F):                                       # max|L⁻¹ F|
        return torch.linalg.solve_triangular(
            L, F.unsqueeze(-1), upper=False).squeeze(-1).abs().amax(-1)

    q1, iters, residual = solver.solve(
        residual_fn, q if q_init is None else q_init, precond=precond,
        cold_start=q, norm_fn=norm_fn)
    G_dq, grad_S = _midpoint_terms(q, q1, eps, evaluate_model)
    return q1, (G_dq - grad_S) / e, iters, residual


# =========================================================================== #
#                                                                             #
#  Chain state                                                                #
#                                                                             #
#  ``U`` and ``metric`` are configuration-bound objects that carry their      #
#  own temperature and retemper themselves under ``reorder``, so the          #
#  state stays agnostic to tempering.  ``max_residual`` and ``fp_iters``      #
#  are integrator diagnostics bound to the slot. The warm-start               #
#  displacements are reset by ``init`` and ``accept`` and carried forward     #
#  by ``step``.                                                               #
#                                                                             #
# =========================================================================== #

class RMHMCState:
    """
    Working state of one RMHMC trajectory, batched over (N,) chains. Purely
    config-bound: every field travels with the configuration, so ``reorder``
    permutes all of them (the integrator's residual / iteration diagnostics are
    slot-bound and live on the sampler instead).

    Attributes
    ----------
    q, p : (N, d)
        Position and momentum.
    U : TemperedAffine or None
        Potential at ``q`` (``U.value`` is the ``(N,)`` energy). Set at
        ``init`` / ``accept`` and by ``integrate`` at its endpoint.
    metric : TemperedMetric or None
        Metric at ``q``. Set with ``U``; ``integrate`` preconditions the next
        substep's solve by it.
    dq, dq_prev : (N, d) or None
        Last two converged position displacements, which warm-start the next
        substep's solve by quadratic extrapolation when the sampler's
        ``warm_start`` is on. None at a trajectory start (dropped by accept).
    """

    def __init__(self, q, p=None, U=None, metric=None, dq=None, dq_prev=None):
        self.q = q
        self.p = p
        self.U = U
        self.metric = metric
        self.dq = dq
        self.dq_prev = dq_prev

    def reorder(self, perm):
        """Permute the batch elements by ``perm`` (an ``(N,)`` long index
        tensor): slot ``i`` of the result carries the configuration from
        ``perm[i]``. Absent (None) fields stay None."""
        return RMHMCState(
            q       = self.q[perm],
            p       = None if self.p is None else self.p[perm],
            U       = None if self.U is None else self.U.reorder(perm),
            metric  = None if self.metric is None else self.metric.reorder(perm),
            dq      = None if self.dq is None else self.dq[perm],
            dq_prev = None if self.dq_prev is None else self.dq_prev[perm],
        )

    def select_accepted(self, accepted, other):
        """Per-chain choice between this endpoint (where ``accepted``) and the
        start ``other``."""
        pick = accepted.unsqueeze(-1)
        return RMHMCState(
            torch.where(pick, self.q, other.q),
            torch.where(pick, self.p, other.p),
            self.U.select(accepted, other.U),
            self.metric.select(accepted, other.metric),
        )


# =========================================================================== #
#                                                                             #
#  RMHMC sampler                                                              #
#                                                                             #
#  The transition machinery (init / step / accept / end_warmup / diagnostics) #
#  is inherited from HamiltonianSampler. RMHMC supplies the integrator and    #
#  energy through the build_initial_state / sample_momentum / integrate /     #
#  acceptance_delta / adapt hooks. All chains run in one batched state.       #
#                                                                             #
# =========================================================================== #

class RMHMC(HamiltonianSampler):
    """
    Riemannian Manifold HMC with the implicit-midpoint integrator, sampling
    q ~ exp(−U(q)) under the position-dependent metric G(q) with Hamiltonian
    H(q, p) = U(q) + ½ pᵀ G⁻¹(q) p + ½ log det G(q).

    Runs on the free variables, so ``q`` is ``theta``. The model is specified
    there and read by :meth:`MCMCSampler.evaluate_model`, which adds the prior's
    potential and its metric. Nothing is transformed.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(theta_full) -> (U_lik, G_lik)``: full variable vector to
        scalar likelihood potential ``-log p(data | theta)`` and (d_full,
        d_full) SPD likelihood metric on the same vector.
    space
        Parameter space: the prior's normal chart and the free/fixed split.
    step_size : float
        Integration step size. Adapted during warmup when adapting.
    num_steps : int
        Number of implicit-midpoint substeps per transition.
    adapt_step_size : bool
        Adapt the step size during warmup via the REINFORCE adapter.
        Default True.
    adaptation_sigma : float
        Perturbation scale of the REINFORCE adapter. Default 0.1.
    fp_max_iter : int
        Maximum fixed-point iterations per substep. Default 100.
    fp_tol : float
        Convergence tolerance for the position solve, on the G(q0)⁻¹ norm of
        the residual, which is the position error in the metric of the
        substep's start. Default 1e-8.
    solver : str
        Fixed-point solver: ``"picard"`` (default) or ``"anderson"``, both
        preconditioned by G(q0)⁻¹. Newton is rejected, since the residual's
        Jacobian carries second derivatives of the metric, which are not
        available here.
    warm_start : bool
        Start each substep's solve by extrapolating the endpoint from the
        last two displacements rather than from ``q0``. Changes the iteration
        count only, not the fixed point. Default False.
    anderson_history : int or None
        History length ``m`` for the Anderson solver (ignored for Picard).
        ``None`` (default) resolves per-solve to ``dim(q)``, the dimension of
        the solve. Must be at least 1 if given.
    damping : float
        Under-relaxation factor β ∈ (0, 1] shared by both solvers.
        Default 1.0 (undamped).
    fallback_damping : tuple of float
        Fallback ladder: on non-convergence, re-solve the failed chains with the
        base solver damped by each factor in turn (each in (0, 1), relative to
        ``damping``, default ``(0.5, 0.25)``). Endpoint-preserving, so it makes
        solver-driven rejections rare. ``()`` disables it.
    fallback_iter_scale : int
        Per-level iteration cap as a multiple of ``fp_max_iter``. Default 4.
    step_normalization : {None, "fixed", "max"}
        Hold the trajectory length ``num_steps * step_size`` at its initial value
        while step sizes adapt. ``"max"`` also re-derives ``num_steps`` from the
        fastest chain. Default None (off).
    divergence_threshold : float
        Raw |delta_H| above which (or non-finite values for which) the step
        is recorded as a divergence. Default 100.

    References
    ----------
    Brofos and Lederman, Evaluating the implicit midpoint integrator for
    Riemannian manifold Hamiltonian Monte Carlo (2021), Algorithm I.M.(a).

    Notes
    -----
    The implicit-midpoint integrator can conserve energy over a wide range of
    step sizes, and exactly so on a Gaussian target up to the fixed-point
    tolerance. That makes acceptance a poor thing to adapt against, because
    whenever the solve converges it saturates near 1 and carries almost no
    gradient on the step size. The true knob is integrator accuracy, so the
    REINFORCE adapter instead targets solver cost and energy error, meaning the
    residual and iteration count per substep together with |delta_H| as described
    in :meth:`adapt`. This keeps acceptance close to 1 while steering the step
    size by how well the trajectory is actually resolved. ``adaptation_sigma``
    sets the exploration scale of that search.
    """

    def __init__(
        self,
        model_fn: Callable,
        space,
        *,
        step_size: float,
        num_steps: int = 10,
        adapt_step_size: bool = True,
        adaptation_sigma: float = 0.1,
        fp_max_iter: int = 100,
        fp_tol: float = 1e-8,
        solver: str = "picard",
        warm_start: bool = False,
        anderson_history: int = None,
        damping: float = 1.0,
        fallback_damping: Tuple[float, ...] = (0.5, 0.25),
        fallback_iter_scale: int = 4,
        step_normalization: str = None,
        divergence_threshold: float = 100.0
    ):
        self._solver = FixedPointSolver(
            solver, damping=damping, anderson_history=anderson_history,
            max_iter=fp_max_iter, tol=fp_tol,
            fallback_damping=fallback_damping,
            fallback_iter_scale=fallback_iter_scale)
        if self._solver.needs_jacobian:
            raise ValueError(
                f"solver {solver!r} needs the Jacobian of the midpoint residual, "
                f"which carries second derivatives of the metric and is not "
                f"available here. Use 'picard' or 'anderson'.")

        # The adapters work on the log step size, so step_size is its exponential.
        log_eps = math.log(step_size)
        if adapt_step_size:
            adapter = Reinforce(sigma=adaptation_sigma, init=log_eps)
        else:
            adapter = NoAdaptation(init=log_eps)
        super().__init__(model_fn, space, requires_metric=True, num_steps=num_steps,
                         adapter=adapter, divergence_threshold=divergence_threshold,
                         trajectory_length=num_steps * step_size,
                         step_normalization=step_normalization)

        self._fp_tol = fp_tol
        self.warm_start = warm_start

        # Solver diagnostics. ``residual`` and ``fp_iters`` take each transition's
        # worst substep, ``fp_iters_total`` the transition's substeps summed; the
        # means are then over transitions.
        self.register_diagnostic("residual_mean", lambda: self._residual_sum / max(self._step, 1))
        self.register_diagnostic("residual_max",  lambda: self._residual_max)
        self.register_diagnostic("fp_iters_mean", lambda: self._fp_iters_sum / max(self._step, 1))
        self.register_diagnostic("fp_iters_max",  lambda: self._fp_iters_max)
        self.register_diagnostic("fp_iters_total", lambda: self._fp_iters_total / max(self._step, 1))
        self.register_logging("|r|", lambda: "{:.2e}".format(float(self._step_residual.max())))

    def build_initial_state(self, q):
        """Evaluate the model at ``q`` and return the initial :class:`RMHMCState`
        (momentum drawn later by :meth:`sample_momentum`). Seeds the
        per-transition solver scratch so the sampler is usable right after init."""
        zeros = torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
        self._step_residual = zeros.clone()
        self._step_iters    = zeros.clone()
        self._step_iters_total = zeros.clone()
        with torch.no_grad():
            U, metric = self.evaluate_model(q)
        return RMHMCState(q, U=U, metric=metric)

    def sample_momentum(self, state):
        """Draw the momentum ``p ~ N(0, G(q))`` on ``state`` and reset the
        per-transition solver scratch (worst residual / iteration count over the
        transition's substeps), read by :meth:`acceptance_delta` and :meth:`adapt`."""
        N = state.q.shape[0]
        zeros = torch.zeros(N, dtype=state.q.dtype, device=state.q.device)
        self._step_residual = zeros.clone()
        self._step_iters    = zeros.clone()
        self._step_iters_total = zeros.clone()
        state.p = state.metric.sample_momentum()
        return state

    def integrate(self, state, step_size):
        """One implicit-midpoint substep at ``step_size``, tracking the worst
        fixed-point residual and iteration count over the transition's substeps
        and their total (read by :meth:`acceptance_delta` and :meth:`adapt`).

        The solve is preconditioned by the metric at the start, ``state.metric``
        (evaluated if absent). The endpoint's potential and metric are evaluated
        on the way out, so the next substep has its preconditioner and
        :meth:`acceptance_delta` its energy.

        With ``warm_start`` the solve starts by extrapolating the endpoint from
        the last two converged displacements: quadratic in general, linear on
        the second substep and trivial on the first. The guess only changes the
        iteration count and not the fixed point, so neither the map nor
        detailed balance is affected. The displacements reset per trajectory
        (``accept`` builds a fresh state without them)."""
        q_init = None                                           # from q
        if self.warm_start and state.dq is not None:
            if state.dq_prev is None:
                q_init = state.q + state.dq                         # linear
            else:
                q_init = state.q + 2.0 * state.dq - state.dq_prev   # quadratic
        q, p, fp_it, residual = _implicit_midpoint_step(
            state.q, state.p, step_size, self.evaluate_model, self._solver,
            metric=state.metric, q_init=q_init)
        it = fp_it.to(step_size.dtype)
        self._step_residual = torch.maximum(self._step_residual, residual)
        self._step_iters    = torch.maximum(self._step_iters, it)
        self._step_iters_total = self._step_iters_total + it
        with torch.no_grad():
            U, metric = self.evaluate_model(q)
        return RMHMCState(q, p, U=U, metric=metric,
                          dq=q - state.q, dq_prev=state.dq)

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) - H(old)``, forced to +inf where the trajectory's
        fixed-point solve did not converge (max residual over ``fp_tol``): a
        non-converged step is not a valid proposal and must be rejected even if
        its energy change is small. Evaluates the endpoint potential/metric
        where :meth:`integrate` has not already."""
        if new.U is None or new.metric is None:
            with torch.no_grad():
                new.U, new.metric = self.evaluate_model(new.q)
        H_new = _hamiltonian(new.q, new.p, new.U.value, new.metric)   # (N,)
        H_old = _hamiltonian(old.q, old.p, old.U.value, old.metric)   # (N,)
        delta = H_new - H_old
        # Fold this transition's worst substep into the run-level summaries.
        self._residual_sum = self._residual_sum + self._step_residual
        self._residual_max = torch.maximum(self._residual_max, self._step_residual)
        self._fp_iters_sum = self._fp_iters_sum + self._step_iters
        self._fp_iters_max = torch.maximum(self._fp_iters_max, self._step_iters)
        self._fp_iters_total = self._fp_iters_total + self._step_iters_total
        solve_failed = self._step_residual > self._fp_tol
        return torch.where(solve_failed, delta.new_full((), float("inf")), delta)

    def adapt(self, accept_prob, delta_H):
        """Derivative-free (REINFORCE) step-size adaptation from this transition's
        energy error ``delta_H`` and worst solver residual / iteration count."""
        # Cost f_t = -log(efficiency), lower = better step size. Efficiency is
        # accepted travel per unit of solver work: step_size (distance) over
        # num_iters (cost), times exp(-|dH|) (the acceptance probability), times
        # exp(-residual/step_size) (the analogous acceptance term for the solver
        # error). The additive floor, normalised by |log(floor)|, bounds the cost
        # so rare failures carry large but finite weight.
        floor = 1.e-3
        num_iters       = self._step_iters
        solver_penalty  = torch.exp(-self._step_residual / self.step_size)
        delta_H_penalty = torch.exp(-delta_H.abs())
        f_t = (-0.5 * torch.log(
                    solver_penalty * delta_H_penalty * self.step_size / num_iters + floor
               ) / abs(math.log(floor)))                                    # (N,)
        self._step_size_adapter.update(f_t)

    def reset_extra_diagnostics(self):
        """Zero the run-level solver summaries. The per-transition scratch
        (``_step_residual`` / ``_step_iters``) is reset each transition in
        :meth:`sample_momentum`, so it is left alone here."""
        N = self.step_size.shape[0]
        zeros = torch.zeros(N, dtype=self.step_size.dtype, device=self.step_size.device)
        self._residual_sum = zeros.clone()
        self._residual_max = zeros.clone()
        self._fp_iters_sum = zeros.clone()
        self._fp_iters_max = zeros.clone()
        self._fp_iters_total = zeros.clone()

