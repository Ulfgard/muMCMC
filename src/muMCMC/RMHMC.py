from typing import Callable, Tuple

import torch
import math

from .HamiltonianSampler import HamiltonianSampler
from .spaces import TemperedMetric
from ._adapters import Reinforce, NoAdaptation
from ._solvers import FixedPointSolver

# =========================================================================== #
#  Why the substep is one root find in 2d                                     #
#                                                                             #
#  The unknown of a substep is the whole endpoint (q_k, p_k), with the        #
#  midpoint derived from it, rather than the midpoint with the endpoint       #
#  derived from that. One root find in 2d replaces a staggered pair of        #
#  solves, and only the values F_q and F_p enter it, so the only derivative   #
#  taken anywhere in the substep is dH/dq at the midpoint.                    #
#                                                                             #
#  That is also what rules Newton out. Its update needs DF, which carries     #
#  the second derivative of H and so of the metric, and the model interface   #
#  exposes only the metric itself.                                            #
#                                                                             #
#  Picard and Anderson drive the same F, so the endpoint is the same for      #
#  either and for any damping. Only the iteration count and the stability of  #
#  getting there differ.                                                      #
# =========================================================================== #

#  ---- Hamiltonian --------------------------------------------------------- #

def _hamiltonian(
    q: torch.Tensor,
    p: torch.Tensor,
    U: torch.Tensor,
    metric: TemperedMetric,
) -> torch.Tensor:
    """
    H(q, p) = U + ½ pᵀ G⁻¹(q) p + ½ log det G(q), of shape ``(N,)``.

    Parameters
    ----------
    q : Tensor, shape (N, d)
        Position. Unused, present so the call reads as H(q, p).
    p : Tensor, shape (N, d)
        Momentum.
    U : Tensor, shape (N,)
        Potential, already evaluated at ``q``.
    metric : TemperedMetric
        Metric ``G``, already evaluated at ``q``.
    """
    Ginv_p = metric.inv_metric_times_vec(p)
    kinetic = 0.5 * (p * Ginv_p).sum(-1)
    return U + kinetic + 0.5 * metric.log_det_metric()


#  ---- Midpoint map -------------------------------------------------------- #

def _midpoint_map(
    q: torch.Tensor,
    p: torch.Tensor,
    q_k: torch.Tensor,
    p_k: torch.Tensor,
    eps,
    evaluate_model: Callable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fixed-point map F(z_k) = (F_q, F_p):

        q_mid = ½(q + q_k)
        p_mid = ½(p + p_k)
        F_q   = q + (ε/2) G⁻¹(q_mid) (p + p_k)
        F_p   = p − ε ∂H/∂q|_{q_mid, p_mid}

    both of shape ``(N, d)``.

    Parameters
    ----------
    q, p : Tensor, shape (N, d)
        Position and momentum at the start of the substep.
    q_k, p_k : Tensor, shape (N, d)
        Current iterate for the endpoint.
    eps : Tensor, shape (N,)
        Per-chain step size.
    evaluate_model : Callable
        Maps ``q`` to ``(potential, metric)``.
    """
    q_mid = (0.5 * (q + q_k)).detach().requires_grad_(True)   # fresh leaf
    p_mid = 0.5 * (p + p_k)

    with torch.enable_grad():
        potential, metric = evaluate_model(q_mid)
        H = _hamiltonian(q_mid, p_mid, potential.value, metric)
        # H has shape (N,) with no cross-chain coupling, so grad of the sum
        # is the per-chain gradient.
        (dHdq,) = torch.autograd.grad(H.sum(), q_mid)

    # eps is the per-chain step size (N,). Trailing axis broadcasts against
    # the (N, d) updates.
    e = eps.unsqueeze(-1)
    with torch.no_grad():
        F_q = q + (e / 2.0) * metric.inv_metric_times_vec(p + p_k)
        F_p = p - e * dHdq
    return F_q, F_p


#  ---- Implicit midpoint step ---------------------------------------------  #

def _implicit_midpoint_step(q, p, eps, evaluate_model, solver, z_init=None):
    """Solve RMHMC's implicit-midpoint equation z = F(z) for the step endpoint
    z = (q_out, p_out), with F the midpoint map in :func:`_midpoint_map`.

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
    z_init : (N, 2d) or None
        Starting iterate. None starts from ``(q, p)``, which is also where the
        fallback ladder restarts.

    Returns
    -------
    tuple
        ``(q_out, p_out, iters, residual)``, the endpoint of shape ``(N, d)``
        each and the solve's iteration count and final residual of shape
        ``(N,)`` each.
    """
    d = q.shape[-1]

    def residual_fn(z):
        F_q, F_p = _midpoint_map(q, p, z[..., :d], z[..., d:], eps, evaluate_model)
        return z - torch.cat([F_q, F_p], dim=-1)

    trivial = torch.cat([q, p], dim=-1)
    z, iters, residual = solver.solve(
        residual_fn, trivial if z_init is None else z_init, cold_start=trivial)
    return z[..., :d], z[..., d:], iters, residual


# =========================================================================== #
#  What the state holds and what the sampler holds                            #
#                                                                             #
#  Every field on the state is a property of the configuration, so a swap     #
#  that relabels a configuration to another slot permutes all of them. U and  #
#  metric are kept as tempered objects, which recombine at the new slot's     #
#  beta, and that is the whole of what the state encodes about tempering.     #
#                                                                             #
#  The solver's residual and iteration count are a property of the substep    #
#  taken at a slot and not of the configuration that was there for it, so     #
#  they are held on the sampler and a swap does not permute them.             #
# =========================================================================== #

class RMHMCState:
    """
    Working state of one RMHMC trajectory, batched over ``N`` chains. Every
    field is a property of the configuration and not of the batch position it
    occupies, so ``reorder`` permutes all of them.

    Attributes
    ----------
    q, p : Tensor, shape (N, d)
        Position and momentum.
    U : TemperedAffine or None
        Potential at ``q``. Set by :meth:`RMHMC.build_initial_state` and by
        :meth:`RMHMC.acceptance_delta`, None on a state the integrator built.
    metric : TemperedMetric or None
        Metric at ``q``, set and unset with ``U``.
    dz, dz_prev : Tensor, shape (N, 2d), or None
        The last two converged endpoint displacements, from which
        :meth:`RMHMC.integrate` extrapolates the next substep's starting
        iterate. Both None at the start of a trajectory.
    """

    def __init__(self, q, p=None, U=None, metric=None, dz=None, dz_prev=None):
        self.q = q
        self.p = p
        self.U = U
        self.metric = metric
        self.dz = dz
        self.dz_prev = dz_prev

    def reorder(self, perm):
        """Permute the batch elements by ``perm`` (an ``(N,)`` long index
        tensor): slot ``i`` of the result carries the configuration from
        ``perm[i]``. Absent (None) fields stay None."""
        return RMHMCState(
            q       = self.q[perm],
            p       = None if self.p is None else self.p[perm],
            U       = None if self.U is None else self.U.reorder(perm),
            metric  = None if self.metric is None else self.metric.reorder(perm),
            dz      = None if self.dz is None else self.dz[perm],
            dz_prev = None if self.dz_prev is None else self.dz_prev[perm],
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


class RMHMC(HamiltonianSampler):
    """
    Riemannian Manifold HMC with the implicit-midpoint integrator, sampling
    ``q`` from ``exp(-U(q))/Z`` under the position-dependent metric ``G(q)``
    with Hamiltonian

        H(q, p) = U(q) + ½ pᵀ G⁻¹(q) p + ½ log det G(q).

    The chain runs on the free variables, so ``q`` is ``theta`` and nothing is
    transformed. Both ``U`` and ``G`` come from
    :meth:`~muMCMC.MCMCSampler.MCMCSampler.evaluate_model`, which adds the
    prior's potential and the metric the prior induces.

    Each substep is a fixed-point solve, so a proposal can fail to be
    well-defined. :meth:`acceptance_delta` rejects a trajectory whose worst
    residual is above ``fp_tol`` and counts it as a divergence.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(theta_full) -> (U_lik, G_lik)``, the likelihood potential
        and an SPD metric of shape ``(d_full, d_full)``, both on the full
        variable vector.
    space : Space
        Parameter space, giving the prior's normal chart and the free/fixed
        split.
    step_size : float
        Integration step size at the start of warmup, and for the whole run
        when ``adapt_step_size`` is False.
    num_steps : int
        Implicit-midpoint substeps per transition.
    adapt_step_size : bool
        Adapt the step size during warmup with the REINFORCE adapter, on the
        cost of :meth:`adapt` rather than on acceptance. See the notes below.
    adaptation_sigma : float
        Scale of the perturbation the REINFORCE adapter explores with. Unused
        when ``adapt_step_size`` is False.
    fp_max_iter : int
        Iteration cap for one fixed-point solve.
    fp_tol : float
        Residual in max norm below which a solve has converged.
    solver : str
        Fixed-point solver, ``"picard"`` or ``"anderson"``. Both drive the same
        map, so the endpoint is the same either way.
    anderson_history : int or None
        History length ``m`` for the Anderson solver, at least 1, and unused by
        Picard. None resolves per solve to the dimension of the solve, which is
        ``2 dim(q)`` because the unknown is the endpoint ``(q, p)``.
    damping : float
        Under-relaxation factor in ``(0, 1]`` for either solver. 1.0 is
        undamped.
    fallback_damping : tuple of float
        Factors in ``(0, 1)``, relative to ``damping``, to re-solve a chain
        with in turn when it did not converge. The fixed point is the same at
        every level, so this changes only whether the solve gets there, and
        ``()`` disables it.
    fallback_iter_scale : int
        Iteration cap at each fallback level, as a multiple of ``fp_max_iter``.
    step_normalization : {None, "fixed", "max"}
        Hold the trajectory length ``num_steps * step_size`` at the value the
        constructor arguments give it while the step size adapts. See
        :class:`~muMCMC.HamiltonianSampler.HamiltonianSampler`. None is off.
    divergence_threshold : float
        A transition counts as a divergence when ``|delta_H|`` exceeds this or
        is not finite, the latter including every non-converged solve.

    Raises
    ------
    ValueError
        If ``solver`` names a rule needing the Jacobian of the midpoint
        residual, which carries second derivatives of the metric and is not
        available from the model interface.

    References
    ----------
    J. Brofos and R. R. Lederman, Evaluating the Implicit Midpoint Integrator
    for Riemannian Manifold Hamiltonian Monte Carlo, ICML 2021,
    Algorithm I.M.(a).

    Notes
    -----
    The implicit midpoint integrator conserves energy over a wide range of step
    sizes, and on a Gaussian target it does so up to the fixed-point tolerance.
    Acceptance is therefore a poor quantity to adapt against here. Whenever the
    solve converges it sits near 1 and barely responds to the step size, so
    dual averaging on it has almost nothing to work with.

    What does respond is the accuracy of the integration, so the REINFORCE
    adapter is driven by the solver's cost and the energy error instead, that
    is by the residual and iteration count per substep together with
    ``|delta_H|``, combined into the cost given in :meth:`adapt`. Acceptance
    stays near 1 and the step size follows how well the trajectory is resolved.
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
                f"available cheaply here. Use 'picard' or 'anderson'.")

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

        # Each transition contributes its worst substep, so a mean here is a
        # mean over transitions of a max over substeps.
        self.register_diagnostic("residual_mean", lambda: self._residual_sum / max(self._step, 1))
        self.register_diagnostic("residual_max",  lambda: self._residual_max)
        self.register_diagnostic("fp_iters_mean", lambda: self._fp_iters_sum / max(self._step, 1))
        self.register_diagnostic("fp_iters_max",  lambda: self._fp_iters_max)
        self.register_logging("|r|", lambda: "{:.2e}".format(float(self._step_residual.max())))

    def build_initial_state(self, q):
        """The initial :class:`RMHMCState` at ``q``, with the potential and the
        metric evaluated. The momentum needs the metric, so it is drawn by
        :meth:`sample_momentum` rather than here. The per-transition solver
        statistics are zeroed too, so :meth:`logging` reads them before any
        substep has run."""
        zeros = torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
        self._step_residual = zeros.clone()
        self._step_iters    = zeros.clone()
        with torch.no_grad():
            U, metric = self.evaluate_model(q)
        return RMHMCState(q, U=U, metric=metric)

    def sample_momentum(self, state):
        """Draw the momentum ``p ~ N(0, G(q))`` on ``state`` and zero the
        per-transition solver statistics, the worst residual and iteration count
        over the transition's substeps, which :meth:`acceptance_delta` and
        :meth:`adapt` read at the end of it."""
        N = state.q.shape[0]
        zeros = torch.zeros(N, dtype=state.q.dtype, device=state.q.device)
        self._step_residual = zeros.clone()
        self._step_iters    = zeros.clone()
        state.p = state.metric.sample_momentum()
        return state

    def integrate(self, state, step_size):
        """One implicit-midpoint substep at ``step_size``, carrying the worst
        fixed-point residual and iteration count of the transition so far.

        The solve starts from an endpoint extrapolated from the last two
        converged displacements, quadratically in general, linearly on the
        second substep and from ``(q, p)`` on the first. The starting iterate
        moves only the iteration count and not the fixed point, so the map and
        detailed balance are unaffected by it. A transition starts without the
        displacements, :meth:`RMHMCState.select_accepted` building a state that
        has none."""
        z_start = torch.cat([state.q, state.p], dim=-1)
        if state.dz is None:
            z_init = None                                       # first substep
        elif state.dz_prev is None:
            z_init = z_start + state.dz                         # linear
        else:
            z_init = z_start + 2.0 * state.dz - state.dz_prev   # quadratic
        q, p, fp_it, residual = _implicit_midpoint_step(
            state.q, state.p, step_size, self.evaluate_model, self._solver,
            z_init=z_init)
        it = fp_it.to(step_size.dtype)
        self._step_residual = torch.maximum(self._step_residual, residual)
        self._step_iters    = torch.maximum(self._step_iters, it)
        dz = torch.cat([q, p], dim=-1) - z_start
        return RMHMCState(q, p, dz=dz, dz_prev=state.dz)

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) - H(old)``, with no Jacobian correction because
        the implicit midpoint preserves volume. The endpoint potential and
        metric are evaluated here and left on ``new``.

        Where the trajectory's worst residual is above ``fp_tol`` the result is
        ``+inf``. Such a proposal is not the endpoint of the map the Metropolis
        test assumes, so it is rejected however small its energy change came
        out."""
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
        solve_failed = self._step_residual > self._fp_tol
        return torch.where(solve_failed, delta.new_full((), float("inf")), delta)

    def adapt(self, accept_prob, delta_H):
        """Update the step size by REINFORCE on a cost built from this
        transition's energy error ``delta_H`` and its worst solver residual and
        iteration count. ``accept_prob`` is unused, for the reason given in the
        class notes."""
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
        """Zero the run-level solver summaries. The per-transition residual and
        iteration count are zeroed by :meth:`sample_momentum` at the start of
        every transition, so they are left alone here."""
        N = self.step_size.shape[0]
        zeros = torch.zeros(N, dtype=self.step_size.dtype, device=self.step_size.device)
        self._residual_sum = zeros.clone()
        self._residual_max = zeros.clone()
        self._fp_iters_sum = zeros.clone()
        self._fp_iters_max = zeros.clone()

