from typing import Callable, Tuple

import torch
import math

from .HamiltonianSampler import HamiltonianSampler
from .spaces import TemperedMetric
from .adapters import Reinforce, NoAdaptation

# =========================================================================== #
#                                                                              #
#  RMHMC helpers  (implicit midpoint integrator)                               #
#                                                                              #
#  The implicit-midpoint step solves a per-chain fixed-point equation          #
#  z = F(z).  Algorithm I.M.(a) of Brofos & Lederman (2021): the unknown       #
#  is the endpoint (q_k, p_k), with the midpoint derived from it.  The         #
#  update rule that drives the solve is pluggable: Picard iteration            #
#  (z_{k+1} = F(z_k)) and Anderson acceleration both solve the same F, so      #
#  the endpoint is solver- and damping-independent. Only the proposal,         #
#  the iteration count, and stability differ.                                  #
#                                                                              #
#  Only the values F_q, F_p are needed (no Jacobian), so the sole              #
#  gradient is the first-order dH/dq at the midpoint.                          #
#                                                                              #
# =========================================================================== #

# ---- Hamiltonian --------------------------------------------------------- #

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


# ---- Midpoint map -------------------------------------------------------- #

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

    Parameters
    ----------
    q, p : torch.Tensor
        Start-of-step position and momentum.
    q_k, p_k : torch.Tensor
        Current endpoint iterate.
    eps : torch.Tensor
        Per-chain step size, shape (N,).
    evaluate_model : Callable
        Maps q to (potential, metric).
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


# ---- Fixed-point update rules ------------------------------------------- #
#
# An updater turns the current iterate/residual pair (z_k, r_k) into the next
# proposal z_{k+1}.  ``r_k`` is the fixed-point residual z_k − F(z_k), so
# ``F(z_k) = z_k − r_k`` and the Anderson residual (Walker & Ni's notation) is
# ``f_k = F(z_k) − z_k = −r_k``.  A fresh updater is built per solve, so any
# internal history it keeps is scoped to a single solve. An updater may carry a
# fixed preconditioner P, stepping along P r_k in place of r_k (P = G_M(η0)⁻¹ for
# the ChartRATTLE position solve, absent for the RMHMC midpoint solve).
#
# Relaxed Picard (β < 1) pulls the iteration eigenvalues (β − 1) + β λ toward
# (1 − β) on the real axis, taming the near-imaginary spectrum of the
# implicit-midpoint map and trading convergence speed for stability.
#
# Anderson(m) (Walker & Ni 2011, Type-II) stacks the last m iterate/residual
# differences and solves a small per-chain least squares.  On a linear map
# Anderson(m ≥ 1) reaches the fixed point in one accelerated step. On the true
# nonlinear map it typically converges in fewer iterations than Picard,
# trading extra model evals for a cheap m×m solve.  β enters only the final
# combination, not the γ least squares, whose conditioning is kept well-posed
# by a scale-aware Tikhonov floor.

class _PicardUpdate:
    """Relaxed Picard iteration: z_{k+1} = z_k − β P r_k = (1−β) z_k + β F(z_k)
    when P = I.

    Stateless.

    Parameters
    ----------
    beta : float
        Under-relaxation factor in (0, 1]. Default 1.0 (undamped).
    precond : callable or None
        Preconditioner P applied to the residual before the step. Default None
        (P = I).
    """

    def __init__(self, beta=1.0, precond=None):
        self.beta = float(beta)
        self._precond = precond

    def new(self, d, precond=None):
        """Fresh per-solve updater for ``d``-dim positions with preconditioner
        ``precond``."""
        return _PicardUpdate(self.beta, precond)

    def propose(self, z, r):
        if self._precond is not None:
            r = self._precond(r)
        return z - self.beta * r

    def damped(self, factor):
        """Copy with β scaled by ``factor`` (fallback ladder)."""
        return _PicardUpdate(self.beta * factor)


class _AndersonUpdate:
    """Anderson(m) acceleration (Type-II, damping β) of the fixed-point map.

    With f_k = F(z_k) − z_k = −r_k and the last ``m`` iterate/residual
    differences stacked column-wise as ΔZ, ΔF, solve the per-chain least
    squares γ = argmin ‖f_k − ΔF γ‖ and take

        z_{k+1} = z_k + β f_k − (ΔZ + β ΔF) γ.

    Parameters
    ----------
    history : int or None
        History length ``m`` (past differences retained), ≥ 1, or None to
        resolve to dim(q) when ``new`` is called.
    beta : float
        Under-relaxation factor in (0, 1]. Default 1.0 (undamped).
    precond : callable or None
        Preconditioner P applied to the residual before the step. Default None
        (P = I).
    """

    # Relative / absolute Tikhonov floors for the least-squares solve.
    reg_rel = 1e-10
    reg_abs = 1e-14

    def __init__(self, history=None, beta=1.0, precond=None):
        self.history = history      # int, or None to resolve to dim(q) in new()
        self.beta = float(beta)
        self._precond = precond
        self._Z = []   # committed iterates z_k        (each (N, 2d))
        self._F = []   # Anderson residuals f_k = −r_k (each (N, 2d))

    def new(self, d, precond=None):
        """Fresh per-solve updater for ``d``-dim positions with preconditioner
        ``precond``, resolving a None ``history`` to ``d``."""
        return _AndersonUpdate(
            d if self.history is None else self.history, self.beta, precond)

    def propose(self, z, r):
        if self._precond is not None:
            r = self._precond(r)
        self._Z.append(z)
        self._F.append(-r)
        if len(self._Z) > self.history + 1:    # keep at most `history` differences
            self._Z.pop(0)
            self._F.pop(0)

        f_k = self._F[-1]                       # (N, 2d)
        if len(self._Z) == 1:                   # no history yet: damped Picard step
            return z + self.beta * f_k

        dZ = torch.stack([self._Z[j] - self._Z[j - 1]
                          for j in range(1, len(self._Z))], dim=-1)   # (N, 2d, mk)
        dF = torch.stack([self._F[j] - self._F[j - 1]
                          for j in range(1, len(self._F))], dim=-1)   # (N, 2d, mk)

        N, _, mk = dF.shape
        # Scale-aware Tikhonov floor for (near-)collinear or zero ΔF columns.
        scale = (dF * dF).sum(-2).mean(-1)                # (N,) mean ‖ΔF_j‖²
        reg   = self.reg_rel * scale + self.reg_abs       # (N,)
        # Solve the damped least squares by QR on the stacked [ΔF; √reg·I], not
        # the normal equations ΔFᵀΔF whose squared condition number overflows
        # float64 at a stiff metric's column spread (collapsing Anderson).
        eye   = torch.eye(mk, dtype=dF.dtype, device=dF.device)
        A_aug = torch.cat([dF, reg.sqrt().view(-1, 1, 1) * eye], dim=-2)
        b_aug = torch.cat([f_k.unsqueeze(-1), f_k.new_zeros(N, mk, 1)], dim=-2)
        Q, R  = torch.linalg.qr(A_aug)
        gamma = torch.linalg.solve_triangular(
            R, Q.transpose(-2, -1) @ b_aug, upper=True)    # (N, mk, 1)

        z_next = z + self.beta * f_k - ((dZ + self.beta * dF) @ gamma).squeeze(-1)
        return z_next

    def damped(self, factor):
        """Copy with β scaled by ``factor``, same history (fallback ladder)."""
        return _AndersonUpdate(self.history, self.beta * factor)


# ---- Implicit midpoint step --------------------------------------------- #

def _fixed_point_solve(residual_fn, z_init, updater, max_iter, tol):
    """Find ``z`` with ``residual_fn(z) = 0``, batched over chains, by
    fixed-point iteration from ``z_init``. Each step the ``updater`` reads the
    current iterate and its residual and returns the next iterate (relaxed
    Picard or Anderson acceleration, optionally preconditioned). A chain stops
    once its residual max-norm drops below ``tol``, or when it diverges. Returns
    the solution, the per-chain iteration count, and the final residual max-norm.

    ``residual_fn(z)`` returns the residual, one row per chain, detached from any
    autograd graph."""
    N = z_init.shape[0]
    z = z_init
    residual = residual_fn(z)
    r0 = residual.abs().amax(-1)

    done      = torch.zeros(N, dtype=torch.bool, device=z.device)
    iters     = torch.full((N,), max_iter, dtype=torch.long, device=z.device)
    residual_norm = r0.clone()

    for i in range(1, max_iter + 1):
        z_next = updater.propose(z, residual)
        residual_next = residual_fn(z_next)
        r = residual_next.abs().amax(-1)

        keep = done[..., None]
        z = torch.where(keep, z, z_next)
        residual = torch.where(keep, residual, residual_next)
        residual_norm = torch.where(done, residual_norm, r)

        # Divergence: non-finite, or grown far beyond the start (the tol
        # conjunct spares a warm start whose residual began sub-tol).
        diverged = ~done & (~torch.isfinite(r) | ((r > 1000.0 * r0) & (r > tol)))
        done = done | diverged

        converged = ~done & (residual_norm < tol)
        iters = torch.where(converged, torch.full_like(iters, i), iters)
        done = done | converged

        if bool(done.all()):
            break

    return z.detach(), iters, residual_norm.detach()


def _implicit_midpoint_step(q, p, eps, evaluate_model, max_iter, tol,
                            solver=None, fallback=(), z_init=None):
    """Solve RMHMC's implicit-midpoint fixed-point equation z = F(z) for the
    step endpoint z = (q_out, p_out), starting from ``(q, p)`` (F is the midpoint
    map in :func:`_midpoint_map`). Returns the endpoint with the per-chain
    iteration count and final residual.

    ``fallback`` is a sequence of ``(damping_factor, max_iter)``: chains that did
    not converge are re-solved with progressively stronger under-relaxation.
    Damping does not move the fixed point, so this resolves a stuck chain rather
    than rejecting it (a solver-driven rejection would break detailed balance).
    ``z_init`` warm-starts the first solve; the fallback re-solves from ``(q, p)``."""
    d = q.shape[-1]
    base = solver if solver is not None else _PicardUpdate()

    def residual_fn(z):
        F_q, F_p = _midpoint_map(q, p, z[..., :d], z[..., d:], eps, evaluate_model)
        return z - torch.cat([F_q, F_p], dim=-1)

    trivial = torch.cat([q, p], dim=-1)
    z, iters, residual = _fixed_point_solve(
        residual_fn, trivial if z_init is None else z_init, base.new(d), max_iter, tol)
    q_out, p_out = z[..., :d], z[..., d:]

    for factor, fb_iter in fallback:
        bad = residual > tol
        if not bool(bad.any()):
            break
        zb, it_n, r_n = _fixed_point_solve(
            residual_fn, trivial, base.damped(factor).new(d), fb_iter, tol)
        commit   = bad.unsqueeze(-1)
        q_out    = torch.where(commit, zb[..., :d], q_out)
        p_out    = torch.where(commit, zb[..., d:], p_out)
        iters    = torch.where(bad, iters + it_n, iters)   # honest cost, bad chains only
        residual = torch.where(bad, r_n, residual)

    return q_out, p_out, iters, residual


# =========================================================================== #
#                                                                              #
#  Chain state                                                                 #
#                                                                              #
#  ``U`` and ``metric`` are configuration-bound objects that carry their       #
#  own temperature and retemper themselves under ``reorder``, so the           #
#  state stays agnostic to tempering.  ``max_residual`` and ``fp_iters``       #
#  are integrator diagnostics bound to the slot. The trajectory                #
#  accumulators are reset by ``init`` and ``accept`` and carried forward       #
#  by ``step``.                                                                #
#                                                                              #
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
        ``init`` / ``accept``, ``None`` after ``step``.
    metric : TemperedMetric or None
        Metric at ``q``. Set at ``init`` / ``accept``, ``None`` after ``step``.
    """

    def __init__(self, q, p=None, U=None, metric=None, dz=None, dz_prev=None):
        self.q = q
        self.p = p
        self.U = U
        self.metric = metric
        # Last two converged endpoint displacements (N, 2d), used to warm-start
        # the next substep's solve by quadratic extrapolation; None at a
        # trajectory start (dropped by accept).
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


# =========================================================================== #
#                                                                              #
#  RMHMC sampler                                                               #
#                                                                              #
#  The transition machinery (init / step / accept / end_warmup / diagnostics)  #
#  is inherited from HamiltonianSampler; RMHMC supplies the integrator and      #
#  energy through the build_initial_state / sample_momentum / integrate /       #
#  acceptance_delta / adapt hooks. All chains run in one batched state.         #
#                                                                              #
#  model_fn is specified in constrained space. MCMCSampler adds the           #
#  prior log-prob and prior metric and pushes the metric forward to free       #
#  unconstrained coordinates (spaces.push_forward_metric).                     #
#                                                                              #
#  Both solvers return the same endpoint up to fp_tol. Anderson                #
#  typically reaches it in fewer iterations on stiff metrics, at the           #
#  cost of a small m x m solve per iteration.  damping (beta) affects          #
#  only stability and iteration count, not the endpoint.                       #
#                                                                              #
# =========================================================================== #

class RMHMC(HamiltonianSampler):
    """
    Riemannian Manifold HMC with the implicit-midpoint integrator, sampling
    q ~ exp(−U(q)) under the position-dependent metric G(q) with Hamiltonian
    H(q, p) = U(q) + ½ pᵀ G⁻¹(q) p + ½ log det G(q).

    Runs in unconstrained space. The model is specified in constrained space
    and pulled back by :meth:`MCMCSampler.evaluate_model`.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(theta_full) -> (U_lik, G_lik)``: full constrained vector
        to scalar likelihood potential ``-log p(data | theta)`` and
        (d_full, d_full) SPD likelihood metric in constrained coordinates.
    space
        Parameter space object (priors, transform, free/fixed split).
    step_size : float
        Integration step size. Adapted during warmup when adapting.
    num_steps : int
        Number of leapfrog substeps per transition.
    adapt_step_size : bool
        Adapt the step size during warmup via the REINFORCE adapter.
        Default True.
    adaptation_sigma : float
        Perturbation scale of the REINFORCE adapter. Default 0.1.
    fp_max_iter : int
        Maximum fixed-point iterations per leapfrog substep. Default 100.
    fp_tol : float
        Convergence tolerance for fixed-point iteration (max norm).
    solver : str
        Fixed-point solver: ``"picard"`` (default) or ``"anderson"``.
    anderson_history : int or None
        History length ``m`` for the Anderson solver (ignored for Picard).
        ``None`` (default) resolves per-solve to ``dim(q)``. Must be ≥ 1 if
        given.
    damping : float
        Under-relaxation factor β ∈ (0, 1] shared by both solvers.
        Default 1.0 (undamped).
    fallback_damping : tuple of float
        Fallback ladder: on non-convergence, re-solve the failed chains with the
        base solver damped by each factor in turn (each in (0, 1), relative to
        ``damping``; default ``(0.5, 0.25)``). Endpoint-preserving, so it removes
        solver-driven rejections. ``()`` disables it.
    fallback_iter_scale : int
        Per-level iteration cap as a multiple of ``fp_max_iter``. Default 4.
    divergence_threshold : float
        Raw |delta_H| above which (or non-finite values for which) the step
        is recorded as a divergence. Default 100.

    Notes
    -----
    Unlike :class:`HMC` / :class:`LMC`, RMHMC exposes no ``target_accept_prob``.
    The implicit-midpoint integrator can conserve energy over a wide range of
    step sizes -- exactly, up to the fixed-point tolerance, on a Gaussian target
    -- so acceptance is a poor thing to adapt against: whenever the solve
    converges it saturates near 1 and carries almost no gradient on the step
    size. The true knob is integrator accuracy, so the REINFORCE adapter instead
    targets solver cost and energy error (residual and iteration count per
    substep together with |delta_H|; see :meth:`adapt`). This keeps acceptance
    close to 1 while steering the step size by how well the trajectory is
    actually resolved; ``adaptation_sigma`` sets the exploration scale of that
    search.
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
        # Resolve the string choice into a configured solver.
        if not 0.0 < damping <= 1.0:
            raise ValueError(f"damping must be in (0, 1], got {damping}")
        if solver == "picard":
            self._solver = _PicardUpdate(damping)
        elif solver == "anderson":
            if anderson_history is not None and anderson_history < 1:
                raise ValueError(
                    f"anderson_history must be >= 1, got {anderson_history}")
            self._solver = _AndersonUpdate(anderson_history, damping)
        else:
            raise ValueError(
                f"unknown solver {solver!r}; expected 'picard' or 'anderson'")

        # Fallback ladder: re-solve non-converged chains with stronger damping.
        if any(not 0.0 < f < 1.0 for f in fallback_damping):
            raise ValueError(
                f"fallback_damping factors must be in (0, 1), got {fallback_damping}")
        self._fallback = [(f, fallback_iter_scale * fp_max_iter)
                          for f in fallback_damping]

        # The adapters work on the log step size; step_size = exp(adapter value).
        log_eps = math.log(step_size)
        if adapt_step_size:
            adapter = Reinforce(sigma=adaptation_sigma, init=log_eps)
        else:
            adapter = NoAdaptation(init=log_eps)
        super().__init__(model_fn, space, requires_metric=True, num_steps=num_steps,
                         adapter=adapter, divergence_threshold=divergence_threshold,
                         trajectory_length=num_steps * step_size,
                         step_normalization=step_normalization)

        self._fp_max_iter = fp_max_iter
        self._fp_tol      = fp_tol

        # Solver diagnostics. Each transition contributes its worst substep;
        # the means are then over transitions.
        self.register_diagnostic("residual_mean", lambda: self._residual_sum / max(self._step, 1))
        self.register_diagnostic("residual_max",  lambda: self._residual_max)
        self.register_diagnostic("fp_iters_mean", lambda: self._fp_iters_sum / max(self._step, 1))
        self.register_diagnostic("fp_iters_max",  lambda: self._fp_iters_max)
        self.register_logging("|r|", lambda: "{:.2e}".format(float(self._step_residual.max())))

    def build_initial_state(self, q):
        """Evaluate the model at ``q`` and return the initial :class:`RMHMCState`
        (momentum drawn later by :meth:`sample_momentum`). Seeds the
        per-transition solver scratch so the sampler is usable right after init."""
        z = torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
        self._step_residual = z.clone()
        self._step_iters    = z.clone()
        with torch.no_grad():
            U, metric = self.evaluate_model(q)
        return RMHMCState(q, U=U, metric=metric)

    def sample_momentum(self, state):
        """Draw the momentum ``p ~ N(0, G(q))`` on ``state`` and reset the
        per-transition solver scratch (worst residual / iteration count over the
        transition's substeps), read by :meth:`acceptance_delta` and :meth:`adapt`."""
        N = state.q.shape[0]
        z = torch.zeros(N, dtype=state.q.dtype, device=state.q.device)
        self._step_residual = z.clone()
        self._step_iters    = z.clone()
        state.p = state.metric.sample_momentum()
        return state

    def integrate(self, state, step_size):
        """One implicit-midpoint substep at ``step_size``, tracking the worst
        fixed-point residual and iteration count over the transition's substeps
        (read by :meth:`acceptance_delta` and :meth:`adapt`).

        The solve is warm-started by extrapolating the endpoint from the last
        two converged displacements (quadratic; linear on the second substep,
        trivial on the first). The guess only changes the iteration count, not
        the fixed point, so the map -- and detailed balance -- are unchanged.
        The displacements reset per trajectory (``accept`` builds a fresh state
        without them)."""
        z_start = torch.cat([state.q, state.p], dim=-1)
        if state.dz is None:
            z_init = None                                       # first substep
        elif state.dz_prev is None:
            z_init = z_start + state.dz                         # linear
        else:
            z_init = z_start + 2.0 * state.dz - state.dz_prev   # quadratic
        q, p, fp_it, residual = _implicit_midpoint_step(
            state.q, state.p, step_size, self.evaluate_model,
            self._fp_max_iter, self._fp_tol, self._solver, self._fallback,
            z_init=z_init)
        it = fp_it.to(step_size.dtype)
        self._step_residual = torch.maximum(self._step_residual, residual)
        self._step_iters    = torch.maximum(self._step_iters, it)
        dz = torch.cat([q, p], dim=-1) - z_start
        return RMHMCState(q, p, dz=dz, dz_prev=state.dz)

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) - H(old)``, forced to +inf where the trajectory's
        fixed-point solve did not converge (max residual over ``fp_tol``): a
        non-converged step is not a valid proposal and must be rejected even if
        its energy change is small. Evaluates the endpoint potential/metric."""
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
        """Derivative-free (REINFORCE) step-size adaptation from this transition's
        energy error ``delta_H`` and worst solver residual / iteration count."""
        # Cost f_t = -log(efficiency), lower = better step size. The efficiency is
        # accepted travel per solver iteration -- exp(-|dH|) ~ accept prob times
        # step_size ~ distance over num_iters ~ solver cost -- weighted by
        # exp(-residual/step_size), an analogous acceptance term for the solver
        # error. The eta floor (normalised by |log eta|) gives rare failures
        # large weight.
        eta = 1.e-3
        num_iters       = self._step_iters
        solver_penalty  = torch.exp(-self._step_residual / self.step_size)
        delta_H_penalty = torch.exp(-delta_H.abs())
        f_t = (-0.5 * torch.log(
                    solver_penalty * delta_H_penalty * self.step_size / num_iters + eta
               ) / abs(math.log(eta)))                                      # (N,)
        self._step_size_adapter.update(f_t)

    def reset_extra_diagnostics(self):
        """Zero the run-level solver summaries. The per-transition scratch
        (``_step_residual`` / ``_step_iters``) is reset each transition in
        :meth:`sample_momentum`, so it is left alone here."""
        N = self.step_size.shape[0]
        z = torch.zeros(N, dtype=self.step_size.dtype, device=self.step_size.device)
        self._residual_sum = z.clone()
        self._residual_max = z.clone()
        self._fp_iters_sum = z.clone()
        self._fp_iters_max = z.clone()

