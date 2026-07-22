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
# internal history it keeps is scoped to a single ``_implicit_midpoint_step``.
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
    """Relaxed Picard iteration: z_{k+1} = z_k − β r_k = (1−β) z_k + β F(z_k).

    Stateless.

    Parameters
    ----------
    beta : float
        Under-relaxation factor in (0, 1]. Default 1.0 (undamped).
    """

    def __init__(self, beta=1.0):
        self.beta = float(beta)

    def new(self, d):
        """Fresh per-solve updater for ``d``-dim positions."""
        return self

    def propose(self, z, r):
        return z - self.beta * r


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
    """

    # Relative / absolute Tikhonov floors for the m×m normal-equation solve.
    reg_rel = 1e-10
    reg_abs = 1e-14

    def __init__(self, history=None, beta=1.0):
        self.history = history      # int, or None to resolve to dim(q) in new()
        self.beta = float(beta)
        self._Z = []   # committed iterates z_k        (each (N, 2d))
        self._F = []   # Anderson residuals f_k = −r_k (each (N, 2d))

    def new(self, d):
        """Fresh per-solve updater for ``d``-dim positions, resolving a None
        ``history`` to ``d``."""
        return _AndersonUpdate(d if self.history is None else self.history, self.beta)

    def propose(self, z, r):
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

        A  = dF.transpose(-2, -1) @ dF                    # (N, mk, mk)
        b  = dF.transpose(-2, -1) @ f_k.unsqueeze(-1)     # (N, mk, 1)
        mk = A.shape[-1]
        # Scale-aware Tikhonov floor for (near-)collinear or zero ΔF columns.
        scale = A.diagonal(dim1=-2, dim2=-1).mean(-1)     # (N,)
        reg   = (self.reg_rel * scale + self.reg_abs).view(-1, 1, 1)
        A = A + reg * torch.eye(mk, dtype=A.dtype, device=A.device)
        gamma = torch.linalg.solve(A, b)                  # (N, mk, 1)

        z_next = z + self.beta * f_k - ((dZ + self.beta * dF) @ gamma).squeeze(-1)
        return z_next


# ---- Implicit midpoint step --------------------------------------------- #

def _implicit_midpoint_step(q, p, eps, evaluate_model, max_iter, tol, solver=None):
    """
    One step of I.M.(a): solve for the endpoint (q', p') via a per-chain
    fixed-point iteration.

    Batched over a leading chain axis (q, p have shape (N, d)). Each chain
    runs its own solve and finishes when it converges (max-norm residual <
    tol) or blows up (residual > 10x its initial value). Finished chains are
    frozen via a mask while live chains keep iterating, until all chains
    finish or max_iter is reached.

    Parameters
    ----------
    q, p : (N, d)
        Start-of-step position and momentum.
    eps : (N,)
        Per-chain step size.
    evaluate_model : Callable
        Maps q to (potential, metric).
    max_iter : int
        Maximum iterations per chain.
    tol : float
        Convergence tolerance (max norm).
    solver : _PicardUpdate or _AndersonUpdate or None
        Configured update rule. None defaults to undamped Picard.

    Returns
    -------
    q_out, p_out : (N, d)
    iters    : (N,) long   per-chain iteration count (max_iter if blown up
               or never converged, else the iteration at which tol was met)
    residual : (N,)        per-chain final max-norm residual
    """
    d = q.shape[-1]
    N = q.shape[0]

    def residual_fn(z):
        F_q, F_p = _midpoint_map(q, p, z[..., :d], z[..., d:], eps, evaluate_model)
        return z - torch.cat([F_q, F_p], dim=-1)

    updater = (solver if solver is not None else _PicardUpdate()).new(d)

    z_k = torch.cat([q, p], dim=-1)            # (N, 2d)
    r_k = residual_fn(z_k)
    r_init_norm = r_k.abs().amax(-1)           # (N,)

    done     = torch.zeros(N, dtype=torch.bool, device=q.device)
    iters    = torch.full((N,), max_iter, dtype=torch.long, device=q.device)
    residual = r_init_norm.clone()             # (N,)

    for i in range(1, max_iter + 1):
        z_next = updater.propose(z_k, r_k)     # Picard or Anderson proposal
        r_next = residual_fn(z_next)
        r_next_norm = r_next.abs().amax(-1)    # (N,)

        # Freeze finished chains: discard their update, keep last state.
        keep = done[..., None]
        z_k = torch.where(keep, z_k, z_next)
        r_k = torch.where(keep, r_k, r_next)
        residual = torch.where(done, residual, r_next_norm)

        live = ~done
        # Blow-up: finish at max_iter semantics (iters already max_iter).
        blew = live & (r_next_norm > 10.0 * r_init_norm)
        done = done | blew

        live = ~done
        # Convergence: finish at this iteration i.
        conv = live & (residual < tol)
        iters = torch.where(conv, torch.full_like(iters, i), iters)
        done = done | conv

        if bool(done.all()):
            break

    return z_k[..., :d].detach(), z_k[..., d:].detach(), iters, residual.detach()


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

    def __init__(self, q, p=None, U=None, metric=None):
        self.q = q
        self.p = p
        self.U = U
        self.metric = metric

    def reorder(self, perm):
        """Permute the batch elements by ``perm`` (an ``(N,)`` long index
        tensor): slot ``i`` of the result carries the configuration from
        ``perm[i]``. Absent (None) fields stay None."""
        return RMHMCState(
            q      = self.q[perm],
            p      = None if self.p is None else self.p[perm],
            U      = None if self.U is None else self.U.reorder(perm),
            metric = None if self.metric is None else self.metric.reorder(perm),
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
        Integration step size (adapted during warmup when adapting).
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
        step_size: float = 0.1,
        num_steps: int = 10,
        adapt_step_size: bool = True,
        adaptation_sigma: float = 0.1,
        fp_max_iter: int = 100,
        fp_tol: float = 1e-8,
        solver: str = "picard",
        anderson_history: int = None,
        damping: float = 1.0,
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

        # The adapters work on the log step size; step_size = exp(adapter value).
        log_eps = math.log(step_size)
        if adapt_step_size:
            adapter = Reinforce(sigma=adaptation_sigma, init=log_eps)
        else:
            adapter = NoAdaptation(init=log_eps)
        super().__init__(model_fn, space, requires_metric=True, num_steps=num_steps,
                         adapter=adapter, divergence_threshold=divergence_threshold)

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
        (read by :meth:`acceptance_delta` and :meth:`adapt`)."""
        q, p, fp_it, residual = _implicit_midpoint_step(
            state.q, state.p, step_size, self.evaluate_model,
            self._fp_max_iter, self._fp_tol, self._solver)
        it = fp_it.to(step_size.dtype)
        self._step_residual = torch.maximum(self._step_residual, residual)
        self._step_iters    = torch.maximum(self._step_iters, it)
        return RMHMCState(q, p)

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

