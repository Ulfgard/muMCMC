from typing import Callable
import math
from contextlib import nullcontext

import torch

from .HamiltonianSampler import HamiltonianSampler
from .adapters import Reinforce, NoAdaptation
from .spaces import TemperedAffine, TemperedMetric
from .RMHMC import _PicardUpdate, _AndersonUpdate, _hamiltonian, _fixed_point_solve

# =========================================================================== #
#                                                                              #
#  ChartRATTLE: constrained HMC for hierarchical posteriors                    #
#                                                                              #
#  Latent z = (η, ε) ~ N(0, I_{n+m}) and a diffeomorphism φ_η with             #
#                                                                              #
#      φ_η(ε) = x,      θ = η,                                                  #
#                                                                              #
#  the non-centered reparameterization of x | η. Conditioning on the data x    #
#  is the constraint g(z) = φ_η(ε) − x = 0 defining the manifold               #
#                                                                              #
#      M = { z : φ_η(ε) = x }.                                                 #
#                                                                              #
#  The η-marginal of the constrained chain is p(θ | x). The co-area Jacobian   #
#  ½ log det Λ is owned here, not by the user.                                 #
#                                                                              #
# =========================================================================== #
#                                                                              #
#  Energy                                                                       #
#                                                                              #
#  With ψ(η) = φ_η⁻¹(x) and W = −∂ψ/∂η = B⁻¹A, the target is e^{−U} with        #
#                                                                              #
#      U(η) = ½‖η‖² + log|det B| + β·½‖ψ‖²,     G_M(η) = I + β Wᵀ W.           #
#                                                                              #
#  U is affine in β (lik = ½‖ψ‖², base = ½‖η‖² + log|det B|) and G_M is affine  #
#  in β (A_lik = Wᵀ W, A_prior = I), so evaluate_model returns them as a        #
#  TemperedAffine and a TemperedMetric. The Hamiltonian                         #
#                                                                              #
#      H = U + ½ πᵀ G_M⁻¹ π + ½ log det G_M                                    #
#                                                                              #
#  keeps e^{−U} invariant. β enters only through U and G_M, so evaluate_model    #
#  gives the tempered target at any temperature and a parallel-tempering swap    #
#  reads the temperature-free U_lik = ½‖ψ‖² off U.lik.                          #
#                                                                              #
# =========================================================================== #
#                                                                              #
#  Step  (η0, π0) -> (η1, π1)                                                  #
#                                                                              #
#      F(η1) = (η1 − η0) − β W0ᵀ(ψ(η1) − ψ0) − h π0 + (h²/2) ∇V(η0) = 0,       #
#      DF(η0) = I + β W0ᵀ W0 = G_M(η0),                                        #
#      η^{k+1} = η^k − G_M(η0)⁻¹ F(η^k),                                       #
#      π1 = (1/h)[(η1 − η0) − β W1ᵀ(ψ1 − ψ0)] − (h/2) ∇V(η1),                  #
#                                                                              #
#  V = U + ½ log det G_M, so the force ∇V is one autograd.grad(V.sum(), η).     #
#  Only ψ(η^k) is evaluated in the loop, preconditioned by one Cholesky of      #
#  G_M(η0) (the metric's own factor). The scheme is self-adjoint, so            #
#  reversible up to the solve. A failed solve is rejected with +inf energy and   #
#  no reverse-projection check is run.                                          #
#                                                                              #
# =========================================================================== #
#                                                                              #
#  Constraint interface (untempered)                                           #
#                                                                              #
#      psi(η)           -> ψ = φ_η⁻¹(x)         the inverse map                #
#      log_abs_det_B(η) -> log|det B|           = ½ log det Σ for a scale       #
#                                                                              #
#  W = −∂ψ/∂η follows from one reverse pass per latent (classic autograd, no   #
#  vmap); override psi_with_jvp with explicit Jacobians to skip it. The         #
#  constraint is temperature-free: β enters only in evaluate_model.            #
#                                                                              #
# =========================================================================== #
#                                                                              #
#  Tempering                                                                   #
#                                                                              #
#  β softens the observation in the scale family, Σ -> Σ/β, which scales the    #
#  data-fit ½‖ψ‖² by β. So β multiplies the likelihood wherever it appears: in  #
#  the potential U (β·½‖ψ‖²), in the metric G_M = I + β WᵀW, and hence in the    #
#  position equation F and its Jacobian above. The prior ½‖η‖² and the volume   #
#  ½ log det Σ stay untempered, so U is finite as β -> 0 while the scale family  #
#  itself needs β > 0. A parallel-tempering swap reads the β-free ½‖ψ‖² off      #
#  U.lik.                                                                       #
#                                                                              #
# =========================================================================== #


def _bcast_beta(beta, ndim):
    """β reshaped to broadcast over ``ndim`` trailing axes (scalar stays scalar)."""
    if torch.is_tensor(beta) and beta.ndim > 0:
        return beta.reshape((-1,) + (1,) * ndim)
    return beta


# ---- Constraint ---------------------------------------------------------- #

class ChartConstraint:
    """Constraint M = {(η, ε) : φ_η(ε) = x} exposed through the inverse map.

    A subclass supplies the batched, temperature-free inverse ``psi(η)`` =
    φ_η⁻¹(x) (shape ``(N, n)`` to ``(N, m)``) and ``log_abs_det_B(η)`` =
    log|det ∂_ε φ|. The sampler derives W = −∂ψ/∂η, the potential, and the
    metric, and applies the inverse temperature. Override ``psi_with_jvp`` to
    return (ψ, W, log|det B|) from explicit Jacobians.

    Parameters
    ----------
    x : (m,)
        Conditioning value the manifold is defined by, shared across chains.
    """

    def __init__(self, x: torch.Tensor):
        self.x = x

    def psi(self, eta: torch.Tensor) -> torch.Tensor:
        """The inverse map ψ(η) = φ_η⁻¹(x): the latent that maps to the
        observation x under the layer φ_η. ``eta`` is ``(N, n)``, the result
        ``(N, m)``. Implemented by a subclass."""
        raise NotImplementedError

    def log_abs_det_B(self, eta: torch.Tensor) -> torch.Tensor:
        """log|det ∂_ε φ_η|, the log Jacobian of the layer in its latent
        argument. ``eta`` is ``(N, n)``, the result ``(N,)``. Implemented by a
        subclass."""
        raise NotImplementedError

    def psi_with_jvp(self, eta: torch.Tensor):
        """(ψ, W, log|det B|) at η, W = −∂ψ/∂η by one reverse pass per latent.
        ``eta`` must require grad. Override with explicit Jacobians."""
        eps = self.psi(eta)
        m = eps.shape[-1]
        cols = [torch.autograd.grad(eps[:, i].sum(), eta, retain_graph=True,
                                    create_graph=True)[0]
                for i in range(m)]
        W = -torch.stack(cols, dim=1)                      # (N, m, n)
        return eps, W, self.log_abs_det_B(eta)


class LocationScaleChart(ChartConstraint):
    """Conditionally-Gaussian layer x | η ~ N(μ(η), Σ(η)).

    φ_η(ε) = μ(η) + L(η) ε with Σ(η) = L(η) L(η)ᵀ, so ψ(η) = L(η)⁻¹(x − μ(η))
    and log|det B| = ½ log det Σ. ``mean`` and ``cov`` are batched callables
    (η shape ``(N, n)`` to ``(N, m)`` and ``(N, m, m)`` SPD), differentiable in η.

    Parameters
    ----------
    mean : callable
        η -> μ(η).
    cov : callable
        η -> Σ(η), SPD.
    x : (m,)
        Observation.
    """

    def __init__(self, mean: Callable, cov: Callable, x: torch.Tensor):
        super().__init__(x)
        self.mean = mean
        self.cov = cov

    def _factor(self, eta):
        return self.mean(eta), torch.linalg.cholesky_ex(self.cov(eta)).L

    def psi(self, eta):
        mu, L = self._factor(eta)
        return torch.linalg.solve_triangular(
            L, (self.x - mu).unsqueeze(-1), upper=False).squeeze(-1)

    def log_abs_det_B(self, eta):
        _, L = self._factor(eta)
        return torch.log(L.diagonal(dim1=-2, dim2=-1).abs()).sum(-1)


# ---- Position solve ------------------------------------------------------ #

def _solve_rattle_step(constraint, eta0, psi0, W0, beta_col, chol_G0, rhs,
                       eta_init, solver, max_iter, tol):
    """Solve the RATTLE position equation F(η) = (η − η0) − β W0ᵀ(ψ(η) − ψ0)
    − rhs = 0 for η1, preconditioned by G_M(η0). Returns (η1, iters,
    residual)."""
    W0t = W0.transpose(-2, -1)                             # (N, n, m)

    def residual_fn(eta):
        with torch.no_grad():                              # solve is derivative-free
            psi = constraint.psi(eta)                      # (N, m), untempered
            corr = (W0t @ (psi - psi0).unsqueeze(-1)).squeeze(-1)
            return (eta - eta0) - beta_col * corr - rhs

    def precond(F):                                        # G_M(η0)⁻¹ F
        return torch.cholesky_solve(F.unsqueeze(-1), chol_G0).squeeze(-1)

    updater = solver.new(eta0.shape[-1], precond=precond)
    return _fixed_point_solve(residual_fn, eta_init, updater, max_iter, tol)


# ---- Chain state --------------------------------------------------------- #

class ChartRATTLEState:
    """Working state of one ChartRATTLE trajectory, batched over ``(N,)`` chains.

    A trajectory keeps the configuration ``q`` and its model (``U, metric, psi,
    W``); the momentum ``p``, force ``grad_V`` and warm-start ``deta`` are
    per-trajectory scratch, set at the start of a transition and dropped at its
    end. A parallel-tempering swap relabels ``q`` to a new temperature slot and
    the model is re-evaluated there.

    Attributes
    ----------
    q, p : (N, n)
        Chart position η and momentum π. ``q`` is read as the sample.
    U : TemperedAffine
        Potential; ``U.value`` is the ``(N,)`` energy, ``U.lik`` the swap statistic.
    metric : TemperedMetric
        Chart metric G_M(η).
    psi, W : (N, m), (N, m, n)
        ψ(η) and W = −∂ψ/∂η.
    grad_V : (N, n) or None
        Chart force ∇V(η).
    deta : (N, n) or None
        Last displacement, warm-starts the next solve.
    """

    def __init__(self, q, p=None, U=None, metric=None, psi=None, W=None,
                 grad_V=None, deta=None):
        self.q = q
        self.p = p
        self.U = U
        self.metric = metric
        self.psi = psi
        self.W = W
        self.grad_V = grad_V
        self.deta = deta

    def reorder(self, perm):
        """Relabel the configuration to a new temperature slot; its potential,
        metric, geometry and force are re-evaluated at the next step."""
        return ChartRATTLEState(self.q[perm],
                                None if self.p is None else self.p[perm])

    def select_accepted(self, accepted, other):
        """Per-chain choice between this endpoint (where ``accepted``) and the
        start ``other``. The force and warm-start displacement are trajectory
        scratch and are dropped."""
        pick = accepted.unsqueeze(-1)
        return ChartRATTLEState(
            torch.where(pick, self.q, other.q),
            torch.where(pick, self.p, other.p),
            self.U.select(accepted, other.U),
            self.metric.select(accepted, other.metric),
            torch.where(pick, self.psi, other.psi),
            torch.where(pick[..., None], self.W, other.W),
        )


# =========================================================================== #
#                                                                              #
#  ChartRATTLE sampler                                                         #
#                                                                              #
#  Runs in the η chart. The N(0, I) top-level prior is baked into U, so the     #
#  space is the identity UnconstrainedSpace over the θ names and the driver     #
#  reads q = η off as θ. evaluate_model builds U (TemperedAffine) and G_M        #
#  (TemperedMetric) from the constraint; integrate performs the RATTLE step.    #
#                                                                              #
# =========================================================================== #

class ChartRATTLE(HamiltonianSampler):
    """RATTLE constrained HMC for the hierarchical posterior p(θ | x), sampled
    in the η chart of the manifold {φ_η(ε) = x}.

    Parameters
    ----------
    constraint : ChartConstraint
        The reparameterization inverse ψ(η) = φ_η⁻¹(x) and its geometry.
    space
        Identity unconstrained space over the θ (= η) names, no prior (the
        N(0, I) prior is in U).
    step_size : float
        Integration step size. When adapting, start it small: the step is grown
        from here, so a too-large start begins above the solver-convergence cliff
        and cannot recover.
    num_steps : int
        RATTLE substeps per transition.
    adapt_step_size : bool
        Adapt the step size during warmup via the REINFORCE adapter. Default True.
    adaptation_sigma : float
        Perturbation scale of the REINFORCE adapter. Default 0.1.
    fp_max_iter : int
        Maximum position-solve iterations per substep. Default 100.
    fp_tol : float
        Convergence tolerance for the position solve (max norm of F). Default 1e-8.
    solver : str
        Position solver: ``"picard"`` (default) or ``"anderson"``.
    anderson_history : int or None
        History length for the Anderson solver. None resolves per-solve to n.
    damping : float
        Under-relaxation factor in (0, 1] shared by both solvers. Default 1.0.
    divergence_threshold : float
        Raw |delta_H| above which (or non-finite for which) the step is a
        divergence. Default 100.

    Notes
    -----
    A proposal whose position solve does not converge is rejected. No
    reverse-projection check is run.
    """

    def __init__(
        self,
        constraint: ChartConstraint,
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
        divergence_threshold: float = 100.0,
    ):
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
                f"unknown solver {solver!r}, expected 'picard' or 'anderson'")

        log_eps = math.log(step_size)
        adapter = NoAdaptation(init=log_eps)
        if adapt_step_size:
            adapter = Reinforce(sigma=adaptation_sigma, init=log_eps)
        super().__init__(None, space, requires_metric=True, num_steps=num_steps,
                         adapter=adapter, divergence_threshold=divergence_threshold,
                         trajectory_length=num_steps * step_size)

        self.constraint = constraint
        self._fp_max_iter = fp_max_iter
        self._fp_tol = fp_tol

        self.register_diagnostic("residual_mean", lambda: self._residual_sum / max(self._step, 1))
        self.register_diagnostic("residual_max", lambda: self._residual_max)
        self.register_diagnostic("fp_iters_mean", lambda: self._fp_iters_sum / max(self._step, 1))
        self.register_diagnostic("fp_iters_max", lambda: self._fp_iters_max)
        self.register_logging("|r|", lambda: "{:.2e}".format(float(self._step_residual.max())))

    # ---- model evaluation (the extension point) ---------------------------- #

    def evaluate_model(self, z_free, beta=None, grad=False):
        """The model at η = ``z_free``, and with ``grad`` the force driving the
        integrator. Returns ``(U, metric, psi, W)``, or ``(U, metric, psi, W,
        grad_V)`` when ``grad`` is set.

        U : TemperedAffine
            Potential U(η) = ½‖η‖² + log|det B| + β·½‖ψ‖². The target is e^{−U}.

        metric : TemperedMetric
            Metric G_M(η) = I + β WᵀW on the η-chart, the pushforward of the
            ambient N(0, I) metric onto the manifold in η coordinates.

        psi : (N, m)
            ψ(η) = φ_η⁻¹(x), the latent on the manifold.

        W : (N, m, n)
            W(η) = −∂ψ/∂η, the chart tangent data.

        grad_V : (N, n)
            Chart force ∇V, where V = U + ½ log det G_M is the potential the
            RATTLE step follows. V differs from U by the metric volume term,
            which cancels in the target but drives the constrained dynamics.

        ``beta`` overrides the sampler temperature (per replica under parallel
        tempering)."""
        beta = self.beta if beta is None else beta
        eta = z_free
        if grad:
            eta = z_free.detach().requires_grad_(True)
        n = eta.shape[-1]
        eye = torch.eye(n, dtype=eta.dtype, device=eta.device)

        with torch.enable_grad() if grad else nullcontext():
            psi, W, log_abs_det_B = self.constraint.psi_with_jvp(eta)
            gram = W.transpose(-2, -1) @ W                 # (N, n, n) = WᵀW
            lik = 0.5 * (psi * psi).sum(-1)                # U_lik = ½‖ψ‖²
            base = 0.5 * (eta * eta).sum(-1) + log_abs_det_B
            if grad:
                G = eye + _bcast_beta(beta, 2) * gram
                V = base + beta * lik + 0.5 * torch.logdet(G)
                (grad_V,) = torch.autograd.grad(V.sum(), eta)

        U = TemperedAffine(lik.detach(), base.detach(), beta)
        metric = TemperedMetric(gram.detach(), eye.expand(eta.shape[0], n, n), beta)
        out = (U, metric, psi.detach(), W.detach())
        if grad:
            out = out + (grad_V.detach(),)
        return out

    # ---- integrator hooks -------------------------------------------------- #

    def build_initial_state(self, q):
        """Evaluate the model at ``q`` = η and return the initial trajectory
        state, with momentum drawn later in :meth:`sample_momentum`."""
        U, metric, psi, W, grad_V = self.evaluate_model(q, grad=True)
        return ChartRATTLEState(q, None, U, metric, psi, W, grad_V, None)

    def sample_momentum(self, state):
        """Evaluate the model and force at η, draw the momentum π ~ N(0, G_M(η)),
        and zero this transition's worst-solve accumulators."""
        z = torch.zeros(state.q.shape[0], dtype=state.q.dtype, device=state.q.device)
        self._step_residual = z
        self._step_iters = z.clone()
        (state.U, state.metric, state.psi, state.W,
         state.grad_V) = self.evaluate_model(state.q, grad=True)
        state.p = state.metric.sample_momentum()
        state.deta = None
        return state

    def integrate(self, state, step_size):
        """One RATTLE substep at ``step_size``, tracking the worst position-solve
        residual and iteration count. The solve is warm-started by the previous
        displacement, which changes only the iteration count, not the fixed point."""
        h = step_size.unsqueeze(-1)                        # (N, 1)
        beta_col = _bcast_beta(self.beta, 1)

        rhs = h * state.p - 0.5 * h * h * state.grad_V
        eta_init = state.q
        if state.deta is not None:
            eta_init = state.q + state.deta
        eta1, iters, residual = _solve_rattle_step(
            self.constraint, state.q, state.psi, state.W, beta_col,
            state.metric.L, rhs, eta_init, self._solver,
            self._fp_max_iter, self._fp_tol)

        # A diverged chain (non-finite η1) is rejected by acceptance_delta. Fall
        # its position back to η0 so the model eval stays finite.
        finite = torch.isfinite(eta1).all(-1, keepdim=True)
        eta1 = torch.where(finite, eta1, state.q)
        U1, metric1, psi1, W1, g1 = self.evaluate_model(eta1, grad=True)

        corr = (W1.transpose(-2, -1) @ (psi1 - state.psi).unsqueeze(-1)).squeeze(-1)
        pi1 = ((eta1 - state.q) - beta_col * corr) / h - 0.5 * h * g1

        self._step_residual = torch.maximum(self._step_residual, residual)
        self._step_iters = torch.maximum(self._step_iters, iters.to(h.dtype))
        return ChartRATTLEState(eta1, pi1, U1, metric1, psi1, W1, g1,
                                deta=(eta1 - state.q))

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) − H(old)``, forced to +inf where the position solve
        did not converge (residual over ``fp_tol``, or non-finite)."""
        H_new = _hamiltonian(new.q, new.p, new.U.value, new.metric)
        H_old = _hamiltonian(old.q, old.p, old.U.value, old.metric)
        delta = H_new - H_old

        self._residual_sum = self._residual_sum + self._step_residual
        self._residual_max = torch.maximum(self._residual_max, self._step_residual)
        self._fp_iters_sum = self._fp_iters_sum + self._step_iters
        self._fp_iters_max = torch.maximum(self._fp_iters_max, self._step_iters)

        solve_failed = ~(self._step_residual <= self._fp_tol)
        return torch.where(solve_failed, delta.new_full((), float("inf")), delta)

    def adapt(self, accept_prob, delta_H):
        """REINFORCE step-size adaptation from this transition's energy error and
        worst solver residual and iteration count. The ``exp(-4|delta_H|)`` term
        brakes the step as the energy error grows, so from a small start the step
        settles below the solver-convergence cliff rather than running away. A
        diverged step leaves a non-finite residual (already rejected); charging it
        zero efficiency points the step down instead of poisoning the adapter."""
        floor = 1.0e-3
        energy_weight = 4.0
        num_iters = self._step_iters
        solver_penalty = torch.exp(-self._step_residual / self.step_size)
        delta_H_penalty = torch.exp(-energy_weight * delta_H.abs())
        efficiency = solver_penalty * delta_H_penalty * self.step_size / num_iters
        efficiency = torch.nan_to_num(efficiency, nan=0.0, posinf=0.0, neginf=0.0)
        f_t = -0.5 * torch.log(efficiency + floor) / abs(math.log(floor))
        self._step_size_adapter.update(f_t)

    def reset_extra_diagnostics(self):
        """Zero the run-level solver summaries and this transition's worst-solve
        accumulators (re-zeroed each transition in :meth:`sample_momentum`)."""
        N = self.step_size.shape[0]
        z = torch.zeros(N, dtype=self.step_size.dtype, device=self.step_size.device)
        self._residual_sum = z.clone()
        self._residual_max = z.clone()
        self._fp_iters_sum = z.clone()
        self._fp_iters_max = z.clone()
        self._step_residual = z.clone()
        self._step_iters = z.clone()
