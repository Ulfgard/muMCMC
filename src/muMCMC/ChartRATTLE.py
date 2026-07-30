from typing import Callable
import math

import torch

from .HamiltonianSampler import HamiltonianSampler
from ._adapters import Reinforce, NoAdaptation
from .spaces import TemperedAffine, TemperedMetric, broadcast_beta
from .RMHMC import _hamiltonian
from ._solvers import FixedPointSolver

# =========================================================================== #
#  ChartRATTLE: constrained HMC for hierarchical posteriors                   #
#                                                                             #
#  q is the library's name for the sampled position. Here it is the free      #
#  vector in the space's normal chart, so the hyperparameter is θ = T(q) and  #
#  q is also the chart coordinate of the manifold below. ε is the inner       #
#  latent and x the observation.                                              #
#                                                                             #
#  With q ~ p(q), ε ~ N(0, I_m) and a diffeomorphism φ_q, the non-centered    #
#  reparameterization of x | q is φ_q(ε) = x, so conditioning on the data is  #
#  the constraint g(q, ε) = φ_q(ε) − x = 0 defining                           #
#                                                                             #
#      M = { (q, ε) : φ_q(ε) = x },                                           #
#                                                                             #
#  whose q-marginal, sampled as a constrained chain, is p(θ | x). Writing     #
#  ψ(q) = φ_q⁻¹(x) and differentiating φ_q(ψ(q)) = x,                         #
#                                                                             #
#      A = ∂φ_q/∂q  (m, n),   B = ∂_ε φ_q  (m, m) invertible,                 #
#      W = −∂ψ/∂q = B⁻¹A.                                                     #
# =========================================================================== #
#  Energy                                                                     #
#                                                                             #
#  Change of variables gives x | q the density N(ψ(q); 0, I)/|det B(q)|,      #
#  which involves the ε block alone, so any prior p(q) carries through:       #
#                                                                             #
#      U(q) = −log p(q) + log|det B| + β·½‖ψ‖²,   G_M(q) = M + β Wᵀ W,        #
#                                                                             #
#  with p(q) and M the space's prior and its metric. Read in the space's      #
#  normal chart the prior is exactly N(0, I), so −log p(q) is ½‖q‖² and M is  #
#  the identity. Both are properties of the chart rather than conditions on   #
#  the model, which is what lets an arbitrary prior be sampled here.          #
#                                                                             #
#  No co-area factor survives into U, and that is not an omission. The        #
#  Hausdorff-measure form carries 1/√(det Λ) with Λ = Dg Dgᵀ, and reading it  #
#  in the q chart multiplies by the chart Jacobian √(det G_M). With Dg = [A,  #
#  B] and A = B W the two collapse:                                           #
#                                                                             #
#      Λ = A Aᵀ + B Bᵀ = B (I + W Wᵀ) Bᵀ,                                     #
#      det Λ = (det B)² det(I + Wᵀ W) = (det B)² det G_M       (Sylvester)    #
#                                                                             #
#  so √(det G_M/det Λ) = 1/|det B|, the log|det B| term above. The user       #
#  supplies that term and the measure bookkeeping is ours.                    #
#                                                                             #
#  M is not a free choice: it is the prior block of G_M, the (q1 − q0) of F   #
#  and of the momentum line, the kinetic term of S_h, and the DF(q0) the      #
#  solve preconditions with, all one matrix. S_h needs it position-           #
#  independent, since it would otherwise need its midpoint value, a different #
#  scheme. The normal chart supplies exactly that, and supplies it as the     #
#  true Hessian of the prior term rather than as a constant standing in for   #
#  one, so no scale is being guessed at.                                      #
#                                                                             #
#  U is affine in β (lik = ½‖ψ‖²) and so is G_M (A_lik = WᵀW), so             #
#  evaluate_model returns a TemperedAffine and a TemperedMetric, giving the   #
#  tempered target at any temperature. The Hamiltonian                        #
#                                                                             #
#      H = U + ½ pᵀ G_M⁻¹ p + ½ log det G_M                                   #
#                                                                             #
#  keeps e^{−U} invariant for any SPD G_M, since marginalizing p ~ N(0,       #
#  G_M(q)) cancels ½ log det G_M against the Gaussian normalizer.             #
# =========================================================================== #
#  Step  (q0, p0) -> (q1, p1)                                                 #
#                                                                             #
#  The step of integrate is the variational integrator of the discrete        #
#  Lagrangian                                                                 #
#                                                                             #
#      S_h(q0, q1) = (q1 − q0)ᵀM(q1 − q0)/(2h) + β‖ψ1 − ψ0‖²/(2h)             #
#                    − (h/2)[V(q0) + V(q1)],   V = U + ½ log det G_M,         #
#                                                                             #
#  in the sense that p0 = −∂S_h/∂q0 is the position equation F(q1) = 0 it     #
#  solves and p1 = +∂S_h/∂q1 is its momentum line. So the map is symplectic,  #
#  hence volume-preserving on (q, p), which is the half of Metropolis         #
#  exactness self-adjointness leaves open, and S_h(q0, q1) = S_h(q1, q0)      #
#  makes it self-adjoint. The kinetic Hessian as q1 -> q0 is G_M(q0)/h, which #
#  is where G_M comes from and what forces the W0 / W1 asymmetry.             #
#                                                                             #
#  Grouping U and ½ log det G_M into one V is what lets the step carry a      #
#  single gradient. The position Jacobian is DF(q1) = M + β W0ᵀ W1, so        #
#  DF(q0) = G_M(q0): preconditioning by the Cholesky of G_M(q0) makes Picard  #
#  the frozen-Jacobian Newton step, and only ψ(q^k) is evaluated in the loop. #
#  The newton rule re-evaluates DF(q1) instead, at a tangent pass per         #
#  iteration.                                                                 #
#                                                                             #
#  Both properties hold up to the solve. Where F has several roots the        #
#  forward and reverse solves can pick different ones, an effect that         #
#  vanishes as h shrinks. A failed solve is rejected with +inf energy, but no #
#  reverse-projection check is run: discarding a step whose reverse solve     #
#  lands elsewhere restores reversibility by turning the discarded set into a #
#  hard barrier, and those are the steps crossing strong nonlinearity that    #
#  the method exists to take. Exact but reducible is the worse failure, and   #
#  the quieter one.                                                           #
# =========================================================================== #
#  Tempering                                                                  #
#                                                                             #
#  β softens the observation in the scale family, Σ -> Σ/β, scaling the data  #
#  fit ½‖ψ‖² by β, so β multiplies the likelihood wherever it appears: in U,  #
#  in G_M, and hence in F and DF. The prior and the volume log|det B| stay    #
#  untempered, which drops the β^{m/2} normalizer the exact scale family      #
#  carries. That is deliberate, keeping U finite as β -> 0 where the family   #
#  degenerates, and it costs the β = 0 rung its normalization: thermodynamic  #
#  integration along the ladder yields log p(x) − log Z_0, so PT's            #
#  log_evidence is an evidence against an unnormalized reference here rather  #
#  than an absolute one.                                                      #
# =========================================================================== #

#  ---- Constraint ---------------------------------------------------------- #

class ChartConstraint:
    """Constraint M = {(q, ε) : φ_q(ε) = x} exposed through the inverse map.

    A subclass supplies the batched, temperature-free inverse ``psi(q)`` =
    φ_q⁻¹(x) (shape ``(N, n)`` to ``(N, m)``) and ``log_abs_det_B(q)`` =
    log|det ∂_ε φ|. The sampler derives W = −∂ψ/∂q, the potential, and the
    metric, and applies the inverse temperature. Override ``psi_with_jvp`` to
    return (ψ, W, log|det B|) from explicit Jacobians.

    Parameters
    ----------
    x : (m,)
        Conditioning value the manifold is defined by, shared across chains.
    """

    # Whether psi_with_jvp needs grad mode enabled around it. True for the
    # default, which differentiates psi. A subclass returning W from explicit or
    # forward-mode Jacobians should set it False, which spares the graph build
    # on every call, and that is once per iteration under the newton solver.
    jvp_needs_grad = True

    def __init__(self, x: torch.Tensor):
        self.x = x

    def psi(self, q: torch.Tensor) -> torch.Tensor:
        """The inverse map ψ(q) = φ_q⁻¹(x): the latent that maps to the
        observation x under the layer φ_q. ``q`` is ``(N, n)``, the result
        ``(N, m)``. Implemented by a subclass."""
        raise NotImplementedError

    def log_abs_det_B(self, q: torch.Tensor) -> torch.Tensor:
        """log|det ∂_ε φ_q|, the log Jacobian of the layer in its latent
        argument. ``q`` is ``(N, n)``, the result ``(N,)``. Implemented by a
        subclass."""
        raise NotImplementedError

    def psi_with_jvp(self, q: torch.Tensor):
        """(ψ, W, log|det B|) at ``q``, with W = −∂ψ/∂q by one reverse pass per
        latent. ``q`` must require grad. Override with explicit Jacobians, and
        set :attr:`jvp_needs_grad` False when the override needs no grad mode."""
        eps = self.psi(q)
        m = eps.shape[-1]
        cols = [torch.autograd.grad(eps[:, i].sum(), q, retain_graph=True,
                                    create_graph=True)[0]
                for i in range(m)]
        W = -torch.stack(cols, dim=1)                      # (N, m, n)
        return eps, W, self.log_abs_det_B(q)


class LocationScaleChart(ChartConstraint):
    """Conditionally-Gaussian layer x | q ~ N(μ(q), Σ(q)).

    φ_q(ε) = μ(q) + L(q) ε with Σ(q) = L(q) L(q)ᵀ, so ψ(q) = L(q)⁻¹(x − μ(q))
    and log|det B| = ½ log det Σ. ``mean`` and ``cov`` are batched callables
    (``q`` shape ``(N, n)`` to ``(N, m)`` and ``(N, m, m)`` SPD), differentiable
    in ``q``. A ``cov`` that is not numerically SPD raises.

    Parameters
    ----------
    mean : callable
        q -> μ(q).
    cov : callable
        q -> Σ(q), SPD.
    x : (m,)
        Observation.
    """

    def __init__(self, mean: Callable, cov: Callable, x: torch.Tensor):
        super().__init__(x)
        self.mean = mean
        self.cov = cov

    def _factor(self, q):
        return self.mean(q), torch.linalg.cholesky(self.cov(q))

    def psi(self, q):
        mu, L = self._factor(q)
        return torch.linalg.solve_triangular(
            L, (self.x - mu).unsqueeze(-1), upper=False).squeeze(-1)

    def log_abs_det_B(self, q):
        _, L = self._factor(q)
        return torch.log(L.diagonal(dim1=-2, dim2=-1).abs()).sum(-1)


class _ChartInNormal:
    """A constraint written on θ, read at the chart coordinate q with θ = T(q),

        W_q = W_θ diag(dθ/dq),

    so the constraint itself never sees the chart.
    """

    def __init__(self, constraint: "ChartConstraint", transform):
        self.constraint = constraint
        self.transform = transform
        self.jvp_needs_grad = constraint.jvp_needs_grad

    def psi(self, q: torch.Tensor) -> torch.Tensor:
        return self.constraint.psi(self.transform.forward(q).mapped_point)

    def psi_with_jvp(self, q: torch.Tensor):
        chart = self.transform.forward(q)
        psi, W, log_abs_det_B = self.constraint.psi_with_jvp(chart.mapped_point)
        return psi, W * chart.jacobian_diag[..., None, :], log_abs_det_B


#  ---- Position solve ------------------------------------------------------ #

def _solve_rattle_step(constraint, q, psi, W, A_prior, beta_col, beta_mat,
                       chol_G, rhs, q_init, solver):
    """Solve the RATTLE position equation for the step endpoint q1,

        F(q1) = A_prior (q1 - q0) - beta W0^T(psi(q1) - psi0) - rhs = 0,

    preconditioned by G_M(q0) = A_prior + beta W0^T W0, whose Cholesky factor is
    ``chol_G``. ``q, psi, W`` are the q0 quantities and ``rhs`` is
    ``h p0 - (h^2/2) grad V(q0)``. Returns a SolveResult.

    The solver's ``needs_jacobian`` picks which residual it gets: the value alone,
    or the value with DF(q1) = A_prior + beta W0^T W(q1), which costs a tangent
    pass per iteration.
    """
    Wt = W.transpose(-2, -1)                               # (N, n, m)

    def drift(q_k):
        return (A_prior @ (q_k - q).unsqueeze(-1)).squeeze(-1)

    def residual_fn(q_k):
        with torch.no_grad():                             # solve is derivative-free
            corr = (Wt @ (constraint.psi(q_k) - psi).unsqueeze(-1)).squeeze(-1)
            return drift(q_k) - beta_col * corr - rhs

    def residual_and_jacobian(q_k):
        # DF(q1) = A_prior + β W0ᵀ W(q1), the true Jacobian rather than its value
        # frozen at q0. W(q1) needs a tangent pass, so this costs more per
        # iteration than the residual alone.
        if constraint.jvp_needs_grad:
            with torch.enable_grad():
                qk = q_k.detach().requires_grad_(True)
                psi_k, W_k, _ = constraint.psi_with_jvp(qk)
        else:
            with torch.no_grad():
                psi_k, W_k, _ = constraint.psi_with_jvp(q_k)
        psi_k, W_k = psi_k.detach(), W_k.detach()
        corr = (Wt @ (psi_k - psi).unsqueeze(-1)).squeeze(-1)
        r = drift(q_k) - beta_col * corr - rhs
        return r, A_prior + beta_mat * (Wt @ W_k)

    def precond(F):                                       # G_M(q0)⁻¹ F
        return torch.cholesky_solve(F.unsqueeze(-1), chol_G).squeeze(-1)

    if solver.needs_jacobian:
        return solver.solve(residual_and_jacobian, q_init)
    return solver.solve(residual_fn, q_init, precond=precond)


#  ---- Chain state --------------------------------------------------------- #

class ChartRATTLEState:
    """Working state of one ChartRATTLE trajectory, batched over ``(N,)`` chains.

    Only ``q``, ``p`` and ``U`` are guaranteed present between transitions. The
    geometry, force and warm-start displacement are trajectory scratch, set by
    :meth:`ChartRATTLE.sample_momentum` and dropped at the end of a transition.

    Attributes
    ----------
    q, p : (N, n)
        Position (the chart coordinate, ``q`` = θ) and momentum. ``q`` is read
        as the sample.
    U : TemperedAffine
        Potential. ``U.value`` is the ``(N,)`` energy and ``U.lik`` the swap
        statistic.
    metric : TemperedMetric
        Chart metric G_M(q).
    psi, W : (N, m), (N, m, n)
        The inverse map ψ(q) and W = −∂ψ/∂q.
    grad_V : (N, n) or None
        Force ∇V(q).
    dq : (N, n) or None
        Last displacement, warm-starts the next solve.
    """

    def __init__(self, q, p=None, U=None, metric=None, psi=None, W=None,
                 grad_V=None, dq=None):
        self.q = q
        self.p = p
        self.U = U
        self.metric = metric
        self.psi = psi
        self.W = W
        self.grad_V = grad_V
        self.dq = dq

    def reorder(self, perm):
        """Relabel the configuration to a new temperature slot. Its potential,
        metric, geometry and force are re-evaluated at the next step."""
        return ChartRATTLEState(self.q[perm],
                                None if self.p is None else self.p[perm])

    def select_accepted(self, accepted, other):
        """Per-chain choice between this endpoint (where ``accepted``) and the
        start ``other``, over the fields that outlive a transition. The metric,
        geometry, force and warm-start displacement are trajectory scratch and
        are dropped."""
        pick = accepted.unsqueeze(-1)
        return ChartRATTLEState(
            torch.where(pick, self.q, other.q),
            torch.where(pick, self.p, other.p),
            self.U.select(accepted, other.U),
        )


class ChartRATTLE(HamiltonianSampler):
    """RATTLE constrained HMC for the hierarchical posterior p(θ | x), sampled
    in the q chart of the manifold {φ_q(ε) = x}.

    Parameters
    ----------
    constraint : ChartConstraint
        The reparameterization inverse ψ(q) = φ_q⁻¹(x) and its geometry.
    space
        Space over the θ names. The chain runs in its normal chart, so the prior
        enters U as ½‖q‖² and the prior block M of G_M = M + β WᵀW is the
        identity. The constraint is evaluated at θ = T(q).
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
        Position solver: ``"picard"`` (default), ``"anderson"`` or ``"newton"``.
    anderson_history : int or None
        History length for the Anderson solver, ignored by the others. None
        resolves per-solve to n.
    damping : float
        Under-relaxation factor in (0, 1] shared by every solver. Default 1.0.
    divergence_threshold : float
        Raw |delta_H| above which (or non-finite for which) the step is a
        divergence. Default 100.

    Raises
    ------
    ValueError
        If ``damping`` is outside (0, 1], if ``solver`` is not recognised, or if
        the space carries no prior.

    Notes
    -----
    A proposal whose position solve does not converge is rejected. No
    reverse-projection check is run: see the Step section at the top of this
    module for why that is a trade in favour of ergodicity.
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
        self._solver = FixedPointSolver(
            solver, damping=damping, anderson_history=anderson_history,
            max_iter=fp_max_iter, tol=fp_tol)

        log_eps = math.log(step_size)
        adapter = NoAdaptation(init=log_eps)
        if adapt_step_size:
            adapter = Reinforce(sigma=adaptation_sigma, init=log_eps)
        if not space.is_proper:
            raise ValueError(
                "ChartRATTLE needs a space with a prior, whose normal chart "
                "supplies the constant prior block M of G_M = M + beta W^T W. "
                "UnnormalizedSpace has no chart and so no M.")
        super().__init__(None, space, requires_metric=True, num_steps=num_steps,
                         adapter=adapter, divergence_threshold=divergence_threshold,
                         trajectory_length=num_steps * step_size)

        self.constraint = constraint
        self._chart = _ChartInNormal(constraint, space.as_transform)
        self._fp_tol = fp_tol

        self.register_diagnostic("residual_mean", lambda: self._residual_sum / max(self._step, 1))
        self.register_diagnostic("residual_max", lambda: self._residual_max)
        self.register_diagnostic("fp_iters_mean", lambda: self._fp_iters_sum / max(self._step, 1))
        self.register_diagnostic("fp_iters_max", lambda: self._fp_iters_max)
        self.register_logging("|r|", lambda: "{:.2e}".format(float(self._step_residual.max())))

    # ---- model evaluation (the extension point) ---------------------------- #

    def evaluate_model(self, z_free, beta=None, grad=False):
        """The model at the position ``z_free`` (the chart coordinate ``q``), and
        with ``grad`` the force driving the integrator. Returns ``(U, metric,
        psi, W)``, or ``(U, metric, psi, W, grad_V)`` when ``grad`` is set.

        U : TemperedAffine
            Potential U(q) = ½‖q‖² + log|det B| + β·½‖ψ‖², the prior read in the
            space's normal chart. The target is e^{−U}.

        metric : TemperedMetric
            Metric G_M(q) = I + β WᵀW on the q-chart: the ambient metric
            pulled back to the chart along q ↦ (q, ψ(q)), whose tangents are
            (v, −W v).

        psi : (N, m)
            ψ(q) = φ_q⁻¹(x), the latent on the manifold.

        W : (N, m, n)
            W(q) = −∂ψ/∂q, the chart tangent data.

        grad_V : (N, n)
            Chart force ∇V, where V = U + ½ log det G_M is the potential the
            RATTLE step follows. V differs from U by the metric volume term,
            which cancels in the target but drives the constrained dynamics.

        ``beta`` overrides the sampler temperature (per replica under parallel
        tempering). All returned tensors are detached."""
        beta = self.beta if beta is None else beta
        # Detached leaf regardless of ``grad``: W comes from a reverse pass
        # through psi, so the input has to carry a graph either way.
        q = z_free.detach().requires_grad_(True)
        A_prior = self.space.prior_metric_normal(q)

        with torch.enable_grad():
            psi, W, log_abs_det_B = self._chart.psi_with_jvp(q)
            gram = W.transpose(-2, -1) @ W                 # (N, n, n) = WᵀW
            lik = 0.5 * (psi * psi).sum(-1)                # U_lik = ½‖ψ‖²
            base = -self.space.prior_log_prob_normal(q) + log_abs_det_B
            if grad:
                G = A_prior + broadcast_beta(beta, 2) * gram
                # Cholesky, not torch.logdet: same factorization the metric uses
                # for G_M⁻¹, and it rejects a non-SPD G instead of returning nan.
                chol = torch.linalg.cholesky(G)
                log_det_G = 2.0 * chol.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
                V = base + beta * lik + 0.5 * log_det_G
                (grad_V,) = torch.autograd.grad(V.sum(), q)

        U = TemperedAffine(lik.detach(), base.detach(), beta)
        metric = TemperedMetric(gram.detach(), A_prior, beta)
        out = (U, metric, psi.detach(), W.detach())
        if grad:
            out = out + (grad_V.detach(),)
        return out

    # ---- integrator hooks -------------------------------------------------- #

    def build_initial_state(self, q):
        """Evaluate the model at ``q`` and return the initial trajectory state,
        with momentum drawn later in :meth:`sample_momentum`."""
        U, metric, psi, W, grad_V = self.evaluate_model(q, grad=True)
        return ChartRATTLEState(q, None, U, metric, psi, W, grad_V, None)

    def sample_momentum(self, state):
        """Evaluate the model, geometry and force at ``q``, draw the momentum
        ``p ~ N(0, G_M(q))``, and zero this transition's worst-solve
        accumulators."""
        # Sole initialization point for the trajectory's model quantities: a PT
        # swap can relabel q to another temperature slot between transitions, so
        # nothing model-dependent may be carried across the accept boundary.
        z = torch.zeros(state.q.shape[0], dtype=state.q.dtype, device=state.q.device)
        self._step_residual = z.clone()
        self._step_iters = z.clone()
        (state.U, state.metric, state.psi, state.W,
         state.grad_V) = self.evaluate_model(state.q, grad=True)
        state.p = state.metric.sample_momentum()
        state.dq = None
        return state

    def integrate(self, state, step_size):
        """One RATTLE substep at ``step_size`` = h, solving

            F(q1) = M(q1 − q0) − β W0ᵀ(ψ(q1) − ψ0) − h p0 + (h²/2) ∇V(q0) = 0

        for the new position and taking the momentum from

            p1 = (1/h)[M(q1 − q0) − β W1ᵀ(ψ1 − ψ0)] − (h/2) ∇V(q1),

        with V = U + ½ log det G_M and M the prior metric. Tracks the worst
        position-solve residual and iteration count over the batch. The solve is
        warm-started by the previous displacement, which changes only the
        iteration count, not the fixed point."""
        h = step_size.unsqueeze(-1)                        # (N, 1)
        beta_col = broadcast_beta(self.beta, 1)
        beta_mat = broadcast_beta(self.beta, 2)
        A_prior = state.metric.base                        # M, off the metric

        rhs = h * state.p - 0.5 * h * h * state.grad_V
        q_init = state.q
        if state.dq is not None:
            q_init = state.q + state.dq
        q, iters, residual = _solve_rattle_step(
            self._chart, state.q, state.psi, state.W, A_prior, beta_col,
            beta_mat, state.metric.L, rhs, q_init, self._solver)

        U, metric, psi, W, grad_V = self.evaluate_model(q, grad=True)

        corr = (W.transpose(-2, -1) @ (psi - state.psi).unsqueeze(-1)).squeeze(-1)
        drift = (A_prior @ (q - state.q).unsqueeze(-1)).squeeze(-1)
        p = (drift - beta_col * corr) / h - 0.5 * h * grad_V

        self._step_residual = torch.maximum(self._step_residual, residual)
        self._step_iters = torch.maximum(self._step_iters, iters.to(h.dtype))
        return ChartRATTLEState(q, p, U, metric, psi, W, grad_V,
                                dq=(q - state.q))

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) − H(old)``, forced to +inf where the position solve
        did not converge (worst residual over ``fp_tol``), so a non-converged step
        is rejected even when its energy change is small."""
        H_new = _hamiltonian(new.q, new.p, new.U.value, new.metric)
        H_old = _hamiltonian(old.q, old.p, old.U.value, old.metric)
        delta = H_new - H_old

        self._residual_sum = self._residual_sum + self._step_residual
        self._residual_max = torch.maximum(self._residual_max, self._step_residual)
        self._fp_iters_sum = self._fp_iters_sum + self._step_iters
        self._fp_iters_max = torch.maximum(self._fp_iters_max, self._step_iters)

        solve_failed = self._step_residual > self._fp_tol
        return torch.where(solve_failed, delta.new_full((), float("inf")), delta)

    def adapt(self, accept_prob, delta_H):
        """REINFORCE step-size adaptation from this transition's energy error and
        worst solver residual and iteration count. The ``exp(-4|delta_H|)`` term
        brakes the step as the energy error grows, so from a small start the step
        settles below the solver-convergence cliff rather than running away."""
        floor = 1.0e-3
        energy_weight = 4.0
        num_iters = self._step_iters
        solver_penalty = torch.exp(-self._step_residual / self.step_size)
        delta_H_penalty = torch.exp(-energy_weight * delta_H.abs())
        efficiency = solver_penalty * delta_H_penalty * self.step_size / num_iters
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
