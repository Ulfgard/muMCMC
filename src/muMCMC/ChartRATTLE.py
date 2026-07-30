from typing import Callable
import math

import torch

from .HamiltonianSampler import HamiltonianSampler
from .adapters import Reinforce, NoAdaptation
from .spaces import TemperedAffine, TemperedMetric, broadcast_beta
from .RMHMC import _hamiltonian
from .solvers import FixedPointSolver

# =========================================================================== #
#                                                                             #
#  ChartRATTLE: constrained HMC for hierarchical posteriors                   #
#                                                                             #
#  q is the library's name for the sampled position. Here it is the           #
#  hyperparameter (q = θ), which doubles as the chart coordinate of the       #
#  manifold below. ε is the inner latent and x the observation.               #
#                                                                             #
#  Latent q ~ p(q) and ε ~ N(0, I_m), and a diffeomorphism φ_q with           #
#                                                                             #
#      φ_q(ε) = x,      θ = q,                                                #
#                                                                             #
#  the non-centered reparameterization of x | q. Conditioning on the data x   #
#  is the constraint g(q, ε) = φ_q(ε) − x = 0 defining the manifold           #
#                                                                             #
#      M = { (q, ε) : φ_q(ε) = x },                                           #
#                                                                             #
#  whose q-marginal, sampled as a constrained chain, is p(θ | x).             #
#                                                                             #
# =========================================================================== #
#                                                                             #
#  Jacobians of the layer, at the point (q, ψ(q)) on M                        #
#                                                                             #
#      A = ∂φ_q/∂q     (m, n)        B = ∂_ε φ_q     (m, m), invertible       #
#                                                                             #
#  ψ(q) = φ_q⁻¹(x) solves φ_q(ψ(q)) = x, so A + B ∂ψ/∂q = 0 and               #
#                                                                             #
#      W = −∂ψ/∂q = B⁻¹A.                                                     #
#                                                                             #
# =========================================================================== #
#                                                                             #
#  Energy                                                                     #
#                                                                             #
#  Change of variables gives x | q the density N(ψ(q); 0, I) / |det B(q)|.    #
#  That involves the ε block alone, so any prior p(q) on the chart coordinate #
#  carries straight through and the target is e^{−U} with                     #
#                                                                             #
#      U(q) = −log p(q) + log|det B| + β·½‖ψ‖²,   G_M(q) = M + β Wᵀ W,        #
#                                                                             #
#  with M the space's prior metric, or I when it has none.                    #
#                                                                             #
#  p(q) is the space's prior, read the same way every other sampler in the    #
#  library reads it. A space with no prior gives a flat one, so the N(0, I)   #
#  latent that the plain non-centered parameterization implies is requested   #
#  by name rather than assumed.                                               #
#                                                                             #
#  M matters, and it is not a free choice. The same matrix appears in five    #
#  coupled places: the prior block of G_M, the (q1 − q0) term of F, the       #
#  (q1 − q0) of the momentum line, the kinetic term of S_h, and DF(q0) that   #
#  the solve preconditions with. They are one object, so they move together   #
#  or not at all. Changing it in G_M alone leaves H no longer the             #
#  integrator's invariant: on the funnel, scaling it by 1.05 raises the worst #
#  |ΔH| over a 20 step trajectory from 8.3e-4 to 8.0e-3.                      #
#                                                                             #
#  Moved together they stay consistent, which is why a constant M is          #
#  supported, and it is what lets the sampler match a prior of the wrong      #
#  scale. A prior N(0, s²I) has precision s⁻²I, and with M = I instead the    #
#  momentum is drawn at unit scale against a target of width s, so the        #
#  trajectory needs about s/h steps to cross what one step should cover.      #
#  M = s⁻²I fixes that.                                                       #
#                                                                             #
#  A position-dependent M cannot be supported by this scheme. S_h would need  #
#  M at the midpoint to stay symmetric, F would pick up ∂M/∂q1 terms, and     #
#  DF(q0) would stop being G_M(q0). One is refused at the first model         #
#  evaluation rather than silently frozen. Correctness never rested on M      #
#  either way: exactness                                                      #
#  needs only that G_M is SPD and that V carries ½ log det G_M, so M is a     #
#  preconditioning choice throughout.                                         #
#                                                                             #
#  Because the constraint and the prior are both evaluated at the sampled     #
#  position, and U carries no change-of-variables Jacobian, the space has to  #
#  map q to θ by the identity. The constructor checks this.                   #
#                                                                             #
#  No co-area factor survives into U, and that is not an omission. In the     #
#  Hausdorff-measure form the ambient density restricted to M carries a       #
#  factor 1/√(det Λ) with Λ = Dg Dgᵀ, and reading it in the q chart           #
#  multiplies by the chart Jacobian √(det G_M). The two collapse exactly:     #
#  with Dg = [A, B] and A = B W,                                              #
#                                                                             #
#      Λ = A Aᵀ + B Bᵀ = B (I + W Wᵀ) Bᵀ,                                     #
#      det Λ = (det B)² det(I + Wᵀ W) = (det B)² det G_M       (Sylvester)    #
#                                                                             #
#  so √(det G_M / det Λ) = 1/|det B|, which is the log|det B| term above. The #
#  user supplies log|det B|. The measure bookkeeping is ours.                 #
#                                                                             #
#  U is affine in β (lik = ½‖ψ‖², base = −log p(q) + log|det B|) and G_M is   #
#  affine in β (A_lik = Wᵀ W, A_prior = M), so evaluate_model returns them as #
#  a TemperedAffine and a TemperedMetric. The Hamiltonian                     #
#                                                                             #
#      H = U + ½ pᵀ G_M⁻¹ p + ½ log det G_M                                   #
#                                                                             #
#  keeps e^{−U} invariant: marginalizing p ~ N(0, G_M(q)) out of e^{−H}       #
#  cancels ½ log det G_M against the Gaussian normalizer and leaves e^{−U}    #
#  dq, for any SPD G_M. β enters only through U and G_M, so evaluate_model    #
#  gives the tempered target at any temperature and a parallel-tempering swap #
#  reads the temperature-free U_lik = ½‖ψ‖² off U.lik.                        #
#                                                                             #
# =========================================================================== #
#                                                                             #
#  Step  (q0, p0) -> (q1, p1)                                                 #
#                                                                             #
#      F(q1) = M(q1 − q0) − β W0ᵀ(ψ(q1) − ψ0) − h p0 + (h²/2) ∇V(q0) = 0,     #
#      DF(q1) = M + β W0ᵀ W1,   so DF(q0) = G_M(q0),                          #
#      p1 = (1/h)[M(q1 − q0) − β W1ᵀ(ψ1 − ψ0)] − (h/2) ∇V(q1),                #
#                                                                             #
#  solved by any rule in solvers.py. Picard preconditioned by G_M(q0) is the  #
#  frozen-Jacobian Newton step, and newton evaluates DF(q1) at each iterate   #
#  instead, at the price of a tangent pass per iteration.                     #
#                                                                             #
#  V = U + ½ log det G_M, so the force ∇V is one autograd.grad(V.sum(), q).   #
#  Only ψ(q^k) is evaluated in the loop, preconditioned by one Cholesky of    #
#  G_M(q0) (the metric's own factor).                                         #
#                                                                             #
#  Why this is a valid proposal. The step is the variational integrator of    #
#  the discrete Lagrangian                                                    #
#                                                                             #
#      S_h(q0, q1) = (q1 − q0)ᵀM(q1 − q0)/(2h) + β‖ψ1 − ψ0‖²/(2h)             #
#                    − (h/2)[V(q0) + V(q1)],                                  #
#                                                                             #
#  in the sense that p0 = −∂S_h/∂q0 is the position equation F = 0 and        #
#  p1 = +∂S_h/∂q1 is the momentum line. Two properties follow for free:       #
#                                                                             #
#    * the map is symplectic, hence volume-preserving on (q, p). This is the  #
#      half of Metropolis exactness that self-adjointness alone leaves open.  #
#    * S_h(q0, q1) = S_h(q1, q0), so the map is self-adjoint: applied to      #
#      (q1, −p1) it returns (q0, −p0).                                        #
#                                                                             #
#  S_h also explains the pieces. Its kinetic term is the ambient chord with   #
#  the q block measured by M and the ε block scaled by β, and its Hessian as  #
#  q1 -> q0 is G_M(q0)/h. So the metric is the discrete kinetic form rather   #
#  than a free choice, and the W0 / W1 asymmetry between the two equations    #
#  is forced (differentiate at the left endpoint, then at the right).         #
#                                                                             #
#  Both properties hold up to the position solve. Where F has several roots   #
#  the forward and reverse solves can pick different ones, and self-          #
#  adjointness holds only up to that. The roots merge as h shrinks, so it is  #
#  a large-step effect.                                                       #
#                                                                             #
#  A failed solve is rejected with +inf energy, but no reverse-projection     #
#  check is run, and that is a deliberate trade rather than an omission.      #
#  Discarding a step whose reverse solve lands elsewhere buys back exact      #
#  reversibility at the cost of irreducibility: the discarded set is a hard   #
#  barrier, and the steps in it are exactly the ones crossing a region of     #
#  strong nonlinearity. Under a metric that already encodes that              #
#  nonlinearity, those are the steps the method exists to take. The checked   #
#  variant is π-reversible but can be reducible, and its failure is silent:   #
#  acceptance stays high while a whole basin goes unvisited. Unchecked, the   #
#  error is instead bounded, visible in the energy and residual diagnostics,  #
#  and shrinks with h. Ergodicity is the property worth testing here, not     #
#  reversibility.                                                             #
#                                                                             #
# =========================================================================== #
#                                                                             #
#  Constraint interface (untempered)                                          #
#                                                                             #
#      psi(q)           -> ψ = φ_q⁻¹(x)         the inverse map               #
#      log_abs_det_B(q) -> log|det B|           = ½ log det Σ for a scale     #
#                                                                             #
#  W = −∂ψ/∂q follows from one reverse pass per latent (classic autograd, no  #
#  vmap). Override psi_with_jvp with explicit Jacobians to skip it. The       #
#  constraint is temperature-free: β enters only in evaluate_model.           #
#                                                                             #
# =========================================================================== #
#                                                                             #
#  Tempering                                                                  #
#                                                                             #
#  β softens the observation in the scale family, Σ -> Σ/β, which scales the  #
#  data-fit ½‖ψ‖² by β. So β multiplies the likelihood wherever it appears:   #
#  in the potential U (β·½‖ψ‖²), in the metric G_M = I + β WᵀW, and hence in  #
#  the position equation F and its Jacobian above. The prior −log p(q) and    #
#  the volume log|det B| stay untempered. That drops the β^{m/2} normalizer   #
#  the                                                                        #
#  exact scale family carries, deliberately: it keeps U finite as β -> 0,     #
#  where the scale family itself degenerates. A parallel-tempering swap reads #
#  the β-free ½‖ψ‖² off U.lik.                                                #
#                                                                             #
#  What the drop costs. The β = 0 rung is p(q)/|det B(q)|, not the prior      #
#  itself, and its normalizer Z_0 is not 1. Thermodynamic integration along   #
#  the ladder therefore yields log p(x) − log Z_0, so PT's ``log_evidence``   #
#  is an evidence against an unnormalized reference here rather than an       #
#  absolute one.                                                              #
#                                                                             #
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
        latent. ``q`` must require grad. Override with explicit Jacobians."""
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


#  ---- Position solve ------------------------------------------------------ #

def _solve_rattle_step(constraint, q, psi, W, A_prior, beta_col, beta_mat,
                       chol_G, rhs, q_init, solver):
    """Solve the RATTLE position equation for the step endpoint q1,

        F(q1) = A_prior (q1 − q0) − β W0ᵀ(ψ(q1) − ψ0) − rhs = 0,

    preconditioned by G_M(q0) = A_prior + β W0ᵀW0.

    Parameters
    ----------
    constraint : ChartConstraint
        Supplies ψ, and W too when the solver wants a Jacobian.
    q, psi, W : (N, n), (N, m), (N, m, n)
        The q0 position and its inverse map and tangent data.
    A_prior : (n, n)
        Constant prior block of the metric.
    beta_col, beta_mat : broadcastable β
        Shaped for an ``(N, n)`` vector and an ``(N, n, n)`` matrix.
    chol_G : (N, n, n)
        Cholesky factor of G_M(q0), the frozen-Jacobian preconditioner.
    rhs : (N, n)
        ``h p0 − (h²/2) ∇V(q0)``.
    q_init : (N, n)
        Warm start.
    solver : FixedPointSolver
        Whose ``needs_jacobian`` picks the residual contract below.

    Returns
    -------
    SolveResult
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
        with torch.enable_grad():
            qk = q_k.detach().requires_grad_(True)
            psi_k, W_k, _ = constraint.psi_with_jvp(qk)
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


# =========================================================================== #
#                                                                             #
#  ChartRATTLE sampler                                                        #
#                                                                             #
#  Runs in the q chart. The space is the identity UnconstrainedSpace over the #
#  θ names and the driver reads the position q off as θ. The space's prior    #
#  goes into U, flat if it has none. evaluate_model builds U (TemperedAffine) #
#  and G_M (TemperedMetric) from the constraint. integrate performs the step. #
#                                                                             #
# =========================================================================== #

class ChartRATTLE(HamiltonianSampler):
    """RATTLE constrained HMC for the hierarchical posterior p(θ | x), sampled
    in the q chart of the manifold {φ_q(ε) = x}.

    Parameters
    ----------
    constraint : ChartConstraint
        The reparameterization inverse ψ(q) = φ_q⁻¹(x) and its geometry.
    space
        Identity unconstrained space over the θ (= q) names. Its prior becomes
        the prior on q, entering U as −log p(q), and is flat when the space
        carries none. Its ``prior_metric_fn`` becomes the prior block M of
        G_M = M + β WᵀW, defaulting to the identity, and has to be independent of
        position (see the Energy section at the top of this module).
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
        Newton re-evaluates DF at every iterate rather than freezing it at the
        step start, which is worth its tangent pass when the Jacobian moves
        quickly along the iteration.
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
        If ``space`` does not map q to θ by the identity, if ``damping`` is
        outside (0, 1], or if ``solver`` is not recognised. A position-dependent
        prior metric is caught later, at the first model evaluation, where the
        dtype and device of the positions are known.

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
        self._require_identity_space(space)
        self._solver = FixedPointSolver(
            solver, damping=damping, anderson_history=anderson_history,
            max_iter=fp_max_iter, tol=fp_tol)

        log_eps = math.log(step_size)
        adapter = NoAdaptation(init=log_eps)
        if adapt_step_size:
            adapter = Reinforce(sigma=adaptation_sigma, init=log_eps)
        super().__init__(None, space, requires_metric=True, num_steps=num_steps,
                         adapter=adapter, divergence_threshold=divergence_threshold,
                         trajectory_length=num_steps * step_size)

        self.constraint = constraint
        self._fp_tol = fp_tol
        self._A_prior = None       # constant prior block of G_M, built on demand

        self.register_diagnostic("residual_mean", lambda: self._residual_sum / max(self._step, 1))
        self.register_diagnostic("residual_max", lambda: self._residual_max)
        self.register_diagnostic("fp_iters_mean", lambda: self._fp_iters_sum / max(self._step, 1))
        self.register_diagnostic("fp_iters_max", lambda: self._fp_iters_max)
        self.register_logging("|r|", lambda: "{:.2e}".format(float(self._step_residual.max())))

    @staticmethod
    def _require_identity_space(space):
        """Reject a space whose z -> theta map is not the identity.

        The constraint reads the sampled position as theta and U carries no
        change-of-variables Jacobian, so a transforming space would silently
        sample the wrong density rather than fail.
        """
        probe = torch.zeros(2, space.d)
        probe[1] = 0.5
        mapped = space.map_to_constrained_vector(probe)
        identity = (torch.equal(mapped.mapped_point, probe)
                    and bool((mapped.jacobian_log_det.abs() < 1e-12).all()))
        if not identity:
            raise ValueError(
                f"ChartRATTLE needs a space that maps q to theta by the "
                f"identity, because the constraint and the prior are both "
                f"evaluated at the sampled position and U carries no transform "
                f"Jacobian. Got {type(space).__name__}.")

    def prior_metric(self, q):
        """``A_prior``, the constant prior block of ``G_M``, shape ``(n, n)``.

        The space's ``prior_metric_fn`` pushed forward to the sampled
        coordinates, or the identity when it has none. Cached after the first
        call.

        Raises
        ------
        ValueError
            If the space's prior metric varies with position. G_M is also the
            Hessian of the integrator's kinetic term, and a position-dependent
            one would need the midpoint form of that term, a different scheme
            from the one implemented here.
        """
        if self._A_prior is not None:
            return self._A_prior
        n = q.shape[-1]
        eye = torch.eye(n, dtype=q.dtype, device=q.device)
        if getattr(self.space, "prior_metric_fn", None) is None:
            self._A_prior = eye
            return eye

        probes = torch.tensor([0.0, 0.75, -1.5], dtype=q.dtype,
                              device=q.device).view(3, 1).expand(3, n)
        mats = [self._pushed_prior_metric(row.unsqueeze(0))[0] for row in probes]
        if not all(torch.allclose(m, mats[0], rtol=1e-9, atol=1e-12)
                   for m in mats[1:]):
            raise ValueError(
                "ChartRATTLE needs a position-independent prior metric, since "
                "G_M doubles as the Hessian of the integrator's kinetic term. "
                "The space's prior_metric_fn varies with position. Drop it to "
                "fall back on the identity, or supply a constant one.")
        self._A_prior = mats[0]
        return self._A_prior

    def _pushed_prior_metric(self, q):
        """The space's prior metric at ``q``, pushed to the sampled coordinates."""
        theta_map = self.space.map_to_constrained_vector(q)
        theta_full = self._free_to_full(theta_map.mapped_point)
        return self.space.push_forward_metric(
            self.space.prior_metric(theta_full), theta_map)

    # ---- model evaluation (the extension point) ---------------------------- #

    def evaluate_model(self, z_free, beta=None, grad=False):
        """The model at the position ``z_free`` (the chart coordinate ``q``), and
        with ``grad`` the force driving the integrator. Returns ``(U, metric,
        psi, W)``, or ``(U, metric, psi, W, grad_V)`` when ``grad`` is set.

        U : TemperedAffine
            Potential U(q) = −log p(q) + log|det B| + β·½‖ψ‖², with p(q) the
            space's prior. The target is e^{−U}.

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
        n = q.shape[-1]
        A_prior = self.prior_metric(q)

        with torch.enable_grad():
            psi, W, log_abs_det_B = self.constraint.psi_with_jvp(q)
            gram = W.transpose(-2, -1) @ W                 # (N, n, n) = WᵀW
            lik = 0.5 * (psi * psi).sum(-1)                # U_lik = ½‖ψ‖²
            base = -self.space.prior_log_prob_vector(q) + log_abs_det_B
            if grad:
                G = A_prior + broadcast_beta(beta, 2) * gram
                # Cholesky, not torch.logdet: same factorization the metric uses
                # for G_M⁻¹, and it rejects a non-SPD G instead of returning nan.
                chol = torch.linalg.cholesky(G)
                log_det_G = 2.0 * chol.diagonal(dim1=-2, dim2=-1).abs().log().sum(-1)
                V = base + beta * lik + 0.5 * log_det_G
                (grad_V,) = torch.autograd.grad(V.sum(), q)

        U = TemperedAffine(lik.detach(), base.detach(), beta)
        metric = TemperedMetric(gram.detach(),
                                A_prior.expand(q.shape[0], n, n), beta)
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
        """One RATTLE substep at ``step_size``, tracking the worst position-solve
        residual and iteration count. The solve is warm-started by the previous
        displacement, which changes only the iteration count, not the fixed point."""
        h = step_size.unsqueeze(-1)                        # (N, 1)
        beta_col = broadcast_beta(self.beta, 1)
        beta_mat = broadcast_beta(self.beta, 2)
        A_prior = self.prior_metric(state.q)

        rhs = h * state.p - 0.5 * h * h * state.grad_V
        q_init = state.q
        if state.dq is not None:
            q_init = state.q + state.dq
        q, iters, residual = _solve_rattle_step(
            self.constraint, state.q, state.psi, state.W, A_prior, beta_col,
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
