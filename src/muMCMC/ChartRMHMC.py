from typing import Callable
import math

import torch

from .HamiltonianSampler import HamiltonianSampler
from ._adapters import Reinforce, NoAdaptation
from .spaces import TemperedAffine, TemperedMetric, broadcast_beta
from .RMHMC import _hamiltonian
from ._solvers import FixedPointSolver

# =========================================================================== #
#  ChartRMHMC                                                                 #
#                                                                             #
#  Sections 1 and 2 are RMHMC read in a chart, which is what the name         #
#  records. Section 3 is the discretization, the part specific to this        #
#  sampler.                                                                   #
# =========================================================================== #
#  1. RMHMC on (θ, x), read in the chart (z, ε)                               #
#                                                                             #
#  The model is the diffeomorphism                                            #
#                                                                             #
#      Φ(z, ε) = (T(z), φ_{T(z)}(ε)) = (θ, x),                                #
#                                                                             #
#  with T the space's normal chart and φ_θ invertible in ε, under which the   #
#  prior is z ~ N(0, I_n) and ε ~ N(0, I_m). The chain runs on z, which the   #
#  code calls q. Fix the metric on the chart as                               #
#                                                                             #
#      g = diag(I_n, β I_m),   β the inverse temperature.                     #
#                                                                             #
#  The conditional. At fixed θ the map ε ↦ x is a diffeomorphism, so by       #
#  change of measure p(x | θ) = N(φ_θ⁻¹(x); 0, I_m) / |det ∂_ε φ_θ|. Write x  #
#  for the observed value and                                                 #
#                                                                             #
#      ψ(z) = φ_{T(z)}⁻¹(x),    B(z) = ∂_ε φ_{T(z)} at ε = ψ(z).              #
#                                                                             #
#  Tempering the conditional whole gives the target on z,                     #
#                                                                             #
#      U(z, β) = ½‖z‖² + (n/2)·log 2π                                         #
#                + β·[½‖ψ(z)‖² + (m/2)·log 2π + log|det B(z)|],               #
#                                                                             #
#  so β = 0 is the prior and β = 1 the posterior.                             #
#                                                                             #
#  The metric. Conditioning confines (z, ε) to the graph Γ = { (z, ψ(z)) },   #
#  charted by c(z) = (z, ψ(z)) with Dc(z) = [I_n ; −W(z)] and W(z) = −∂ψ/∂z.  #
#  Pulling g back along c gives the metric on z,                              #
#                                                                             #
#      G_M(z) = Dc(z)ᵀ g Dc(z) = M + β W(z)ᵀW(z),   M = I_n,                  #
#                                                                             #
#  the Gauss-Newton Hessian of U up to the volume term, and at β = 1 the      #
#  metric Γ inherits. Any SPD G_M is exact, so g is a free choice.            #
#                                                                             #
#  The Hamiltonian is then the usual one, with V its position-only part,      #
#                                                                             #
#      H(z, p) = U(z, β) + ½ pᵀG_M(z)⁻¹p + ½ log det G_M(z),                  #
#      V(z)    = U(z, β) + ½ log det G_M(z).                                  #
# =========================================================================== #
#  2. The dynamics as motion in (z, ε)                                        #
#                                                                             #
#  By the Legendre transform the Lagrangian of H is L(z, ż) = ½ żᵀG_M(z)ż −   #
#  V(z). Lifting z(t) to Γ by P(t) = c(z(t)) gives Ṗ = Dc(z)ż, hence          #
#                                                                             #
#      ½ żᵀG_M(z)ż = ½‖Ṗ‖²_g                                                  #
#                                                                             #
#  by the definition of G_M. The dynamics is that of a particle in the flat   #
#  space (ℝⁿ⁺ᵐ, g) confined to Γ, not a curved-space problem, and it is       #
#  discretized as one.                                                        #
# =========================================================================== #
#  3. The discretization                                                      #
#                                                                             #
#  In (ℝⁿ⁺ᵐ, g) free motion is a straight line, whose exact discrete          #
#  Lagrangian is the chord ‖P1 − P0‖²_g / (2h). Restricting it to Γ by        #
#  P_i = c(z_i) and taking the potential by the trapezoidal rule,             #
#                                                                             #
#      S_h(z0, z1) = ‖c(z1) − c(z0)‖²_g / (2h) − (h/2)[V(z0) + V(z1)]         #
#                  = [(z1−z0)ᵀM(z1−z0) + β‖ψ(z1)−ψ(z0)‖²] / (2h)              #
#                    − (h/2)[V(z0) + V(z1)].                                  #
#                                                                             #
#  This is the constrained variational integrator whose discrete              #
#  Euler-Lagrange equations are RATTLE. Γ is a graph, so substituting the     #
#  chart imposes the constraint exactly and no Lagrange multiplier appears.   #
#                                                                             #
#  Two properties make it an exact Metropolis proposal for e^{−H}. Define the #
#  step by the discrete Legendre transforms                                   #
#                                                                             #
#      p0 = −∂S_h/∂z0,    p1 = +∂S_h/∂z1.                                     #
#                                                                             #
#  Those two are exactly the statement dS_h = −p0 dz0 + p1 dz1, so            #
#                                                                             #
#      0 = d²S_h = −dp0 ∧ dz0 + dp1 ∧ dz1,                                    #
#                                                                             #
#  and the map preserves the symplectic form, hence volume. Both terms of S_h #
#  are symmetric in (z0, z1) and odd in h, so S_h(z0, z1) = −S_{−h}(z1, z0),  #
#  which is Φ_h⁻¹ = Φ_{−h}. H is even in p, so a momentum flip followed by a  #
#  step is an involution. Acceptance is min(1, e^{−ΔH}) with no correction    #
#  term.                                                                      #
#                                                                             #
#  The chord and not its linearization (z1−z0)ᵀG_M(z0)(z1−z0) is what buys    #
#  that symmetry. The two agree as h → 0.                                     #
#                                                                             #
#  Taking the two derivatives,                                                #
#                                                                             #
#      F(z1)  = M(z1−z0) − β W(z0)ᵀ(ψ(z1)−ψ(z0))                              #
#               − h p0 + (h²/2)∇V(z0) = 0,                                    #
#      p1     = (1/h)[M(z1−z0) − β W(z1)ᵀ(ψ(z1)−ψ(z0))] − (h/2)∇V(z1),        #
#      DF(z1) = M + β W(z0)ᵀW(z1),   so   DF(z0) = G_M(z0),                   #
#                                                                             #
#  the unknown being z1 alone, of dimension n whatever m is.                  #
#                                                                             #
#  Both properties hold up to the solve. Where F has several roots the        #
#  forward and reverse solves can pick different ones, an effect that         #
#  vanishes with h. A non-converged solve is rejected with +inf energy, and   #
#  no reverse-projection check is run: discarding a step whose reverse solve  #
#  lands elsewhere would restore reversibility exactly, at the cost of making #
#  the discarded set a barrier, and those are the steps across strong         #
#  nonlinearity the method exists to take.                                    #
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
        Conditioning value the constraint set is defined by, shared across
        chains.

    Attributes
    ----------
    jvp_needs_grad : bool
        Whether :meth:`psi_with_jvp` has to be called under grad mode. True for
        the default, which differentiates ``psi``. An override returning W from
        explicit or forward-mode Jacobians sets it False, which spares a graph
        build on every call, and the newton solver makes one call per iteration.

    Raises
    ------
    NotImplementedError
        From :meth:`psi` and :meth:`log_abs_det_B`, which a subclass supplies.
    """

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

    ``φ_q(ε) = μ(q) + L(q) ε`` with ``Σ(q) = L(q) L(q)ᵀ``, so
    ``ψ(q) = L(q)⁻¹(x − μ(q))`` and ``log|det B| = ½ log det Σ``.

    Parameters
    ----------
    mean : callable
        ``q -> μ(q)``, taking ``(N, n)`` to ``(N, m)`` and differentiable in
        ``q``.
    cov : callable
        ``q -> Σ(q)``, taking ``(N, n)`` to an SPD ``(N, m, m)`` and
        differentiable in ``q``.
    x : (m,)
        Observation.

    Raises
    ------
    RuntimeError
        From :meth:`psi` and :meth:`log_abs_det_B`, where ``cov`` returns a
        matrix that is not numerically SPD.
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
    """A constraint written on θ, read at the chart coordinate q with θ = T(q).

    The chart is elementwise, so its Jacobian is the diagonal dθ/dq and the
    chain rule on ψ(T(q)) is

        W_q = W_θ diag(dθ/dq).

    A constraint is therefore written once, on the variables, and never has to
    know which chart the chain runs in.
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

def _solve_position_step(constraint, q, psi, W, A_prior, beta_col, beta_mat,
                       chol_G, rhs, q_init, solver):
    """Solve the position equation of the step, derived at the top of this
    module, for the endpoint q1,

        F(q1) = M (q1 - q0) - beta W0^T(psi(q1) - psi0) - rhs = 0,

    with M the constant prior block, ``q, psi, W`` the q0 quantities and
    ``rhs = h p0 - (h^2/2) grad V(q0)``. Returns a SolveResult of shape (N, n),
    so the solve is over the hyperparameters alone.

    The Jacobian is DF(q1) = M + beta W0^T W(q1), which at q1 = q0 is the
    metric G_M(q0) itself. Two of the update rules follow from that.

    Picard is preconditioned by G_M(q0), whose Cholesky factor is ``chol_G``,
    which makes each iteration the Newton step with the Jacobian frozen at q0.
    Only psi(q^k) is evaluated in the loop. Newton instead re-evaluates
    DF(q1) at every iterate, which costs a tangent pass each time and is what
    ``solver.needs_jacobian`` selects between here.

    Damping and the update rule change how the iteration reaches the root and
    not which root it is, so neither affects the step the sampler takes.
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

class ChartRMHMCState:
    """Working state of one ChartRMHMC trajectory, batched over ``N`` chains.

    ``q``, ``p`` and ``U`` are kept across a transition and :meth:`reorder`
    keeps only ``q`` and ``p``. Every other field is valid for one trajectory,
    :meth:`ChartRMHMC.sample_momentum` evaluating it at the start of each.

    Attributes
    ----------
    q, p : (N, n)
        Position and momentum. The position is the chart coordinate, so ``q`` is
        the latent ``z`` of the space's transform and the variables are
        θ = T(q). A run reports θ, not ``q``.
    U : TemperedAffine or None
        Potential. ``U.value`` is the ``(N,)`` energy and ``U.lik`` the part a
        tempering driver swaps on.
    metric : TemperedMetric or None
        Chart metric G_M(q).
    psi, W : (N, m) and (N, m, n), or None
        The inverse map ψ(q) and W = −∂ψ/∂q.
    grad_V : (N, n) or None
        Force ∇V(q).
    dq : (N, n) or None
        Displacement of the last substep, from which :meth:`ChartRMHMC.integrate`
        starts the next position solve. None at the start of a trajectory.
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
        """Move each configuration to the batch position ``perm`` gives it.

        Only ``q`` and ``p`` are carried over. Every model quantity is a
        function of the temperature the configuration is now at, so keeping the
        old one would be wrong, and :meth:`ChartRMHMC.sample_momentum`
        re-evaluates all of them at the start of the next transition."""
        return ChartRMHMCState(self.q[perm],
                                None if self.p is None else self.p[perm])

    def select_accepted(self, accepted, other):
        """Per-chain choice between this endpoint where ``accepted`` and the
        start ``other`` where not, over ``q``, ``p`` and ``U``. The rest belongs
        to the trajectory that has just ended and is dropped."""
        pick = accepted.unsqueeze(-1)
        return ChartRMHMCState(
            torch.where(pick, self.q, other.q),
            torch.where(pick, self.p, other.p),
            self.U.select(accepted, other.U),
        )


class ChartRMHMC(HamiltonianSampler):
    """RMHMC for the hierarchical posterior p(θ | x), run on the latent q of
    the prior's normal chart, where observing x is a constraint that q charts.

    There is no ``model_fn`` here. The target is given by ``constraint`` and
    ``space`` together, and :meth:`evaluate_model` is overridden to build it,
    so it also returns the geometry the integrator needs and takes the chart
    coordinate rather than the variables.

    Parameters
    ----------
    constraint : ChartConstraint
        The reparameterization inverse ψ(q) = φ_q⁻¹(x) and its geometry.
    space : Space
        Space over the θ names. The chain runs in its normal chart, so the prior
        enters U as ½‖q‖² and the prior block M of G_M = M + β WᵀW is the
        identity. The constraint is evaluated at θ = T(q).
    step_size : float
        Integration step size at the start of warmup, and for the whole run
        when ``adapt_step_size`` is False. When adapting, start it small. The
        adaptation grows it from here, and a start above the largest step size
        the position solve converges at is not recovered from.
    num_steps : int
        Integrator substeps per transition.
    adapt_step_size : bool
        Adapt the step size during warmup with the REINFORCE adapter, on the
        cost of :meth:`adapt`.
    adaptation_sigma : float
        Scale of the perturbation the REINFORCE adapter explores with. Unused
        when ``adapt_step_size`` is False.
    fp_max_iter : int
        Iteration cap for one position solve.
    fp_tol : float
        Residual of F in max norm below which the position solve has converged.
    solver : str
        Position solver, ``"picard"``, ``"anderson"`` or ``"newton"``. All
        three are available because DF is a tangent pass here, unlike in
        :class:`~muMCMC.RMHMC.RMHMC`.
    anderson_history : int or None
        History length for the Anderson solver, unused by the others. None
        resolves per solve to ``n``.
    damping : float
        Under-relaxation factor in ``(0, 1]`` for any of the solvers. 1.0 is
        undamped.
    divergence_threshold : float
        A transition counts as a divergence when ``|delta_H|`` exceeds this or
        is not finite, the latter including every non-converged position solve.

    Raises
    ------
    ValueError
        If ``damping`` is outside ``(0, 1]``, if ``solver`` is not one of the
        three names, or if the space carries no prior and so no constant prior
        block M.

    Notes
    -----
    A proposal whose position solve did not converge is rejected. No
    reverse-projection check is run, which trades exact reversibility for the
    ability to cross strong nonlinearity. The end of section 3 of the
    derivation at the top of this module gives the argument.
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
        if space.prior_metric(space.as_transform.interior_point) is None:
            raise ValueError(
                "ChartRMHMC needs a space with a prior, whose chart supplies "
                "the constant prior block M of G_M = M + beta W^T W. A space "
                "with no prior has no M.")
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

    def evaluate_model(self, q_free, beta=None, grad=False):
        """The model and the geometry at the chart coordinate ``q``, and with
        ``grad`` the force the integrator follows.

        This takes the chart coordinate where
        :meth:`~muMCMC.MCMCSampler.MCMCSampler.evaluate_model` takes the
        variables, and returns four values where that one returns two. Map
        between the two coordinates with :meth:`to_position` and
        :meth:`to_variables`.

        Parameters
        ----------
        q_free : Tensor, shape (N, n)
            Chart coordinate.
        beta : float or Tensor, optional
            Inverse temperature. Default :attr:`beta`.
        grad : bool
            Whether to also return ``grad_V``.

        Returns
        -------
        U : TemperedAffine
            Potential

                U(q) = ½‖q‖² + (n/2)·log 2π
                       + β·[½‖ψ‖² + (m/2)·log 2π + log|det B|],

            of shape ``(N,)``. The first line is −log p(q) for the prior read
            in the space's normal chart, where it is exactly N(0, I), and the
            bracket is −log p(x | q). The target e^{−U} is therefore the prior
            at β = 0 and the posterior at β = 1. Read on the variables by
            :meth:`potential` it is normalized, so at β = 1 its integral there
            is the evidence.
        metric : TemperedMetric
            Metric G_M(q) = I + β WᵀW of shape ``(N, n, n)``, the pullback of
            the latent metric g = diag(I_n, β I_m) along the chart
            q ↦ (q, ψ(q)). At β = 1 that is the metric the constraint set
            inherits.
        psi : Tensor, shape (N, m)
            ψ(q) = φ_q⁻¹(x), the latent on the constraint set.
        W : Tensor, shape (N, m, n)
            W(q) = −∂ψ/∂q, which spans the tangent space along with the
            identity.
        grad_V : Tensor, shape (N, n)
            ∇V with V = U + ½ log det G_M, the potential the step follows.
            V differs from U by the metric volume term, which cancels in the
            target but not in the dynamics. Returned only when ``grad`` is
            True.

        Every returned tensor is detached."""
        beta = self.beta if beta is None else beta
        # Detached leaf regardless of ``grad``: W comes from a reverse pass
        # through psi, so the input has to carry a graph either way.
        q = q_free.detach().requires_grad_(True)
        # In the chart the prior is exactly N(0, I), so its potential is ½‖q‖²
        # and the prior block M of G_M is the identity. Both hold for any prior,
        # which is why the chain runs in the chart.
        A_prior = torch.eye(q.shape[-1], dtype=q.dtype, device=q.device).expand(
            q.shape[:-1] + (q.shape[-1], q.shape[-1]))

        with torch.enable_grad():
            psi, W, log_abs_det_B = self._chart.psi_with_jvp(q)
            gram = W.transpose(-2, -1) @ W                 # (N, n, n) = WᵀW
            # lik is the whole conditional -log p(x | q), volume term included.
            # |det B| comes from the change of variables eps -> x, so it belongs
            # to the conditional and tempers out with it. Left in base it would
            # survive to beta = 0 and leave p(q)/|det B| there, not the prior.
            lik = (0.5 * (psi * psi).sum(-1)
                   + 0.5 * psi.shape[-1] * math.log(2.0 * math.pi)
                   + log_abs_det_B)
            base = (0.5 * (q * q).sum(-1)
                    + 0.5 * q.shape[-1] * math.log(2.0 * math.pi))
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

    # ---- the chain's coordinates ------------------------------------------- #

    def potential(self, theta_free, beta=None):
        """``U(theta, beta)`` on the variables, the chart potential of
        :meth:`evaluate_model` pulled back along ``q = T⁻¹(theta)``,

            U(theta) = U(q) − log|det dq/dtheta|
                     = −log p(theta) + β·[½‖ψ‖² + (m/2)·log 2π + log|det B|].

        At ``beta = 1`` that is −log[p(theta) p(x | theta)], so ``Z_1`` is the
        evidence p(x). The coarea factor and the change of variables on
        ``x | theta`` are the same log|det B|, so no measure factor is left
        over.
        """
        chart = self.space.as_transform.inverse(theta_free)
        return (self.evaluate_model(chart.mapped_point, beta)[0].value
                - chart.jacobian_log_det)

    def to_position(self, theta_free):
        """The chart coordinate q at the variables theta, so q = T⁻¹(theta).
        The chain runs in the chart, where the prior block M of G_M is constant,
        which is the one thing the step needs and the variables cannot give."""
        return self.space.as_transform.inverse(theta_free).mapped_point

    def to_variables(self, q_free):
        """The variables theta = T(q) at the chart coordinate q, which is what a
        run reports."""
        return self.space.as_transform.forward(q_free).mapped_point

    # ---- integrator hooks -------------------------------------------------- #

    def build_initial_state(self, q):
        """The initial :class:`ChartRMHMCState` at ``q``, with the potential,
        the metric, the geometry and the force evaluated. The momentum needs
        the metric, so it is drawn by :meth:`sample_momentum` rather than
        here."""
        U, metric, psi, W, grad_V = self.evaluate_model(q, grad=True)
        return ChartRMHMCState(q, None, U, metric, psi, W, grad_V, None)

    def sample_momentum(self, state):
        """Evaluate the potential, the metric, the geometry and the force at
        ``q``, draw the momentum ``p ~ N(0, G_M(q))``, and zero the residual and
        iteration count of the transition about to start.

        This is the only place the trajectory's model quantities are set. A PT
        swap can relabel ``q`` to another temperature slot between transitions,
        so nothing that depends on the model may be kept across a transition."""
        zeros = torch.zeros(state.q.shape[0], dtype=state.q.dtype, device=state.q.device)
        self._step_residual = zeros.clone()
        self._step_iters = zeros.clone()
        (state.U, state.metric, state.psi, state.W,
         state.grad_V) = self.evaluate_model(state.q, grad=True)
        state.p = state.metric.sample_momentum()
        state.dq = None
        return state

    def integrate(self, state, step_size):
        """One substep at ``step_size`` = h, solving

            F(q1) = M(q1 − q0) − β W0ᵀ(ψ(q1) − ψ0) − h p0 + (h²/2) ∇V(q0) = 0

        for the new position and taking the momentum from

            p1 = (1/h)[M(q1 − q0) − β W1ᵀ(ψ1 − ψ0)] − (h/2) ∇V(q1),

        with V = U + ½ log det G_M and M the prior block of G_M. Section 3 of the
        derivation at the top of this module derives both equations. This carries
        the worst position-solve residual and iteration count of the transition
        so far. The solve starts from the previous substep's displacement, which
        changes the iteration count and not the fixed point."""
        h = step_size.unsqueeze(-1)                        # (N, 1)
        beta_col = broadcast_beta(self.beta, 1)
        beta_mat = broadcast_beta(self.beta, 2)
        A_prior = state.metric.base                        # M, off the metric

        rhs = h * state.p - 0.5 * h * h * state.grad_V
        q_init = state.q
        if state.dq is not None:
            q_init = state.q + state.dq
        q, iters, residual = _solve_position_step(
            self._chart, state.q, state.psi, state.W, A_prior, beta_col,
            beta_mat, state.metric.L, rhs, q_init, self._solver)

        U, metric, psi, W, grad_V = self.evaluate_model(q, grad=True)

        corr = (W.transpose(-2, -1) @ (psi - state.psi).unsqueeze(-1)).squeeze(-1)
        drift = (A_prior @ (q - state.q).unsqueeze(-1)).squeeze(-1)
        p = (drift - beta_col * corr) / h - 0.5 * h * grad_V

        self._step_residual = torch.maximum(self._step_residual, residual)
        self._step_iters = torch.maximum(self._step_iters, iters.to(h.dtype))
        return ChartRMHMCState(q, p, U, metric, psi, W, grad_V,
                                dq=(q - state.q))

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) − H(old)``, with no Jacobian correction because
        the step is symplectic.

        Where the trajectory's worst residual is above ``fp_tol`` the result is
        ``+inf``. Such a proposal is not the endpoint of the map the Metropolis
        test assumes, so it is rejected however small its energy change came
        out."""
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
        """Update the step size by REINFORCE on a cost built from this
        transition's energy error and its worst solver residual and iteration
        count. ``accept_prob`` is unused.

        The cost weights ``|delta_H|`` by 4 rather than by 1 as
        :meth:`~muMCMC.RMHMC.RMHMC.adapt` does, so the energy error dominates
        sooner. Started small, the step size then settles below the size at
        which the position solve stops converging."""
        floor = 1.0e-3
        energy_weight = 4.0
        num_iters = self._step_iters
        solver_penalty = torch.exp(-self._step_residual / self.step_size)
        delta_H_penalty = torch.exp(-energy_weight * delta_H.abs())
        efficiency = solver_penalty * delta_H_penalty * self.step_size / num_iters
        f_t = -0.5 * torch.log(efficiency + floor) / abs(math.log(floor))
        self._step_size_adapter.update(f_t)

    def reset_extra_diagnostics(self):
        """Zero the run-level solver summaries, and the per-transition residual
        and iteration count as well, so :meth:`logging` reads them before any
        substep has run. :meth:`sample_momentum` zeroes those two again at the
        start of every transition."""
        N = self.step_size.shape[0]
        zeros = torch.zeros(N, dtype=self.step_size.dtype, device=self.step_size.device)
        self._residual_sum = zeros.clone()
        self._residual_max = zeros.clone()
        self._fp_iters_sum = zeros.clone()
        self._fp_iters_max = zeros.clone()
        self._step_residual = zeros.clone()
        self._step_iters = zeros.clone()
