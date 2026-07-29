from typing import Callable, Tuple
import math

import torch

from .HamiltonianSampler import HamiltonianSampler
from .adapters import Reinforce, NoAdaptation
from .RMHMC import _PicardUpdate, _AndersonUpdate

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
#  Constrained dynamics on M with potential U leaves e^{−U} σ_M invariant.     #
#  The co-area formula gives the law of z given X = x as                       #
#                                                                              #
#      e^{−‖z‖²/2} det Λ^{−1/2},      Λ = ∇g ∇gᵀ = A Aᵀ + B Bᵀ,                #
#      A = ∂_η φ,   B = ∂_ε φ,                                                  #
#                                                                              #
#  so U(z) = ½‖z‖² + ½ log det Λ, and the η-marginal of the constrained        #
#  chain is p(θ | x). The ½ log det Λ term is the co-area Jacobian of the      #
#  conditioning. It is owned here, not by the user.                            #
#                                                                              #
# =========================================================================== #
#                                                                              #
#  Chart                                                                       #
#                                                                              #
#  M is a graph over η. With ψ(η) = φ_η⁻¹(x) and Z(η) = (η, ψ(η)),             #
#                                                                              #
#      T(η) = ∂Z/∂η = [ I ; −W ],    W = −∂ψ/∂η = B⁻¹A,                        #
#      G_M(η) = Tᵀ T = I + Wᵀ W.                                               #
#                                                                              #
#  The chart potential is                                                      #
#                                                                              #
#      V(η) = ½‖η‖² + ½‖ψ‖² + log|det B| + ½ log det G_M,                      #
#                                                                              #
#  and RATTLE in the chart reads e^{−V} √det G_M dη ∝ p(θ | x) dη. The         #
#  Hamiltonian is H(η, π) = V(η) + ½ πᵀ G_M(η)⁻¹ π with momentum               #
#  π ~ N(0, G_M(η)), whose Gaussian normalizer supplies the √det G_M.          #
#                                                                              #
# =========================================================================== #
#                                                                              #
#  Step  (η0, π0) -> (η1, π1)                                                  #
#                                                                              #
#  Position solve, n equations in η1, the RATTLE orthogonality of the move:    #
#                                                                              #
#      F(η1) = (η1 − η0) − W0ᵀ(ψ(η1) − ε0) − h π0 + (h²/2) ∇V(η0) = 0,         #
#      DF(η1) = I + W0ᵀ W1,    DF(η0) = G_M(η0).                              #
#                                                                              #
#  Frozen-Jacobian iteration, one Cholesky of G_M(η0) reused across steps:     #
#                                                                              #
#      η^{k+1} = η^k − G_M(η0)⁻¹ F(η^k).                                       #
#                                                                              #
#  Only ψ(η^k) is evaluated in the loop. The updater is pluggable: Picard      #
#  and Anderson both drive the preconditioned residual G_M(η0)⁻¹ F to zero.    #
#  Momentum is explicit,                                                        #
#                                                                              #
#      π1 = (1/h)[(η1 − η0) − W1ᵀ(ε1 − ε0)] − (h/2) ∇V(η1).                    #
#                                                                              #
#  The scheme is self-adjoint, so reversible up to the solve. v1 runs no       #
#  reverse-projection check and rejects a failed solve (+inf energy), as       #
#  RMHMC does.                                                                  #
#                                                                              #
# =========================================================================== #
#                                                                              #
#  Constraint interface                                                        #
#                                                                              #
#      psi(η)           -> ε                    inner-loop workhorse           #
#      log_abs_det_B(η) -> log|det B|           = ½ log det Σ for a scale       #
#                                                                              #
#  W = −∂ψ/∂η follows from one reverse pass per latent (classic autograd, no   #
#  vmap). A subclass with explicit Jacobians overrides psi_with_jvp to return  #
#  (ε, W, log|det B|) directly. ∇V is one autograd.grad(V.sum(), η), the same  #
#  machinery as RMHMC's dH/dq, and the sole gradient in the scheme.            #
#                                                                              #
# =========================================================================== #
#                                                                              #
#  Tempering                                                                   #
#                                                                              #
#  Inverse temperature β lives on the constraint and softens the observation.  #
#  In the scale family Σ -> Σ/β, so ψ_β = √β ψ₁ and the chart target is         #
#                                                                              #
#      e^{−½‖η‖² − ½ log det Σ}·e^{−β U_lik},      U_lik = ½‖ψ₁‖² = ½‖ψ_β‖²/β,  #
#                                                                              #
#  base·e^{−β U_lik} with a temperature-free U_lik, so parallel tempering      #
#  swaps are valid. The sampler pushes its per-replica β into the constraint   #
#  and exposes U_lik on the state. Note β = 0 is the Σ-weighted prior, not the #
#  N(0, I) prior (the ½ log det Σ term stays in the base).                     #
#                                                                              #
# =========================================================================== #


# ---- Chart geometry ------------------------------------------------------ #

class ChartLocal:
    """Per-endpoint bundle from one endpoint evaluation, batched over chains.

    Attributes
    ----------
    eps : (N, m)
        ψ(η), the latent on M.
    W : (N, m, n)
        −∂ψ/∂η. The induced metric is G_M = I + Wᵀ W.
    chol_G : (N, n, n)
        Lower Cholesky of G_M(η), the position-solve preconditioner.
    V : (N,)
        Chart potential V(η).
    """

    def __init__(self, eps: torch.Tensor, W: torch.Tensor,
                 chol_G: torch.Tensor, V: torch.Tensor):
        self.eps = eps
        self.W = W
        self.chol_G = chol_G
        self.V = V

    def select(self, mask: torch.Tensor, other: "ChartLocal") -> "ChartLocal":
        """This bundle where ``mask`` is True, ``other`` where False, per chain."""
        m1 = mask[..., None]
        m2 = mask[..., None, None]
        return ChartLocal(
            torch.where(m1, self.eps, other.eps),
            torch.where(m2, self.W, other.W),
            torch.where(m2, self.chol_G, other.chol_G),
            torch.where(mask, self.V, other.V),
        )

    def reorder(self, perm: torch.Tensor) -> "ChartLocal":
        return ChartLocal(self.eps[perm], self.W[perm], self.chol_G[perm], self.V[perm])


class ChartConstraint:
    """Constraint M = {(η, ε) : φ_η(ε) = x} exposed through the inverse map.

    A subclass supplies the batched inverse ``psi(η)`` = φ_η⁻¹(x) (shape
    ``(N, n)`` to ``(N, m)``) and ``log_abs_det_B(η)`` = log|det ∂_ε φ|. The
    base derives W = −∂ψ/∂η by classic autograd, the metric G_M and its
    Cholesky, and ∇V. Override ``psi_with_jvp`` to return (ε, W, log|det B|)
    from explicit Jacobians and skip the autograd of W.

    Parameters
    ----------
    x : (m,)
        Conditioning value the manifold is defined by, shared across chains.
    beta : float or (N,) Tensor
        Inverse temperature, per chain. A subclass softens the constraint by
        ``beta`` (the scale family divides Sigma by ``beta``), so ``beta = 1`` is
        the posterior and ``beta`` -> 0 flattens the data fit. Set by the sampler
        from its ``beta`` (parallel tempering fills it per replica).
    """

    def __init__(self, x: torch.Tensor, beta=1.0):
        self.x = x
        self.beta = beta

    # ---- subclass hooks ---------------------------------------------------- #

    def psi(self, eta: torch.Tensor) -> torch.Tensor:
        """ε = φ_η⁻¹(x) at the bound x, batched. Inner-loop workhorse."""
        raise NotImplementedError

    def log_abs_det_B(self, eta: torch.Tensor) -> torch.Tensor:
        """log|det ∂_ε φ| at η, batched ``(N,)``."""
        raise NotImplementedError

    # ---- derived ----------------------------------------------------------- #

    def psi_with_jvp(self, eta: torch.Tensor):
        """(ε, W, log|det B|) at η, W = −∂ψ/∂η computed by one reverse pass per
        latent. ``eta`` must require grad. Override with explicit Jacobians."""
        eps = self.psi(eta)
        m = eps.shape[-1]
        cols = [torch.autograd.grad(eps[:, i].sum(), eta, retain_graph=True,
                                    create_graph=True)[0]
                for i in range(m)]
        W = -torch.stack(cols, dim=1)                      # (N, m, n)
        return eps, W, self.log_abs_det_B(eta)

    def endpoint(self, eta: torch.Tensor) -> Tuple[ChartLocal, torch.Tensor]:
        """One evaluation at η returning ``(ChartLocal, ∇V)``.

        ∇V = ∂V/∂η by a single ``autograd.grad(V.sum(), η)``, as RMHMC takes
        dH/dq. Everything is detached into the bundle, so no graph is pinned
        across steps."""
        eta = eta.detach().requires_grad_(True)
        with torch.enable_grad():
            eps, W, log_abs_det_B = self.psi_with_jvp(eta)
            n = eta.shape[-1]
            eye = torch.eye(n, dtype=eta.dtype, device=eta.device)
            G = eye + W.transpose(-2, -1) @ W
            V = (0.5 * (eta * eta).sum(-1) + 0.5 * (eps * eps).sum(-1)
                 + log_abs_det_B + 0.5 * torch.logdet(G))
            (grad_V,) = torch.autograd.grad(V.sum(), eta)
        chol_G = torch.linalg.cholesky_ex(G.detach()).L
        local = ChartLocal(eps.detach(), W.detach(), chol_G, V.detach())
        return local, grad_V.detach()


class LocationScaleChart(ChartConstraint):
    """Conditionally-Gaussian layer x | η ~ N(μ(η), Σ(η)).

    φ_η(ε) = μ(η) + L(η) ε with Σ(η) = L(η) L(η)ᵀ, so ψ(η) = L(η)⁻¹(x − μ(η))
    and log|det B| = ½ log det Σ. ``mean`` and ``cov`` are batched callables
    (η shape ``(N, n)`` to ``(N, m)`` and ``(N, m, m)`` SPD), differentiable in
    η. W is taken by the autograd default. ``cholesky_ex`` returns a non-finite
    factor rather than raising on a diverged η, so a bad chain is rejected
    downstream instead of crashing the batch.

    Parameters
    ----------
    mean : callable
        η -> μ(η).
    cov : callable
        η -> Σ(η), SPD.
    x : (m,)
        Observation.
    """

    def __init__(self, mean: Callable, cov: Callable, x: torch.Tensor, beta=1.0):
        super().__init__(x, beta)
        self.mean = mean
        self.cov = cov

    def _factor(self, eta):
        # Tempering softens the observation: Sigma_beta = Sigma / beta.
        b = self.beta
        Sigma = self.cov(eta)
        if torch.is_tensor(b) and b.ndim > 0:
            Sigma = Sigma / b.reshape(-1, 1, 1)
        elif b != 1.0:
            Sigma = Sigma / b
        return self.mean(eta), torch.linalg.cholesky_ex(Sigma).L

    def psi(self, eta):
        mu, L = self._factor(eta)
        return torch.linalg.solve_triangular(
            L, (self.x - mu).unsqueeze(-1), upper=False).squeeze(-1)

    def log_abs_det_B(self, eta):
        _, L = self._factor(eta)
        return torch.log(L.diagonal(dim1=-2, dim2=-1).abs()).sum(-1)


# ---- Position solve ------------------------------------------------------ #

def _solve_position(constraint, eta0, eps0, W0, chol_G0, rhs, eta_init,
                    solver, max_iter, tol):
    """Solve F(η) = (η − η0) − W0ᵀ(ψ(η) − ε0) − rhs = 0, preconditioned by
    G_M(η0), batched over chains. ``rhs`` folds the momentum half step,
    ``rhs = h π0 − (h²/2) ∇V(η0)``. Each chain iterates until ‖F‖∞ < tol or it
    blows up (non-finite, or ‖F‖∞ over 10x its start), frozen thereafter, up to
    ``max_iter``. ``eta_init`` seeds the iterate. Returns (η1, iters, residual)."""
    N, n = eta0.shape
    W0t = W0.transpose(-2, -1)                             # (N, n, m)

    def residual_fn(eta):
        eps = constraint.psi(eta)                          # (N, m)
        F = (eta - eta0) \
            - (W0t @ (eps - eps0).unsqueeze(-1)).squeeze(-1) \
            - rhs
        pre = torch.cholesky_solve(F.unsqueeze(-1), chol_G0).squeeze(-1)   # G⁻¹ F
        return F, pre

    updater = solver.new(n)
    eta = eta_init.clone()
    with torch.no_grad():
        F, pre = residual_fn(eta)
        F_init = F.abs().amax(-1)                           # (N,)

        done = torch.zeros(N, dtype=torch.bool, device=eta0.device)
        iters = torch.full((N,), max_iter, dtype=torch.long, device=eta0.device)
        residual = F_init.clone()

        for i in range(1, max_iter + 1):
            eta_next = updater.propose(eta, pre)           # Picard or Anderson
            F_next, pre_next = residual_fn(eta_next)
            F_norm = F_next.abs().amax(-1)

            keep = done[..., None]
            eta = torch.where(keep, eta, eta_next)
            pre = torch.where(keep, pre, pre_next)
            residual = torch.where(done, residual, F_norm)

            live = ~done
            blew = live & (~torch.isfinite(F_norm)
                           | ((F_norm > 10.0 * F_init) & (F_norm > tol)))
            done = done | blew

            live = ~done
            conv = live & (residual < tol)
            iters = torch.where(conv, torch.full_like(iters, i), iters)
            done = done | conv

            if bool(done.all()):
                break

    return eta.detach(), iters, residual.detach()


# ---- Chart energy -------------------------------------------------------- #

def _chart_hamiltonian(local: ChartLocal, pi: torch.Tensor) -> torch.Tensor:
    """H = V(η) + ½ πᵀ G_M(η)⁻¹ π, with G_M via its Cholesky."""
    Ginv_pi = torch.cholesky_solve(pi.unsqueeze(-1), local.chol_G).squeeze(-1)
    return local.V + 0.5 * (pi * Ginv_pi).sum(-1)


# ---- Chain state --------------------------------------------------------- #

class _LikPotential:
    """Temperature-free likelihood potential U_lik = ½‖ψ₁‖², carried on the state
    for parallel tempering. Exposes ``lik`` and travels under reorder / select.
    ``base`` is absent (the sampler forms energies directly), present only so the
    PT swap reads ``U.lik``."""

    __slots__ = ("lik",)

    def __init__(self, lik):
        self.lik = lik

    def reorder(self, perm):
        return _LikPotential(self.lik[perm])

    def select(self, mask, other):
        return _LikPotential(torch.where(mask, self.lik, other.lik))


class ChartRATTLEState:
    """Working state of one ChartRATTLE trajectory, batched over ``(N,)`` chains.

    Attributes
    ----------
    q : (N, n)
        Chart position η. Named ``q`` for the driver, which reads it as the
        sample and maps it through the identity space to θ.
    pi : (N, n) or None
        Chart momentum. Drawn by :meth:`ChartRATTLE.sample_momentum`.
    local : ChartLocal
        Endpoint bundle at η.
    grad_V : (N, n)
        ∇V(η).
    deta : (N, n) or None
        Last position displacement, used to warm-start the next substep's solve.
        None at a trajectory start.
    """

    def __init__(self, q, pi=None, local=None, grad_V=None, deta=None, U=None):
        self.q = q
        self.pi = pi
        self.local = local
        self.grad_V = grad_V
        self.deta = deta
        self.U = U                                     # _LikPotential, for PT swaps

    def reorder(self, perm):
        # A swap relabels a config to a new temperature slot. q, U (the
        # temperature-free likelihood) and the momentum permute directly. The
        # bundle and grad_V are retempered by the next sample_momentum, which
        # re-evaluates the endpoint at the slot's beta.
        return ChartRATTLEState(
            q=self.q[perm],
            pi=None if self.pi is None else self.pi[perm],
            local=None if self.local is None else self.local.reorder(perm),
            grad_V=None if self.grad_V is None else self.grad_V[perm],
            deta=None if self.deta is None else self.deta[perm],
            U=None if self.U is None else self.U.reorder(perm),
        )

    def select_accepted(self, accepted, other):
        """Per-chain choice between this endpoint (where ``accepted``) and the
        start ``other``. ``deta`` is dropped, so the next trajectory starts from
        the trivial guess."""
        pick = accepted.unsqueeze(-1)
        return ChartRATTLEState(
            torch.where(pick, self.q, other.q),
            torch.where(pick, self.pi, other.pi),
            self.local.select(accepted, other.local),
            torch.where(pick, self.grad_V, other.grad_V),
            deta=None,
            U=None if self.U is None else self.U.select(accepted, other.U),
        )


# =========================================================================== #
#                                                                              #
#  ChartRATTLE sampler                                                         #
#                                                                              #
#  Runs in the η chart. The N(0, I) top-level prior is baked into V, so the    #
#  space is the identity UnconstrainedSpace over the θ names and the driver     #
#  reads q = η off as θ. The model is a ChartConstraint, so evaluate_model of   #
#  the MCMCSampler base is unused.                                             #
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
        N(0, I) prior is in V).
    step_size : float
        Integration step size, required (no default). When adapting, start it
        small: the step is grown from here, so a too-large start begins above the
        solver-convergence cliff and cannot recover.
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
    A failed position solve (max residual over ``fp_tol``, or non-finite) is
    rejected with +inf energy, so a non-reversible move never enters the chain.
    No reverse-projection check is run in v1.
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
        adapter = (Reinforce(sigma=adaptation_sigma, init=log_eps)
                   if adapt_step_size else NoAdaptation(init=log_eps))
        super().__init__(None, space, requires_metric=False, num_steps=num_steps,
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

    # ---- integrator hooks -------------------------------------------------- #

    def _u_lik(self, eps):
        """Temperature-free likelihood potential U_lik = ½‖ψ₁‖² = ½‖ψ_β‖² / β,
        the parallel-tempering swap statistic (``ψ_β = √β ψ₁`` in the scale
        family)."""
        b = self.constraint.beta
        b = b.reshape(-1) if (torch.is_tensor(b) and b.ndim > 0) else b
        return 0.5 * (eps * eps).sum(-1) / b

    def build_initial_state(self, q):
        """Evaluate the constraint at ``q`` = η and return the initial state. The
        sampler's per-replica ``beta`` is pushed into the constraint here."""
        self.constraint.beta = self.beta
        z = torch.zeros(q.shape[0], dtype=q.dtype, device=q.device)
        self._step_residual = z.clone()
        self._step_iters = z.clone()
        local, grad_V = self.constraint.endpoint(q)
        return ChartRATTLEState(q, None, local, grad_V, None,
                                U=_LikPotential(self._u_lik(local.eps)))

    def sample_momentum(self, state):
        """Re-evaluate the endpoint at ``q`` (retempering it to the current
        replica ``beta`` after a swap), draw the chart momentum π ~ N(0, G_M(η)),
        and reset the per-transition solver scratch and warm-start displacement."""
        self.constraint.beta = self.beta
        state.local, state.grad_V = self.constraint.endpoint(state.q)
        state.U = _LikPotential(self._u_lik(state.local.eps))
        N, n = state.q.shape
        z = torch.zeros(N, dtype=state.q.dtype, device=state.q.device)
        self._step_residual = z.clone()
        self._step_iters = z.clone()
        xi = torch.randn(N, n, dtype=state.q.dtype, device=state.q.device)
        state.pi = (state.local.chol_G @ xi.unsqueeze(-1)).squeeze(-1).detach()
        state.deta = None
        return state

    def integrate(self, state, step_size):
        """One RATTLE substep at ``step_size``, tracking the worst position-solve
        residual and iteration count over the transition. The solve is
        warm-started by the previous substep's displacement, which changes only
        the iteration count and not the fixed point."""
        h = step_size.unsqueeze(-1)                        # (N, 1)
        eta0 = state.q
        L0 = state.local
        g0 = state.grad_V
        pi0 = state.pi
        eps0, W0, chol_G0 = L0.eps, L0.W, L0.chol_G

        rhs = h * pi0 - 0.5 * h * h * g0
        eta_init = eta0 if state.deta is None else eta0 + state.deta
        eta1, iters, residual = _solve_position(
            self.constraint, eta0, eps0, W0, chol_G0, rhs, eta_init,
            self._solver, self._fp_max_iter, self._fp_tol)

        # A diverged chain (non-finite η1, residual over tol) is rejected by
        # acceptance_delta. Fall its position back to η0 so the endpoint eval
        # stays finite and never crashes the batch.
        finite = torch.isfinite(eta1).all(-1, keepdim=True)
        eta1 = torch.where(finite, eta1, eta0)
        L1, g1 = self.constraint.endpoint(eta1)
        eps1, W1 = L1.eps, L1.W
        W1t_de = (W1.transpose(-2, -1) @ (eps1 - eps0).unsqueeze(-1)).squeeze(-1)
        pi1 = ((eta1 - eta0) - W1t_de) / h - 0.5 * h * g1

        self._step_residual = torch.maximum(self._step_residual, residual)
        self._step_iters = torch.maximum(self._step_iters, iters.to(h.dtype))
        return ChartRATTLEState(eta1, pi1, L1, g1, deta=(eta1 - eta0),
                                U=_LikPotential(self._u_lik(eps1)))

    def acceptance_delta(self, new, old):
        """``delta_H = H(new) − H(old)``, forced to +inf where the position solve
        did not converge (max residual over ``fp_tol``, or non-finite)."""
        H_new = _chart_hamiltonian(new.local, new.pi)
        H_old = _chart_hamiltonian(old.local, old.pi)
        delta = H_new - H_old

        self._residual_sum = self._residual_sum + self._step_residual
        self._residual_max = torch.maximum(self._residual_max, self._step_residual)
        self._fp_iters_sum = self._fp_iters_sum + self._step_iters
        self._fp_iters_max = torch.maximum(self._fp_iters_max, self._step_iters)

        # ``~(res <= tol)`` also flags a non-finite residual as failed.
        solve_failed = ~(self._step_residual <= self._fp_tol)
        return torch.where(solve_failed, delta.new_full((), float("inf")), delta)

    def adapt(self, accept_prob, delta_H):
        """REINFORCE step-size adaptation from this transition's energy error and
        worst solver residual and iteration count. The ``exp(-w|delta_H|)`` term
        brakes the step as the energy error grows, so from a small start the step
        settles below the solver-convergence cliff rather than running away.

        The energy weight ``w`` is heavier than RMHMC's ``w = 1`` because the
        chart solve stays cheap (Anderson keeps the iteration count low even as
        the step grows), so the throughput reward would otherwise push the step
        past the divergence edge before the brake engages. ``w = 4`` settles the
        funnel near 0.99 acceptance with no divergences.

        A diverged trajectory leaves a non-finite residual (the step is already
        rejected). Charging it zero efficiency, so maximum finite cost, points the
        step down instead of poisoning the adapter with a non-finite update."""
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
        """Zero the run-level solver summaries. The per-transition scratch is
        reset each transition in :meth:`sample_momentum`."""
        N = self.step_size.shape[0]
        z = torch.zeros(N, dtype=self.step_size.dtype, device=self.step_size.device)
        self._residual_sum = z.clone()
        self._residual_max = z.clone()
        self._fp_iters_sum = z.clone()
        self._fp_iters_max = z.clone()
