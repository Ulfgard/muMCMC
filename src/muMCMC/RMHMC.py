from typing import Callable, Tuple
from collections import OrderedDict

import torch
import math

from .BaseSampler import BaseSampler
from .spaces import TransformedMetric
from .adapters import REINFORCEAdapter

# =========================================================================== #
#                                                                             #
#  RMHMC helpers  (implicit midpoint integrator)                              #
#                                                                             #
#  The implicit-midpoint step solves a per-chain fixed-point equation         #
#  z = F(z).  The *update rule* that drives that solve is pluggable: the      #
#  default Picard iteration (z_{k+1} = F(z_k)) and Anderson acceleration      #
#  (z_{k+1} = a residual-minimising combination of the last m iterates) both  #
#  attack the same F and share the same convergence / freeze / blow-up        #
#  bookkeeping in ``_implicit_midpoint_step`` -- only the proposal differs.    #
#                                                                             #
# =========================================================================== #

# ---- Hamiltonian --------------------------------------------------------- #

def _hamiltonian(
    q: torch.Tensor,
    p: torch.Tensor,
    U: torch.Tensor,
    metric: TransformedMetric,
) -> torch.Tensor:
    """
    H(q, p) = U + ½ pᵀ G⁻¹(q) p + ½ log det G(q).

    All position-dependent quantities (U, metric) must be pre-evaluated
    at q.  The argument q is included for interface clarity (mirroring
    the mathematical H(q,p)) but is not used in the computation.
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
    Evaluate the I.M.(a) fixed-point map F(z_k) and return (F_q, F_p).

    Implements algorithm I.M.(a) from Brofos & Lederman (2021): the unknown
    is the endpoint (q_k, p_k), with the midpoint derived from it:

        q_mid = ½(q + q_k)
        p_mid = ½(p + p_k)
        F_q   = q + (ε/2) G⁻¹(q_mid) (p + p_k)
        F_p   = p − ε ∂H/∂q|_{q_mid, p_mid}

    The Picard fixed-point iteration only needs the *values* of F_q, F_p
    (no Jacobian), so the sole gradient required is the first-order
    ∂H/∂q at the midpoint.  We therefore compute that gradient in a local
    enable_grad block on a fresh leaf q_mid (create_graph defaults to
    False) and build F_q, F_p under no_grad.  Nothing carries an autograd
    graph out of this function -- there is no second-order graph and no
    cross-iteration accumulation.
    """
    q_mid = (0.5 * (q + q_k)).detach().requires_grad_(True)   # fresh leaf
    p_mid = 0.5 * (p + p_k)

    with torch.enable_grad():
        U, metric = evaluate_model(q_mid)
        H = _hamiltonian(q_mid, p_mid, U, metric)
        # H has shape (N,); each H[i] depends only on q_mid[i] (no cross-chain
        # coupling), so grad of the sum is the per-chain gradient with no mixing.
        (dHdq,) = torch.autograd.grad(H.sum(), q_mid)         # first-order only

    # eps is the per-chain step size, shape (N,); RMHMC.init() tensorizes
    # step_size per chain, so eps is always a vector here -- we just add the
    # trailing axis to broadcast against the (N, d) updates, with no scalar
    # special-casing inside the hot math.
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
# ``propose`` is always called with the *committed* (post-freeze) (z_k, r_k):
# a frozen chain feeds an unchanged z_k in, so any history it accumulates has
# zero differences for that chain and contributes nothing -- the freeze mask in
# ``_implicit_midpoint_step`` stays authoritative regardless of the updater.

class _PicardUpdate:
    """Relaxed Picard iteration: z_{k+1} = z_k − β r_k = (1−β) z_k + β F(z_k).

    Stateless.  ``beta`` is the under-relaxation / damping factor; β = 1 is the
    plain iteration (the cold-started first iterate is then already an explicit
    Euler step, so Picard needs no separate predictor to warm-start).  β < 1
    trades speed for stability: it pulls the iteration eigenvalues (β − 1) + β λ
    toward the point (1 − β) on the real axis, which is what tames the (near-)
    imaginary spectrum of the implicit-midpoint map.

    An instance is both the configured solver (held by ``RMHMC``) and, being
    stateless, its own per-solve working copy -- ``new`` just returns ``self``.

    Parameters
    ----------
    beta : float
        Damping factor in (0, 1]; 1.0 is the undamped iteration.
    """

    def __init__(self, beta=1.0):
        self.beta = float(beta)

    def new(self, d):
        """Fresh working updater for a solve over ``d``-dim positions; stateless,
        so ``self`` is reused."""
        return self

    def propose(self, z, r):
        return z - self.beta * r


class _AndersonUpdate:
    """Anderson(m) acceleration of the same fixed-point map (Walker & Ni 2011,
    Type-II, damping ``beta``).

    With f_k = F(z_k) − z_k = −r_k and the last ``m`` iterate/residual
    differences stacked column-wise as ΔZ, ΔF, solve the small per-chain least
    squares γ = argmin ‖f_k − ΔF γ‖ and take

        z_{k+1} = z_k + β f_k − (ΔZ + β ΔF) γ.

    ``beta`` is the same under-relaxation factor as ``_PicardUpdate`` (β = 1 is
    undamped; β < 1 stabilises the near-imaginary spectrum).  It enters only
    this final combination -- the γ least squares is β-independent -- so damping
    changes stability, NOT the inner-solve conditioning: the Tikhonov floor
    below is still what keeps the m×m solve well-posed (a frozen chain
    contributes zero ΔF columns, giving γ = 0 → a plain damped Picard step that
    is masked out anyway).

    On a linear map Anderson(m≥1) reaches the fixed point in one accelerated
    step; on the true nonlinear map it typically converges in fewer iterations
    than Picard, trading extra model evals for a cheap m×m solve.

    A configured instance (held by ``RMHMC``) carries its history/damping but no
    live buffers; ``new`` returns a fresh working copy for a single solve, at
    which point a ``history`` of None resolves to dim(q).

    Parameters
    ----------
    history : int or None
        History length ``m`` (number of past differences retained), ≥ 1, or
        None to resolve to dim(q) when ``new`` is called.
    beta : float
        Damping factor in (0, 1]; 1.0 is the undamped iteration.
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
        """Fresh, empty-buffer working updater for a solve over ``d``-dim
        positions, resolving a None ``history`` to ``d``."""
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
        # Scale-aware Tikhonov floor so the solve is well-posed even when a
        # chain's ΔF columns are (near-)collinear or all zero.  Independent of
        # beta, which does not enter this solve.
        scale = A.diagonal(dim1=-2, dim2=-1).mean(-1)     # (N,)
        reg   = (self.reg_rel * scale + self.reg_abs).view(-1, 1, 1)
        A = A + reg * torch.eye(mk, dtype=A.dtype, device=A.device)
        gamma = torch.linalg.solve(A, b)                  # (N, mk, 1)

        z_next = z + self.beta * f_k - ((dZ + self.beta * dF) @ gamma).squeeze(-1)
        return z_next


# ---- Implicit midpoint step --------------------------------------------- #

def _implicit_midpoint_step(q, p, eps, evaluate_model, max_iter, tol, solver=None):
    """
    One step of I.M.(a): solve for the endpoint (q', p') directly via a
    per-chain fixed-point iteration, with the midpoint derived from the current
    iterate and the start-of-step point.  ``solver`` is a configured update rule
    (``_PicardUpdate`` or ``_AndersonUpdate``, defaulting to undamped Picard);
    ``solver.new(d)`` supplies the fresh per-solve updater, so this function
    never re-validates the choice.  Every such rule solves the identical
    fixed-point equation, so the returned endpoint, reversibility, and
    symplecticity are solver- and damping-independent; only the iteration count
    (and stability) differs.

    Batched over a leading chain axis (q, p have shape (N, d)).  Each chain
    runs its own fixed-point solve; a chain "finishes" when it either
    converges (max-norm residual < tol) or blows up (residual > 10x its
    initial value).  Finished chains are frozen via a mask -- their state
    and residual stop updating -- while live chains keep iterating, so the
    loop runs until all chains finish or max_iter is reached.  (Finished
    chains are still re-evaluated each iteration since the model call is
    batched; only their *updates* are discarded.)

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


# ---- Residual Jacobian (for Newton-type inner solves) -------------------- #
#
# The fixed-point solvers above (Picard / Anderson) only need residual *values*.
# A Newton-type corrector instead needs the Jacobian d r / d z of the residual
# r(z) = z - F(z) at the endpoint iterate z = [q_k, p_k].  For a stiff,
# strongly position-varying metric the fixed-point contraction factor ~ eps * L
# is large and Picard/Anderson stall, whereas Newton is spectrum-agnostic -- so
# this Jacobian is the building block for a simplified/frozen-Jacobian Newton
# corrector.
#
# Structure exploited (see _implicit_midpoint_residual_jacobian): with the
# metric available in closed form, only ONE block of the Jacobian is genuinely
# second-order in the model (the force/position Hessian Hqq); every other block
# is a first derivative of the metric.  Hqq is taken with exact second-order
# autodiff by default (the model's metric is a closed-form forward map, so this
# is a plain double backward, not a nested autograd-of-autograd); a
# finite-difference of the first-order force is kept as a robust fallback for
# models where the second-order graph is fragile.

def _vector_field_jacobian(outputs, inputs, vectorized=True):
    """Jacobian of a batched vector field: ``outputs`` (N, d) w.r.t. ``inputs``
    (N, d), returning ``(N, d, d)`` with ``[n, k, i] = d outputs[n,k]/d inputs[n,i]``.

    ``vectorized=True`` takes all ``d`` output rows in ONE reverse pass via
    ``is_grads_batched`` (vmap over the output basis) instead of a Python loop of
    ``d`` separate backwards.  The per-row loop is kept as a fallback for models
    whose backward is not vmap-compatible.  ``retain_graph`` is set so the caller
    can reuse ``inputs``'s graph afterwards.
    """
    d = outputs.shape[-1]
    if vectorized:
        eye = torch.eye(d, dtype=outputs.dtype, device=outputs.device)
        grad_outputs = eye.unsqueeze(1).expand(d, *outputs.shape)      # (d, N, d)
        (J,) = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs,
                                   is_grads_batched=True, retain_graph=True)
        return J.movedim(0, -2)                                        # (N, d, d)
    rows = []
    for k in range(d):
        (gk,) = torch.autograd.grad(outputs[..., k].sum(), inputs,
                                    retain_graph=True, allow_unused=True)
        rows.append(torch.zeros_like(inputs) if gk is None else gk)
    return torch.stack(rows, dim=-2)


def _dHdq(q_mid, p_mid, evaluate_model, create_graph=False):
    """First-order force dH/dq at (q_mid, p_mid).

    Mirrors the gradient inside ``_midpoint_map``.  ``create_graph=True`` keeps
    the graph so a second derivative (Hqq) can be taken through it.
    """
    qm = q_mid.detach().requires_grad_(True)
    with torch.enable_grad():
        U, metric = evaluate_model(qm)
        H = _hamiltonian(qm, p_mid, U, metric)
        (g,) = torch.autograd.grad(H.sum(), qm, create_graph=create_graph)
    return g, qm


def _force_position_hessian_autodiff(q_mid, p_mid, evaluate_model, vectorized=True):
    """Hqq = d(dH/dq)/dq at the midpoint via exact second-order autodiff.

    ``Hqq[..., k, i] = d^2 H / dq_k dq_i`` (``p_mid`` held fixed) -- symmetric up
    to autodiff round-off.  The force is taken with a graph and its Jacobian in
    one batched second backward (``_vector_field_jacobian``); the returned tensor
    is detached (no graph escapes).
    """
    with torch.enable_grad():
        g, qm = _dHdq(q_mid, p_mid, evaluate_model, create_graph=True)
        Hqq = _vector_field_jacobian(g, qm, vectorized)
    return Hqq.detach()


def _force_position_hessian_fd(q_mid, p_mid, evaluate_model, fd_step, central):
    """Fallback for ``_force_position_hessian_autodiff``: Hqq by finite-
    differencing the first-order force in ``q_mid`` (``p_mid`` fixed).  Central
    uses 2d force evals, one-sided d+1 (reusing the base force).  No second-order
    autodiff.
    """
    d = q_mid.shape[-1]
    Hqq = torch.empty(*q_mid.shape[:-1], d, d, dtype=q_mid.dtype, device=q_mid.device)
    if central:
        for i in range(d):
            e = torch.zeros_like(q_mid); e[..., i] = fd_step
            gp, _ = _dHdq(q_mid + e, p_mid, evaluate_model)
            gm, _ = _dHdq(q_mid - e, p_mid, evaluate_model)
            Hqq[..., :, i] = (gp - gm).detach() / (2 * fd_step)
    else:
        base, _ = _dHdq(q_mid, p_mid, evaluate_model)
        base = base.detach()
        for i in range(d):
            e = torch.zeros_like(q_mid); e[..., i] = fd_step
            gp, _ = _dHdq(q_mid + e, p_mid, evaluate_model)
            Hqq[..., :, i] = (gp.detach() - base) / fd_step
    return Hqq


def _implicit_midpoint_residual_jacobian(
    q, p, eps, evaluate_model, z, *, force_hessian="autodiff",
    vectorized=True, fd_step=1e-6, fd_central=False,
):
    """Per-chain Jacobian ``d r / d z`` of the implicit-midpoint residual
    ``r(z) = z - F(z)`` at the endpoint iterate ``z = [q_k, p_k]``.

    Shapes: ``q, p`` are ``(N, d)``; ``z`` is ``(N, 2d)``; ``eps`` is ``(N,)``.
    Returns ``(N, 2d, 2d)``.

    Derivation.  With ``q_mid = (q+q_k)/2``, ``p_mid = (p+p_k)/2``,
    ``w = p + p_k = 2 p_mid`` and ``e = eps``,

        F_q = q + (e/2) G^{-1}(q_mid) w
        F_p = p - e * dH/dq(q_mid, p_mid)

    so, writing ``Ginv = G^{-1}(q_mid)`` and the vector-field Jacobian
    ``Da = d/dq[ G^{-1}(q) w ]|_{q_mid}`` (first order in the metric),

        Hqp = d(dH/dq)/dp|_mid = Da^T / 2        (first order, since p_mid = w/2)
        Hqq = d(dH/dq)/dq|_mid                    (second order -> finite diff)

        J_r = [[ I - (e/4) Da ,  -(e/2) Ginv    ],
               [ (e/2) Hqq    ,   I + (e/2) Hqp ]].

    Only ``Hqq`` needs a second derivative of the model.  ``force_hessian``
    selects how it is taken: ``"autodiff"`` (default) uses exact second-order
    autodiff (a double backward through the closed-form metric); ``"fd"`` falls
    back to finite-differencing the first-order force (``fd_step`` / ``fd_central``)
    for models where the second-order graph is fragile.  Both are verified
    against a brute-force finite-difference Jacobian of the residual.

    ``vectorized=True`` builds the per-block Jacobians (``Da`` and, for
    ``"autodiff"``, ``Hqq``) in one batched reverse pass each rather than a
    Python loop over the ``d`` output rows; set it False for models whose
    backward is not vmap-compatible.
    """
    d = q.shape[-1]
    N = q.shape[0]
    q_k, p_k = z[..., :d], z[..., d:]
    q_mid = 0.5 * (q + q_k)
    p_mid = 0.5 * (p + p_k)
    w = (p + p_k).detach()
    e = eps.reshape(N, *([1] * 2))
    eye = torch.eye(d, dtype=q.dtype, device=q.device).expand(N, d, d)

    # First-order blocks from a single grad-enabled metric evaluation at q_mid.
    qm = q_mid.detach().requires_grad_(True)
    with torch.enable_grad():
        _, metric = evaluate_model(qm)
        a = metric.inv_metric_times_vec(w)                      # (N, d) = Ginv w
    if a.requires_grad:                                         # (N,d,d): [:,k,i]=da_k/dq_i
        Da = _vector_field_jacobian(a, qm, vectorized).detach()
    else:                                                       # metric independent of q
        Da = torch.zeros(N, d, d, dtype=q.dtype, device=q.device)
    with torch.no_grad():                                       # dense Ginv, reuses metric
        Ginv = torch.stack(
            [metric.inv_metric_times_vec(eye[..., j]) for j in range(d)], dim=-1)
    Hqp = Da.transpose(-2, -1) * 0.5

    # The one second-order block: exact autodiff by default, FD fallback.
    if force_hessian == "autodiff":
        Hqq = _force_position_hessian_autodiff(q_mid, p_mid, evaluate_model, vectorized)
    elif force_hessian == "fd":
        Hqq = _force_position_hessian_fd(q_mid, p_mid, evaluate_model, fd_step, fd_central)
    else:
        raise ValueError(
            f"force_hessian must be 'autodiff' or 'fd', got {force_hessian!r}")

    top = torch.cat([eye - (e / 4) * Da,  -(e / 2) * Ginv],     dim=-1)
    bot = torch.cat([(e / 2) * Hqq,        eye + (e / 2) * Hqp], dim=-1)
    return torch.cat([top, bot], dim=-2)


# =========================================================================== #
#                                                                             #
#  Chain state                                                                #
#                                                                             #
# =========================================================================== #

class RMHMCState:
    """
    Internal working state of one RMHMC trajectory, batched over (N,) chains.

    Fields
    ------
    q, p : (N, d)
        Position and momentum.
    U : (N,) or None
        Potential at ``q``.
    metric : TransformedMetric or None
        Metric at ``q``.  ``U`` and ``metric`` are present after
        ``init`` / ``accept`` and ``None`` right
        after ``step`` (the integrator evaluates the model only at
        midpoints, so the endpoint's ``U``/``metric`` are not free); they
        are filled lazily by :meth:`complete`.
    max_residual : (N,) or None
        Running max fixed-point residual over the current trajectory.
    fp_iters : list of (N,)
        Per-midpoint-step fixed-point iteration counts for the trajectory.

    The trajectory accumulators (``max_residual``, ``fp_iters``) are reset
    by ``init`` and ``accept``; ``step`` carries them forward.  The
    state is operator-internal, so it stores whatever is cheap to keep.
    """

    def __init__(self, q, p=None, U=None, metric=None,
                 max_residual=None, fp_iters=None):
        self.q = q
        self.p = p
        self.U = U
        self.metric = metric
        self.max_residual = max_residual
        self.fp_iters = [] if fp_iters is None else fp_iters

    def complete(self, evaluate_model):
        """Fill ``U`` and ``metric`` by evaluating the model at ``q`` if they
        are not already present; a no-op on an already-complete state."""
        if self.U is None or self.metric is None:
            # Endpoint evaluations only feed energies/diagnostics and are never
            # backpropagated (the integrator takes its gradient at midpoints in
            # ``_midpoint_map``'s own ``enable_grad`` block).  Evaluating under
            # ``no_grad`` keeps a model whose U/G carry an autograd graph from
            # pinning that graph via ``delta_H`` in the diagnostics list, which
            # would otherwise accumulate CUDA memory across the whole run.
            with torch.no_grad():
                self.U, self.metric = evaluate_model(self.q)
        return self

    def reorder(self, perm):
        """Permute the *state* (configuration) across chain slots: slot ``i`` of
        the result carries the configuration ``q, p, U, metric`` from slot
        ``perm[i]`` (e.g. a PT rung swap).  ``perm`` is an ``(N,)`` long index
        tensor.  Pure -- returns a new state; absent (None) fields stay None.

        ``max_residual`` and ``fp_iters`` are deliberately NOT permuted: they
        are integrator diagnostics belonging to the chain slot (its step size,
        and under PT its own distribution), not to the configuration that
        occupies it, so they do not travel when configurations are exchanged.
        """
        return RMHMCState(
            q            = self.q[perm],
            p            = None if self.p is None else self.p[perm],
            U            = None if self.U is None else self.U[perm],
            metric       = None if self.metric is None else self.metric.reorder(perm),
            max_residual = self.max_residual,   # slot-bound: not permuted (see docstring)
            fp_iters     = self.fp_iters,       # slot-bound: not permuted
        )


# =========================================================================== #
#                                                                             #
#  RMHMC sampler                                                              #
#                                                                             #
# =========================================================================== #

class RMHMC(BaseSampler):
    """
    Riemannian Manifold HMC with the implicit midpoint integrator.

    The sampler operates in unconstrained space (via the space object) while
    the user specifies the model in **constrained** space; the pull-back is
    handled by :meth:`BaseSampler.evaluate_model`.

    It both *is* the sampler (``run_mcmc`` via the own batched driver) and
    implements the operator interface the driver composes:

        init(q) -> state    # initialize per-chain step_size/adapter/counters
                            #   AND return the initial chain state (not stateless)
        step(s) -> state    # one transition: num_steps leapfrogs + accept
        end_warmup()        # freeze step_size to the adapter average

    with ``leapfrog_step`` / ``accept`` / ``_bookkeep`` as implementation
    detail (not part of the base contract).  All chains run in one batched
    state; a PT swap would ``state.reorder`` between transitions.

    User contract
    -------------
    ``model_fn(theta_full) -> (U_lik, G_lik)`` where ``theta_full`` is the
    full constrained vector, ``U_lik`` the scalar likelihood potential
    ``-log p(data | theta)``, and ``G_lik`` the (d_full, d_full) SPD
    likelihood metric in constrained coordinates.  BaseSampler adds the prior
    log-prob and prior metric, Choleskys, restricts to free coordinates, and
    pulls back through the Jacobian (see ``spaces.TransformedMetric``).

    Parameters
    ----------
    model_fn : callable
        See above.
    space
        Parameter space object (priors, transform, free/fixed split).
    step_size : float
        Integration step size (adapted during warmup when adapting).
    num_steps : int
        Number of leapfrog substeps per transition.
    adapt_step_size : bool
        Whether to adapt the step size during warmup via the REINFORCE
        adapter.  Default True.
    adaptation_sigma : float
        Perturbation scale of the REINFORCE adapter.  Default 0.1.
    fp_max_iter : int
        Maximum fixed-point iterations per leapfrog substep.
    fp_tol : float
        Convergence tolerance for fixed-point iteration (max norm).
    solver : str
        Fixed-point solver driving the implicit-midpoint solve: ``"picard"``
        (default) or ``"anderson"``.  Both solve the identical equation and
        return the same endpoint up to ``fp_tol``; Anderson typically reaches
        it in fewer iterations (hence fewer model evals) on stiff metrics, at
        the cost of a small m×m solve per iteration.
    anderson_history : int or None
        History length ``m`` for the Anderson solver (ignored for Picard).
        ``None`` (default) resolves per-solve to ``dim(q)`` (the free-parameter
        dimension): a safe, model-agnostic choice because the m×m least-squares
        overhead is negligible next to a single likelihood/metric evaluation,
        which dominates RMHMC cost.  Must be ≥ 1 if given.
    damping : float
        Under-relaxation factor β ∈ (0, 1] shared by both solvers (default 1.0
        = undamped).  The step becomes ``(1−β) z + β·(solver step)``; β < 1
        trades convergence speed for stability and is the lever for the
        (near-)imaginary iteration spectrum of the implicit-midpoint map.  It
        affects only stability/iteration count, not the endpoint, nor the
        Anderson inner solve's conditioning (which the Tikhonov floor handles).
    divergence_threshold : float
        Raw |delta_H| above which (or non-finite values for which) the step
        is recorded as a divergence.  Default 100.
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
        fp_max_iter: int = 0,
        fp_tol: float = 1e-8,
        solver: str = "picard",
        anderson_history: int = None,
        damping: float = 1.0,
        divergence_threshold: float = 100.0
    ):
        super().__init__(potential_fn=model_fn, space=space, requires_metric=True)

        if fp_max_iter == 0:
            fp_max_iter = 100

        # Resolve the string choice into a configured solver here, once, so the
        # integrator never re-checks it and RMHMC holds no solver-specific
        # scalars (history/damping live inside the solver object).
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

        self._step_size_init       = step_size       # scalar; tensorized per-chain in init()
        self.step_size             = step_size
        self.num_steps             = num_steps
        self._adapt_step_size      = adapt_step_size
        self._adaptation_sigma     = adaptation_sigma
        self._fp_max_iter          = fp_max_iter
        self._fp_tol               = fp_tol
        self._divergence_threshold = divergence_threshold

    @property
    def trajectory_length(self):
        eps = self.step_size
        eps = float(eps.mean()) if torch.is_tensor(eps) else eps
        return eps * self.num_steps

    def init(self, q):
        """Initialize a run and return the initial chain state.

        NOT stateless: sizes the per-chain ``step_size``, adapter, and
        diagnostic counters from ``q``'s batch and arms adaptation
        (``_adapting = adapt_step_size``).  Then builds the initial chain
        state (evaluate (U, metric), sample momentum, reset accumulators).

        There is no ``num_warmup_steps`` here -- the driver decides when to
        call :meth:`end_warmup`.  For a zero-warmup run the driver calls
        ``end_warmup`` before the first transition; since the adapter
        round-trips with no updates, ``step_size`` is left at its
        constructor value.
        """
        N = q.shape[0]
        dtype, device = q.dtype, q.device

        # per-chain step size (N,)
        self.step_size = torch.full((N,), float(self._step_size_init),
                                    dtype=dtype, device=device)

        # reset running statistics (per-chain where it matters)
        self._step = 0
        self._accepted = torch.zeros(N, dtype=torch.long, device=device)  # (N,) count
        self._num_divergences = torch.zeros(N, dtype=torch.long, device=device)  # (N,) count
        # Running per-chain integrator summaries (O(1) memory), folded in by
        # _bookkeep each transition.  These replace the old unbounded per-step
        # lists (one (N,) tensor -- plus, for fp_iters, num_steps of them --
        # appended every transition), so the diagnostics footprint stays
        # constant over an arbitrarily long run instead of growing with it.
        self._reset_diagnostics()

        # arm adaptation; the driver flips it off via end_warmup()
        self._adapting = self._adapt_step_size
        if self._adapt_step_size:
            self._adapter = REINFORCEAdapter(N, self._adaptation_sigma)
            self._adapter.prox_center = torch.log(self.step_size)   # (N,)
            self._adapter.reset()

        # initial chain state at q
        s = RMHMCState(q).complete(self.evaluate_model)
        s.p = s.metric.sample_momentum()
        s.max_residual = torch.zeros(N, dtype=dtype, device=device)
        return s

    # ---- Operator interface (composed by run_mcmc) ---------------------- #
    #
    # init(q) -> step -> step -> ...  is the chain; each step is one
    # transition (num_steps leapfrog substeps + accept).  leapfrog_step is the
    # HMC-internal substep that variants may build on; it is not the
    # transition.  The driver composes these; the split lets a PT swap
    # (state.reorder) sit between transitions, or variants refresh p between
    # substeps.

    def step(self, s):
        """One chain transition from the ready state ``s``: integrate
        ``num_steps`` leapfrog substeps, then Metropolis accept/reject.
        Returns the chosen ready-to-step state.  This is the transition verb
        -- ``num_steps`` is the sampler's own business, so a driver just calls
        ``s = step(s)`` repeatedly."""
        new = s
        for _ in range(self.num_steps):
            new = self.leapfrog_step(new)
        return self.accept(new, s)

    def leapfrog_step(self, s):
        """One implicit-midpoint substep.  Returns a new state whose U/metric
        are ``None`` (the integrator evaluates only at midpoints) and whose
        trajectory accumulators are carried forward.  HMC-internal; ``step``
        composes ``num_steps`` of these."""
        q, p, fp_it, residual = _implicit_midpoint_step(
            s.q, s.p, self.step_size,
            self.evaluate_model,
            self._fp_max_iter, self._fp_tol,
            self._solver
        )
        out = RMHMCState(q, p)
        out.max_residual = torch.maximum(s.max_residual, residual)
        out.fp_iters = s.fp_iters + [fp_it]
        return out

    def accept(self, new, old):
        """Per-chain Metropolis accept/reject between the trajectory endpoint
        ``new`` and its start ``old``, plus bookkeeping and (while adapting)
        the step-size update.  Returns the chosen ready-to-step state: U/metric
        are carried over per chain via ``TransformedMetric.select`` (no model
        eval) and a fresh momentum is sampled at the chosen point."""
        new.complete(self.evaluate_model)      # endpoint eval (1 model eval)
        old.complete(self.evaluate_model)       # no-op: start already complete

        H_new = _hamiltonian(new.q, new.p, new.U, new.metric)   # (N,)
        H_old = _hamiltonian(old.q, old.p, old.U, old.metric)   # (N,)
        delta_H_raw = H_new - H_old                              # (N,)

        # Divergence: raw delta_H non-finite or over threshold.  Clamping
        # below is for Metropolis-ratio safety only; it does not affect
        # accounting.
        is_divergent = (~torch.isfinite(delta_H_raw)) | (delta_H_raw > self._divergence_threshold)
        delta_H = torch.where(torch.isfinite(delta_H_raw),
                              delta_H_raw, delta_H_raw.new_full((), 300.0))
        delta_H = delta_H.clamp(-300.0, 300.0)                   # (N,)

        N = new.q.shape[0]
        accepted = torch.log(torch.rand(N, device=new.q.device, dtype=new.q.dtype)) < -delta_H
        chosen_q = torch.where(accepted.unsqueeze(-1), new.q, old.q)   # (N, d)

        self._bookkeep(accepted, delta_H, is_divergent, new.max_residual, new.fp_iters)

        # Ready-to-step state: mix U/metric per chain (free -- both already
        # computed) and resample momentum at the chosen point.
        chosen_U = torch.where(accepted, new.U, old.U)               # (N,)
        chosen_metric = new.metric.select(accepted, old.metric)
        out = RMHMCState(chosen_q, None, chosen_U, chosen_metric)
        out.p = chosen_metric.sample_momentum()
        out.max_residual = torch.zeros(N, dtype=chosen_q.dtype, device=chosen_q.device)
        return out

    def _reset_diagnostics(self):
        """(Re)initialize the running per-chain integrator summaries to empty.

        Called by ``init`` and ``end_warmup`` to start a phase.  Every summary
        is an ``(N,)`` accumulator sized from ``step_size`` (already tensorized
        per chain by then), so the diagnostics cost is O(num_chains), not
        O(num_chains * num_steps) as the old append-per-step lists were.
        """
        N = self.step_size.shape[0]
        dtype, device = self.step_size.dtype, self.step_size.device
        z = torch.zeros(N, dtype=dtype, device=device)
        self._delta_H_last     = z.clone()   # most recent delta_H          (logging |dH|)
        self._delta_H_abs_sum  = z.clone()   # running sum of |delta_H|     (-> mean)
        self._delta_H_abs_max  = z.clone()   # running max |delta_H|
        self._residual_last    = z.clone()   # most recent max-residual     (logging |r|)
        self._residual_sum     = z.clone()   # running sum of max-residual  (-> mean)
        self._residual_max      = z.clone()  # running max max-residual
        self._fp_iters_sum     = z.clone()   # running sum of per-transition mean iters (-> mean)
        self._fp_iters_max     = z.clone()   # running max per-transition worst-substep iters

    def _bookkeep(self, accepted, delta_H, is_divergent, max_residual, fp_iters):
        # Fold this transition into the O(1) running per-chain summaries rather
        # than appending to unbounded lists.  Everything folded in is detached
        # (delta_H explicitly; max_residual and fp_iters carry no graph), so no
        # per-step model graph is pinned across the run.
        dH = delta_H.detach()                                         # (N,)
        fp = torch.stack(fp_iters).to(self.step_size.dtype)           # (num_steps, N)
        it_mean = fp.mean(0)                                          # (N,) per-substep mean
        it_max  = fp.amax(0)                                          # (N,) worst substep

        self._delta_H_last     = dH
        self._delta_H_abs_sum += dH.abs()
        self._delta_H_abs_max  = torch.maximum(self._delta_H_abs_max, dH.abs())
        self._residual_last    = max_residual
        self._residual_sum    += max_residual
        self._residual_max     = torch.maximum(self._residual_max, max_residual)
        self._fp_iters_sum    += it_mean
        self._fp_iters_max     = torch.maximum(self._fp_iters_max, it_max)

        self._accepted += accepted
        self._step += 1
        # per-chain divergence count (post-warmup once reset at end_warmup,
        # mirroring Pyro NUTS conventions).
        self._num_divergences += is_divergent.long()

        # step-size adaptation (per chain), while the driver leaves us adapting
        if self._adapting:
            # mean fp iters per chain across the trajectory's midpoint steps
            num_iters = it_mean                                        # (N,), computed above
            eta = 1.e-3

            # f_t is the per-step cost the adapter minimises (lower = better
            # step size).  The bracketed product is a per-step efficiency:
            #     delta_H_penalty * step_size / num_iters
            # is accepted travel distance per solver iteration (exp(-|dH|) ~
            # accept prob, step_size ~ distance, num_iters ~ solver cost),
            # times solver_penalty = exp(-residual/step_size), an analogous
            # acceptance term for the fixed-point solver error.  Taking -log of
            # that product (floored at eta, normalised by |log eta|) turns the
            # efficiency into a cost AND gives rare solver/Metropolis failures
            # large weight -- a plain average would let infrequent failures be
            # drowned out.
            solver_penalty  = torch.exp(-max_residual / self.step_size)        # (N,)
            delta_H_penalty = torch.exp(-delta_H.abs())                         # (N,)
            f_t = (-0.5 * torch.log(
                        solver_penalty * delta_H_penalty * self.step_size / num_iters + eta
                   ) / abs(math.log(eta)))                                      # (N,)

            self._adapter.step(f_t)
            self.step_size = torch.exp(self._adapter.get_state()[0])            # (N,)

    def end_warmup(self):
        """Transition from warmup to sampling: freeze ``step_size`` to the
        adapter's running average and stop adapting, then reset the diagnostic
        counters for the sampling phase.  Driver-timed.  For a zero-warmup run
        this is called before any transition; the adapter, never updated,
        reports its seed, so ``step_size`` is left at its constructor value
        (up to the float ``exp(log(.))`` round-trip)."""
        if self._adapt_step_size:
            # final step size = slow-moving average, per chain
            self.step_size = torch.exp(self._adapter.get_state()[1])   # (N,)
        self._adapting = False
        # restart counters for the sampling phase
        self._accepted = torch.zeros_like(self._accepted)
        self._num_divergences = torch.zeros_like(self._num_divergences)
        self._step = 0
        self._reset_diagnostics()

    def logging(self):
        if self._step == 0:
            return {}
        # Reduce per-chain stats to scalar summaries for the progress bar:
        # mean step size / accept prob, worst-chain |dH| and |r|.
        eps   = float(self.step_size.mean())
        dH    = float(self._delta_H_last.abs().max())
        res   = float(self._residual_last.max())
        accpr = float((self._accepted / self._step).mean())
        return OrderedDict(
            [
                ("eps", "{:.2e}".format(eps)),
                ("|dH|", "{:.2e}".format(dH)),
                ("|r|", "{:.2e}".format(res)),
                ("acc. prob", "{:.3f}".format(accpr)),
            ]
        )

    def diagnostics(self):
        """Per-chain diagnostics.  The common schema -- ``accept_rate``,
        ``num_divergences``, ``step_size`` (each a ``(num_chains,)`` tensor) --
        matches the Pyro path.

        The RMHMC-specific integrator extras are running per-chain *summaries*
        (each an ``(num_chains,)`` tensor), not the full per-step history: the
        history grew without bound over a run (and its churn of many tiny
        tensors fragmented the heap), so it is folded online into O(1)
        accumulators instead.  Reported over the current phase (reset at
        ``end_warmup``):

          ``delta_H_abs_mean`` / ``delta_H_abs_max`` -- mean / max ``|delta_H|``
          ``residual_mean``    / ``residual_max``    -- mean / max fixed-point
                                                        max-residual per transition
          ``fp_iters_mean``    / ``fp_iters_max``    -- mean per-substep iters /
                                                        worst single substep
        """
        steps = max(self._step, 1)
        return {
            "accept_rate": self._accepted / steps,    # (N,) per chain
            "num_divergences": self._num_divergences,  # (N,) per-chain count
            "step_size": self.step_size,              # (N,) per chain
            # RMHMC-specific running summaries (no Pyro equivalent), (N,) each:
            "delta_H_abs_mean": self._delta_H_abs_sum / steps,
            "delta_H_abs_max":  self._delta_H_abs_max,
            "residual_mean":    self._residual_sum / steps,
            "residual_max":     self._residual_max,
            "fp_iters_mean":    self._fp_iters_sum / steps,
            "fp_iters_max":     self._fp_iters_max,
        }

