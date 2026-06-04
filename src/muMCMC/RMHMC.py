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


# ---- Implicit midpoint step --------------------------------------------- #

def _implicit_midpoint_step(q, p, eps, evaluate_model, max_iter, tol):
    """
    One step of I.M.(a): solve for the endpoint (q', p') directly via Picard
    fixed-point iteration (z_{k+1} = z_k − r_k), with the midpoint derived
    from the current iterate and the start-of-step point.

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

    z_k = torch.cat([q, p], dim=-1)            # (N, 2d)
    r_k = residual_fn(z_k)
    r_init_norm = r_k.abs().amax(-1)           # (N,)

    done     = torch.zeros(N, dtype=torch.bool, device=q.device)
    iters    = torch.full((N,), max_iter, dtype=torch.long, device=q.device)
    residual = r_init_norm.clone()             # (N,)

    for i in range(1, max_iter + 1):
        z_next = z_k - r_k                     # Picard fixed-point update
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
        divergence_threshold: float = 100.0,
    ):
        super().__init__(potential_fn=model_fn, space=space, requires_metric=True)

        if fp_max_iter == 0:
            fp_max_iter = 100

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
        self._delta_Hs = []          # list of (N,) tensors
        self._residuals = []         # list of (N,) tensors
        self._fp_iters = []          # list of lists-of-(N,)  (per leapfrog substep)

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
            self._fp_max_iter, self._fp_tol
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

    def _bookkeep(self, accepted, delta_H, is_divergent, max_residual, fp_iters):
        # update running statistics (all per-chain, shape (N,))
        self._delta_Hs.append(delta_H)
        self._residuals.append(max_residual)
        self._fp_iters.append(fp_iters)
        self._accepted += accepted
        self._step += 1
        # per-chain divergence count (post-warmup once reset at end_warmup,
        # mirroring Pyro NUTS conventions).
        self._num_divergences += is_divergent.long()

        # step-size adaptation (per chain), while the driver leaves us adapting
        if self._adapting:
            # mean fp iters per chain across the trajectory's midpoint steps
            num_iters = torch.stack(fp_iters).to(self.step_size.dtype).mean(0)  # (N,)
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
        self._delta_Hs = []
        self._residuals = []
        self._fp_iters = []

    def logging(self):
        if self._step == 0:
            return {}
        # Reduce per-chain stats to scalar summaries for the progress bar:
        # mean step size / accept prob, worst-chain |dH| and |r|.
        eps   = float(self.step_size.mean())
        dH    = float(self._delta_Hs[-1].abs().max())
        res   = float(self._residuals[-1].max())
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
        matches the Pyro path; ``delta_Hs`` / ``residuals`` / ``fp_iters`` are
        RMHMC-specific integrator extras."""
        steps = max(self._step, 1)
        return {
            "accept_rate": self._accepted / steps,    # (N,) per chain
            "num_divergences": self._num_divergences,  # (N,) per-chain count
            "step_size": self.step_size,              # (N,) per chain
            # RMHMC-specific extras (no Pyro equivalent):
            "delta_Hs": self._delta_Hs,
            "residuals": self._residuals,
            "fp_iters": self._fp_iters,
        }

