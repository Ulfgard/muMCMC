from collections import namedtuple

import torch

# =========================================================================== #
#                                                                             #
#  Batched solvers for r(z) = 0                                               #
#                                                                             #
#  One problem per row of z, solved to a per-row tolerance. Every implicit    #
#  integrator in the library reaches for this: RMHMC's midpoint equation and  #
#  ChartRATTLE's position equation are both a root find in the step endpoint. #
#                                                                             #
#  An update rule turns the current iterate and its residual into the next    #
#  iterate. Three are available, chosen by name.                              #
#                                                                             #
#      picard     z_{k+1} = z_k − β P r_k                                     #
#      anderson   Picard plus Walker and Ni's Type-II acceleration            #
#      newton     z_{k+1} = z_k − β J(z_k)⁻¹ r_k                              #
#                                                                             #
#  β ∈ (0, 1] under-relaxes all three. It does not move the root, so it buys  #
#  stability at the cost of iterations. P is an optional fixed                #
#  preconditioner.                                                            #
#                                                                             #
#  Picard converges when the map is a contraction, which for an implicit      #
#  integrator means a step size small enough. Relaxing it pulls the iteration #
#  eigenvalues (β − 1) + β λ toward (1 − β) on the real axis, taming the      #
#  near-imaginary spectrum of the midpoint map.                               #
#                                                                             #
#  Anderson(m) stacks the last m iterate and residual differences and solves  #
#  a small per-row least squares. On a linear map Anderson(m ≥ 1) reaches the #
#  root in one accelerated step. On a nonlinear one it usually beats Picard,  #
#  trading extra residual evaluations for a cheap m by m solve. It is not     #
#  monotone in the residual, so a transient rise is normal.                   #
#                                                                             #
#  Newton needs the Jacobian J = ∂r/∂z and converges quadratically near the   #
#  root, which the other two do not. It is the rule to reach for when a fixed #
#  preconditioner underperforms because J moves quickly along the iteration,  #
#  since Picard with P ≈ J(z_0)⁻¹ is exactly Newton with a frozen Jacobian.   #
#  Whether that trade pays depends on the cost of J against the cost of the   #
#  extra iterations the frozen version needs.                                 #
#                                                                             #
#  J is not assumed symmetric, so the step goes through a general LU solve.   #
#  A singular J gives non-finite entries rather than raising, which the       #
#  divergence test below then catches for that row alone.                     #
#                                                                             #
# =========================================================================== #
#                                                                             #
#  The residual interface                                                     #
#                                                                             #
#  A solver that reports needs_jacobian expects                               #
#                                                                             #
#      residual_fn(z) -> (r, J)      r is (N, d), J is (N, d, d)              #
#                                                                             #
#  and one that does not expects                                              #
#                                                                             #
#      residual_fn(z) -> r                                                    #
#                                                                             #
#  so the contract follows from the chosen rule rather than from a flag on    #
#  the call. Either way the result is detached from any autograd graph, since #
#  the solve is derivative-free in the outer sense: only the root matters,    #
#  not how the iteration reached it.                                          #
#                                                                             #
# =========================================================================== #

SolveResult = namedtuple("SolveResult", ["z", "iters", "residual"])

# Growth factor over the starting residual past which a row counts as diverged.
# Loose enough to let Anderson overshoot on its way to the root.
_DIVERGENCE_GROWTH = 1000.0


class _PicardUpdate:
    """Relaxed Picard iteration, ``z_{k+1} = z_k − β P r_k``.

    Stateless.

    Parameters
    ----------
    beta : float
        Under-relaxation factor in (0, 1].
    precond : callable or None
        Preconditioner P applied to the residual before the step, or None for
        the identity.
    """

    needs_jacobian = False

    def __init__(self, beta=1.0, precond=None):
        self.beta = float(beta)
        self._precond = precond

    def new(self, d, precond=None):
        """Fresh per-solve updater for ``d``-dimensional rows. ``precond``
        overrides the preconditioner. Omitted, this updater's own is kept."""
        return _PicardUpdate(self.beta, self._precond if precond is None else precond)

    def propose(self, z, r, jac=None):
        if self._precond is not None:
            r = self._precond(r)
        return z - self.beta * r

    def damped(self, factor):
        """Copy with β scaled by ``factor``, same preconditioner."""
        return _PicardUpdate(self.beta * factor, self._precond)


class _AndersonUpdate:
    """Anderson(m) acceleration, Type-II, of the fixed-point map.

    With ``f_k = −r_k`` and the last ``m`` iterate and residual differences
    stacked column-wise as ΔZ and ΔF, solve the per-row least squares
    ``γ = argmin ‖f_k − ΔF γ‖`` and take

        z_{k+1} = z_k + β f_k − (ΔZ + β ΔF) γ.

    Parameters
    ----------
    history : int or None
        History length ``m``, at least 1, or None to resolve to ``d`` when
        :meth:`new` is called.
    beta : float
        Under-relaxation factor in (0, 1].
    precond : callable or None
        Preconditioner P applied to the residual before the step, or None for
        the identity.

    References
    ----------
    Walker and Ni, Anderson acceleration for fixed-point iterations, 2011.
    """

    needs_jacobian = False

    # Relative and absolute Tikhonov floors for the least-squares solve.
    reg_rel = 1e-10
    reg_abs = 1e-14

    def __init__(self, history=None, beta=1.0, precond=None):
        self.history = history
        self.beta = float(beta)
        self._precond = precond
        self._Z = []
        self._F = []

    def new(self, d, precond=None):
        """Fresh per-solve updater for ``d``-dimensional rows, resolving a None
        ``history`` to ``d``. ``precond`` overrides the preconditioner. Omitted,
        this updater's own is kept."""
        return _AndersonUpdate(
            d if self.history is None else self.history, self.beta,
            self._precond if precond is None else precond)

    def propose(self, z, r, jac=None):
        if self._precond is not None:
            r = self._precond(r)
        self._Z.append(z)
        self._F.append(-r)
        if len(self._Z) > self.history + 1:     # keep at most `history` differences
            self._Z.pop(0)
            self._F.pop(0)

        f_k = self._F[-1]
        if len(self._Z) == 1:                   # no history yet, damped Picard step
            return z + self.beta * f_k

        dZ = torch.stack([self._Z[j] - self._Z[j - 1]
                          for j in range(1, len(self._Z))], dim=-1)
        dF = torch.stack([self._F[j] - self._F[j - 1]
                          for j in range(1, len(self._F))], dim=-1)

        N, _, mk = dF.shape
        # Scale-aware Tikhonov floor for (near-)collinear or zero dF columns.
        scale = (dF * dF).sum(-2).mean(-1)
        reg = self.reg_rel * scale + self.reg_abs
        # Solve the damped least squares by QR on the stacked [dF; sqrt(reg) I],
        # not the normal equations dF^T dF whose squared condition number
        # overflows float64 at a stiff metric's column spread.
        eye = torch.eye(mk, dtype=dF.dtype, device=dF.device)
        A_aug = torch.cat([dF, reg.sqrt().view(-1, 1, 1) * eye], dim=-2)
        b_aug = torch.cat([f_k.unsqueeze(-1), f_k.new_zeros(N, mk, 1)], dim=-2)
        Q, R = torch.linalg.qr(A_aug)
        gamma = torch.linalg.solve_triangular(
            R, Q.transpose(-2, -1) @ b_aug, upper=True)

        return z + self.beta * f_k - ((dZ + self.beta * dF) @ gamma).squeeze(-1)

    def damped(self, factor):
        """Copy with β scaled by ``factor``, same history and preconditioner."""
        return _AndersonUpdate(self.history, self.beta * factor, self._precond)


class _NewtonUpdate:
    """Damped Newton iteration, ``z_{k+1} = z_k − β J(z_k)⁻¹ r_k``.

    Stateless. Requires the residual function to return the Jacobian alongside
    the residual. Any preconditioner is ignored, since the Jacobian solve is
    already the exact one.

    Parameters
    ----------
    beta : float
        Under-relaxation factor in (0, 1].
    """

    needs_jacobian = True

    def __init__(self, beta=1.0, precond=None):
        self.beta = float(beta)

    def new(self, d, precond=None):
        """Fresh per-solve updater for ``d``-dimensional rows."""
        return _NewtonUpdate(self.beta)

    def propose(self, z, r, jac=None):
        step, _ = torch.linalg.solve_ex(jac, r.unsqueeze(-1))
        return z - self.beta * step.squeeze(-1)

    def damped(self, factor):
        """Copy with β scaled by ``factor``."""
        return _NewtonUpdate(self.beta * factor)


_RULES = {"picard": _PicardUpdate, "anderson": _AndersonUpdate,
          "newton": _NewtonUpdate}


def _iterate(residual_fn, z_init, updater, max_iter, tol):
    """Iterate ``updater`` from ``z_init`` until every row's residual max-norm
    is below ``tol`` or the row diverges. Returns a :class:`SolveResult`."""
    wants_jac = updater.needs_jacobian

    def evaluate(z):
        out = residual_fn(z)
        return out if wants_jac else (out, None)

    N = z_init.shape[0]
    z = z_init
    residual, jac = evaluate(z)
    r0 = residual.abs().amax(-1)

    done = torch.zeros(N, dtype=torch.bool, device=z.device)
    iters = torch.full((N,), max_iter, dtype=torch.long, device=z.device)
    residual_norm = r0.clone()

    for i in range(1, max_iter + 1):
        z_next = updater.propose(z, residual, jac)
        residual_next, jac_next = evaluate(z_next)
        r = residual_next.abs().amax(-1)

        keep = done[..., None]
        z = torch.where(keep, z, z_next)
        residual = torch.where(keep, residual, residual_next)
        if wants_jac:
            jac = torch.where(keep[..., None], jac, jac_next)
        residual_norm = torch.where(done, residual_norm, r)

        # Divergence: non-finite, or grown far beyond the start. The tol
        # conjunct spares a warm start whose residual began sub-tol.
        diverged = ~done & (~torch.isfinite(r)
                            | ((r > _DIVERGENCE_GROWTH * r0) & (r > tol)))
        done = done | diverged

        converged = ~done & (residual_norm < tol)
        iters = torch.where(converged, torch.full_like(iters, i), iters)
        done = done | converged

        if bool(done.all()):
            break

    return SolveResult(z.detach(), iters, residual_norm.detach())


class FixedPointSolver:
    """Batched root finder for ``r(z) = 0``, one problem per row of ``z``.

    Parameters
    ----------
    kind : {"picard", "anderson", "newton"}
        Update rule. See the design note at the top of this module.
    damping : float
        Under-relaxation factor β in (0, 1]. Default 1.0, undamped.
    anderson_history : int or None
        History length for ``"anderson"``, at least 1, ignored by the others.
        None resolves per solve to the row dimension.
    max_iter : int
        Iteration cap per solve.
    tol : float
        Convergence tolerance on the residual max-norm, per row.
    fallback_damping : tuple of float
        On non-convergence, re-solve the failed rows with β scaled by each
        factor in turn, each in (0, 1). Empty disables the ladder. Damping does
        not move the root, so this rescues a stuck row rather than rejecting it.
    fallback_iter_scale : int
        Iteration cap of each fallback level, as a multiple of ``max_iter``.

    Attributes
    ----------
    needs_jacobian : bool
        Whether ``residual_fn`` must return ``(r, J)`` rather than ``r``.

    Raises
    ------
    ValueError
        For an unknown ``kind``, a ``damping`` outside (0, 1], an
        ``anderson_history`` below 1, or a ``fallback_damping`` factor outside
        (0, 1).
    """

    def __init__(self, kind="picard", *, damping=1.0, anderson_history=None,
                 max_iter=100, tol=1e-8, fallback_damping=(),
                 fallback_iter_scale=4):
        if kind not in _RULES:
            raise ValueError(
                f"unknown solver {kind!r}, expected one of {sorted(_RULES)}")
        if not 0.0 < damping <= 1.0:
            raise ValueError(f"damping must be in (0, 1], got {damping}")
        if anderson_history is not None and anderson_history < 1:
            raise ValueError(
                f"anderson_history must be >= 1, got {anderson_history}")
        if any(not 0.0 < f < 1.0 for f in fallback_damping):
            raise ValueError(
                f"fallback_damping factors must be in (0, 1), got {fallback_damping}")

        self.kind = kind
        self.max_iter = max_iter
        self.tol = tol
        if kind == "anderson":
            self._rule = _AndersonUpdate(anderson_history, damping)
        else:
            self._rule = _RULES[kind](damping)
        self._fallback = [(f, fallback_iter_scale * max_iter)
                          for f in fallback_damping]

    @property
    def needs_jacobian(self):
        return self._rule.needs_jacobian

    def solve(self, residual_fn, z_init, *, precond=None, cold_start=None):
        """Drive ``residual_fn`` to zero from ``z_init``.

        Parameters
        ----------
        residual_fn : callable
            ``z -> r``, or ``z -> (r, J)`` when :attr:`needs_jacobian`. ``z``
            and ``r`` are ``(N, d)`` and ``J`` is ``(N, d, d)``.
        z_init : (N, d)
            Starting iterate. A warm start changes the iteration count and not
            the root, provided the root is unique.
        precond : callable or None
            Fixed preconditioner applied to the residual, ``(N, d) -> (N, d)``.
            Ignored by ``"newton"``.
        cold_start : (N, d) or None
            Iterate the fallback ladder restarts from. Defaults to ``z_init``.

        Returns
        -------
        SolveResult
            ``z`` the root, ``iters`` the per-row iteration count, and
            ``residual`` the per-row final max-norm. A row whose ``residual``
            exceeds :attr:`tol` did not converge, and its ``z`` is not a root.
        """
        d = z_init.shape[-1]
        result = _iterate(residual_fn, z_init, self._rule.new(d, precond),
                          self.max_iter, self.tol)
        if not self._fallback:
            return result

        z, iters, residual = result
        restart = z_init if cold_start is None else cold_start
        for factor, fb_iter in self._fallback:
            bad = residual > self.tol
            if not bool(bad.any()):
                break
            retry = _iterate(residual_fn, restart,
                             self._rule.damped(factor).new(d, precond),
                             fb_iter, self.tol)
            commit = bad.unsqueeze(-1)
            z = torch.where(commit, retry.z, z)
            # Honest cost: the failed rows paid for both passes.
            iters = torch.where(bad, iters + retry.iters, iters)
            residual = torch.where(bad, retry.residual, residual)
        return SolveResult(z, iters, residual)
