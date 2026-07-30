from collections import namedtuple

import torch

# =========================================================================== #
#                                                                             #
#  Batched root finding for r(z) = 0, one problem per row of z                 #
#                                                                             #
#  Internal. A caller picks the update rule by name and the solver builds it,  #
#  so nothing here is part of the package surface.                            #
#                                                                             #
#  A rule reporting needs_jacobian expects residual_fn(z) -> (r, J), with r    #
#  shaped (N, d) and J shaped (N, d, d). The others expect residual_fn(z) -> r #
#  so the contract follows from the rule rather than from a flag on the call.  #
#  Either way only the root matters, and the result comes back detached.       #
#                                                                             #
#      picard     z_{k+1} = z_k − β P r_k                                     #
#      anderson   Picard plus Type-II Anderson acceleration                   #
#      newton     z_{k+1} = z_k − β J(z_k)⁻¹ r_k                              #
#                                                                             #
#  β ∈ (0, 1] under-relaxes every rule without moving the root, so a fallback  #
#  ladder can re-solve a stuck row rather than reject it. P preconditions the  #
#  residual, and is meaningless for newton, whose solve is already exact. J is #
#  not assumed symmetric, and a singular one gives non-finite entries rather   #
#  than raising, which the divergence test catches for that row alone.         #
#                                                                             #
# =========================================================================== #

SolveResult = namedtuple("SolveResult", ["z", "iters", "residual"])

_RULES = ("picard", "anderson", "newton")

# Growth over the starting residual past which a row counts as diverged. Loose,
# since Anderson is not monotone on its way to the root.
_DIVERGENCE_GROWTH = 1000.0

# Relative and absolute Tikhonov floors for Anderson's least squares.
_REG_REL, _REG_ABS = 1e-10, 1e-14


def _picard(beta, precond, _history):
    def step(z, r, jac):
        return z - beta * (r if precond is None else precond(r))
    return step


def _newton(beta, precond, _history):
    def step(z, r, jac):
        dz, _ = torch.linalg.solve_ex(jac, r.unsqueeze(-1))
        return z - beta * dz.squeeze(-1)
    return step


def _anderson(beta, precond, history):
    """Anderson(m), Type II. With f_k = −r_k and the last m iterate and residual
    differences stacked column-wise as ΔZ and ΔF, solve the per-row least squares
    γ = argmin ‖f_k − ΔF γ‖ and take

        z_{k+1} = z_k + β f_k − (ΔZ + β ΔF) γ.
    """
    Z, F = [], []

    def step(z, r, jac):
        Z.append(z)
        F.append(-(r if precond is None else precond(r)))
        if len(Z) > history + 1:                  # keep at most `history` diffs
            Z.pop(0)
            F.pop(0)
        f_k = F[-1]
        if len(Z) == 1:
            return z + beta * f_k

        dZ = torch.stack([Z[j] - Z[j - 1] for j in range(1, len(Z))], dim=-1)
        dF = torch.stack([F[j] - F[j - 1] for j in range(1, len(F))], dim=-1)
        n, _, mk = dF.shape
        # QR on [ΔF; √reg I] rather than the normal equations, whose squared
        # condition number overflows float64 at a stiff column spread. The floor
        # covers near-collinear and zero columns.
        reg = _REG_REL * (dF * dF).sum(-2).mean(-1) + _REG_ABS
        eye = torch.eye(mk, dtype=dF.dtype, device=dF.device)
        Q, R = torch.linalg.qr(torch.cat([dF, reg.sqrt().view(-1, 1, 1) * eye], -2))
        rhs = torch.cat([f_k.unsqueeze(-1), f_k.new_zeros(n, mk, 1)], -2)
        gamma = torch.linalg.solve_triangular(R, Q.transpose(-2, -1) @ rhs,
                                              upper=True)
        return z + beta * f_k - ((dZ + beta * dF) @ gamma).squeeze(-1)
    return step


_BUILD = {"picard": _picard, "anderson": _anderson, "newton": _newton}


def _iterate(residual_fn, z_init, step, wants_jac, max_iter, tol):
    """Drive ``step`` from ``z_init`` until every row is below ``tol`` or has
    diverged. Returns a :class:`SolveResult`."""
    def evaluate(z):
        out = residual_fn(z)
        return out if wants_jac else (out, None)

    z = z_init
    residual, jac = evaluate(z)
    r0 = residual.abs().amax(-1)
    done = torch.zeros_like(r0, dtype=torch.bool)
    iters = torch.full(r0.shape, max_iter, dtype=torch.long, device=z.device)
    norm = r0.clone()

    for i in range(1, max_iter + 1):
        z_next = step(z, residual, jac)
        residual_next, jac_next = evaluate(z_next)
        r = residual_next.abs().amax(-1)

        keep = done[..., None]
        z = torch.where(keep, z, z_next)
        residual = torch.where(keep, residual, residual_next)
        if wants_jac:
            jac = torch.where(keep[..., None], jac, jac_next)
        norm = torch.where(done, norm, r)

        # The tol conjunct spares a warm start whose residual began sub-tol.
        done = done | (~done & (~torch.isfinite(r)
                                | ((r > _DIVERGENCE_GROWTH * r0) & (r > tol))))
        converged = ~done & (norm < tol)
        iters = torch.where(converged, torch.full_like(iters, i), iters)
        done = done | converged
        if bool(done.all()):
            break

    return SolveResult(z.detach(), iters, norm.detach())


class FixedPointSolver:
    """Batched root finder for ``r(z) = 0``, one problem per row of ``z``.

    Parameters
    ----------
    kind : {"picard", "anderson", "newton"}
        Update rule. ``"newton"`` sets :attr:`needs_jacobian`.
    damping : float
        Under-relaxation β in (0, 1]. Default 1.0.
    anderson_history : int or None
        History length for ``"anderson"``, at least 1, ignored by the others.
        None resolves per solve to the row dimension.
    max_iter : int
        Iteration cap per solve.
    tol : float
        Convergence tolerance on the residual max-norm, per row.
    fallback_damping : tuple of float
        On non-convergence, re-solve the failed rows with β scaled by each factor
        in turn, each in (0, 1). Empty disables the ladder.
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
            raise ValueError(f"unknown solver {kind!r}, expected one of {_RULES}")
        if not 0.0 < damping <= 1.0:
            raise ValueError(f"damping must be in (0, 1], got {damping}")
        if anderson_history is not None and anderson_history < 1:
            raise ValueError(
                f"anderson_history must be >= 1, got {anderson_history}")
        if any(not 0.0 < f < 1.0 for f in fallback_damping):
            raise ValueError(
                f"fallback_damping factors must be in (0, 1), got {fallback_damping}")

        self.kind = kind
        self.damping = damping
        self.history = anderson_history
        self.max_iter = max_iter
        self.tol = tol
        self.fallback = [(f, fallback_iter_scale * max_iter)
                         for f in fallback_damping]

    @property
    def needs_jacobian(self):
        return self.kind == "newton"

    def solve(self, residual_fn, z_init, *, precond=None, cold_start=None):
        """Drive ``residual_fn`` to zero from ``z_init``.

        Parameters
        ----------
        residual_fn : callable
            ``z -> r``, or ``z -> (r, J)`` when :attr:`needs_jacobian`.
        z_init : (N, d)
            Starting iterate. A warm start changes the iteration count and not
            the root, provided the root is unique.
        precond : callable or None
            Preconditioner applied to the residual, ``(N, d) -> (N, d)``.
        cold_start : (N, d) or None
            Iterate the fallback ladder restarts from. Defaults to ``z_init``.

        Returns
        -------
        SolveResult
            ``z`` the root, with the per-row iteration count and final residual
            max-norm. A row whose ``residual`` exceeds :attr:`tol` did not
            converge and its ``z`` is not a root.
        """
        d = z_init.shape[-1]
        history = d if self.history is None else self.history
        build = _BUILD[self.kind]

        def run(beta, start, cap):
            return _iterate(residual_fn, start, build(beta, precond, history),
                            self.needs_jacobian, cap, self.tol)

        z, iters, residual = run(self.damping, z_init, self.max_iter)
        restart = z_init if cold_start is None else cold_start
        for factor, cap in self.fallback:
            bad = residual > self.tol
            if not bool(bad.any()):
                break
            retry = run(self.damping * factor, restart, cap)
            z = torch.where(bad.unsqueeze(-1), retry.z, z)
            # Honest cost: the failed rows paid for both passes.
            iters = torch.where(bad, iters + retry.iters, iters)
            residual = torch.where(bad, retry.residual, residual)
        return SolveResult(z, iters, residual)
