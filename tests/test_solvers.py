"""Tests for the shared batched root finders in muMCMC.solvers.

The three update rules solve the same equation, so the checks are:

1. Every rule reaches the same root, and reports the iteration count and final
   residual per row.
2. Newton converges quadratically and takes the exact step on a linear system.
3. Damping does not move the root, and rescues a divergent undamped iteration.
4. The fallback ladder only re-solves the rows that failed.
5. Rows are independent, and a diverged row is reported through its residual
   rather than raising.
6. The residual contract follows from the rule, through needs_jacobian.
"""
import torch
import pytest

from muMCMC.solvers import FixedPointSolver, SolveResult

torch.set_default_dtype(torch.float64)

RULES = ["picard", "anderson", "newton"]


# A nonlinear system with a root that is easy to state. For row n with shift
# s_n, r(z) = z - a * tanh(z) - s_n, whose Jacobian is I - a diag(sech^2 z).
A_NL = 0.6


def _system(shifts):
    """(residual_fn, residual_and_jacobian) for the tanh system above."""
    def r_of(z):
        return z - A_NL * torch.tanh(z) - shifts

    def residual_fn(z):
        return r_of(z)

    def with_jac(z):
        d = z.shape[-1]
        sech2 = 1.0 - torch.tanh(z) ** 2
        jac = torch.eye(d, dtype=z.dtype).expand(z.shape[0], d, d) \
            - A_NL * torch.diag_embed(sech2)
        return r_of(z), jac
    return residual_fn, with_jac


def _rotation_system(c, shift):
    """r(z) = z - R z - shift with R = [[0, -c], [c, 0]].

    The Picard map is z <- R z + shift, whose eigenvalues are +-i c. Undamped it
    diverges for c > 1, while relaxing by beta gives eigenvalues (1-beta) +- i
    beta c, back inside the unit circle for small enough beta. This is the shape
    the implicit-midpoint iteration has, so it is what damping is for.
    """
    R = torch.tensor([[0.0, -c], [c, 0.0]])

    def residual_fn(z):
        return z - (R @ z.unsqueeze(-1)).squeeze(-1) - shift
    return residual_fn


def _linear_system(M, b):
    """(residual_fn, residual_and_jacobian) for r(z) = M z - b."""
    d = M.shape[-1]

    def residual_fn(z):
        return (M @ z.unsqueeze(-1)).squeeze(-1) - b

    def with_jac(z):
        return residual_fn(z), M.expand(z.shape[0], d, d)
    return residual_fn, with_jac


def _solve(kind, shifts, z0, **kw):
    s = FixedPointSolver(kind, max_iter=kw.pop("max_iter", 200),
                         tol=kw.pop("tol", 1e-12), **kw)
    plain, with_jac = _system(shifts)
    return s.solve(with_jac if s.needs_jacobian else plain, z0)


# ========================================================================== #
#  1. Every rule finds the same root                                         #
# ========================================================================== #

@pytest.mark.parametrize("kind", RULES)
def test_rule_finds_the_root(kind):
    shifts = torch.tensor([[0.3, -0.7], [1.2, 0.05], [-2.0, 3.0]])
    out = _solve(kind, shifts, torch.zeros_like(shifts))
    assert isinstance(out, SolveResult)
    # Residual recomputed from the system, not read off the solver's own value.
    plain, _ = _system(shifts)
    assert float(plain(out.z).abs().max()) < 1e-11
    assert torch.all(out.residual < 1e-12)
    assert torch.all(out.iters >= 1)


def test_rules_agree_on_the_root():
    shifts = torch.tensor([[0.3, -0.7], [1.2, 0.05]])
    roots = [_solve(k, shifts, torch.zeros_like(shifts)).z for k in RULES]
    for other in roots[1:]:
        assert torch.allclose(roots[0], other, atol=1e-10)


# ========================================================================== #
#  2. Newton: quadratic convergence, and exact on a linear system            #
# ========================================================================== #

def test_newton_converges_quadratically():
    # Newton squares the residual each step near the root, so it needs far fewer
    # iterations than the linearly convergent Picard on the same system.
    shifts = torch.tensor([[1.5, -1.5]])
    z0 = torch.zeros_like(shifts)
    it_newton = int(_solve("newton", shifts, z0).iters.max())
    it_picard = int(_solve("picard", shifts, z0).iters.max())
    assert it_newton < it_picard
    assert it_newton <= 8            # quadratic from a cold start on this system


def test_newton_solves_a_linear_system_in_one_step():
    # r(z) = M z - b has a constant Jacobian M, so the first Newton step lands
    # on the root exactly.
    torch.manual_seed(0)
    d = 3
    M = torch.eye(d) + 0.3 * torch.randn(d, d)
    b = torch.randn(2, d)
    _, with_jac = _linear_system(M, b)

    s = FixedPointSolver("newton", max_iter=50, tol=1e-12)
    out = s.solve(with_jac, torch.zeros(2, d))
    exact = torch.linalg.solve(M, b.transpose(-2, -1)).transpose(-2, -1)
    assert torch.allclose(out.z, exact, atol=1e-12)
    assert torch.all(out.iters == 1)


# ========================================================================== #
#  3. Damping                                                                #
# ========================================================================== #

@pytest.mark.parametrize("kind", RULES)
def test_damping_leaves_the_root_alone(kind):
    shifts = torch.tensor([[0.4, -1.1]])
    z0 = torch.zeros_like(shifts)
    full = _solve(kind, shifts, z0).z
    damped = _solve(kind, shifts, z0, damping=0.5).z
    assert torch.allclose(full, damped, atol=1e-10)


def test_damping_rescues_a_divergent_picard_iteration():
    shift = torch.tensor([[0.5, -0.2]])
    r = _rotation_system(1.5, shift)
    z0 = torch.zeros(1, 2)

    undamped = FixedPointSolver("picard", max_iter=300, tol=1e-10).solve(r, z0)
    damped = FixedPointSolver("picard", max_iter=300, tol=1e-10,
                              damping=0.3).solve(r, z0)
    assert float(undamped.residual) > 1e-10          # |+-i 1.5| > 1
    assert float(damped.residual) < 1e-10
    # ...and to the genuine root, which damping does not move.
    assert float(r(damped.z).abs().max()) < 1e-9


# ========================================================================== #
#  4. The fallback ladder                                                    #
# ========================================================================== #

def test_fallback_ladder_only_resolves_the_failed_rows():
    # Row 0 has c < 1 and converges undamped. Row 1 has c > 1 and does not, so
    # only it should be re-solved.
    shift = torch.tensor([[0.5, -0.2], [0.5, -0.2]])
    R = torch.stack([torch.tensor([[0.0, -0.4], [0.4, 0.0]]),
                     torch.tensor([[0.0, -1.5], [1.5, 0.0]])])

    def r(z):
        return z - (R @ z.unsqueeze(-1)).squeeze(-1) - shift

    z0 = torch.zeros(2, 2)
    base = FixedPointSolver("picard", max_iter=80, tol=1e-10).solve(r, z0)
    ladder = FixedPointSolver("picard", max_iter=80, tol=1e-10,
                              fallback_damping=(0.3,),
                              fallback_iter_scale=6).solve(r, z0)

    assert float(base.residual[1]) > 1e-10           # failed without the ladder
    assert float(ladder.residual[1]) < 1e-10         # rescued by it
    assert torch.equal(base.z[0], ladder.z[0])       # converged row untouched
    assert int(base.iters[0]) == int(ladder.iters[0])


def test_empty_fallback_is_a_plain_single_pass():
    shifts = torch.tensor([[0.3, -0.7]])
    z0 = torch.zeros_like(shifts)
    a = _solve("picard", shifts, z0)
    b = _solve("picard", shifts, z0, fallback_damping=())
    assert torch.equal(a.z, b.z) and torch.equal(a.iters, b.iters)


# ========================================================================== #
#  5. Row independence and divergence reporting                              #
# ========================================================================== #

def test_a_diverged_row_is_reported_not_raised():
    # A singular Jacobian sends the Newton step non-finite. That is the row's
    # own problem: it comes back through the residual, the others still solve.
    d = 2

    def with_jac(z):
        jac = torch.eye(d, dtype=z.dtype).expand(z.shape[0], d, d).clone()
        jac[1] = 0.0                                  # row 1 singular
        return z - 1.0, jac

    s = FixedPointSolver("newton", max_iter=20, tol=1e-12)
    out = s.solve(with_jac, torch.zeros(2, d))
    assert float(out.residual[0]) < 1e-12             # row 0 fine
    assert not bool(out.residual[1] <= s.tol)         # row 1 flagged
    assert torch.isfinite(out.z[0]).all()


@pytest.mark.parametrize("kind", RULES)
def test_rows_are_independent(kind):
    # Solving rows together must give what solving them one at a time gives.
    shifts = torch.tensor([[0.3, -0.7], [1.2, 0.05], [-2.0, 3.0]])
    together = _solve(kind, shifts, torch.zeros_like(shifts)).z
    for i in range(shifts.shape[0]):
        alone = _solve(kind, shifts[i:i + 1], torch.zeros(1, 2)).z
        assert torch.allclose(together[i:i + 1], alone, atol=1e-10)


# ========================================================================== #
#  6. The residual contract, and constructor validation                      #
# ========================================================================== #

def test_needs_jacobian_follows_the_rule():
    assert FixedPointSolver("newton").needs_jacobian
    assert not FixedPointSolver("picard").needs_jacobian
    assert not FixedPointSolver("anderson").needs_jacobian


def test_preconditioner_leaves_the_root_alone():
    # P rescales the step, so it changes the iteration count and not the root.
    # On a linear system P = M^-1 makes Picard a frozen-Jacobian Newton, which
    # lands in one step.
    torch.manual_seed(2)
    d = 3
    M = torch.eye(d) + 0.25 * torch.randn(d, d)
    b = torch.randn(2, d)
    plain, _ = _linear_system(M, b)
    Minv = torch.linalg.inv(M)

    s = FixedPointSolver("picard", max_iter=400, tol=1e-12)
    bare = s.solve(plain, torch.zeros(2, d))
    pre = s.solve(plain, torch.zeros(2, d),
                  precond=lambda r: (Minv @ r.unsqueeze(-1)).squeeze(-1))

    exact = torch.linalg.solve(M, b.transpose(-2, -1)).transpose(-2, -1)
    assert torch.allclose(pre.z, exact, atol=1e-12)
    assert torch.allclose(bare.z, pre.z, atol=1e-9)
    assert torch.all(pre.iters == 1) and int(bare.iters.max()) > 1


@pytest.mark.parametrize("kwargs, match", [
    ({"kind": "secant"}, "unknown solver"),
    ({"damping": 0.0}, "damping"),
    ({"damping": 1.5}, "damping"),
    ({"anderson_history": 0}, "anderson_history"),
    ({"fallback_damping": (1.5,)}, "fallback_damping"),
])
def test_invalid_configuration_raises(kwargs, match):
    kind = kwargs.pop("kind", "picard")
    with pytest.raises(ValueError, match=match):
        FixedPointSolver(kind, **kwargs)
