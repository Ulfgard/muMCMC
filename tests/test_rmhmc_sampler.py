"""Sampler-level tests for RMHMC: the operator interface around the integrator.

The integrator internals are covered in test_rmhmc_integrator.py; here we test
the transition machinery that wraps them -- init, step/accept, divergence
accounting, warmup freeze, and the parallel-tempering reorder -- using
controlled inputs (e.g. forcing a huge energy gap so the Metropolis decision is
deterministic) rather than statistical recovery (which test_samplers.py covers).
"""
import torch
import pytest

from muMCMC.RMHMC import RMHMC, RMHMCState
from muMCMC.adapters import Reinforce, NoAdaptation
from muMCMC.spaces import UnconstrainedSpace

torch.set_default_dtype(torch.float64)

D = 3


def model_simple(theta):
    """U = 1/2 |theta|^2 with a constant identity metric."""
    U = 0.5 * (theta ** 2).sum(-1)
    n = theta.shape[-1]
    return U, torch.eye(n, dtype=theta.dtype).expand(*theta.shape[:-1], n, n)


def model_qdep(theta):
    """Likelihood and metric both depend on q (metric SPD by construction)."""
    U = 0.5 * (theta ** 2).sum(-1)
    n = theta.shape[-1]
    G = torch.eye(n, dtype=theta.dtype) + 0.3 * theta[..., :, None] * theta[..., None, :]
    return U, G


def make_sampler(model_fn=model_simple, *, adapt=False, **kw):
    space = UnconstrainedSpace([f"x{i}" for i in range(D)])
    return RMHMC(model_fn, space, adapt_step_size=adapt, **kw)


def _endpoint(q_val, N):
    """A bare endpoint state at a constant position (config-bound fields only;
    the trajectory residual / iteration scratch lives on the sampler, zeroed by
    init())."""
    s = RMHMCState(torch.full((N, D), float(q_val)))
    s.p = torch.zeros(N, D)
    return s


# ========================================================================== #
#  init                                                                      #
# ========================================================================== #

def test_init_sizes_step_size_and_resets_counters():
    s = make_sampler(step_size=0.25, adapt=False, num_steps=4)
    state = s.init(torch.zeros(5, D))
    assert s.step_size.shape == (5,)
    assert torch.allclose(s.step_size, torch.full((5,), 0.25))
    assert s._step == 0
    assert torch.equal(s._accepted, torch.zeros(5, dtype=torch.long))
    assert torch.equal(s._num_divergences, torch.zeros(5, dtype=torch.long))
    # initial state is complete; step() samples the momentum
    assert state.U is not None and state.metric is not None
    assert state.p is None
    # run-level solver summaries live on the sampler, zeroed
    assert torch.allclose(s._residual_sum, torch.zeros(5))
    assert torch.allclose(s._fp_iters_sum, torch.zeros(5))


def test_init_selects_adapter_by_flag():
    assert isinstance(make_sampler(adapt=True)._step_size_adapter, Reinforce)
    assert isinstance(make_sampler(adapt=False)._step_size_adapter, NoAdaptation)


# ========================================================================== #
#  RMHMCState: reorder                                                       #
# ========================================================================== #

def test_reorder_permutes_config():
    # The state is purely config-bound, so every field follows the permutation
    # (the integrator's residual / iteration diagnostics are slot-bound and live
    # on the sampler, not the state).
    s = make_sampler(model_qdep)
    state = s.init(torch.randn(3, D))             # q, U, metric present
    state.p = state.metric.sample_momentum()      # give it momentum, as step() would
    perm = torch.tensor([2, 0, 1])
    r = state.reorder(perm)

    assert torch.equal(r.q, state.q[perm])
    assert torch.equal(r.p, state.p[perm])
    assert torch.equal(r.U.value, state.U.value[perm])
    # metric travels with the configuration (log-det follows the permutation)
    assert torch.allclose(r.metric.log_det_metric(),
                          state.metric.log_det_metric()[perm])


def test_reorder_leaves_absent_fields_none():
    state = RMHMCState(torch.randn(3, D))         # p, U, metric all None
    r = state.reorder(torch.tensor([2, 0, 1]))
    assert torch.equal(r.q, state.q[[2, 0, 1]])
    assert r.p is None and r.U is None and r.metric is None


# ========================================================================== #
#  accept: Metropolis decision, accounting, momentum                         #
# ========================================================================== #

def test_accept_rejects_when_endpoint_energy_far_higher():
    s = make_sampler(adapt=False, num_steps=1)
    old = s.init(torch.zeros(2, D))               # low energy at the origin
    old.p = torch.zeros_like(old.q)               # start momentum (step() would sample it)
    new = _endpoint(50.0, N=2)                    # huge U -> delta_H >> 0
    out = s.accept(new, old)
    assert torch.allclose(out.q, old.q)           # rejected: kept the start
    assert torch.equal(s._accepted, torch.zeros(2, dtype=torch.long))


def test_accept_accepts_when_endpoint_energy_far_lower():
    s = make_sampler(adapt=False, num_steps=1)
    old = s.init(torch.full((2, D), 50.0))        # high energy far out
    old.p = torch.zeros_like(old.q)
    new = _endpoint(0.0, N=2)                     # low U at origin
    out = s.accept(new, old)
    assert torch.allclose(out.q, new.q)           # accepted: moved to endpoint
    assert torch.equal(s._accepted, torch.ones(2, dtype=torch.long))


def test_accept_counts_divergence_over_threshold():
    s = make_sampler(adapt=False, num_steps=1, divergence_threshold=100.0)
    old = s.init(torch.zeros(2, D))
    old.p = torch.zeros_like(old.q)
    new = _endpoint(50.0, N=2)                    # delta_H >> 100
    s.accept(new, old)
    assert torch.equal(s._num_divergences, torch.ones(2, dtype=torch.long))


def test_accept_counts_large_energy_drop_as_divergence():
    # Divergence is on |delta_H|: a huge energy *drop* (not only a rise) is
    # numerically suspect and must register, even though it is accepted.
    s = make_sampler(adapt=False, num_steps=1, divergence_threshold=100.0)
    old = s.init(torch.full((2, D), 50.0))        # high energy
    old.p = torch.zeros_like(old.q)
    new = _endpoint(0.0, N=2)                      # delta_H << -100
    out = s.accept(new, old)
    assert torch.allclose(out.q, new.q)           # accepted (energy dropped)
    assert torch.equal(s._num_divergences, torch.ones(2, dtype=torch.long))


def test_failed_solve_is_rejected_even_when_energy_matches():
    # A fixed-point solve that did not converge (max residual over fp_tol) is not
    # a valid proposal: it must be rejected and counted divergent even when the
    # energy change is small. acceptance_delta forces +inf for those chains.
    s = make_sampler(adapt=False, num_steps=1, fp_tol=1e-8)
    old = s.init(torch.zeros(2, D))
    old.p = old.metric.sample_momentum()
    new = _endpoint(0.0, N=2)                      # same position: delta_H ~ 0
    s._step_residual = torch.full((2,), 1e-3)      # solve failed (>> fp_tol)
    out = s.accept(new, old)
    assert torch.allclose(out.q, old.q)           # rejected despite small |delta_H|
    assert torch.equal(s._num_divergences, torch.ones(2, dtype=torch.long))


def test_accept_returns_complete_state():
    torch.manual_seed(0)
    s = make_sampler(adapt=False, num_steps=1)
    old = s.init(torch.zeros(3, D))
    old.p = old.metric.sample_momentum()          # start momentum, as step() sets it
    new = _endpoint(0.5, N=3)
    out = s.accept(new, old)
    assert out.U is not None and out.metric is not None
    assert out.p is not None                      # momentum carried; resampled next step


# ========================================================================== #
#  step: composes num_steps leapfrogs + accept                               #
# ========================================================================== #

def test_step_runs_exactly_num_steps_integration_steps():
    s = make_sampler(adapt=False, num_steps=4)
    state = s.init(torch.zeros(2, D))
    calls = {"n": 0}
    original = s.integrate

    def counting(x, step_size):
        calls["n"] += 1
        return original(x, step_size)

    s.integrate = counting
    s.step(state)
    assert calls["n"] == 4


def test_step_returns_complete_state():
    s = make_sampler(adapt=False, num_steps=3)
    state = s.init(torch.zeros(2, D))
    out = s.step(state)
    assert out.U is not None and out.metric is not None and out.p is not None
    assert out.q.shape == (2, D)


# ========================================================================== #
#  end_warmup: freeze step size, reset counters                             #
# ========================================================================== #

def test_end_warmup_freezes_to_adapter_average_and_resets():
    s = make_sampler(adapt=True, num_steps=2)
    state = s.init(torch.zeros(2, D))
    for _ in range(5):
        state = s.step(state)
    s.end_warmup()
    assert torch.allclose(s.step_size, torch.exp(s._step_size_adapter.get_state()[1]))
    assert torch.equal(s._accepted, torch.zeros(2, dtype=torch.long))
    assert torch.equal(s._num_divergences, torch.zeros(2, dtype=torch.long))
    assert s._step == 0
    # running diagnostic summaries are reset to zeros for the sampling phase
    z = torch.zeros(2)
    assert torch.equal(s._delta_H_abs_sum, z) and torch.equal(s._delta_H_abs_max, z)
    assert torch.equal(s._residual_sum, z) and torch.equal(s._residual_max, z)
    assert torch.equal(s._fp_iters_sum, z) and torch.equal(s._fp_iters_max, z)


def test_end_warmup_without_adaptation_keeps_step_size():
    s = make_sampler(adapt=False, step_size=0.37, num_steps=2)
    state = s.init(torch.zeros(2, D))
    for _ in range(3):
        state = s.step(state)
    s.end_warmup()
    assert torch.allclose(s.step_size, torch.full((2,), 0.37))
    assert s._step == 0


# ========================================================================== #
#  logging                                                                   #
# ========================================================================== #

def test_logging_reports_registry_entries():
    s = make_sampler(adapt=False, num_steps=2)
    state = s.init(torch.zeros(2, D))
    # Registry is populated at construction, so logging() reports its keys as
    # soon as init has sized the state -- same contract before and after a step.
    assert set(s.logging()) == {"eps", "acc. prob", "|r|"}
    s.step(state)
    assert set(s.logging()) == {"eps", "acc. prob", "|r|"}


# ========================================================================== #
#  solver argument: picard vs anderson                                       #
# ========================================================================== #

def test_invalid_solver_raises():
    with pytest.raises(ValueError, match="unknown solver"):
        make_sampler(solver="newton")


def test_invalid_anderson_history_raises():
    with pytest.raises(ValueError, match="anderson_history"):
        make_sampler(solver="anderson", anderson_history=0)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_damping_raises(bad):
    with pytest.raises(ValueError, match="damping"):
        make_sampler(damping=bad)


def test_damping_default_matches_undamped_transition():
    # damping=1.0 is the default, so an explicit 1.0 must reproduce it exactly.
    def run(**kw):
        torch.manual_seed(0)
        s = make_sampler(model_qdep, adapt=False, num_steps=3, **kw)
        return s.step(s.init(torch.zeros(4, D))).q

    assert torch.allclose(run(), run(damping=1.0), atol=1e-12)


def test_anderson_solver_runs_and_matches_picard_endpoint():
    # With adaptation off and a shared seed, a full transition (num_steps
    # leapfrogs + accept) must land in the same place under either solver,
    # since both solve the same implicit-midpoint equations.
    def run(solver):
        torch.manual_seed(0)
        s = make_sampler(model_qdep, adapt=False, num_steps=3, solver=solver)
        state = s.init(torch.zeros(4, D))
        state = s.step(state)
        return state.q

    q_picard = run("picard")
    q_anderson = run("anderson")
    assert torch.allclose(q_anderson, q_picard, atol=1e-7)


# ========================================================================== #
#  memory leak: endpoint evals must not pin an autograd graph                #
# ========================================================================== #

def test_endpoint_state_carries_no_autograd_graph():
    """A model whose U/G carry an autograd graph must not leak that graph into
    the sampler: endpoint U/metric, momentum, and the accumulated delta_H
    diagnostics must all be detached so the per-step graph is freed each step
    instead of accumulating over the run (CUDA OOM regression)."""
    scale = torch.tensor(1.5, requires_grad=True)

    def model_grad(theta):
        U, G = model_qdep(theta)
        return scale * U, scale * G

    s = make_sampler(model_grad, adapt=False, num_steps=2)
    state = s.init(torch.zeros(2, D))
    for _ in range(3):
        state = s.step(state)

    assert not s._delta_H_last.requires_grad
    assert not s._delta_H_abs_sum.requires_grad and not s._delta_H_abs_max.requires_grad
    assert not state.U.value.requires_grad
    assert not state.metric.L.requires_grad
    assert not state.p.requires_grad               # momentum carries no graph either


def test_diagnostics_footprint_is_constant_over_steps():
    """The integrator diagnostics are folded into O(num_chains) running
    summaries, so their memory footprint must not grow with the number of
    transitions (regression against the old append-per-step lists that grew
    unbounded and fragmented the heap)."""
    N, num_steps = 3, 4
    summaries = ["_delta_H_last", "_delta_H_abs_sum", "_delta_H_abs_max",
                 "_step_residual", "_step_iters", "_residual_sum", "_residual_max",
                 "_fp_iters_sum", "_fp_iters_max"]

    def footprint(sampler):
        # every accumulator is a fixed (N,) tensor -- no lists, no per-step
        # growth; count elements so a regression to lists would blow this up.
        for name in summaries:
            t = getattr(sampler, name)
            assert isinstance(t, torch.Tensor) and t.shape == (N,)
        return sum(getattr(sampler, name).numel() for name in summaries)

    s = make_sampler(model_qdep, adapt=True, num_steps=num_steps)
    state = s.init(torch.zeros(N, D))
    state = s.step(state)
    after_1 = footprint(s)
    for _ in range(30):
        state = s.step(state)
    after_31 = footprint(s)

    assert after_1 == after_31 == len(summaries) * N

    # diagnostics() exposes the summaries as (num_chains,) tensors under the
    # documented keys (no per-step history).
    diag = s.diagnostics()
    for key in ("delta_H_abs_mean", "delta_H_abs_max", "residual_mean",
                "residual_max", "fp_iters_mean", "fp_iters_max"):
        assert diag[key].shape == (N,)
