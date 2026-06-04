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
    """A bare endpoint state at a constant position, with the accumulators
    accept()/_bookkeep expect already populated."""
    s = RMHMCState(torch.full((N, D), float(q_val)))
    s.p = torch.zeros(N, D)
    s.max_residual = torch.zeros(N)
    s.fp_iters = [torch.zeros(N, dtype=torch.long)]
    return s


# ========================================================================== #
#  init                                                                      #
# ========================================================================== #

def test_init_tensorises_step_size_and_resets_counters():
    s = make_sampler(step_size=0.25, adapt=True, num_steps=4)
    state = s.init(torch.zeros(5, D))
    assert s.step_size.shape == (5,)
    assert torch.allclose(s.step_size, torch.full((5,), 0.25))
    assert s._step == 0
    assert torch.equal(s._accepted, torch.zeros(5, dtype=torch.long))
    assert torch.equal(s._num_divergences, torch.zeros(5, dtype=torch.long))
    assert s._adapting is True
    # adapter seeded at log(step_size)
    assert torch.allclose(s._adapter.prox_center, torch.log(s.step_size))
    # initial state is complete and has momentum
    assert state.U is not None and state.metric is not None
    assert state.p.shape == (5, D)
    assert torch.allclose(state.max_residual, torch.zeros(5))


def test_init_without_adaptation_does_not_arm_adapter():
    s = make_sampler(adapt=False)
    s.init(torch.zeros(2, D))
    assert s._adapting is False


# ========================================================================== #
#  RMHMCState: complete / reorder                                            #
# ========================================================================== #

def test_complete_fills_then_is_idempotent():
    s = make_sampler(model_qdep)
    st = RMHMCState(torch.randn(2, D))
    assert st.U is None and st.metric is None
    st.complete(s.evaluate_model)
    U_ref, metric_ref = st.U, st.metric
    assert U_ref is not None and metric_ref is not None
    st.complete(s.evaluate_model)                 # no-op on a complete state
    assert st.U is U_ref and st.metric is metric_ref


def test_reorder_permutes_config_but_not_slot_diagnostics():
    s = make_sampler(model_qdep)
    state = s.init(torch.randn(3, D))             # q, p, U, metric all present
    state.max_residual = torch.tensor([1.0, 2.0, 3.0])
    state.fp_iters = [torch.tensor([4, 5, 6])]
    perm = torch.tensor([2, 0, 1])
    r = state.reorder(perm)

    assert torch.equal(r.q, state.q[perm])
    assert torch.equal(r.p, state.p[perm])
    assert torch.equal(r.U, state.U[perm])
    # metric travels with the configuration (log-det follows the permutation)
    assert torch.allclose(r.metric.log_det_metric(),
                          state.metric.log_det_metric()[perm])
    # integrator diagnostics are slot-bound: NOT permuted
    assert torch.equal(r.max_residual, state.max_residual)
    assert r.fp_iters is state.fp_iters


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
    new = _endpoint(50.0, N=2)                    # huge U -> delta_H >> 0
    out = s.accept(new, old)
    assert torch.allclose(out.q, old.q)           # rejected: kept the start
    assert torch.equal(s._accepted, torch.zeros(2, dtype=torch.long))


def test_accept_accepts_when_endpoint_energy_far_lower():
    s = make_sampler(adapt=False, num_steps=1)
    old = s.init(torch.full((2, D), 50.0))        # high energy far out
    new = _endpoint(0.0, N=2)                     # low U at origin
    out = s.accept(new, old)
    assert torch.allclose(out.q, new.q)           # accepted: moved to endpoint
    assert torch.equal(s._accepted, torch.ones(2, dtype=torch.long))


def test_accept_counts_divergence_over_threshold():
    s = make_sampler(adapt=False, num_steps=1, divergence_threshold=100.0)
    old = s.init(torch.zeros(2, D))
    new = _endpoint(50.0, N=2)                    # delta_H >> 100
    s.accept(new, old)
    assert torch.equal(s._num_divergences, torch.ones(2, dtype=torch.long))


def test_accept_returns_complete_ready_state_with_fresh_momentum():
    torch.manual_seed(0)
    s = make_sampler(adapt=False, num_steps=1)
    old = s.init(torch.zeros(3, D))
    p_old = old.p.clone()
    new = _endpoint(0.5, N=3)
    out = s.accept(new, old)
    assert out.U is not None and out.metric is not None
    assert out.p is not None and not torch.allclose(out.p, p_old)   # resampled
    assert torch.allclose(out.max_residual, torch.zeros(3))


# ========================================================================== #
#  step: composes num_steps leapfrogs + accept                               #
# ========================================================================== #

def test_step_runs_exactly_num_steps_leapfrogs():
    s = make_sampler(adapt=False, num_steps=4)
    state = s.init(torch.zeros(2, D))
    calls = {"n": 0}
    original = s.leapfrog_step

    def counting(x):
        calls["n"] += 1
        return original(x)

    s.leapfrog_step = counting
    s.step(state)
    assert calls["n"] == 4


def test_step_returns_ready_state():
    s = make_sampler(adapt=False, num_steps=3)
    state = s.init(torch.zeros(2, D))
    out = s.step(state)
    assert out.U is not None and out.metric is not None and out.p is not None
    assert out.q.shape == (2, D)
    assert torch.allclose(out.max_residual, torch.zeros(2))


# ========================================================================== #
#  end_warmup: freeze step size, reset counters                             #
# ========================================================================== #

def test_end_warmup_freezes_to_adapter_average_and_resets():
    s = make_sampler(adapt=True, num_steps=2)
    state = s.init(torch.zeros(2, D))
    for _ in range(5):
        state = s.step(state)
    s.end_warmup()
    assert s._adapting is False
    assert torch.allclose(s.step_size, torch.exp(s._adapter.get_state()[1]))
    assert torch.equal(s._accepted, torch.zeros(2, dtype=torch.long))
    assert torch.equal(s._num_divergences, torch.zeros(2, dtype=torch.long))
    assert s._step == 0
    assert s._delta_Hs == [] and s._residuals == [] and s._fp_iters == []


def test_end_warmup_without_adaptation_keeps_step_size():
    s = make_sampler(adapt=False, step_size=0.37, num_steps=2)
    state = s.init(torch.zeros(2, D))
    for _ in range(3):
        state = s.step(state)
    s.end_warmup()
    assert s._adapting is False
    assert torch.allclose(s.step_size, torch.full((2,), 0.37))
    assert s._step == 0


# ========================================================================== #
#  trajectory_length / logging                                              #
# ========================================================================== #

def test_trajectory_length_scalar_and_tensor_step_size():
    s = make_sampler(step_size=0.2, num_steps=5)
    assert abs(s.trajectory_length - 1.0) < 1e-12        # scalar before init
    s.init(torch.zeros(2, D))
    assert abs(s.trajectory_length - 1.0) < 1e-12        # mean over (N,) after


def test_logging_empty_before_steps_then_populated():
    s = make_sampler(adapt=False, num_steps=2)
    state = s.init(torch.zeros(2, D))
    assert s.logging() == {}                              # _step == 0
    s.step(state)
    assert set(s.logging()) == {"eps", "|dH|", "|r|", "acc. prob"}
