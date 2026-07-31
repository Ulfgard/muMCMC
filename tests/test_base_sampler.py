"""Tests for ``MCMCSampler`` -- the shared posterior assembly and batched driver.

Two responsibilities live here, independent of any concrete sampler:

* :meth:`evaluate_model` assembles, on the free variables the chain runs on,

      U(theta) = U_lik(theta_full) - log p(theta)

  (plus, when ``requires_metric``, the free block of G_lik plus the prior's
  metric). We check each term of that composition in isolation: the prior
  term, the full-width vector the user's ``potential_fn`` is handed,
  fixed-coordinate splicing, and the metric assembly (with and without a
  prior metric).

* :meth:`run_mcmc` is the batched driver: ``init`` once, ``step`` per
  iteration, ``end_warmup`` exactly at the warmup boundary, collecting only
  post-warmup states and returning them keyed by name.  We drive it with a
  tiny recording sampler so the mechanics are testable without a real
  integrator.
"""
import torch
import pytest
from pyro.distributions import Normal

from muMCMC.MCMCSampler import MCMCSampler
from muMCMC.spaces import NormalSpace, LogNormalSpace, UnnormalizedSpace

torch.set_default_dtype(torch.float64)

ATOL = 1e-9


# --------------------------------------------------------------------------- #
#  Minimal concrete sampler: records driver calls; step adds a constant.      #
# --------------------------------------------------------------------------- #

class _State:
    def __init__(self, q):
        self.q = q


class _RecordingSampler(MCMCSampler):
    def __init__(self, space, potential_fn=None, *, requires_metric=False, delta=1.0):
        super().__init__(
            potential_fn=potential_fn
            or (lambda th: torch.zeros(th.shape[:-1], dtype=th.dtype)),
            space=space,
            requires_metric=requires_metric,
        )
        self.calls = {"init": 0, "step": 0, "end_warmup": 0}
        self.end_warmup_at_step = None
        self.delta = delta

    def init(self, q):
        self.calls["init"] += 1
        return _State(q.clone())

    def step(self, s):
        self.calls["step"] += 1
        return _State(s.q + self.delta)

    def end_warmup(self):
        self.calls["end_warmup"] += 1
        # how many steps had run when warmup ended
        self.end_warmup_at_step = self.calls["step"]


# ========================================================================== #
#  evaluate_model: potential composition                                     #
# ========================================================================== #

def test_potential_adds_prior_on_identity_space():
    # Nothing is transformed, so U = U_lik - log prior.
    names = ["a", "b"]
    space = NormalSpace(names)
    s = _RecordingSampler(space, potential_fn=lambda th: 0.5 * (th ** 2).sum(-1))

    theta = torch.randn(5, 2)
    U = s.evaluate_model(theta)[0].value
    u_lik = 0.5 * (theta ** 2).sum(-1)
    prior_lp = Normal(0.0, 1.0).log_prob(theta).sum(-1)   # computed independently
    assert U.shape == (5,)
    assert torch.allclose(U, u_lik - prior_lp, atol=ATOL)


def test_potential_is_the_likelihood_plus_the_prior_on_the_variables():
    # The chain runs on the variables, so U = U_lik(theta) - log_prior(theta)
    # with nothing transformed and no Jacobian anywhere.
    space = LogNormalSpace(["x", "y"])
    s = _RecordingSampler(space, potential_fn=lambda th: th.sum(-1))

    theta = torch.rand(6, 2) + 0.5
    expected = theta.sum(-1) - space.prior_log_prob_vector(theta)
    assert torch.allclose(s.evaluate_model(theta)[0].value, expected, atol=ATOL)


def test_potential_splices_fixed_coordinate_and_skips_its_prior():
    # c is fixed at 2.0: potential_fn sees the full vector (so its sum includes
    # the +2.0), while the prior sums over the free names a, b only.
    names = ["a", "b", "c"]
    space = NormalSpace(names, fixed={"c": 2.0})
    s = _RecordingSampler(space, potential_fn=lambda th: th.sum(-1))

    theta = torch.randn(4, 2)                    # free coords a, b
    U = s.evaluate_model(theta)[0].value
    u_lik = theta.sum(-1) + 2.0                  # fixed c spliced in
    prior_lp = Normal(0.0, 1.0).log_prob(theta).sum(-1)   # free names only
    assert torch.allclose(U, u_lik - prior_lp, atol=ATOL)


def test_potential_fn_receives_full_width_vector_with_fixed():
    names = ["a", "b", "c"]
    space = UnnormalizedSpace(names, fixed={"c": 2.0})
    seen = {}

    def potential_fn(theta_full):
        seen["width"] = theta_full.shape[-1]
        return torch.zeros(theta_full.shape[:-1], dtype=theta_full.dtype)

    s = _RecordingSampler(space, potential_fn=potential_fn)
    s.evaluate_model(torch.randn(3, 2))
    assert seen["width"] == 3                    # a, b, c -- fixed included


# ---- metric branch ---------------------------------------------------------

def _metric_model(scale):
    def model(theta):
        n = theta.shape[-1]
        U = 0.5 * (theta ** 2).sum(-1)
        G = scale * torch.eye(n, dtype=theta.dtype).expand(*theta.shape[:-1], n, n)
        return U, G
    return model


def test_metric_branch_returns_the_likelihood_metric():
    # No prior metric, so the assembled metric is just G_lik.
    space = UnnormalizedSpace(["a", "b"])        # no priors, no prior metric
    s = _RecordingSampler(space, potential_fn=_metric_model(2.0), requires_metric=True)

    theta = torch.randn(4, 2)
    potential, metric = s.evaluate_model(theta)
    # U = U_lik, since the space carries no prior
    assert torch.allclose(potential.value, 0.5 * (theta ** 2).sum(-1), atol=ATOL)
    v = torch.randn(4, 2)
    assert torch.allclose(metric.inv_metric_times_vec(v), v / 2.0, atol=ATOL)   # G = 2 I


def test_metric_branch_adds_prior_metric():
    # A Normal(0, s) prior has precision 1/s^2 on the variables, so with a
    # likelihood metric of 2 I the total is (2 + 1/4) I.
    space = NormalSpace(["a", "b"], sigma=2.0)
    s = _RecordingSampler(space, potential_fn=_metric_model(2.0), requires_metric=True)

    theta = torch.randn(4, 2)
    _, metric = s.evaluate_model(theta)
    v = torch.randn(4, 2)
    assert torch.allclose(metric.inv_metric_times_vec(v), v / 2.25, atol=ATOL)


# ========================================================================== #
#  vector <-> coordinate helpers                                             #
# ========================================================================== #

def test_to_full_splices_fixed():
    space = UnnormalizedSpace(["a", "b", "c"], fixed={"c": 2.0})
    theta_free = torch.randn(5, 2)
    full = space.to_full(theta_free)
    assert full.shape == (5, 3)
    assert torch.allclose(full[..., 0], theta_free[..., 0], atol=ATOL)
    assert torch.allclose(full[..., 1], theta_free[..., 1], atol=ATOL)
    assert torch.allclose(full[..., 2], torch.full((5,), 2.0), atol=ATOL)


def test_init_position_is_the_variables_themselves():
    # The chain runs on the variables, so the starting position is the caller's
    # point stacked in free-name order, with no map applied.
    s = _RecordingSampler(LogNormalSpace(["x", "y"]))
    point = {"x": torch.tensor(0.3), "y": torch.tensor(2.0)}
    assert torch.allclose(s._init_position(point),
                          torch.tensor([0.3, 2.0]), atol=ATOL)


def test_init_position_drops_fixed_names():
    s = _RecordingSampler(UnnormalizedSpace(["a", "b", "c"], fixed={"c": 9.0}))
    point = {"a": torch.tensor(1.0), "b": torch.tensor(2.0),
             "c": torch.tensor(9.0)}
    q = s._init_position(point)
    assert q.shape == (2,)                         # only free a, b
    assert torch.allclose(q, torch.tensor([1.0, 2.0]), atol=ATOL)


def test_init_position_rejects_a_missing_name():
    s = _RecordingSampler(UnnormalizedSpace(["a", "b"]))
    with pytest.raises(ValueError, match="missing"):
        s._init_position({"a": torch.tensor(1.0)})


# ========================================================================== #
#  run_mcmc: batched driver mechanics                                        #
# ========================================================================== #

def test_driver_calls_and_warmup_boundary():
    space = UnnormalizedSpace(["a", "b"])
    s = _RecordingSampler(space)
    out = s.run_mcmc({n: torch.tensor(0.0) for n in s.space.free_names}, num_samples=5, num_warmup_steps=3,
                     num_chains=4, disable_progbar=True)
    assert s.calls["init"] == 1
    assert s.calls["step"] == 3 + 5              # warmup + sampling
    assert s.calls["end_warmup"] == 1
    assert s.end_warmup_at_step == 3             # exactly at the boundary
    assert set(out) == {"a", "b"}
    assert out["a"].shape == (4, 5)              # (num_chains, num_samples)


def test_driver_collects_only_post_warmup_states_in_order():
    # step adds delta=1 each call from q0=0, so the j-th collected sample is
    # (num_warmup + 1 + j): a deterministic check of "collect post-warmup,
    # grouped (chain, sample)" plus the identity map-back.
    space = UnnormalizedSpace(["a", "b"])
    s = _RecordingSampler(space, delta=1.0)
    W, S = 4, 6
    out = s.run_mcmc({n: torch.tensor(0.0) for n in s.space.free_names}, num_samples=S, num_warmup_steps=W,
                     num_chains=2, disable_progbar=True)
    expected_row = torch.arange(W + 1, W + S + 1, dtype=torch.get_default_dtype())
    assert torch.allclose(out["a"][0], expected_row, atol=ATOL)
    assert torch.allclose(out["a"][1], expected_row, atol=ATOL)


def test_driver_zero_warmup_is_clean():
    space = UnnormalizedSpace(["a", "b"])
    s = _RecordingSampler(space)
    out = s.run_mcmc({n: torch.tensor(0.0) for n in s.space.free_names}, num_samples=4, num_warmup_steps=0,
                     num_chains=3, disable_progbar=True)
    assert s.calls["step"] == 4
    assert s.calls["end_warmup"] == 1
    assert s.end_warmup_at_step == 0             # called before the first step
    assert out["a"].shape == (3, 4)


def test_driver_default_single_chain_shape():
    space = UnnormalizedSpace(["a", "b"])
    s = _RecordingSampler(space)
    out = s.run_mcmc({n: torch.tensor(0.0) for n in s.space.free_names}, num_samples=4, num_warmup_steps=2,
                     disable_progbar=True)
    assert out["a"].shape == (1, 4)


def test_driver_splices_fixed_into_output():
    space = UnnormalizedSpace(["a", "b", "c"], fixed={"c": 7.0})
    s = _RecordingSampler(space)
    out = s.run_mcmc({n: torch.tensor(0.0) for n in s.space.free_names}, num_samples=4, num_warmup_steps=2,
                     num_chains=2, disable_progbar=True)
    assert set(out) == {"a", "b", "c"}
    assert torch.allclose(out["c"], torch.full((2, 4), 7.0), atol=ATOL)


def test_driver_accepts_and_ignores_extra_kwargs():
    # The Pyro path takes mp_context; the base driver must tolerate it.
    space = UnnormalizedSpace(["a", "b"])
    s = _RecordingSampler(space)
    out = s.run_mcmc({n: torch.tensor(0.0) for n in s.space.free_names}, num_samples=3, num_warmup_steps=1,
                     num_chains=2, disable_progbar=True, mp_context="spawn")
    assert out["a"].shape == (2, 3)


# ========================================================================== #
#  default hooks                                                             #
# ========================================================================== #

def test_logging_and_diagnostics_default_empty():
    space = UnnormalizedSpace(["a", "b"])
    s = _RecordingSampler(space)
    assert s.logging() == {}
    assert s.diagnostics() == {}
