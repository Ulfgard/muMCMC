"""Tests for ``BaseSampler`` -- the shared posterior assembly and batched driver.

Two responsibilities live here, independent of any concrete sampler:

* :meth:`evaluate_model` pulls the user's constrained-space ``potential_fn``
  back to unconstrained coordinates and assembles

      U(z) = U_lik(theta(z)) + U_prior(theta(z)) - log|det dtheta/dz|

  (plus, when ``requires_metric``, the pulled-back metric G_lik + G_prior).
  We check each term of that composition in isolation: the prior term on an
  identity space, the Jacobian term on a box space, fixed-coordinate splicing,
  and the metric assembly (with and without a prior metric).

* :meth:`run_mcmc` is the batched driver: ``init`` once, ``step`` per
  iteration, ``end_warmup`` exactly at the warmup boundary, collecting only
  post-warmup states and mapping them back to constrained space.  We drive it
  with a tiny recording sampler so the mechanics are testable without a real
  integrator.
"""
import math

import torch
import pytest
from pyro.distributions import Normal

from muMCMC.BaseSampler import BaseSampler
from muMCMC.spaces import UnconstrainedSpace, UniformBoxSpace, transforms

torch.set_default_dtype(torch.float64)

ATOL = 1e-9


# --------------------------------------------------------------------------- #
#  Minimal concrete sampler: records driver calls; step adds a constant.      #
# --------------------------------------------------------------------------- #

class _State:
    def __init__(self, q):
        self.q = q


class _RecordingSampler(BaseSampler):
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


def _matvec(M, v):
    return (M @ v[..., None])[..., 0]


# ========================================================================== #
#  evaluate_model: potential composition                                     #
# ========================================================================== #

def test_potential_adds_prior_on_identity_space():
    # Identity transform (Jacobian log-det = 0): U = U_lik - log prior.
    names = ["a", "b"]
    space = UnconstrainedSpace(names, priors={n: Normal(0.0, 1.0) for n in names})
    s = _RecordingSampler(space, potential_fn=lambda th: 0.5 * (th ** 2).sum(-1))

    z = torch.randn(5, 2)
    U = s.evaluate_model(z)
    u_lik = 0.5 * (z ** 2).sum(-1)
    prior_lp = Normal(0.0, 1.0).log_prob(z).sum(-1)   # computed independently
    assert U.shape == (5,)
    assert torch.allclose(U, u_lik - prior_lp, atol=ATOL)


def test_potential_subtracts_jacobian_log_det_on_box_space():
    # Uniform box: prior log-prob is 0, so U = U_lik(theta) - log|det J|, and
    # theta passed to potential_fn must be the *constrained* (box) value.
    space = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 4.0)}, ["x", "y"],
                            device="cpu")
    s = _RecordingSampler(space, potential_fn=lambda th: th.sum(-1))

    z = torch.randn(6, 2)
    theta_map = space.map_to_constrained_vector(z)
    theta = theta_map.mapped_point
    expected = theta.sum(-1) - theta_map.jacobian_log_det
    assert torch.allclose(s.evaluate_model(z), expected, atol=ATOL)


def test_potential_splices_fixed_coordinate_and_skips_its_prior():
    # c is fixed at 2.0: potential_fn sees the full vector (so its sum includes
    # the +2.0), while the prior sums over the free names a, b only.
    names = ["a", "b", "c"]
    space = UnconstrainedSpace(names, priors={n: Normal(0.0, 1.0) for n in names},
                               fixed={"c": 2.0})
    s = _RecordingSampler(space, potential_fn=lambda th: th.sum(-1))

    z = torch.randn(4, 2)                       # free coords a, b
    U = s.evaluate_model(z)
    u_lik = z.sum(-1) + 2.0                      # fixed c spliced in
    prior_lp = Normal(0.0, 1.0).log_prob(z).sum(-1)   # free names only
    assert torch.allclose(U, u_lik - prior_lp, atol=ATOL)


def test_potential_fn_receives_full_width_vector_with_fixed():
    names = ["a", "b", "c"]
    space = UnconstrainedSpace(names, fixed={"c": 2.0})
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


def test_metric_branch_returns_pulled_back_likelihood_metric():
    # No prior metric, identity transform: the pulled-back metric is just G_lik.
    space = UnconstrainedSpace(["a", "b"])        # no priors, no prior metric
    s = _RecordingSampler(space, potential_fn=_metric_model(2.0), requires_metric=True)

    z = torch.randn(4, 2)
    U, metric = s.evaluate_model(z)
    # U = U_lik (no prior, identity Jacobian)
    assert torch.allclose(U, 0.5 * (z ** 2).sum(-1), atol=ATOL)
    v = torch.randn(4, 2)
    assert torch.allclose(metric.metric_times_vec(v), 2.0 * v, atol=ATOL)
    assert torch.allclose(metric.inv_metric_times_vec(v), v / 2.0, atol=ATOL)


def test_metric_branch_adds_prior_metric():
    # G_full = G_lik + G_prior = (2 + 3) I, so metric_times_vec(v) == 5 v.
    def prior_metric_fn(theta_full):
        n = theta_full.shape[-1]
        return 3.0 * torch.eye(n, dtype=theta_full.dtype).expand(
            *theta_full.shape[:-1], n, n)

    space = UnconstrainedSpace(["a", "b"], prior_metric_fn=prior_metric_fn)
    s = _RecordingSampler(space, potential_fn=_metric_model(2.0), requires_metric=True)

    z = torch.randn(4, 2)
    _, metric = s.evaluate_model(z)
    v = torch.randn(4, 2)
    assert torch.allclose(metric.metric_times_vec(v), 5.0 * v, atol=ATOL)


# ========================================================================== #
#  vector <-> coordinate helpers                                             #
# ========================================================================== #

def test_free_to_full_splices_fixed():
    space = UnconstrainedSpace(["a", "b", "c"], fixed={"c": 2.0})
    s = _RecordingSampler(space)
    theta_free = torch.randn(5, 2)
    full = s._free_to_full(theta_free)
    assert full.shape == (5, 3)
    assert torch.allclose(full[..., 0], theta_free[..., 0], atol=ATOL)
    assert torch.allclose(full[..., 1], theta_free[..., 1], atol=ATOL)
    assert torch.allclose(full[..., 2], torch.full((5,), 2.0), atol=ATOL)


def test_init_z_free_identity_space_is_passthrough():
    space = UnconstrainedSpace(["a", "b"])
    s = _RecordingSampler(space)
    theta = torch.randn(2)
    assert torch.allclose(s._init_z_free(theta), theta, atol=ATOL)


def test_init_z_free_box_space_unconstrains():
    space = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 4.0)}, ["x", "y"],
                            device="cpu")
    s = _RecordingSampler(space)
    theta = torch.tensor([0.3, 2.0])
    z = s._init_z_free(theta)
    expected = transforms.box_inv(theta, space.l, space.u).mapped_point
    assert torch.allclose(z, expected, atol=ATOL)


def test_init_z_free_drops_fixed_coordinates():
    space = UnconstrainedSpace(["a", "b", "c"], fixed={"c": 9.0})
    s = _RecordingSampler(space)
    theta_full = torch.tensor([1.0, 2.0, 9.0])
    z = s._init_z_free(theta_full)
    assert z.shape == (2,)                        # only free a, b
    assert torch.allclose(z, torch.tensor([1.0, 2.0]), atol=ATOL)


# ========================================================================== #
#  run_mcmc: batched driver mechanics                                        #
# ========================================================================== #

def test_driver_calls_and_warmup_boundary():
    space = UnconstrainedSpace(["a", "b"])
    s = _RecordingSampler(space)
    out = s.run_mcmc(torch.zeros(2), num_samples=5, num_warmup_steps=3,
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
    space = UnconstrainedSpace(["a", "b"])
    s = _RecordingSampler(space, delta=1.0)
    W, S = 4, 6
    out = s.run_mcmc(torch.zeros(2), num_samples=S, num_warmup_steps=W,
                     num_chains=2, disable_progbar=True)
    expected_row = torch.arange(W + 1, W + S + 1, dtype=torch.get_default_dtype())
    assert torch.allclose(out["a"][0], expected_row, atol=ATOL)
    assert torch.allclose(out["a"][1], expected_row, atol=ATOL)


def test_driver_zero_warmup_is_clean():
    space = UnconstrainedSpace(["a", "b"])
    s = _RecordingSampler(space)
    out = s.run_mcmc(torch.zeros(2), num_samples=4, num_warmup_steps=0,
                     num_chains=3, disable_progbar=True)
    assert s.calls["step"] == 4
    assert s.calls["end_warmup"] == 1
    assert s.end_warmup_at_step == 0             # called before the first step
    assert out["a"].shape == (3, 4)


def test_driver_default_single_chain_shape():
    space = UnconstrainedSpace(["a", "b"])
    s = _RecordingSampler(space)
    out = s.run_mcmc(torch.zeros(2), num_samples=4, num_warmup_steps=2,
                     disable_progbar=True)
    assert out["a"].shape == (1, 4)


def test_driver_splices_fixed_into_output():
    space = UnconstrainedSpace(["a", "b", "c"], fixed={"c": 7.0})
    s = _RecordingSampler(space)
    out = s.run_mcmc(torch.zeros(3), num_samples=4, num_warmup_steps=2,
                     num_chains=2, disable_progbar=True)
    assert set(out) == {"a", "b", "c"}
    assert torch.allclose(out["c"], torch.full((2, 4), 7.0), atol=ATOL)


def test_driver_accepts_and_ignores_extra_kwargs():
    # The Pyro path takes mp_context; the base driver must tolerate it.
    space = UnconstrainedSpace(["a", "b"])
    s = _RecordingSampler(space)
    out = s.run_mcmc(torch.zeros(2), num_samples=3, num_warmup_steps=1,
                     num_chains=2, disable_progbar=True, mp_context="spawn")
    assert out["a"].shape == (2, 3)


# ========================================================================== #
#  default hooks                                                             #
# ========================================================================== #

def test_logging_and_diagnostics_default_empty():
    space = UnconstrainedSpace(["a", "b"])
    s = _RecordingSampler(space)
    assert s.logging() == {}
    assert s.diagnostics() == {}
