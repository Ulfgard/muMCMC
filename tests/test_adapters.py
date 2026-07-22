"""Tests for the stochastic optimizers in ``adapters``.

Both follow an ask/tell contract and are vectorised over a leading axis (N
independent problems, no cross-coupling).  We test them as optimizers, not as
step-size adapters:

* ``DualAveraging`` is a first-order subgradient method -- feed it the
  subgradient of a convex objective and the averaged iterate must converge to
  the minimiser, even with noisy subgradients.  It also claims to be byte-for-
  byte equal to ``pyro.ops.dual_averaging`` on scalar inputs, which we pin as a
  regression anchor.
* ``REINFORCEAdapter`` is derivative-free: it minimises a Gaussian-smoothed
  objective from noisy *evaluations* only.  Most tests minimise the bounded,
  smooth f(x) = 1 - exp(-(x-a)**2/2) (min at x = a) at the default learning
  rate.  A steep/unbounded objective (e.g. x**2) can blow up at the default
  gamma; raising gamma (gentler steps) stabilises it -- exercised separately.
"""
import math

import torch
import pytest

from muMCMC.adapters import DualAveraging, REINFORCEAdapter

torch.set_default_dtype(torch.float64)


# ========================================================================== #
#  DualAveraging                                                              #
# ========================================================================== #

def _da_minimise(target, steps, *, prox=0.0, noise=0.0, seed=0):
    """Drive DualAveraging on 0.5*(x-target)^2 (subgradient x-target)."""
    torch.manual_seed(seed)
    da = DualAveraging()
    da.prox_center = prox
    da.reset()
    for _ in range(steps):
        x_t, _ = da.get_state()
        g = x_t - target
        if noise:
            g = g + noise * torch.randn(torch.as_tensor(target).shape)
        da.step(g)
    return da.get_state()


def test_reset_round_trips_before_any_step():
    da = DualAveraging()
    da.prox_center = 1.3
    da.reset()
    x_t, x_avg = da.get_state()
    assert x_t == 1.3 and x_avg == 1.3


def test_zero_subgradient_keeps_iterate_at_prox_center():
    # g == 0 throughout: the dual sequence stays 0, so x_t never leaves prox.
    da = DualAveraging()
    da.prox_center = 2.0
    da.reset()
    for _ in range(50):
        da.step(0.0)
    x_t, x_avg = da.get_state()
    assert x_t == 2.0
    assert abs(x_avg - 2.0) < 1e-12


def test_first_step_overwrites_the_average_seed():
    # At t=1 the averaging weight is 1^-kappa = 1, so x_avg == x_t after one
    # step regardless of the reset seed -- stepped runs don't depend on it.
    da = DualAveraging()
    da.prox_center = 5.0
    da.reset()
    da.step(0.3)
    x_t, x_avg = da.get_state()
    assert x_avg == x_t


def test_converges_to_minimum_exact_subgradients():
    _, x_avg = _da_minimise(3.0, steps=1500)
    assert abs(float(x_avg) - 3.0) < 0.03


def test_converges_under_noisy_subgradients():
    # Averaging is the whole point: zero-mean noise must wash out.
    _, x_avg = _da_minimise(3.0, steps=3000, noise=0.5)
    assert abs(float(x_avg) - 3.0) < 0.08


def test_vectorised_problems_are_independent():
    # Three problems in one (N,) call; chains already at their optimum (g=0)
    # must stay put while the others converge -- i.e. no cross-coupling.
    target = torch.tensor([0.0, 4.0, 0.0])
    torch.manual_seed(0)
    da = DualAveraging()
    da.prox_center = torch.zeros(3)
    da.reset()
    for _ in range(1500):
        x_t, _ = da.get_state()
        da.step(x_t - target)
    _, x_avg = da.get_state()
    assert abs(float(x_avg[0])) < 0.02
    assert abs(float(x_avg[2])) < 0.02
    assert abs(float(x_avg[1]) - 4.0) < 0.05


def test_matches_pyro_dual_averaging_on_scalars():
    # Regression anchor: identical output to Pyro for the same gradient stream.
    PyroDA = pytest.importorskip("pyro.ops.dual_averaging").DualAveraging
    torch.manual_seed(1)
    grads = torch.randn(60)
    ours, theirs = DualAveraging(), PyroDA()
    ours.reset()
    for g in grads:
        ours.step(g)
        theirs.step(g)
        ox_t, ox_avg = ours.get_state()
        tx_t, tx_avg = theirs.get_state()
        assert abs(float(ox_t) - float(tx_t)) < 1e-12
        assert abs(float(ox_avg) - float(tx_avg)) < 1e-12


# ========================================================================== #
#  REINFORCEAdapter                                                           #
# ========================================================================== #

def _bounded_objective(x, a):
    """Bounded, smooth, minimised at x = a (value 0 there, ->1 far away)."""
    return 1.0 - torch.exp(-0.5 * (x - a) ** 2)


def _reinforce_minimise(target, steps, *, sigma=0.3, seed=0):
    torch.manual_seed(seed)
    n = target.shape[0]
    ad = REINFORCEAdapter(n, sigma=sigma)
    ad.prox_center = torch.zeros(n)
    ad.reset()
    for _ in range(steps):
        x, _ = ad.get_state()
        ad.step(_bounded_objective(x, target))
    return ad.get_state()[1]


def test_reset_round_trips_to_prox_center():
    # mu after reset is the prox-center; in adapter use that is log(step0).
    step0 = torch.tensor([0.05, 0.3, 1.7])
    ad = REINFORCEAdapter(3, sigma=0.1)
    ad.prox_center = torch.log(step0)
    ad.reset()
    _, mu = ad.get_state()
    assert torch.allclose(torch.exp(mu), step0)


def test_proposal_is_fixed_between_get_state_and_step():
    torch.manual_seed(0)
    ad = REINFORCEAdapter(2, sigma=0.1)
    ad.prox_center = torch.zeros(2)
    ad.reset()
    x1, mu1 = ad.get_state()
    x2, _ = ad.get_state()
    assert torch.equal(x1, x2)                       # stable until step()
    assert torch.allclose(x1, mu1 + 0.1 * ad._eps)   # proposal = mu + sigma*eps
    ad.step(torch.zeros(2))
    x3, _ = ad.get_state()
    assert not torch.equal(x1, x3)                   # fresh eps after step()


def test_first_step_initialises_baseline_to_observation():
    torch.manual_seed(0)
    ad = REINFORCEAdapter(2, sigma=0.1)
    ad.reset()
    assert ad._g is None
    f0 = torch.tensor([0.5, 1.5])
    ad.step(f0)
    assert torch.equal(ad._g, f0)


def test_minimises_bounded_objective_to_shifted_optimum():
    mu = _reinforce_minimise(torch.tensor([1.0]), steps=3000)
    assert abs(float(mu) - 1.0) < 0.1


def test_minimises_at_zero():
    mu = _reinforce_minimise(torch.tensor([0.0]), steps=3000)
    assert abs(float(mu)) < 0.1


def test_vectorised_problems_converge_independently():
    targets = torch.tensor([1.0, -2.0, 0.5])
    mu = _reinforce_minimise(targets, steps=5000, sigma=0.3)
    assert torch.allclose(mu, targets, atol=0.15)


def test_exposed_gamma_stabilises_unbounded_objective():
    # x**2 diverges at the default gamma (0.05); the exposed learning-rate
    # knob (larger gamma => gentler steps) makes it converge.
    torch.manual_seed(0)
    target = torch.tensor([1.0])
    ad = REINFORCEAdapter(1, sigma=0.1, gamma=0.5)
    ad.prox_center = torch.zeros(1)
    ad.reset()
    for _ in range(4000):
        x, _ = ad.get_state()
        ad.step((x - target) ** 2)          # unbounded objective
    mu = ad.get_state()[1]
    assert math.isfinite(float(mu))
    assert abs(float(mu) - 1.0) < 0.1


def test_reinforce_perturbation_follows_prox_center_dtype():
    # Regression: eps is drawn on prox_center's device/dtype, not CPU/default,
    # so a GPU/float32 run does not mix devices in the score estimate. (Default
    # dtype here is float64; a float32 prox-center must produce float32 eps.)
    ad = REINFORCEAdapter(3, sigma=0.1)
    ad.prox_center = torch.zeros(3, dtype=torch.float32)
    ad.reset()
    assert ad._eps.dtype == torch.float32
    x, mu = ad.get_state()
    assert x.dtype == torch.float32 and mu.dtype == torch.float32
    ad.step(torch.zeros(3, dtype=torch.float32))
    assert ad._eps.dtype == torch.float32


def test_reproducible_with_fixed_seed():
    a = torch.tensor([1.0, -1.0])

    def run():
        torch.manual_seed(7)
        ad = REINFORCEAdapter(2, sigma=0.3)
        ad.prox_center = torch.zeros(2)
        ad.reset()
        for _ in range(200):
            x, _ = ad.get_state()
            ad.step(_bounded_objective(x, a))
        return ad.get_state()[1]

    assert torch.equal(run(), run())
