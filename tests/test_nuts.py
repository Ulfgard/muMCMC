"""Behaviour / regression tests for the NUTS sampler.

NUTS currently delegates the actual transitions to Pyro, so it is correct by
construction.  What is *ours* -- and what these tests pin down -- is the
constrained-space reparameterization layered on top: the potential Pyro sees is

    U(z) = U_lik(theta(z)) + U_prior(theta(z)) - log|det dtheta/dz|

assembled in ``MCMCSampler.evaluate_model``, plus the free/fixed splicing and the
output schema.  The statistical tests below (sample the prior; sample a flat
target on a box and recover a *uniform* marginal) exercise exactly the prior and
Jacobian terms that a future non-Pyro kernel would have to reproduce, so they
double as a behaviour spec should we ever cut the Pyro tether.

Runs are single-chain (Pyro spawns a worker process per chain; single-chain
keeps these in-process, fast, and deterministic) and seed-fixed.  Expensive
sampler runs are shared across assertions via module-scoped fixtures.
"""
import math

import torch
import pytest
import pyro
from pyro.distributions import Normal

from muMCMC import NUTS, UnconstrainedSpace, UniformBoxSpace

torch.set_default_dtype(torch.float64)

COMMON_KEYS = {"accept_rate", "num_divergences", "step_size"}

N_SAMPLES = 500
N_WARMUP = 250


def _flat_likelihood(theta):
    """Zero likelihood potential: the target is then prior x Jacobian only."""
    return torch.zeros(theta.shape[:-1], dtype=theta.dtype)


# --------------------------------------------------------------------------- #
#  Shared (expensive) sampler runs                                            #
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def prior_run():
    """Flat likelihood under N(0,1) priors on an unconstrained space: NUTS
    should reproduce the prior as its stationary distribution."""
    torch.manual_seed(0)
    names = ["a", "b"]
    space = UnconstrainedSpace(names, priors={n: Normal(0.0, 1.0) for n in names})
    nuts = NUTS(_flat_likelihood, space)
    out = nuts.run_mcmc(torch.zeros(2), num_samples=N_SAMPLES,
                        num_warmup_steps=N_WARMUP, num_chains=1,
                        disable_progbar=True)
    return out


@pytest.fixture(scope="module")
def box_run():
    """Flat likelihood on a box: the tanh transform's Jacobian must turn the
    flat constrained target into a *uniform* distribution on the box."""
    torch.manual_seed(0)
    limits = {"x": (-2.0, 1.0), "y": (0.0, 4.0)}
    space = UniformBoxSpace(limits, ["x", "y"], device="cpu")
    nuts = NUTS(_flat_likelihood, space)
    out = nuts.run_mcmc(torch.tensor([0.0, 2.0]), num_samples=N_SAMPLES,
                        num_warmup_steps=N_WARMUP, num_chains=1,
                        disable_progbar=True)
    return space, nuts, out


# --------------------------------------------------------------------------- #
#  Statistical behaviour: the reparameterization terms                        #
# --------------------------------------------------------------------------- #

def test_prior_recovery_marginals(prior_run):
    # With a flat likelihood the prior term is the whole target: each
    # coordinate's marginal must come back as N(0, 1).
    for n in ["a", "b"]:
        x = prior_run[n]
        assert x.shape == (1, N_SAMPLES)
        assert abs(float(x.mean())) < 0.15
        assert abs(float(x.std()) - 1.0) < 0.15


def test_box_samples_stay_in_bounds(box_run):
    _, _, out = box_run
    assert torch.all(out["x"] > -2.0) and torch.all(out["x"] < 1.0)
    assert torch.all(out["y"] > 0.0) and torch.all(out["y"] < 4.0)


def test_box_flat_target_is_uniform(box_run):
    # This is the Jacobian-correctness anchor.  A flat target in constrained
    # coordinates becomes Uniform(l, u) only because the -log|det J| term is
    # included; dropping or mis-signing it would pile mass at the box edges
    # (tanh saturates), inflating the std and shifting the mean.
    _, _, out = box_run
    for name, (lo, hi) in {"x": (-2.0, 1.0), "y": (0.0, 4.0)}.items():
        x = out[name]
        midpoint = 0.5 * (lo + hi)
        uniform_std = (hi - lo) / math.sqrt(12.0)
        assert abs(float(x.mean()) - midpoint) < 0.2
        assert abs(float(x.std()) - uniform_std) < 0.15


# --------------------------------------------------------------------------- #
#  Output schema and free/fixed splicing                                      #
# --------------------------------------------------------------------------- #

def test_output_keys_and_grouping(box_run):
    _, _, out = box_run
    assert set(out) == {"x", "y"}
    # single chain -> (num_chains, num_samples)
    assert out["x"].shape == (1, N_SAMPLES)
    assert out["y"].shape == (1, N_SAMPLES)


def test_fixed_parameter_is_spliced_as_constant():
    torch.manual_seed(0)
    names = ["a", "b", "c"]
    space = UnconstrainedSpace(names, priors={n: Normal(0.0, 1.0) for n in names},
                               fixed={"c": 1.5})
    nuts = NUTS(_flat_likelihood, space)
    out = nuts.run_mcmc(torch.zeros(3), num_samples=40, num_warmup_steps=20,
                        num_chains=1, disable_progbar=True)
    assert set(out) == {"a", "b", "c"}
    assert out["a"].shape == (1, 40)
    # the fixed coordinate is not sampled -- it is spliced back as its constant
    assert torch.allclose(out["c"], torch.full((1, 40), 1.5))


# --------------------------------------------------------------------------- #
#  Diagnostics                                                                #
# --------------------------------------------------------------------------- #

def test_diagnostics_schema(box_run):
    _, nuts, _ = box_run
    d = nuts.diagnostics()
    assert set(d) == COMMON_KEYS
    for k in COMMON_KEYS:
        assert torch.is_tensor(d[k]) and d[k].shape == (1,)
    assert d["num_divergences"].dtype == torch.long


def test_diagnostics_values_are_sane(box_run):
    _, nuts, _ = box_run
    d = nuts.diagnostics()
    assert 0.0 <= float(d["accept_rate"][0]) <= 1.0
    assert float(d["step_size"][0]) > 0.0 and math.isfinite(float(d["step_size"][0]))
    assert int(d["num_divergences"][0]) >= 0


def test_diagnostics_empty_before_run():
    space = UnconstrainedSpace(["a"], priors={"a": Normal(0.0, 1.0)})
    nuts = NUTS(_flat_likelihood, space)
    assert nuts.diagnostics() == {}


# --------------------------------------------------------------------------- #
#  Determinism (regression anchor)                                            #
# --------------------------------------------------------------------------- #

def test_reproducible_with_fixed_seed():
    names = ["a", "b"]

    def run():
        pyro.set_rng_seed(123)
        space = UnconstrainedSpace(names, priors={n: Normal(0.0, 1.0) for n in names})
        nuts = NUTS(_flat_likelihood, space)
        return nuts.run_mcmc(torch.zeros(2), num_samples=30, num_warmup_steps=20,
                             num_chains=1, disable_progbar=True)

    first, second = run(), run()
    for n in names:
        assert torch.equal(first[n], second[n])
