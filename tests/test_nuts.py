"""Behaviour / regression tests for the NUTS sampler.

NUTS currently delegates the actual transitions to Pyro, so it is correct by
construction.  What is *ours* -- and what these tests pin down -- is the
potential layered on top: the one Pyro sees is

    U(theta) = U_lik(theta_full) - log p(theta)

assembled in ``MCMCSampler.evaluate_model``, plus the free/fixed splicing and the
output schema.  The statistical tests below sample the prior under a flat
likelihood, which exercises the prior term a future non-Pyro kernel would have
to reproduce, so they double as a behaviour spec should we ever cut the Pyro
tether.

Runs are single-chain (Pyro spawns a worker process per chain; single-chain
keeps these in-process, fast, and deterministic) and seed-fixed.  Expensive
sampler runs are shared across assertions via module-scoped fixtures.
"""
import math

import torch
import pytest
import pyro

from muMCMC import NUTS, LogNormalSpace, NormalSpace

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
    """Flat likelihood under N(0,1) priors: NUTS should reproduce the prior as
    its stationary distribution."""
    torch.manual_seed(0)
    names = ["a", "b"]
    space = NormalSpace(names)
    nuts = NUTS(_flat_likelihood, space)
    out = nuts.run_mcmc({n: torch.tensor(0.0) for n in nuts.space.free_names}, num_samples=N_SAMPLES,
                        num_warmup_steps=N_WARMUP, num_chains=1,
                        disable_progbar=True)
    return out


@pytest.fixture(scope="module")
def chart_run():
    """Flat likelihood under a log-normal prior, whose support is the positive
    half line. The chain runs on the variables, so it meets that boundary
    directly and pays for it in divergences and mixing, which is why this draws
    more than the unbounded fixtures."""
    torch.manual_seed(0)
    space = LogNormalSpace(["x", "y"])
    nuts = NUTS(_flat_likelihood, space)
    out = nuts.run_mcmc({n: torch.tensor(1.0) for n in nuts.space.free_names}, num_samples=8 * N_SAMPLES,
                        num_warmup_steps=4 * N_WARMUP, num_chains=1,
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


def test_chart_samples_stay_in_the_support(chart_run):
    _, _, out = chart_run
    assert torch.all(out["x"] > 0.0) and torch.all(out["y"] > 0.0)


def test_chart_flat_target_is_the_prior(chart_run):
    # With a flat likelihood the prior is the whole target, so a bounded support
    # must still come back as the prior it was drawn from.
    _, _, out = chart_run
    for name in ["x", "y"]:
        log_x = torch.log(out[name])
        assert abs(float(log_x.mean())) < 0.15
        assert abs(float(log_x.std()) - 1.0) < 0.15


# --------------------------------------------------------------------------- #
#  Output schema and free/fixed splicing                                      #
# --------------------------------------------------------------------------- #

def test_output_keys_and_grouping(chart_run):
    _, _, out = chart_run
    assert set(out) == {"x", "y"}
    # single chain -> (num_chains, num_samples)
    assert out["x"].shape == (1, 8 * N_SAMPLES)
    assert out["y"].shape == (1, 8 * N_SAMPLES)


def test_fixed_parameter_is_spliced_as_constant():
    torch.manual_seed(0)
    names = ["a", "b", "c"]
    space = NormalSpace(names,
                               fixed={"c": 1.5})
    nuts = NUTS(_flat_likelihood, space)
    out = nuts.run_mcmc({n: torch.tensor(0.0) for n in nuts.space.free_names}, num_samples=40, num_warmup_steps=20,
                        num_chains=1, disable_progbar=True)
    assert set(out) == {"a", "b", "c"}
    assert out["a"].shape == (1, 40)
    # the fixed coordinate is not sampled -- it is spliced back as its constant
    assert torch.allclose(out["c"], torch.full((1, 40), 1.5))


# --------------------------------------------------------------------------- #
#  Diagnostics                                                                #
# --------------------------------------------------------------------------- #

def test_diagnostics_schema(chart_run):
    _, nuts, _ = chart_run
    d = nuts.diagnostics()
    assert set(d) == COMMON_KEYS
    for k in COMMON_KEYS:
        assert torch.is_tensor(d[k]) and d[k].shape == (1,)
    assert d["num_divergences"].dtype == torch.long


def test_diagnostics_values_are_sane(chart_run):
    _, nuts, _ = chart_run
    d = nuts.diagnostics()
    assert 0.0 <= float(d["accept_rate"][0]) <= 1.0
    assert float(d["step_size"][0]) > 0.0 and math.isfinite(float(d["step_size"][0]))
    assert int(d["num_divergences"][0]) >= 0


def test_diagnostics_empty_before_run():
    space = NormalSpace(["a"])
    nuts = NUTS(_flat_likelihood, space)
    assert nuts.diagnostics() == {}


# --------------------------------------------------------------------------- #
#  Determinism (regression anchor)                                            #
# --------------------------------------------------------------------------- #

def test_reproducible_with_fixed_seed():
    names = ["a", "b"]

    def run():
        pyro.set_rng_seed(123)
        space = NormalSpace(names)
        nuts = NUTS(_flat_likelihood, space)
        return nuts.run_mcmc({n: torch.tensor(0.0) for n in nuts.space.free_names}, num_samples=30, num_warmup_steps=20,
                             num_chains=1, disable_progbar=True)

    first, second = run(), run()
    for n in names:
        assert torch.equal(first[n], second[n])
