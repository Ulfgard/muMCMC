"""Correctness tests for the recursive NUTS oracle (``nuts_reference``).

These pin the *kernel* behaviour we later want the batched/iterative
implementation to reproduce: unbiased sampling of a Gaussian (mean, variance,
and full covariance under correlation), dynamic trajectory lengths that both
grow and terminate, divergence detection on a stiff target, and seed
determinism. Everything runs single-chain on a bare potential, so nothing here
depends on the library's spaces/tempering/batch interfaces.
"""
import math

import torch
import pytest

from nuts_reference import nuts_sample, nuts_step, _no_uturn

torch.set_default_dtype(torch.float64)


def _gaussian_potential(cov: torch.Tensor):
    """Return ``U(q) = ½ qᵀ Σ⁻¹ q`` for a zero-mean Gaussian with covariance
    ``cov``."""
    prec = torch.linalg.inv(cov)
    return lambda q: 0.5 * (q @ prec @ q)


# --------------------------------------------------------------------------- #
#  Statistical recovery                                                       #
# --------------------------------------------------------------------------- #

def test_standard_normal_1d_moments():
    U = lambda q: 0.5 * (q * q).sum()
    out = nuts_sample(U, torch.zeros(1), num_samples=4000, num_warmup=500,
                      step_size=0.9, seed=0)
    x = out["samples"][:, 0]
    assert abs(float(x.mean())) < 0.08
    assert abs(float(x.var(unbiased=True)) - 1.0) < 0.12
    assert out["num_divergences"] == 0


def test_correlated_gaussian_covariance():
    cov = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    out = nuts_sample(_gaussian_potential(cov), torch.zeros(2),
                      num_samples=6000, num_warmup=1000, step_size=0.4, seed=1)
    s = out["samples"]
    emp = torch.cov(s.T)
    assert torch.allclose(emp, cov, atol=0.12)
    assert torch.allclose(s.mean(0), torch.zeros(2), atol=0.08)


# --------------------------------------------------------------------------- #
#  Trajectory behaviour                                                       #
# --------------------------------------------------------------------------- #

def test_tree_depth_grows_and_terminates():
    # On a smooth standard normal the sampler should build non-trivial trees
    # (depth > 0) yet U-turn well before the cap (depth < max_tree_depth).
    U = lambda q: 0.5 * (q * q).sum()
    out = nuts_sample(U, torch.zeros(2), num_samples=1500, num_warmup=300,
                      step_size=0.7, max_tree_depth=10, seed=2)
    depth = out["tree_depth"].to(torch.float64)
    assert float(depth.mean()) > 0.5
    assert int(out["tree_depth"].max()) < 10


def test_accept_stat_is_high_on_easy_target():
    U = lambda q: 0.5 * (q * q).sum()
    out = nuts_sample(U, torch.zeros(1), num_samples=1000, num_warmup=200,
                      step_size=0.6, seed=3)
    assert float(out["accept_stat"].mean()) > 0.6


# --------------------------------------------------------------------------- #
#  Divergences                                                                #
# --------------------------------------------------------------------------- #

def test_divergences_on_stiff_target_with_large_step():
    # A sharply peaked direction + a large step makes the leapfrog blow up in
    # energy, which must register as divergences.
    cov = torch.tensor([[1.0, 0.0], [0.0, 1e-4]])
    out = nuts_sample(_gaussian_potential(cov), torch.zeros(2),
                      num_samples=400, num_warmup=0, step_size=0.8,
                      max_delta_H=1000.0, seed=4)
    assert out["num_divergences"] > 0


# --------------------------------------------------------------------------- #
#  Determinism + unit pieces                                                  #
# --------------------------------------------------------------------------- #

def test_seed_determinism():
    U = lambda q: 0.5 * (q * q).sum()
    a = nuts_sample(U, torch.zeros(2), num_samples=200, step_size=0.7, seed=7)
    b = nuts_sample(U, torch.zeros(2), num_samples=200, step_size=0.7, seed=7)
    c = nuts_sample(U, torch.zeros(2), num_samples=200, step_size=0.7, seed=8)
    assert torch.equal(a["samples"], b["samples"])
    assert not torch.equal(a["samples"], c["samples"])


def test_no_uturn_criterion():
    # Momenta pointing back across the span -> U-turn (False); pointing along
    # the span -> keep going (True).
    q_minus = torch.tensor([0.0, 0.0])
    q_plus = torch.tensor([1.0, 0.0])
    assert _no_uturn(q_minus, q_plus, torch.tensor([1.0, 0.0]),
                     torch.tensor([1.0, 0.0])) is True
    assert _no_uturn(q_minus, q_plus, torch.tensor([-1.0, 0.0]),
                     torch.tensor([-1.0, 0.0])) is False


def test_single_step_returns_finite_state():
    U = lambda q: 0.5 * (q * q).sum()
    gen = torch.Generator().manual_seed(0)
    q, info = nuts_step(U, torch.zeros(3), 0.5, max_tree_depth=10,
                        max_delta_H=1000.0, gen=gen)
    assert q.shape == (3,)
    assert torch.isfinite(q).all()
    assert 1 <= info.tree_depth <= 10
    assert 0.0 <= info.accept_stat <= 1.0
