"""Tests for the iterative single-chain NUTS (``nuts_iterative``).

Two jobs: (1) confirm the iterative doubling-loop kernel samples correctly --
the same statistical battery the recursive oracle passes -- and (2) unit-test
the generalized U-turn logic that is new here and that the batched kernel will
inherit. A cross-check that the iterative and recursive samplers agree on a
target (despite different U-turn criteria) ties the two references together.
"""
import torch
import pytest

import nuts_iterative as it
import nuts_reference as rec
from nuts_iterative import generalized_turn, _subtree_turns

torch.set_default_dtype(torch.float64)


def _gaussian_potential(cov: torch.Tensor):
    prec = torch.linalg.inv(cov)
    return lambda q: 0.5 * (q @ prec @ q)


# --------------------------------------------------------------------------- #
#  Generalized U-turn logic                                                   #
# --------------------------------------------------------------------------- #

def test_generalized_turn_expanding_vs_turning():
    e = torch.tensor([1.0, 0.0])
    # Momenta aligned with the accumulated direction -> still expanding.
    assert generalized_turn(rho=e, r_left=e, r_right=e) is False
    # Right end points back against the span -> turning.
    assert generalized_turn(rho=e, r_left=e, r_right=-e) is True
    # Left end points back -> turning.
    assert generalized_turn(rho=e, r_left=-e, r_right=e) is True


def test_subtree_turns_detects_reversal():
    # Four leaves all moving + : no reversal.
    fwd = [torch.tensor([1.0, 0.0]) for _ in range(4)]
    assert _subtree_turns(fwd) is False
    # Second half reverses -> the length-4 span turns.
    mixed = [torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]),
             torch.tensor([-1.0, 0.0]), torch.tensor([-1.0, 0.0])]
    assert _subtree_turns(mixed) is True


def test_subtree_turns_single_leaf_never_turns():
    assert _subtree_turns([torch.tensor([1.0, 0.0])]) is False


# --------------------------------------------------------------------------- #
#  Statistical recovery                                                       #
# --------------------------------------------------------------------------- #

def test_standard_normal_1d_moments():
    U = lambda q: 0.5 * (q * q).sum()
    out = it.nuts_sample(U, torch.zeros(1), num_samples=3000, num_warmup=500,
                         step_size=0.9, seed=0)
    x = out["samples"][:, 0]
    assert abs(float(x.mean())) < 0.08
    assert abs(float(x.var(unbiased=True)) - 1.0) < 0.12
    assert out["num_divergences"] == 0


def test_correlated_gaussian_covariance():
    cov = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    out = it.nuts_sample(_gaussian_potential(cov), torch.zeros(2),
                         num_samples=5000, num_warmup=1000, step_size=0.4, seed=1)
    s = out["samples"]
    assert torch.allclose(torch.cov(s.T), cov, atol=0.12)
    assert torch.allclose(s.mean(0), torch.zeros(2), atol=0.08)


# --------------------------------------------------------------------------- #
#  Trajectory behaviour                                                       #
# --------------------------------------------------------------------------- #

def test_tree_depth_grows_and_terminates():
    U = lambda q: 0.5 * (q * q).sum()
    out = it.nuts_sample(U, torch.zeros(2), num_samples=1500, num_warmup=300,
                         step_size=0.7, max_tree_depth=10, seed=2)
    assert float(out["tree_depth"].to(torch.float64).mean()) > 0.5
    assert int(out["tree_depth"].max()) < 10


def test_divergences_on_stiff_target_with_large_step():
    cov = torch.tensor([[1.0, 0.0], [0.0, 1e-4]])
    out = it.nuts_sample(_gaussian_potential(cov), torch.zeros(2),
                         num_samples=400, num_warmup=0, step_size=0.8, seed=4)
    assert out["num_divergences"] > 0


def test_seed_determinism():
    U = lambda q: 0.5 * (q * q).sum()
    a = it.nuts_sample(U, torch.zeros(2), num_samples=200, step_size=0.7, seed=7)
    b = it.nuts_sample(U, torch.zeros(2), num_samples=200, step_size=0.7, seed=7)
    c = it.nuts_sample(U, torch.zeros(2), num_samples=200, step_size=0.7, seed=8)
    assert torch.equal(a["samples"], b["samples"])
    assert not torch.equal(a["samples"], c["samples"])


# --------------------------------------------------------------------------- #
#  Agreement with the recursive oracle                                        #
# --------------------------------------------------------------------------- #

def test_agrees_with_recursive_oracle_on_correlated_gaussian():
    # Different U-turn criteria, same target: run both references on the same
    # target and check each recovers it. (Comparing the two noisy estimates
    # directly would compound their Monte Carlo errors, so both are checked
    # against the known truth instead.)
    cov = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
    U = _gaussian_potential(cov)
    kw = dict(num_samples=5000, num_warmup=1000, step_size=0.5)
    ri = it.nuts_sample(U, torch.zeros(2), seed=11, **kw)["samples"]
    rr = rec.nuts_sample(U, torch.zeros(2), seed=11, **kw)["samples"]
    for s in (ri, rr):
        assert torch.allclose(s.mean(0), torch.zeros(2), atol=0.08)
        assert torch.allclose(torch.cov(s.T), cov, atol=0.15)
