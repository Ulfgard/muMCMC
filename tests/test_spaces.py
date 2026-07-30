"""Contract tests for the space classes.

A space owns the parameter naming, the free/fixed split, the chart between the
constrained variables and the standard normal, and the prior. The samplers drive
spaces through a fixed protocol -- ``to_vector`` / ``to_free_vector``,
``from_full_vector`` / ``from_free_vector``, ``to_full`` / ``to_free``,
``add_fixed`` / ``remove_fixed``, ``as_transform``, ``prior_log_prob[_vector]``,
``prior_metric``, ``free_block``, ``sample`` -- so these tests exercise that
protocol's invariants.
"""
import math

import torch
import pytest
from torch.distributions import Normal

from muMCMC.spaces import LogNormalSpace, NormalSpace, UnnormalizedSpace

torch.set_default_dtype(torch.float64)

ATOL = 1e-10
NAMES = ["a", "b", "c"]

PROPER = [
    lambda **kw: NormalSpace(NAMES, mu=0.3, sigma=1.7, **kw),
    lambda **kw: LogNormalSpace(NAMES, mu=-0.2, sigma=0.8, **kw),
]


# --------------------------------------------------------------------------- #
#  Naming and the free/fixed split                                            #
# --------------------------------------------------------------------------- #

def test_dimensions_without_fixed():
    s = NormalSpace(NAMES)
    assert s.d == 3 and s.d_full == 3
    assert s.free_names == NAMES and s.free_indices == [0, 1, 2]
    assert s.fixed_indices == []


def test_dimensions_with_interior_fixed():
    # A fixed name keeps its place in the full layout, so the free block is not
    # a prefix of it.
    s = NormalSpace(NAMES, fixed={"b": 0.5})
    assert s.d == 2 and s.d_full == 3
    assert s.free_names == ["a", "c"]
    assert s.free_indices == [0, 2] and s.fixed_indices == [1]


def test_fixed_names_must_appear_in_names():
    with pytest.raises(ValueError, match="fixed name"):
        NormalSpace(NAMES, fixed={"z": 1.0})


# --------------------------------------------------------------------------- #
#  Prior parameters                                                           #
# --------------------------------------------------------------------------- #

def test_scalar_parameter_is_shared_across_the_free_variables():
    s = NormalSpace(NAMES, mu=2.0, sigma=0.5)
    theta = s.as_transform.forward(torch.zeros(1, 3)).mapped_point
    assert torch.allclose(theta, torch.full((1, 3), 2.0), atol=ATOL)


def test_dict_parameter_is_read_by_name_not_by_position():
    # b is fixed, so the free order is a, c. Keying by name is what keeps the
    # prior from being transposed onto the wrong variable.
    s = NormalSpace(NAMES, mu={"a": 10.0, "c": -10.0}, sigma=1.0,
                    fixed={"b": 0.0})
    theta = s.as_transform.forward(torch.zeros(1, 2)).mapped_point
    assert torch.allclose(theta, torch.tensor([[10.0, -10.0]]), atol=ATOL)


def test_dict_parameter_missing_a_free_name_raises():
    with pytest.raises(ValueError, match="missing an entry"):
        NormalSpace(NAMES, mu={"a": 0.0, "b": 0.0})


def test_tensor_parameter_of_the_wrong_length_raises():
    with pytest.raises(ValueError, match="shape"):
        NormalSpace(NAMES, sigma=torch.ones(2))


# --------------------------------------------------------------------------- #
#  Vector layouts                                                             #
# --------------------------------------------------------------------------- #

def test_to_full_splices_fixed_and_to_free_removes_it():
    s = NormalSpace(NAMES, fixed={"b": 7.0})
    theta_free = torch.randn(5, 2)
    full = s.to_full(theta_free)
    assert full.shape == (5, 3)
    assert torch.allclose(full[:, 1], torch.full((5,), 7.0), atol=ATOL)
    assert torch.allclose(s.to_free(full), theta_free, atol=ATOL)


def test_to_full_is_a_no_op_without_fixed():
    s = NormalSpace(NAMES)
    v = torch.randn(4, 3)
    assert s.to_full(v) is v and s.to_free(v) is v


def test_to_full_preserves_dtype_and_leading_axes():
    s = NormalSpace(NAMES, fixed={"c": 1.0}, dtype=torch.float32)
    full = s.to_full(torch.randn(2, 6, 2, dtype=torch.float32))
    assert full.shape == (2, 6, 3) and full.dtype == torch.float32


def test_dict_and_vector_round_trip():
    s = NormalSpace(NAMES, fixed={"c": 9.0})
    theta_free = torch.randn(4, 2)
    full = s.to_full(theta_free)
    as_dict = s.from_full_vector(full)
    assert set(as_dict) == set(NAMES)
    assert torch.allclose(s.to_vector(as_dict), full, atol=ATOL)
    assert torch.allclose(s.to_free_vector(as_dict), theta_free, atol=ATOL)


def test_from_free_vector_keys_on_the_free_names():
    s = NormalSpace(NAMES, fixed={"b": 1.0})
    assert set(s.from_free_vector(torch.randn(4, 2))) == {"a", "c"}


def test_to_vector_fills_in_a_missing_fixed_name():
    s = NormalSpace(NAMES, fixed={"c": 4.0})
    full = s.to_vector({"a": torch.zeros(3), "b": torch.zeros(3)})
    assert torch.allclose(full[:, 2], torch.full((3,), 4.0), atol=ATOL)


@pytest.mark.parametrize("width", [1, 4])
def test_from_vector_rejects_the_wrong_width(width):
    s = NormalSpace(NAMES, fixed={"c": 1.0})
    with pytest.raises(ValueError, match="vector of size"):
        s.from_full_vector(torch.zeros(2, width))


# --------------------------------------------------------------------------- #
#  Fixed variables at the dict level                                          #
# --------------------------------------------------------------------------- #

def test_add_and_remove_fixed_round_trip():
    s = NormalSpace(NAMES, fixed={"c": 4.0})
    free = {"a": torch.randn(3), "b": torch.randn(3)}
    added = s.add_fixed(free)
    assert set(added) == set(NAMES)
    assert torch.allclose(added["c"], torch.full((3,), 4.0), atol=ATOL)
    assert set(s.remove_fixed(added)) == {"a", "b"}


def test_add_fixed_preserves_dtype_and_shape():
    s = NormalSpace(NAMES, fixed={"c": 4.0}, dtype=torch.float32)
    added = s.add_fixed({"a": torch.zeros(2, 5, dtype=torch.float32),
                         "b": torch.zeros(2, 5, dtype=torch.float32)})
    assert added["c"].shape == (2, 5) and added["c"].dtype == torch.float32


def test_add_and_remove_fixed_are_no_ops_without_fixed():
    s = NormalSpace(NAMES)
    d = {"a": torch.zeros(2), "b": torch.zeros(2), "c": torch.zeros(2)}
    assert s.add_fixed(d) is d and s.remove_fixed(d) is d


def test_remove_fixed_tolerates_an_absent_name():
    s = NormalSpace(NAMES, fixed={"c": 1.0})
    assert set(s.remove_fixed({"a": torch.zeros(2)})) == {"a"}


# --------------------------------------------------------------------------- #
#  The prior and its chart                                                    #
# --------------------------------------------------------------------------- #

def test_prior_log_prob_matches_the_reference_density():
    s = NormalSpace(NAMES, mu=0.3, sigma=1.7)
    theta = 0.3 + 1.7 * torch.randn(5, 3)
    ref = Normal(0.3, 1.7).log_prob(theta).sum(-1)
    assert torch.allclose(s.prior_log_prob_vector(theta), ref, atol=ATOL)


def test_prior_log_prob_dict_form_matches_the_vector_form():
    s = NormalSpace(NAMES, sigma=2.0)
    theta = 2.0 * torch.randn(5, 3)
    assert torch.allclose(s.prior_log_prob(s.from_free_vector(theta)),
                          s.prior_log_prob_vector(theta), atol=ATOL)


def test_prior_log_prob_of_a_subset_is_the_marginal():
    # The prior factorizes over the coordinate axis, so a subset gives the
    # marginal over exactly those names.
    s = NormalSpace(NAMES, mu={"a": 0.0, "b": 1.0, "c": -1.0}, sigma=1.0)
    y = {"a": torch.randn(4), "c": torch.randn(4)}
    expected = Normal(0.0, 1.0).log_prob(y["a"]) + Normal(-1.0, 1.0).log_prob(y["c"])
    assert torch.allclose(s.prior_log_prob(y), expected, atol=ATOL)


def test_prior_log_prob_ignores_a_fixed_name():
    s = NormalSpace(NAMES, fixed={"c": 2.0})
    y = {"a": torch.randn(4), "b": torch.randn(4)}
    with_fixed = dict(y, c=torch.full((4,), 2.0))
    assert torch.allclose(s.prior_log_prob(with_fixed), s.prior_log_prob(y),
                          atol=ATOL)


def test_prior_log_prob_without_any_free_name_raises():
    s = NormalSpace(NAMES, fixed={"c": 2.0})
    with pytest.raises(ValueError, match="none of the free"):
        s.prior_log_prob({"c": torch.zeros(2)})


@pytest.mark.parametrize("build", PROPER)
def test_prior_metric_is_the_pullback_of_the_identity(build):
    # M = J^-T J^-1 on the variables, which is what a chain running there reads,
    # and its pushforward to the chart is the identity, which is what a chain
    # running in the chart reads.
    s = build()
    z = torch.randn(5, 3)
    m = s.as_transform.forward(z)
    M = s.prior_metric(m.mapped_point)
    assert M.shape == (5, 3, 3)
    diag = M.diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag * m.jacobian_diag ** 2, torch.ones_like(diag),
                          atol=1e-8)
    assert torch.allclose(diag, 1.0 / m.jacobian_diag ** 2, atol=1e-8)


def test_free_block_drops_the_fixed_rows_and_columns():
    s = LogNormalSpace(NAMES, fixed={"b": 1.0})
    G = torch.arange(9.0).reshape(3, 3).expand(4, 3, 3)
    A = s.free_block(G)
    assert A.shape == (4, 2, 2)
    assert torch.allclose(A, G[:, [[0], [2]], [[0, 2]]], atol=ATOL)


# --------------------------------------------------------------------------- #
#  Sampling                                                                   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("build", PROPER)
def test_sample_shapes_and_keys(build):
    s = build(fixed={"c": 3.0})
    draw = s.sample(64)
    assert set(draw) == set(NAMES)
    assert draw["a"].shape == (64,)
    assert torch.allclose(draw["c"], torch.full((64,), 3.0), atol=ATOL)


def test_sample_reproduces_from_a_generator():
    s = NormalSpace(NAMES)
    a = s.sample(16, generator=torch.Generator().manual_seed(5))
    b = s.sample(16, generator=torch.Generator().manual_seed(5))
    assert torch.allclose(a["a"], b["a"], atol=ATOL)


def test_sample_does_not_disturb_the_global_rng():
    s = NormalSpace(NAMES)
    torch.manual_seed(0)
    before = torch.randn(3)
    torch.manual_seed(0)
    s.sample(8, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(torch.randn(3), before, atol=ATOL)


def test_sample_lands_in_the_support():
    draw = LogNormalSpace(NAMES).sample(256)
    assert all(bool((draw[n] > 0).all()) for n in NAMES)


def test_sample_recovers_the_prior_moments():
    draw = NormalSpace(NAMES, mu=1.0, sigma=2.0).sample(
        40000, generator=torch.Generator().manual_seed(3))
    assert abs(float(draw["a"].mean()) - 1.0) < 0.05
    assert abs(float(draw["a"].std()) - 2.0) < 0.05


# --------------------------------------------------------------------------- #
#  UnnormalizedSpace                                                          #
# --------------------------------------------------------------------------- #

def test_unnormalized_chart_is_the_identity():
    s = UnnormalizedSpace(NAMES)
    z = torch.randn(4, 3)
    m = s.as_transform.forward(z)
    assert torch.allclose(m.mapped_point, z, atol=ATOL)
    assert torch.allclose(m.jacobian_log_det, torch.zeros(4), atol=ATOL)


def test_unnormalized_contributes_no_potential_and_no_metric():
    s = UnnormalizedSpace(NAMES)
    z = torch.randn(4, 3)
    assert not s.is_proper
    assert torch.allclose(s.prior_log_prob_vector(z), torch.zeros(4), atol=ATOL)
    assert s.prior_metric(z) is None


def test_unnormalized_refuses_the_prior_density_and_a_draw():
    s = UnnormalizedSpace(NAMES)
    with pytest.raises(ValueError, match="no prior"):
        s.prior_log_prob({"a": torch.zeros(2)})
    with pytest.raises(ValueError, match="no prior"):
        s.sample(4)


def test_unnormalized_keeps_the_fixed_machinery():
    s = UnnormalizedSpace(NAMES, fixed={"b": 5.0})
    assert s.d == 2 and s.free_names == ["a", "c"]
    full = s.to_full(torch.randn(3, 2))
    assert torch.allclose(full[:, 1], torch.full((3,), 5.0), atol=ATOL)


# --------------------------------------------------------------------------- #
#  Batched leading axes                                                       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("build", PROPER)
def test_protocol_accepts_arbitrary_leading_axes(build):
    s = build()
    z = torch.randn(2, 5, 3)
    m = s.as_transform.forward(z)
    assert m.mapped_point.shape == (2, 5, 3)
    assert s.prior_log_prob_vector(m.mapped_point).shape == (2, 5)
    assert s.prior_metric(m.mapped_point).shape == (2, 5, 3, 3)
