"""Contract tests for ``UnconstrainedSpace`` and ``UniformBoxSpace``.

A ``space`` owns the parameter naming, the free/fixed split, the transform
between constrained and unconstrained coordinates, and the prior.  The samplers
drive spaces through a fixed protocol -- ``to_vector`` / ``from_vector`` /
``to_free_vector``, ``map_to_(un)constrained_vector``, ``add_fixed`` /
``remove_fixed``, ``prior_log_prob[_vector]``, ``sample`` -- so these tests
exercise that protocol's invariants (round trips, shapes, fixed-coordinate
splicing, prior arithmetic, and the documented error cases).
"""
import torch
import pytest
from pyro.distributions import Normal

from muMCMC.spaces import UnconstrainedSpace, UniformBoxSpace

torch.set_default_dtype(torch.float64)

ATOL = 1e-10
NAMES = ["a", "b", "c"]


def _priors(names=NAMES):
    return {n: Normal(0.0, 1.0) for n in names}


# ========================================================================== #
#  UnconstrainedSpace                                                         #
# ========================================================================== #

def test_dimensions_without_fixed():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    assert s.d == 3
    assert s.d_full == 3
    assert s.free_names == NAMES
    assert s.free_indices == [0, 1, 2]
    assert s.fixed_indices == []


def test_dimensions_with_trailing_fixed():
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"c": 1.0})
    assert s.d == 2
    assert s.d_full == 3
    assert s.free_names == ["a", "b"]
    assert s.free_indices == [0, 1]
    assert s.fixed_indices == [2]


def test_dimensions_with_interior_fixed():
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"b": 0.5})
    assert s.free_names == ["a", "c"]
    assert s.free_indices == [0, 2]
    assert s.fixed_indices == [1]


def test_priors_must_cover_all_names():
    with pytest.raises(ValueError):
        UnconstrainedSpace(NAMES, priors={"a": Normal(0.0, 1.0)})


def test_fixed_names_must_appear_in_names():
    with pytest.raises(ValueError):
        UnconstrainedSpace(NAMES, priors=_priors(), fixed={"z": 1.0})


def test_to_from_vector_round_trip_full():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    samples = {"a": torch.tensor([1.0, 2.0]),
               "b": torch.tensor([3.0, 4.0]),
               "c": torch.tensor([5.0, 6.0])}
    vec = s.to_vector(samples)          # (2, 3) in full name order
    assert vec.shape == (2, 3)
    back = s.from_vector(vec)           # full-size vec -> free dict (here all)
    for n in NAMES:
        assert torch.allclose(back[n], samples[n], atol=ATOL)


def test_to_free_vector_and_from_free_vector():
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"c": 9.0})
    samples = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0]),
               "c": torch.tensor([9.0])}
    free = s.to_free_vector(samples)    # (1, 2), only a,b
    assert free.shape == (1, 2)
    d = s.from_vector(free)             # free-size vec -> {a, b}
    assert set(d) == {"a", "b"}
    assert torch.allclose(d["a"], samples["a"], atol=ATOL)
    assert torch.allclose(d["b"], samples["b"], atol=ATOL)


def test_from_vector_full_size_uses_free_indices():
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"b": 7.0})
    full = torch.tensor([[10.0, 7.0, 30.0]])   # a, b(fixed), c
    d = s.from_vector(full)
    assert set(d) == {"a", "c"}
    assert torch.allclose(d["a"], torch.tensor([10.0]), atol=ATOL)
    assert torch.allclose(d["c"], torch.tensor([30.0]), atol=ATOL)


def test_from_vector_rejects_wrong_size():
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"c": 1.0})
    with pytest.raises(ValueError):
        s.from_vector(torch.zeros(1, 5))


def test_add_remove_fixed_round_trip():
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"c": 4.0})
    free = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0, 4.0])}
    full = s.add_fixed(free)
    assert set(full) == {"a", "b", "c"}
    assert torch.allclose(full["c"], torch.tensor([4.0, 4.0]), atol=ATOL)
    # add_fixed is pure: original dict untouched
    assert "c" not in free
    back = s.remove_fixed(full)
    assert set(back) == {"a", "b"}


def test_add_fixed_preserves_dtype():
    # Fixed columns must match the samples' dtype, not the default dtype.
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"c": 4.0})
    free = {"a": torch.tensor([1.0], dtype=torch.float32),
            "b": torch.tensor([3.0], dtype=torch.float32)}
    assert s.add_fixed(free)["c"].dtype == torch.float32


def test_add_fixed_is_noop_without_fixed():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    free = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0]),
            "c": torch.tensor([3.0])}
    assert s.add_fixed(free) is free       # documented no-op (same object)
    assert s.remove_fixed(free) is free


def test_map_to_unconstrained_is_identity_round_trip():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    theta = torch.randn(4, 3)
    z = s.map_to_unconstrained_vector(theta).mapped_point
    theta_back = s.map_to_constrained_vector(z).mapped_point
    assert torch.allclose(theta_back, theta, atol=ATOL)
    # identity transform: z equals theta
    assert torch.allclose(z, theta, atol=ATOL)


def test_map_to_unconstrained_drops_fixed_coords():
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"c": 1.0})
    theta_full = torch.randn(4, 3)
    z = s.map_to_unconstrained_vector(theta_full).mapped_point
    assert z.shape == (4, 2)               # only free a, b
    assert torch.allclose(z, theta_full[..., [0, 1]], atol=ATOL)


def test_prior_log_prob_sums_over_free_names():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    y = {"a": torch.tensor([0.5]), "b": torch.tensor([-1.0]),
         "c": torch.tensor([2.0])}
    expected = sum(Normal(0.0, 1.0).log_prob(y[n]).squeeze(-1) for n in NAMES)
    assert torch.allclose(s.prior_log_prob(y), expected, atol=ATOL)


def test_prior_log_prob_skips_fixed_names():
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"c": 2.0})
    y = {"a": torch.tensor([0.5]), "b": torch.tensor([-1.0])}
    expected = sum(Normal(0.0, 1.0).log_prob(y[n]).squeeze(-1) for n in ["a", "b"])
    assert torch.allclose(s.prior_log_prob(y), expected, atol=ATOL)


def test_prior_log_prob_marginal_subset():
    # A subset of names returns the marginal prior over just those names.
    s = UnconstrainedSpace(NAMES, priors=_priors())
    y = {"a": torch.tensor([0.5]), "c": torch.tensor([2.0])}
    expected = sum(Normal(0.0, 1.0).log_prob(y[n]).squeeze(-1) for n in ["a", "c"])
    assert torch.allclose(s.prior_log_prob(y), expected, atol=ATOL)


def test_prior_log_prob_vector_matches_dict_form():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    theta_free = torch.randn(6, 3)
    vec = s.prior_log_prob_vector(theta_free)
    dct = s.prior_log_prob(s.from_vector(theta_free))
    assert vec.shape == (6,)
    assert torch.allclose(vec, dct, atol=ATOL)


def test_prior_log_prob_vector_is_zero_without_priors():
    s = UnconstrainedSpace(NAMES)               # no priors
    theta_free = torch.randn(5, 3)
    z = s.prior_log_prob_vector(theta_free)
    assert z.shape == (5,)
    assert torch.allclose(z, torch.zeros(5), atol=ATOL)


def test_prior_log_prob_without_priors_raises():
    s = UnconstrainedSpace(NAMES)
    with pytest.raises(ValueError):
        s.prior_log_prob({"a": torch.zeros(1)})


def test_sample_without_priors_raises():
    s = UnconstrainedSpace(NAMES)
    with pytest.raises(ValueError):
        s.sample(4)


def test_sample_shapes_and_fixed_value():
    # Scalar priors (Normal(0.0, 1.0)) are the convention used elsewhere
    # (e.g. test_samplers.py); each free name yields an (n_samples,) column.
    s = UnconstrainedSpace(NAMES, priors=_priors(), fixed={"c": 3.0})
    samples = s.sample(8)
    assert set(samples) == {"a", "b", "c"}
    assert samples["a"].shape == (8,)
    assert samples["b"].shape == (8,)
    assert torch.allclose(samples["c"], torch.full((8,), 3.0), atol=ATOL)


def test_sample_tolerates_trailing_singleton_priors():
    # Priors built with a trailing singleton dim (Normal(zeros(1), ones(1)))
    # must also yield (n_samples,) columns, not (n_samples, 1).
    priors = {n: Normal(torch.zeros(1), torch.ones(1)) for n in NAMES}
    s = UnconstrainedSpace(NAMES, priors=priors)
    samples = s.sample(8)
    assert samples["a"].shape == (8,)


def test_sample_rejects_multivariate_prior():
    # A name maps to a single scalar coordinate, so a multivariate prior
    # cannot be reshaped to (n_samples,) and must fail loudly.
    priors = {n: Normal(torch.zeros(2), torch.ones(2)) for n in NAMES}
    s = UnconstrainedSpace(NAMES, priors=priors)
    with pytest.raises(RuntimeError):
        s.sample(8)


def test_prior_metric_default_none():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    assert s.prior_metric(torch.randn(2, 3)) is None


def test_prior_metric_fn_is_used():
    def metric_fn(theta_full):
        return torch.eye(3).expand(theta_full.shape[0], 3, 3)
    s = UnconstrainedSpace(NAMES, priors=_priors(), prior_metric_fn=metric_fn)
    G = s.prior_metric(torch.randn(2, 3))
    assert G.shape == (2, 3, 3)


def test_point_inside_always_true():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    assert s.point_inside({"a": torch.tensor([1e9])})


# ========================================================================== #
#  UniformBoxSpace                                                            #
# ========================================================================== #

def _box(limits=None, names=None):
    names = names or ["x", "y"]
    limits = limits or {"x": (-1.0, 1.0), "y": (0.0, 10.0)}
    return UniformBoxSpace(limits, names, device="cpu")


def test_box_dimensions_and_limits():
    s = _box()
    assert s.d == 2
    assert s.d_full == 2
    assert s.free_names == ["x", "y"]
    assert torch.allclose(s.l, torch.tensor([-1.0, 0.0]), atol=ATOL)
    assert torch.allclose(s.u, torch.tensor([1.0, 10.0]), atol=ATOL)


def test_box_degenerate_limit_becomes_fixed():
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (5.0, 5.0)}, ["x", "y"],
                        device="cpu")
    assert s.fixed == {"y": 5.0}
    assert s.free_names == ["x"]
    assert s.d == 1
    assert s.d_full == 2
    assert s.fixed_indices == [1]


def test_box_map_round_trip():
    s = _box()
    theta = torch.tensor([[0.0, 5.0], [0.5, 2.0], [-0.7, 9.0]])
    z = s.map_to_unconstrained_vector(theta).mapped_point
    theta_back = s.map_to_constrained_vector(z).mapped_point
    assert torch.allclose(theta_back, theta, atol=1e-9)


def test_box_map_to_constrained_stays_inside():
    s = _box()
    z = torch.randn(20, 2) * 3.0
    theta = s.map_to_constrained_vector(z).mapped_point
    assert torch.all(theta > s.l)
    assert torch.all(theta < s.u)


def test_box_prior_log_prob_is_normalized_uniform():
    # No explicit prior -> uniform on the box, normalized: each free coordinate
    # contributes -log(u_i - l_i). Limits x:(-1,1), y:(0,10) -> -log(2)-log(10).
    s = _box()
    import math
    norm = -math.log(2.0) - math.log(10.0)
    y = {"x": torch.tensor([0.1, -0.2]), "y": torch.tensor([3.0, 4.0])}
    lp = s.prior_log_prob(y)
    assert lp.shape == (2,)
    assert torch.allclose(lp, torch.full((2,), norm), atol=ATOL)
    v = s.prior_log_prob_vector(torch.randn(7, 2))
    assert v.shape == (7,)
    assert torch.allclose(v, torch.full((7,), norm), atol=ATOL)


def test_box_prior_log_prob_marginal_subset():
    # Passing a subset of names returns the marginal prior over that subset:
    # only the provided coordinates' uniform normalizers are summed.
    import math
    s = _box()                                   # x:(-1,1), y:(0,10)
    lp_x = s.prior_log_prob({"x": torch.tensor([0.1, -0.2])})
    assert torch.allclose(lp_x, torch.full((2,), -math.log(2.0)), atol=ATOL)
    lp_y = s.prior_log_prob({"y": torch.tensor([3.0, 4.0])})
    assert torch.allclose(lp_y, torch.full((2,), -math.log(10.0)), atol=ATOL)


def test_box_prior_log_prob_marginal_subset_mixed():
    # Mixed: an explicit prior on x, uniform on y. The x-marginal is the user
    # density; the y-marginal is the uniform normalizer -log(u - l).
    import math
    priors = {"x": Normal(0.0, 1.0)}
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 10.0)}, ["x", "y"],
                        device="cpu", priors=priors)
    xv = torch.tensor([0.1, -0.2])
    lp_x = s.prior_log_prob({"x": xv})
    assert torch.allclose(lp_x, Normal(0.0, 1.0).log_prob(xv).squeeze(-1), atol=ATOL)
    lp_y = s.prior_log_prob({"y": torch.tensor([3.0, 4.0])})
    assert torch.allclose(lp_y, torch.full((2,), -math.log(10.0)), atol=ATOL)


def test_box_sample_generator_is_reproducible():
    s = _box()
    a = s.sample(32, generator=torch.Generator().manual_seed(0))
    b = s.sample(32, generator=torch.Generator().manual_seed(0))
    c = s.sample(32, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(a["x"], b["x"]) and torch.allclose(a["y"], b["y"])
    assert not torch.allclose(a["x"], c["x"])


def test_unconstrained_sample_generator_is_reproducible():
    s = UnconstrainedSpace(NAMES, priors=_priors())
    a = s.sample(32, generator=torch.Generator().manual_seed(0))
    b = s.sample(32, generator=torch.Generator().manual_seed(0))
    assert all(torch.allclose(a[n], b[n]) for n in NAMES)


def test_sample_generator_does_not_disturb_global_rng():
    # The forked RNG must leave the global stream untouched.
    s = _box()
    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    s.sample(16, generator=torch.Generator().manual_seed(7))
    after = torch.rand(3)
    assert torch.allclose(before, after)


def test_box_prior_metric_default_none():
    s = _box()
    assert s.prior_metric(torch.randn(3, 2)) is None


def test_box_prior_metric_fn_is_used():
    def metric_fn(theta_full):
        return torch.eye(2).expand(theta_full.shape[0], 2, 2)
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 10.0)}, ["x", "y"],
                        device="cpu", prior_metric_fn=metric_fn)
    G = s.prior_metric(torch.randn(3, 2))
    assert G.shape == (3, 2, 2)


def test_box_sample_inside_box():
    torch.manual_seed(0)
    s = _box()
    samples = s.sample(64)
    assert set(samples) == {"x", "y"}
    assert samples["x"].shape == (64,)
    assert torch.all(samples["x"] > -1.0) and torch.all(samples["x"] < 1.0)
    assert torch.all(samples["y"] > 0.0) and torch.all(samples["y"] < 10.0)


def test_box_sample_fills_fixed_value():
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (5.0, 5.0)}, ["x", "y"],
                        device="cpu")
    samples = s.sample(10)
    assert torch.allclose(samples["y"], torch.full((10,), 5.0), atol=ATOL)
    assert samples["x"].shape == (10,)


def test_box_point_inside_bounds_check():
    s = _box()
    assert s.point_inside({"x": torch.tensor([0.0]), "y": torch.tensor([5.0])})
    assert not s.point_inside({"x": torch.tensor([2.0]), "y": torch.tensor([5.0])})
    # boundary is treated as outside (strict inequalities)
    assert not s.point_inside({"x": torch.tensor([1.0]), "y": torch.tensor([5.0])})


def test_box_to_from_vector_round_trip():
    s = _box()
    samples = {"x": torch.tensor([0.1, 0.2]), "y": torch.tensor([3.0, 4.0])}
    vec = s.to_vector(samples)
    assert vec.shape == (2, 2)
    back = s.from_vector(vec)
    for n in ["x", "y"]:
        assert torch.allclose(back[n], samples[n], atol=ATOL)


def test_box_add_fixed_preserves_dtype():
    # Regression: the box space must fill fixed columns in the samples' dtype,
    # matching UnconstrainedSpace (it previously defaulted, dropping float32).
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (5.0, 5.0)}, ["x", "y"],
                        device="cpu")
    free = {"x": torch.tensor([0.3, -0.4], dtype=torch.float32)}
    assert s.add_fixed(free)["y"].dtype == torch.float32


def test_box_add_remove_fixed():
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (5.0, 5.0)}, ["x", "y"],
                        device="cpu")
    free = {"x": torch.tensor([0.3, -0.4])}
    full = s.add_fixed(free)
    assert torch.allclose(full["y"], torch.full((2,), 5.0), atol=ATOL)
    assert "y" not in free                   # purity
    assert set(s.remove_fixed(full)) == {"x"}


# -------------------------------------------------------------------------- #
#  UniformBoxSpace with per-name priors (truncated to the box)               #
# -------------------------------------------------------------------------- #

def test_box_prior_log_prob_uses_user_density():
    # With priors set, prior_log_prob sums the user's densities over free
    # coords, evaluated as-is (unnormalized truncated is fine).
    priors = {"x": Normal(0.0, 1.0), "y": Normal(2.0, 3.0)}
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 10.0)}, ["x", "y"],
                        device="cpu", priors=priors)
    y = {"x": torch.tensor([0.1, -0.2]), "y": torch.tensor([3.0, 4.0])}
    expected = (Normal(0.0, 1.0).log_prob(y["x"]).squeeze(-1)
                + Normal(2.0, 3.0).log_prob(y["y"]).squeeze(-1))
    assert torch.allclose(s.prior_log_prob(y), expected, atol=ATOL)


def test_box_prior_log_prob_vector_matches_dict_form():
    priors = {"x": Normal(0.0, 1.0), "y": Normal(2.0, 3.0)}
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 10.0)}, ["x", "y"],
                        device="cpu", priors=priors)
    theta_free = torch.tensor([[0.2, 5.0], [-0.5, 1.0], [0.9, 9.0]])
    vec = s.prior_log_prob_vector(theta_free)
    dct = s.prior_log_prob(s.from_vector(theta_free))
    assert vec.shape == (3,)
    assert torch.allclose(vec, dct, atol=ATOL)


def test_box_prior_skips_fixed_names():
    # A degenerate (fixed) coord contributes nothing even if a prior is given.
    priors = {"x": Normal(0.0, 1.0), "y": Normal(2.0, 3.0)}
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (5.0, 5.0)}, ["x", "y"],
                        device="cpu", priors=priors)
    assert s.free_names == ["x"]
    y = {"x": torch.tensor([0.1, -0.2])}
    expected = Normal(0.0, 1.0).log_prob(y["x"]).squeeze(-1)
    assert torch.allclose(s.prior_log_prob(y), expected, atol=ATOL)


def test_box_sample_with_prior_stays_inside_box():
    torch.manual_seed(0)
    priors = {"x": Normal(0.0, 1.0), "y": Normal(2.0, 3.0)}
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 10.0)}, ["x", "y"],
                        device="cpu", priors=priors)
    samples = s.sample(256)
    assert samples["x"].shape == (256,)
    assert torch.all(samples["x"] > -1.0) and torch.all(samples["x"] < 1.0)
    assert torch.all(samples["y"] > 0.0) and torch.all(samples["y"] < 10.0)


def test_box_sample_mixed_prior_and_uniform_coord():
    # A free name without a prior falls back to uniform on its interval.
    torch.manual_seed(0)
    priors = {"x": Normal(0.0, 1.0)}
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 10.0)}, ["x", "y"],
                        device="cpu", priors=priors)
    samples = s.sample(128)
    assert torch.all(samples["x"] > -1.0) and torch.all(samples["x"] < 1.0)
    assert torch.all(samples["y"] > 0.0) and torch.all(samples["y"] < 10.0)


def test_box_sample_raises_when_prior_misses_box():
    # Prior with negligible mass inside the box -> rejection cannot fill.
    priors = {"x": Normal(1000.0, 1.0)}
    s = UniformBoxSpace({"x": (-1.0, 1.0)}, ["x"], device="cpu", priors=priors)
    with pytest.raises(RuntimeError):
        s.sample(16)


def test_box_without_priors_is_uniform_normalized():
    # No priors -> uniform on the box, normalized to -log(volume) per coord.
    import math
    s = _box()
    norm = -math.log(2.0) - math.log(10.0)
    y = {"x": torch.tensor([0.1, -0.2]), "y": torch.tensor([3.0, 4.0])}
    assert torch.allclose(s.prior_log_prob(y), torch.full((2,), norm), atol=ATOL)
