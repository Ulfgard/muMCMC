"""Tests for ``muMCMC.validation.coverage``: PIT of a statistic and coverage.

``coverage_ci`` is checked against hand-computed cases; ``pit`` against Gaussian
quantiles and an arbitrary statistic; and the pair end to end for the SBC
property that correct calibration yields coverage at the nominal level.
"""
import math

import numpy as np

from muMCMC.validation.coverage import pit, coverage_ci, Coverage


# --------------------------------------------------------------------------- #
#  coverage_ci                                                                 #
# --------------------------------------------------------------------------- #

def test_coverage_ci_deterministic():
    # central-50% interval is [0.25, 0.75]; two of four PITs fall inside.
    c = coverage_ci([0.1, 0.5, 0.9, 0.5], 0.5)
    assert isinstance(c, Coverage)
    assert c.n_objects == 4
    assert abs(c.coverage - 0.5) < 1e-12
    assert c.low <= 0.5 <= c.high


def test_coverage_ci_drops_nonfinite():
    c = coverage_ci([0.5, np.nan, 0.5], 0.5)
    assert c.n_objects == 2 and abs(c.coverage - 1.0) < 1e-12


def test_coverage_ci_empty():
    c = coverage_ci([np.nan, np.nan], 0.5)
    assert c.n_objects == 0 and math.isnan(c.coverage)


def test_coverage_ci_weighted_equal_matches_unweighted():
    pits = [0.1, 0.5, 0.9, 0.5]
    c = coverage_ci(pits, 0.5, weights=np.ones(4))
    assert c.n_objects == 4 and abs(c.coverage - 0.5) < 1e-12


def test_coverage_ci_interval_matches_scipy_binomial():
    # The Beta-quantile Clopper-Pearson must reproduce scipy's exact binomial CI
    # for integer counts (unweighted). 3 of 5 PITs inside the central-50% band.
    from scipy.stats import binomtest
    pits = [0.5, 0.5, 0.5, 0.1, 0.9]           # 3 covered at level 0.5
    c = coverage_ci(pits, 0.5)
    ci = binomtest(3, 5).proportion_ci(confidence_level=0.95, method="exact")
    assert abs(c.low - ci.low) < 1e-9 and abs(c.high - ci.high) < 1e-9


def test_coverage_ci_weighted_reports_weighted_coverage():
    # Weighting the two covered PITs up raises the coverage above the unweighted
    # 0.5, and the interval is valid.
    pits = [0.5, 0.5, 0.1, 0.9]
    w = np.array([3.0, 3.0, 1.0, 1.0])
    c = coverage_ci(pits, 0.5, weights=w)
    assert abs(c.coverage - 0.75) < 1e-12
    assert 0.0 <= c.low <= c.high <= 1.0 and c.n_objects == 4


# --------------------------------------------------------------------------- #
#  pit                                                                         #
# --------------------------------------------------------------------------- #

def test_pit_matches_gaussian_quantile():
    s = np.random.default_rng(0).standard_normal((4, 5000))
    ident = lambda a: a
    assert abs(pit(s, 0.0, ident, thin=False) - 0.5) < 0.02
    assert abs(pit(s, 1.0, ident, thin=False) - 0.8413) < 0.02


def test_pit_arbitrary_statistic():
    # T(y) = y^2: P(y^2 < 1) = P(|y| < 1) ~ 0.6827 for N(0, 1).
    s = np.random.default_rng(1).standard_normal((4, 5000))
    assert abs(pit(s, 1.0, lambda a: a ** 2, thin=False) - 0.6827) < 0.02


def test_pit_thinning_iid_matches_unthinned():
    # iid draws -> arviz ESS ~ N -> tau ~ 1 -> thinned ~ unthinned.
    s = np.random.default_rng(2).standard_normal((4, 4000))
    ident = lambda a: a
    assert abs(pit(s, 0.5, ident, thin=True) - pit(s, 0.5, ident, thin=False)) < 0.03


def test_pit_explicit_thin_factor():
    s = np.random.default_rng(3).standard_normal((4, 4000))
    p = pit(s, 0.0, lambda a: a, thin=2)
    assert abs(p - 0.5) < 0.03


# --------------------------------------------------------------------------- #
#  SBC: correct calibration -> nominal coverage                               #
# --------------------------------------------------------------------------- #

def test_coverage_is_calibrated_under_sbc():
    # truth ~ N(0,1), draws ~ N(0,1) => PIT = Phi(truth) ~ Uniform, so empirical
    # coverage matches the nominal level (within the exact interval).
    rng = np.random.default_rng(4)
    M = 600
    ident = lambda a: a
    truths = rng.standard_normal(M)
    pits = [pit(rng.standard_normal((2, 400)), float(t), ident, thin=False)
            for t in truths]
    for level in (0.5, 0.95):
        c = coverage_ci(pits, level)
        assert c.low <= level <= c.high, (level, c)
