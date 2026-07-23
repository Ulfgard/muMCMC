"""Tests for ``muMCMC.validation.calibration``: SBC ranks and histogram band.

``sbc_histogram`` is checked against hand-computed cases and the binomial band;
``Calibration`` for its streaming accumulation, multi-statistic tracking,
under-resolved discarding, and the SBC property that correct calibration keeps
the rank histogram inside the band while an overconfident posterior breaches it.
"""
import numpy as np

from muMCMC.validation import Calibration
from muMCMC.validation.calibration import _sbc_histogram, SBCHistogram, Coverage


def _coord(k):
    """Statistic T(y) = y_k, valid on both a (chains, draws, d) trace and a
    (d,) truth (coordinate on the last axis)."""
    return lambda s: s[..., k]


# --------------------------------------------------------------------------- #
#  sbc_histogram                                                              #
# --------------------------------------------------------------------------- #

def test_sbc_histogram_mechanics():
    h = _sbc_histogram([0, 1, 2, 3, 99, 50, 50], L=99, n_bins=10)
    assert isinstance(h, SBCHistogram)
    assert h.counts.sum() == 7 and h.n_objects == 7
    assert abs(h.expected - 0.7) < 1e-12 and h.low <= h.high


def test_sbc_histogram_drops_nonfinite():
    h = _sbc_histogram([0.0, np.nan, 99.0], L=99, n_bins=10)
    assert h.n_objects == 2 and h.counts.sum() == 2


# --------------------------------------------------------------------------- #
#  Calibration accumulator                                                    #
# --------------------------------------------------------------------------- #

def test_calibration_accumulates_and_discards():
    cal = Calibration({"y0": _coord(0)}, L=99, thin=False)
    rng = np.random.default_rng(10)
    cal.add(rng.standard_normal((2, 100, 1)), rng.standard_normal(1))      # 200 draws -> kept
    cal.add(rng.standard_normal((2, 40, 1)), rng.standard_normal(1))       # 80 draws  -> discarded
    assert cal.n_objects == 1 and cal.n_discarded == 1
    assert cal.ranks("y0").shape == (1,) and 0 <= cal.ranks("y0")[0] <= 99


def test_calibration_tracks_multiple_statistics():
    cal = Calibration({"y0": _coord(0), "y1": _coord(1)}, L=99, thin=False)
    rng = np.random.default_rng(11)
    for _ in range(5):
        cal.add(rng.standard_normal((2, 120, 2)), rng.standard_normal(2))
    assert cal.n_objects == 5
    assert cal.ranks("y0").shape == (5,) and cal.ranks("y1").shape == (5,)
    # the two statistics are ranked independently.
    assert not np.array_equal(cal.ranks("y0"), cal.ranks("y1"))


def test_calibration_uniform_under_calibration():
    # truth ~ N(0,I), draws ~ N(0,I) -> ranks discrete-uniform -> counts in band.
    cal = Calibration({"y0": _coord(0)}, L=99, thin=False)
    rng = np.random.default_rng(12)
    for _ in range(1000):
        cal.add(rng.standard_normal((2, 120, 1)), rng.standard_normal(1))
    h = cal.sbc_histogram("y0", n_bins=10)
    outside = int(np.sum((h.counts < h.low) | (h.counts > h.high)))
    assert outside <= 1


def test_coverage_matches_target_under_calibration():
    # calibrated draws -> coverage at each level sits at the finite-L target p_L,
    # which for L=99 is within 0.02 of the nominal level.
    cal = Calibration({"y0": _coord(0)}, L=99, thin=False)
    rng = np.random.default_rng(20)
    for _ in range(1500):
        cal.add(rng.standard_normal((2, 120, 1)), rng.standard_normal(1))
    for level in (0.5, 0.75, 0.95):
        c = cal.coverage("y0", level)
        assert isinstance(c, Coverage) and c.n_objects == 1500
        assert abs(c.target - level) < 0.02
        assert abs(c.coverage - c.target) < 0.04       # a few binomial SE at M=1500
        assert c.low <= c.coverage <= c.high


def test_coverage_flags_overconfident_posterior():
    # posterior too narrow: the central interval misses the truth far more often,
    # so coverage falls well below the target and the target leaves the interval.
    cal = Calibration({"y0": _coord(0)}, L=99, thin=False)
    rng = np.random.default_rng(21)
    for _ in range(1000):
        cal.add(0.5 * rng.standard_normal((2, 120, 1)), rng.standard_normal(1))
    c = cal.coverage("y0", 0.95)
    assert c.coverage < c.target - 0.05
    assert not (c.low <= c.target <= c.high)


def test_calibration_flags_overconfident_posterior():
    # posterior too narrow (std 0.5) vs truth ~ N(0,1): ranks pile at the edges,
    # so the histogram breaches the uniform band.
    cal = Calibration({"y0": _coord(0)}, L=99, thin=False)
    rng = np.random.default_rng(13)
    for _ in range(1000):
        cal.add(0.5 * rng.standard_normal((2, 120, 1)), rng.standard_normal(1))
    h = cal.sbc_histogram("y0", n_bins=10)
    outside = int(np.sum((h.counts < h.low) | (h.counts > h.high)))
    assert outside >= 2


def test_calibration_thin_true_uses_arviz():
    # smoke test of the arviz thinning path on iid draws (ESS ~ n -> tau ~ 1).
    cal = Calibration({"y0": _coord(0)}, L=50, thin=True)
    rng = np.random.default_rng(14)
    cal.add(rng.standard_normal((2, 400, 1)), rng.standard_normal(1))
    assert cal.n_objects == 1 and 0 <= cal.ranks("y0")[0] <= 50
