"""Calibration of a sampler across many inference problems.

For each object there are posterior draws and a known truth. The PIT of a scalar
statistic ``T`` is the fraction of draws whose ``T`` falls below the truth's,
``P(T(y_s) < T(y*))``. Under correct calibration these PIT values are uniform
over objects (simulation-based calibration, Talts et al. 2018), so
``coverage_ci`` reads the empirical central-interval coverage at a nominal level
with an exact Clopper-Pearson interval.

``T`` is any user mapping from points to a scalar. A coordinate ``T(y) = y_k``
recovers the per-parameter rank, a likelihood ``T(y) = loglik(y)`` the likelihood
rank, and any other statistic works the same way.
"""
from collections import namedtuple

import numpy as np
from scipy import stats as sstats


def _ess(trace, method):
    """arviz effective sample size of a ``(chains, draws)`` trace."""
    import arviz as az
    idata = az.from_dict(posterior={"v": np.asarray(trace)})
    return float(az.ess(idata, method=method).to_dict()["data_vars"]["v"]["data"])


def pit(samples, truth, statistic, *, thin=True, ess_method="bulk"):
    """PIT of the truth for one object under ``statistic``.

    ``statistic`` maps the draws to a per-draw value and the truth to a scalar.
    Returns ``P(statistic(draw) < statistic(truth))``. With ``thin=True`` the
    statistic trace is thinned to about independence by its arviz ESS, which the
    SBC uniformity relies on. Pass ``thin=int`` to force the factor or
    ``thin=False`` to keep every draw.

    Parameters
    ----------
    samples
        Posterior draws for one object, shaped so ``statistic`` returns a
        ``(chains, draws)`` trace.
    truth
        The true parameters, shaped so ``statistic`` returns a scalar.
    statistic : callable
        Mapping from points to a scalar statistic ``T``.
    """
    t = np.asarray(statistic(samples), dtype=np.float64)
    t_truth = float(statistic(truth))
    if thin is True:
        ess = _ess(t, ess_method)
        tau = max(1, round(t.size / ess)) if np.isfinite(ess) and ess > 0 else 1
    else:
        tau = 1 if thin is False else max(1, int(thin))
    flat = t.reshape(-1)[::tau]
    return float(np.mean(flat < t_truth)) if flat.size else float("nan")


Coverage = namedtuple("Coverage", ["coverage", "low", "high", "n_objects"])


def coverage_ci(pit_values, level, *, confidence=0.95, weights=None):
    """Empirical central-``level`` coverage over objects, with an exact
    Clopper-Pearson interval.

    An object is covered iff its PIT lies in the central-``level`` interval
    ``[alpha/2, 1 - alpha/2]``. Non-finite PIT values are dropped. With
    ``weights`` (one per object) the coverage is weighted and the interval is the
    exact test at the Kish effective count ``(sum w)^2 / sum w^2``.

    Returns ``Coverage(coverage, low, high, n_objects)``.
    """
    p = np.asarray(pit_values, dtype=np.float64)
    finite = np.isfinite(p)
    p = p[finite]
    M = p.size
    if M == 0:
        return Coverage(np.nan, np.nan, np.nan, 0)

    alpha = 1.0 - level
    covered = (p >= alpha / 2.0) & (p <= 1.0 - alpha / 2.0)
    if weights is None:
        k, n = int(np.count_nonzero(covered)), M
        cov = k / n
    else:
        w = np.asarray(weights, dtype=np.float64)[finite]
        w = w / w.sum()
        cov = float(np.sum(w * covered))
        n = max(1, int(round(1.0 / float(np.sum(w * w)))))     # Kish effective count
        k = min(max(int(round(cov * n)), 0), n)

    ci = sstats.binomtest(k, n).proportion_ci(confidence_level=confidence, method="exact")
    return Coverage(cov, float(ci.low), float(ci.high), M)
