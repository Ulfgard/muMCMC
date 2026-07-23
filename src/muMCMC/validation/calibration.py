"""Simulation-based calibration across many inference problems (Talts et al. 2018).

For each object there are posterior draws and a known truth. The SBC rank of a
scalar statistic ``T`` is ``r = #{draws : T(draw) < T(truth)}``, an integer in
``{0, ..., L}`` that is discrete-uniform under correct calibration (their
Theorem 1). Feed objects one at a time to a :class:`Calibration`; the accumulated
ranks per statistic give the SBC rank histogram with its discrete-uniform
confidence band (:meth:`Calibration.sbc_histogram`).

``T`` is any user mapping from points to a scalar. A coordinate ``T(y) = y_k``
recovers the per-parameter rank, a likelihood ``T(y) = loglik(y)`` the likelihood
rank, and any other statistic works the same way.
"""
from collections import namedtuple

import numpy as np
from scipy.stats import binom


def _ess(trace, method):
    """arviz effective sample size of a ``(chains, draws)`` trace."""
    import arviz as az
    idata = az.from_dict(posterior={"v": np.asarray(trace)})
    return float(az.ess(idata, method=method).to_dict()["data_vars"]["v"]["data"])


SBCHistogram = namedtuple(
    "SBCHistogram", ["counts", "bin_edges", "expected", "low", "high", "n_objects"])


def _sbc_histogram(ranks, L, *, n_bins=None, confidence=0.99):
    """SBC rank histogram with a discrete-uniform confidence band (Talts et al.
    2018).

    Bins the ranks over ``{0, ..., L}`` and returns the per-bin counts with the
    band under the discrete-uniform null: each bin count is ``Binomial(N, 1/B)``,
    and ``[low, high]`` are its central-``confidence`` quantiles. Counts drifting
    outside the band flag miscalibration. Non-finite ranks are dropped.

    Parameters
    ----------
    ranks : array-like
        One SBC rank per object, each in ``{0, ..., L}``.
    L : int
        Rank scale the ranks were computed on.
    n_bins : int, optional
        Number of equal bins over ``{0, ..., L}``. Default is ``L + 1`` (one bin
        per rank). Rebin (e.g. to keep ``N / n_bins`` around 20) to cut noise.
    confidence : float
        Central mass of the band under the uniform null.

    Returns
    -------
    SBCHistogram
        ``(counts, bin_edges, expected, low, high, n_objects)``. ``expected`` is
        ``N / n_bins`` and ``low`` / ``high`` are the band, shared by all bins.
    """
    r = np.asarray(ranks, dtype=np.float64)
    r = r[np.isfinite(r)]
    N = r.size
    B = (L + 1) if n_bins is None else int(n_bins)
    edges = np.linspace(-0.5, L + 0.5, B + 1)
    counts, _ = np.histogram(r, bins=edges)
    a = 1.0 - confidence
    low = int(binom.ppf(a / 2.0, N, 1.0 / B)) if N else 0
    high = int(binom.ppf(1.0 - a / 2.0, N, 1.0 / B)) if N else 0
    return SBCHistogram(counts, edges, N / B if N else float("nan"), low, high, N)


class Calibration:
    """Accumulate SBC ranks for several statistics over a stream of objects.

    Feed one object at a time with :meth:`add`. Each object's chain is thinned to
    about independence and truncated to a common ``L`` (Talts et al. 2018,
    Algorithm 2), so ranks from different objects share one scale. The chain is
    thinned once, by the largest per-statistic factor, so every statistic uses
    ~independent draws. An object with fewer than ``L`` effective draws is
    discarded and counted in :attr:`n_discarded`. :meth:`sbc_histogram` reads the
    accumulated ranks of a statistic.

    Parameters
    ----------
    statistics : dict[str, callable]
        Named statistics. Each maps the draws to a ``(chains, draws)`` trace and
        the truth to a scalar.
    L : int
        Common number of thinned draws to rank against (the rank scale).
    thin : bool or int
        ``True`` thins by ``max_k ceil(n / ESS_k)`` over the statistics, ``False``
        keeps every draw, an int forces the factor.
    ess_method : str
        arviz ESS method used when ``thin=True``.
    """

    def __init__(self, statistics, L, *, thin=True, ess_method="bulk"):
        self.statistics = dict(statistics)
        self.L = int(L)
        self._thin = thin
        self._ess_method = ess_method
        self._ranks = {name: [] for name in self.statistics}
        self.n_discarded = 0

    def add(self, samples, truth):
        """Rank one object and accumulate, or discard it if under-resolved."""
        traces = {n: np.asarray(f(samples), dtype=np.float64)
                  for n, f in self.statistics.items()}
        truths = {n: float(f(truth)) for n, f in self.statistics.items()}

        if self._thin is True:
            tau = 1
            for t in traces.values():
                ess = _ess(t, self._ess_method)
                if np.isfinite(ess) and ess > 0:
                    tau = max(tau, int(np.ceil(t.size / ess)))
        else:
            tau = 1 if self._thin is False else max(1, int(self._thin))

        thinned = {n: t.reshape(-1)[::tau] for n, t in traces.items()}
        if min(v.size for v in thinned.values()) < self.L:
            self.n_discarded += 1
            return self

        for n in self.statistics:
            self._ranks[n].append(int(np.count_nonzero(thinned[n][:self.L] < truths[n])))
        return self

    @property
    def n_objects(self):
        """Number of accumulated (non-discarded) objects."""
        return len(next(iter(self._ranks.values()))) if self._ranks else 0

    def ranks(self, name):
        """Accumulated SBC ranks for statistic ``name`` (shape ``(n_objects,)``)."""
        return np.array(self._ranks[name], dtype=int)

    def sbc_histogram(self, name, *, n_bins=None, confidence=0.99):
        """SBC rank histogram + band for statistic ``name``, as an
        ``SBCHistogram(counts, bin_edges, expected, low, high, n_objects)``.

        Bins the accumulated ranks over ``{0, ..., L}``; under the discrete-
        uniform null each bin count is ``Binomial(N, 1/n_bins)`` and
        ``[low, high]`` are its central-``confidence`` quantiles (the band).
        ``n_bins`` defaults to ``L + 1``; rebin to keep ``N / n_bins`` around 20.
        """
        return _sbc_histogram(self.ranks(name), self.L,
                              n_bins=n_bins, confidence=confidence)
