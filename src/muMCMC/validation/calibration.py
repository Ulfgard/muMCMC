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
from scipy.stats import binom, binomtest


def _as_numpy(x):
    """Float64 numpy view of ``x``, detaching a torch tensor (and moving it off
    device) first so a grad-carrying statistic does not error or leak a graph.
    Duck-typed to avoid importing torch here."""
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x, dtype=np.float64)


def _ess(trace, method):
    """arviz effective sample size of a ``(chains, draws)`` trace."""
    import arviz as az
    idata = az.from_dict(posterior={"v": np.asarray(trace)})
    return float(az.ess(idata, method=method).to_dict()["data_vars"]["v"]["data"])


SBCHistogram = namedtuple(
    "SBCHistogram", ["counts", "bin_edges", "expected", "low", "high", "n_objects"])

Coverage = namedtuple("Coverage", ["coverage", "low", "high", "target", "n_objects"])


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

    Feed one object at a time with :meth:`add`. Each object is ranked at a common
    ``L`` draws spread evenly across the whole chain (Talts et al. 2018, Algorithm
    2, "uniformly thin to ``L`` states"), so ranks from different objects share one
    scale. The spacing is the largest the chain allows and at least the ESS
    thinning factor (taken once, as the largest over the statistics), so every
    statistic uses ~independent draws. An object that cannot supply ``L`` draws
    that far apart is discarded and counted in :attr:`n_discarded`.
    :meth:`coverage` reads the central-interval coverage at a level and
    :meth:`sbc_histogram` the full rank histogram, both from the accumulated
    ranks of a statistic.

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
        self._weights = []
        self.n_discarded = 0

    def add(self, samples, truth, weight=1.0):
        """Rank one object and accumulate, or discard it if under-resolved.

        ``weight`` is a per-object weight carried into :meth:`coverage` (e.g. an
        importance weight for reweighting the test set); discarded objects drop
        their weight with them. Equal weights give the unweighted result.
        """
        traces = {n: _as_numpy(f(samples)) for n, f in self.statistics.items()}
        truths = {n: float(f(truth)) for n, f in self.statistics.items()}

        if self._thin is True:
            tau = 1
            for t in traces.values():
                ess = _ess(t, self._ess_method)
                if np.isfinite(ess) and ess > 0:
                    tau = max(tau, int(np.ceil(t.size / ess)))
        else:
            tau = 1 if self._thin is False else max(1, int(self._thin))

        # Place the L ranked draws as far apart as the chain allows, spanning the
        # whole chain rather than its first L*tau samples. The spacing n // L is
        # at least tau (the gate below), so with ESS >> L the draws end up far
        # more separated than tau: more nearly independent, and less sensitive to
        # an over-optimistic ESS.
        n = next(iter(traces.values())).size
        step = n // self.L
        if step < tau:                       # cannot place L draws >= tau apart
            self.n_discarded += 1
            return self
        idx = np.arange(self.L) * step
        for name, t in traces.items():
            self._ranks[name].append(
                int(np.count_nonzero(t.reshape(-1)[idx] < truths[name])))
        self._weights.append(float(weight))
        return self

    @property
    def n_objects(self):
        """Number of accumulated (non-discarded) objects."""
        return len(next(iter(self._ranks.values()))) if self._ranks else 0

    def ranks(self, name):
        """Accumulated SBC ranks for statistic ``name`` (shape ``(n_objects,)``)."""
        return np.array(self._ranks[name], dtype=int)

    def coverage(self, name, level, *, confidence=0.95):
        """Empirical central-``level`` coverage for statistic ``name``.

        The fraction of objects whose rank falls in the central-``level`` band
        (``rank / L in [alpha/2, 1 - alpha/2]``), with an exact Clopper-Pearson
        interval over the objects. ``target`` is the discrete-uniform reference
        ``p_L``: the coverage a calibrated sampler produces at this finite ``L``
        (it tends to ``level`` as ``L`` grows). Compare the coverage to ``target``,
        not to ``level`` -- that is how the finite-draw (ESS) uncertainty of the
        ranks enters, as a shift of the reference rather than a wider interval.

        The per-object weights from :meth:`add` reweight the coverage, and the
        interval is the Clopper-Pearson at the Kish effective count
        ``floor((sum w)^2 / sum w^2)`` (floored, which widens it). Equal weights
        give the unweighted coverage over all objects.

        Parameters
        ----------
        name : str
            Statistic to score.
        level : float
            Nominal central-interval level, in ``(0, 1)``.
        confidence : float
            Confidence level of the Clopper-Pearson interval.

        Returns
        -------
        Coverage
            ``(coverage, low, high, target, n_objects)``. ``n_objects`` is the raw
            retained count.
        """
        r = self.ranks(name)
        M = r.size
        alpha = 1.0 - level
        lo_edge, hi_edge = alpha / 2.0, 1.0 - alpha / 2.0
        grid = np.arange(self.L + 1) / self.L
        target = float(np.mean((grid >= lo_edge) & (grid <= hi_edge)))
        if M == 0:
            return Coverage(float("nan"), float("nan"), float("nan"), target, 0)
        covered = (r / self.L >= lo_edge) & (r / self.L <= hi_edge)
        w = np.asarray(self._weights, dtype=np.float64)
        w = w / w.sum()
        cov = float(np.sum(w * covered))
        n = max(1, int(1.0 / float(np.sum(w * w))))    # Kish effective count, floored
        k = min(max(int(round(cov * n)), 0), n)
        ci = binomtest(k, n).proportion_ci(confidence_level=confidence, method="exact")
        return Coverage(cov, float(ci.low), float(ci.high), target, M)

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
