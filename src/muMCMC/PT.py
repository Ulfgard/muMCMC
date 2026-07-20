from __future__ import annotations

import torch

from .BaseSampler import BaseSampler


class _PTState:
    """The wrapped kernel's state over ``L * K`` replica slots.  ``q`` projects
    out the temperature axis, exposing the target chain for the base driver."""

    def __init__(self, inner, L: int, K: int):
        self.inner = inner
        self.L, self.K = L, K

    @property
    def q(self) -> torch.Tensor:
        return self.inner.q.reshape(self.L, self.K, -1)[:, -1, :]   # target chain


class PT(BaseSampler):
    """
    Parallel tempering.  A :class:`BaseSampler` that wraps another
    :class:`BaseSampler` as its exploration kernel.

    The state is replicated along an axis of K inverse temperatures ``betas``;
    replica k targets

        pi_{beta_k}(theta)  ~  prior(theta) * p(data | theta) ** beta_k

    so ``beta = 1`` is the posterior and ``beta = 0`` the prior.  Each step
    explores every replica with one kernel transition at its temperature, then
    sweeps the even and odd adjacent pairs, exchanging the configurations of
    replicas ``a`` and ``a+1`` with probability

        min(1, exp((beta_{a+1} - beta_a) (U_lik[a+1] - U_lik[a]))),

    ``U_lik = -log p(data | theta)``.  Each replica keeps its temperature for the
    whole run, so the kernel's per-temperature step size adapts during warmup.
    A swap only relabels configurations across temperature slots: the kept
    kernel state is permuted with :meth:`reorder`, which retempers each moved
    configuration to its new slot temperature (no model re-evaluation).

    The wrapped kernel's state must expose ``q`` and a ``reorder`` that
    retempers under a temperature change (as ``RMHMCState`` does).

    Parameters
    ----------
    sampler
        Exploration kernel (a :class:`BaseSampler`).
    betas
        Increasing inverse temperatures as a 1-D tensor; the target chain is
        ``betas[-1]``.
    """

    def __init__(self, sampler: BaseSampler, betas: torch.Tensor):
        super().__init__(sampler.potential_fn, sampler.space,
                         requires_metric=sampler.requires_metric)
        self.sampler = sampler
        self.betas = betas

    def init(self, q: torch.Tensor) -> _PTState:
        self.L, self.K = q.shape[0], len(self.betas)
        M = self.L * self.K
        self.sampler.beta = self.betas.unsqueeze(0).expand(self.L, -1).reshape(M)
        z = q.unsqueeze(1).expand(self.L, self.K, -1).reshape(M, q.shape[-1])
        self._reset_stats()
        return _PTState(self.sampler.init(z), self.L, self.K)

    def _reset_stats(self):
        L, K = self.L, self.K
        dtype, device = self.betas.dtype, self.betas.device
        self._swap_acc = torch.zeros(L, K - 1, dtype=dtype, device=device)
        self._swap_cnt = torch.zeros(L, K - 1, dtype=dtype, device=device)
        self._u_lik_sum = torch.zeros(L, K, dtype=dtype, device=device)
        self._nstep = 0

    def end_warmup(self):
        self.sampler.end_warmup()          # freeze the kernel's step-size adaptation
        self._reset_stats()

    def _swap(self, u, parity):
        """One even (parity 0) or odd (parity 1) sweep over adjacent pairs.
        Records per-pair acceptance and returns the per-ladder column
        permutation ``(L, K)`` together with the likelihood potentials gathered
        through it (config-bound, so ``u`` rides along)."""
        L, K = self.L, self.K
        device = u.device
        a = torch.arange(parity, K - 1, 2, device=device)
        b = a + 1
        logr = (self.betas[b] - self.betas[a]) * (u[:, b] - u[:, a])       # (L, P)
        accepted = torch.log(torch.rand(L, a.shape[0], dtype=u.dtype, device=device)) < logr
        perm = torch.arange(K, device=device).expand(L, K).clone()
        for p in range(a.shape[0]):
            m = accepted[:, p]
            perm[m, a[p]] = b[p]
            perm[m, b[p]] = a[p]
        self._swap_acc[:, a] += accepted.to(u.dtype)
        self._swap_cnt[:, a] += 1
        return perm, torch.gather(u, 1, perm)

    def step(self, s: _PTState) -> _PTState:
        L, K, M = self.L, self.K, self.L * self.K

        inner = self.sampler.step(s.inner)                 # explore every replica at its temperature
        u = self.potential_likelihood(inner.q).reshape(L, K)   # U_lik per temperature
        self._u_lik_sum += u                               # for thermodynamic integration

        # even then odd swap sweep, composed into one relabeling of the replicas
        perm0, u = self._swap(u, 0)
        perm1, u = self._swap(u, 1)
        perm = torch.gather(perm0, 1, perm1)               # apply even, then odd
        flat = (torch.arange(L, device=perm.device).unsqueeze(1) * K + perm).reshape(M)

        inner = inner.reorder(flat)                        # retemper the moved configs
        self._nstep += 1
        return _PTState(inner, L, K)

    def logging(self) -> dict:
        if self._nstep == 0:
            return {}
        rate = float((self._swap_acc / self._swap_cnt.clamp(min=1.0)).min())
        return {"swap": f"{rate:.2f}"}

    def diagnostics(self) -> dict:
        """Post-warmup diagnostics: the ladder, per-pair ``swap_accept_rate`` and
        per-chain ``explore_accept_rate`` (averaged over ladders), the global
        ``communication_barrier`` (sum of per-pair mean rejection), and a
        thermodynamic-integration ``log_evidence`` (absolute when beta_min=0)."""
        if self._nstep == 0:
            return {}
        swap_rate = (self._swap_acc / self._swap_cnt.clamp(min=1.0)).mean(0)
        u_mean = self._u_lik_sum / self._nstep
        db = self.betas[1:] - self.betas[:-1]
        log_ev = -(0.5 * (u_mean[:, 1:] + u_mean[:, :-1]) * db).sum(-1)     # (L,)
        explore = self.sampler.diagnostics()["accept_rate"].reshape(self.L, self.K)
        return {
            "betas": self.betas,
            "swap_accept_rate": swap_rate,
            "explore_accept_rate": explore.mean(0),
            "communication_barrier": float((1.0 - swap_rate).sum()),
            "log_evidence": float(log_ev.mean()),
        }
