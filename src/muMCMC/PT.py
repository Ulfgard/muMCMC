from __future__ import annotations

import torch

from .MCMCSampler import MCMCSampler


# =========================================================================== #
#  Why a swap is a relabeling                                                 #
#                                                                             #
#  A temperature slot keeps its temperature for the whole run, so the         #
#  kernel's step size for that slot adapts over every transition taken there  #
#  rather than over a temperature that keeps changing under it.               #
#                                                                             #
#  A swap therefore moves configurations between slots instead of moving      #
#  temperatures between configurations. reorder permutes the kernel state     #
#  and leaves each slot's beta where it is, which recombines the two parts    #
#  of every tempered quantity at the new temperature. No model evaluation is  #
#  needed for that, so a swap sweep costs one permutation.                    #
# =========================================================================== #


class _PTState:
    """Kernel state over ``L * K`` replica slots, laid out ladder-major so slot
    ``l * K + k`` is rung ``k`` of ladder ``l``.

    Parameters
    ----------
    inner
        State of the wrapped kernel, over all ``L * K`` slots at once.
    L, K : int
        Number of ladders and number of temperatures on each.
    """

    def __init__(self, inner, L: int, K: int):
        self.inner = inner
        self.L, self.K = L, K

    @property
    def q(self) -> torch.Tensor:
        """Positions of the ``betas[-1]`` rung of each ladder, of shape
        ``(L, d)``. These are the draws a run collects. The lower rungs are
        there to move them and are not reported."""
        return self.inner.q.reshape(self.L, self.K, -1)[:, -1, :]


class PT(MCMCSampler):
    """Parallel tempering around a :class:`MCMCSampler` exploration kernel.

    Rung ``k`` targets

        pi_{beta_k}(theta) ~ prior(theta) * p(data | theta)**beta_k,

    so ``beta = 1`` is the posterior and ``beta = 0`` the prior. One
    :meth:`step` explores every rung with one kernel transition, then sweeps
    the even and then the odd adjacent pairs, exchanging rungs ``a`` and
    ``a+1`` with probability

        min(1, exp((beta_{a+1} - beta_a) (U_lik[a+1] - U_lik[a]))),

    where ``U_lik = -log p(data | theta)``. Only ``betas[-1]`` is sampled from,
    the rest of the ladder being there to move it.

    ``num_chains`` in :meth:`run_mcmc` is the number of ladders. Each is run at
    the full set of temperatures, so the kernel carries ``num_chains * K``
    chains.

    The kernel is driven rung by rung through its own ``beta``, which this
    class takes over for the duration of the run. Its state must expose ``q``,
    the tempered potential ``U``, whose ``lik`` part is the likelihood potential
    the swap ratio is built from, and a ``reorder`` that leaves each slot's
    temperature in place.

    Parameters
    ----------
    sampler : MCMCSampler
        Exploration kernel, used for every rung of every ladder at once.
    betas : Tensor, shape (K,)
        Increasing inverse temperatures. ``betas[-1]`` is the target.
    """

    def __init__(self, sampler: MCMCSampler, betas: torch.Tensor):
        super().__init__(sampler.potential_fn, sampler.space,
                         requires_metric=sampler.requires_metric)
        self.sampler = sampler
        self.betas = betas

    def to_position(self, theta_free: torch.Tensor) -> torch.Tensor:
        """The kernel's own, the rungs being the kernel's chains and running in
        whatever coordinates it runs in."""
        return self.sampler.to_position(theta_free)

    def to_variables(self, q_free: torch.Tensor) -> torch.Tensor:
        """The kernel's own, the inverse of :meth:`to_position`."""
        return self.sampler.to_variables(q_free)

    def init(self, q: torch.Tensor) -> _PTState:
        """The initial :class:`_PTState` at the positions ``q`` of shape
        ``(L, d)``, one per ladder. Each is copied to all ``K`` rungs of its
        ladder, and the kernel's ``beta`` is bound to the ladder-major slot
        layout for the rest of the run."""
        self.L, self.K = q.shape[0], len(self.betas)
        M = self.L * self.K
        self.sampler.beta = self.betas.unsqueeze(0).expand(self.L, -1).reshape(M)
        q_rep = q.unsqueeze(1).expand(self.L, self.K, -1).reshape(M, q.shape[-1])
        self._reset_stats()
        return _PTState(self.sampler.init(q_rep), self.L, self.K)

    def _reset_stats(self):
        L, K = self.L, self.K
        dtype, device = self.betas.dtype, self.betas.device
        self._swap_acc = torch.zeros(L, K - 1, dtype=dtype, device=device)
        self._swap_cnt = torch.zeros(L, K - 1, dtype=dtype, device=device)
        self._u_lik_sum = torch.zeros(L, K, dtype=dtype, device=device)
        self._nstep = 0

    def end_warmup(self):
        """Freeze the kernel's per-rung adaptation and zero the swap and
        potential statistics, so what :meth:`diagnostics` reports covers the
        sampling phase alone."""
        self.sampler.end_warmup()
        self._reset_stats()

    def _swap(self, u, parity):
        """One sweep over the adjacent pairs of one parity, as the per-ladder
        rung permutation of shape ``(L, K)`` it accepted and the likelihood
        potentials ``u`` gathered through that permutation.

        Parameters
        ----------
        u : Tensor, shape (L, K)
            Likelihood potentials, rung by rung.
        parity : int
            0 sweeps the pairs starting at an even rung, 1 those starting at an
            odd one. The two together cover every adjacent pair once.
        """
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
        """One kernel transition on every rung, then an even and an odd swap
        sweep composed into a single relabeling of the rungs."""
        L, K, M = self.L, self.K, self.L * self.K

        inner = self.sampler.step(s.inner)                 # explore every replica at its temperature
        u = inner.U.lik.reshape(L, K)                      # U_lik per temperature (grad-free, from state)
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
        """``swap``, the lowest mean swap rate over the adjacent pairs, which is
        the pair the ladder communicates worst across."""
        if self._nstep == 0:
            return {}
        rate = float((self._swap_acc / self._swap_cnt.clamp(min=1.0)).mean(0).min())
        return {"swap": f"{rate:.2f}"}

    def diagnostics(self) -> dict:
        """Diagnostics over the steps since the last :meth:`init` or
        :meth:`end_warmup`, empty before the first of them.

        The entries are ``betas``, ``swap_accept_rate`` per adjacent pair of
        shape ``(K-1,)``, ``explore_accept_rate`` per rung of shape ``(K,)``
        and averaged over the ladders, ``communication_barrier`` as the sum of
        the per-pair mean rejection, and ``log_evidence`` by thermodynamic
        integration,

            log_evidence = -sum_i 0.5 (u[i+1] + u[i]) (beta[i+1] - beta[i]),

        with ``u`` the run's mean likelihood potential per rung.

        ``log_evidence`` estimates ``log Z_last - log Z_first``, so it is an
        evidence in its own right only when ``betas[0]`` is 0 and the target at
        ``beta = 0`` integrates to one. A space carrying no prior leaves it
        offset by that target's own ``log Z``.
        """
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
