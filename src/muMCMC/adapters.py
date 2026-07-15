"""
Stochastic optimizers.

Two self-contained stochastic optimizers, vectorised over a leading axis so
that N independent problems are solved in parallel without coupling.

* :class:`DualAveraging` -- a first-order stochastic subgradient optimizer
  (Nesterov dual averaging; Nesterov [1], Hoffman & Gelman [2]).  Minimises
  a convex objective given a stream of (noisy) subgradients fed one per
  step.  Every update is elementwise, so the parameter and subgradients may
  be ``(N,)`` tensors.

* :class:`REINFORCEAdapter` -- a derivative-free (zeroth-order) stochastic
  optimizer: it minimises a Gaussian-smoothed objective using only noisy
  *evaluations* of the objective (no gradient access), estimating the
  gradient with the score-function / REINFORCE estimator and driving the
  parameter with a dual-averaging optimizer.

References
----------
[1] Nesterov, "Primal-dual subgradient methods for convex problems".
[2] Hoffman & Gelman, "The No-U-Turn Sampler: adaptively setting path
    lengths in Hamiltonian Monte Carlo".
"""

import torch


class DualAveraging:
    """
    First-order stochastic subgradient optimizer (Nesterov dual averaging).

    Minimises a convex objective ``J(x)`` over a parameter ``x`` using a
    stream of (possibly noisy) subgradients, one per iteration.  The
    *averaged* iterate ``x_avg`` converges to the optimum even though
    individual subgradients are noisy.

    Optimizer contract (ask/tell)
    -----------------------------
    Repeat::

        x_t, x_avg = opt.get_state()   # current iterate and its running average
        g = subgradient_of_J(x_t)      # caller evaluates a subgradient at x_t
        opt.step(g)                    # tell the optimizer

    ``g`` is a subgradient (or unbiased noisy estimate of one) of the
    objective.  Use ``x_t`` while optimising and ``x_avg`` as the final
    answer.

    Vectorisation
    -------------
    Every update is elementwise, so ``prox_center`` and the subgradient
    ``g`` may be scalars or ``(N,)`` tensors -- the latter solves ``N``
    independent problems in parallel.  The iteration counter is shared.

    Parameters
    ----------
    prox_center : float or Tensor
        Reference point the primal sequence is pulled toward (a soft initial
        guess).  May be set after construction (it is read in
        :meth:`step`).  Default 0.
    t0 : float
        Stabilises the early iterations.  Default 10.
    kappa : float
        Averaging-weight exponent, in (0.5, 1].  Smaller forgets early
        iterates faster.  Default 0.75.
    gamma : float
        Step-scale parameter controlling convergence speed.  Default 0.05.
    """

    def __init__(self, prox_center=0.0, t0=10, kappa=0.75, gamma=0.05):
        self.prox_center = prox_center
        self.t0 = t0
        self.kappa = kappa
        self.gamma = gamma
        self.reset()

    def reset(self):
        # average of the primal sequence; seeded at prox_center so
        # get_state() is valid before any step.
        self._x_avg = self.prox_center
        self._g_avg = 0.0   # average of dual sequence
        self._t = 0
        # latest primal point; equals prox_center before the first step.
        self._x_t = self.prox_center

    def step(self, g):
        """
        Tell the optimizer a new subgradient ``g`` (scalar or ``(N,)``)
        evaluated at the current iterate, and advance one iteration.
        """
        self._t += 1
        # g_avg = (g_1 + ... + g_t) / t   (running, t0-stabilised)
        self._g_avg = (1 - 1 / (self._t + self.t0)) * self._g_avg + g / (
            self._t + self.t0
        )
        # x_t = argmin{ g_avg . x + loc_t |x - x0|^2 }, loc_t := (gamma/2) sqrt(t) / t
        self._x_t = self.prox_center - (self._t ** 0.5) / self.gamma * self._g_avg
        # weighted average of the primal sequence
        weight_t = self._t ** (-self.kappa)
        self._x_avg = (1 - weight_t) * self._x_avg + weight_t * self._x_t

    def get_state(self):
        """Return ``(x_t, x_avg)`` -- latest primal point and its average."""
        return self._x_t, self._x_avg


class REINFORCEAdapter:
    """
    Derivative-free (zeroth-order) stochastic optimizer.

    Minimises the Gaussian-smoothed objective

        J(mu) = E_{eps ~ N(0, I)} [ f(mu + sigma * eps) ]

    over ``mu``, using only noisy *evaluations* of ``f`` -- no gradient of
    ``f`` is required.  The gradient of ``J`` is estimated unbiasedly by the
    score-function / REINFORCE estimator

        grad J ≈ (f_t − b_t) · eps_t / sigma,

    where ``eps_t`` is the perturbation used at this step and ``b_t`` is an
    EMA baseline that reduces variance without introducing bias.  That
    gradient drives a dual-averaging optimizer on ``mu``.

    Optimizer contract (ask/tell)
    -----------------------------
    Each step evaluates ``f`` once, at the point the optimizer proposes::

        x, mu = opt.get_state()    # x = mu_t + sigma*eps_t (point to probe), mu = best estimate
        f_t = f(x)                 # caller evaluates the objective at x
        opt.step(f_t)              # tell the optimizer; it draws a fresh eps for next step

    The proposal ``x`` is fixed between :meth:`get_state` and :meth:`step`,
    so ``f_t`` must be ``f`` evaluated at exactly that ``x``.  Use the
    returned ``mu`` (the dual-averaged estimate) as the final optimum.

    Vectorisation
    -------------
    Solves ``n`` independent problems in parallel: ``f_t`` is ``(N,)`` and
    the returned state is ``(N,)``.  Per-problem signals are never mixed.

    Parameters
    ----------
    n : int
        Number of independent problems optimised in parallel.
    sigma : float
        Perturbation / smoothing radius, and the denominator of the
        gradient estimate.  Larger explores more but smooths the objective
        more.  Default 0.1.
    ema_decay : float
        Decay factor of the EMA baseline used for variance reduction.
        Default 0.2.
    gamma : float
        Step-scale of the underlying dual averaging (the learning rate; the
        update is ``-sqrt(t)/gamma * g_avg``, so *larger* gamma means
        gentler steps).  Default 0.05.  A steeper or unbounded objective may
        need a larger gamma to stay stable.
    """

    def __init__(self, n: int, sigma: float = 0.1, ema_decay: float = 0.2,
                 gamma: float = 0.05):
        self.n           = n
        self.sigma       = sigma
        self.ema_decay   = ema_decay
        self.prox_center = torch.zeros(n)   # (N,) log initial step; set externally
        self._dual       = DualAveraging(gamma=gamma)
        self._g          = None             # EMA baseline (N,), None until first step

    def reset(self):
        self._dual.prox_center = self.prox_center    # (N,)
        self._dual.reset()
        self._g   = None
        self._eps = torch.randn(self.n)     # (N,)

    def step(self, f_t: torch.Tensor):
        """Tell the optimizer the objective value f_t (shape (N,)) observed at
        the current proposal; draw a fresh perturbation for the next step."""
        if self._g is None:
            self._g = f_t.clone()
        else:
            self._g = self.ema_decay * self._g + (1.0 - self.ema_decay) * f_t

        # REINFORCE gradient estimate, per chain (N,)
        stat = (f_t - self._g) / self.sigma * self._eps
        self._dual.step(stat)               # vectorised, (N,) input

        self._eps = torch.randn(self.n)

    def get_state(self):
        """Return (proposal, mu) each of shape (N,):

        - proposal = mu_t + sigma * eps_t, the point at which the caller
          must next evaluate the objective f;
        - mu       = the dual-averaged estimate of the optimum.

        The proposal is fixed until the next step() call.
        """
        x_t, x_avg = self._dual.get_state()
        return (x_t + self.sigma * self._eps, x_avg)
