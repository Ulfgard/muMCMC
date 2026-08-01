import torch

# =========================================================================== #
#                                                                             #
#  Value adapters                                                             #
#                                                                             #
#  The three classes share one protocol, reset / update / finalize /          #
#  get_state, without a common base class. Each is a minimiser in its own     #
#  right and the adapter role is a second reading of the same state, so a     #
#  base class would have to own the intersection of two interfaces and would  #
#  buy nothing back.                                                          #
#                                                                             #
#  An adapter has no notion of what its value means. The caller owns that, so #
#  the same three classes serve a step size, a mass scale or a temperature.   #
#  Every value is vectorised over a leading axis, so N independent problems   #
#  adapt elementwise in parallel.                                             #
#                                                                             #
#  Internal. A sampler builds its own adapter from its own arguments, so      #
#  nothing here is part of the package surface.                               #
#                                                                             #
# =========================================================================== #


class NoAdaptation:
    """Adapter that holds the value fixed at its initial value.

    Parameters
    ----------
    init : float
        The fixed value, broadcast to every problem at :meth:`reset`.
    """

    def __init__(self, init):
        self._init = float(init)
        self._value = None

    def reset(self, N, dtype, device):
        """Size the (constant) value to ``(N,)``."""
        self._value = torch.full((N,), self._init, dtype=dtype, device=device)

    def set_upper_bound(self, ub):
        """Does nothing. The value is constant at ``init``, so an ``init``
        greater than ``ub`` is reported unchanged."""
        pass

    def update(self, signal):
        """Does nothing. The value is constant."""
        pass

    def finalize(self):
        """Does nothing. There is no estimate to freeze."""
        pass

    def get_state(self):
        """Return ``(value, value)``, constant in both entries."""
        return self._value, self._value


class DualAveraging:
    """Dual-averaging minimiser of a convex objective, usable as an adapter.

    As a minimiser, each :meth:`step` folds a subgradient into the running mean
    ``g_avg`` and updates

        x_t   = prox_center - sqrt(t)/gamma * g_avg
        x_avg = (1 - t^-kappa) x_avg + t^-kappa x_t

    ``prox_center``, the subgradient, and the state are scalars or ``(N,)``
    tensors, updated elementwise.

    As an adapter, :meth:`reset` with a batch size seeds ``prox_center`` at
    ``init`` over ``(N,)``, :meth:`update` folds one subgradient, and
    :meth:`finalize` freezes the estimate, after which :meth:`get_state` reports
    ``(x_avg, x_avg)``.

    Parameters
    ----------
    prox_center : float or Tensor
        Reference point the primal sequence is pulled toward (minimiser use).
        Default 0.
    t0 : float
        Offset in the subgradient average's weight ``1/(t + t0)``, which damps
        the first iterations. Default 10.
    kappa : float
        Averaging-weight exponent in (0.5, 1]. Default 0.75.
    gamma : float
        Gain. ``x_t`` is displaced from ``prox_center`` by ``sqrt(t)/gamma``
        times the averaged subgradient, so a smaller value is a larger
        response. Default 0.05.
    init : float
        Initial value seeded over ``(N,)`` at :meth:`reset` in the adapter role.
        Ignored by the minimiser, which uses ``prox_center`` instead. Default 0.

    References
    ----------
    Nesterov, Primal-dual subgradient methods for convex problems.
    Hoffman and Gelman, The No-U-Turn Sampler.
    """

    def __init__(self, prox_center=0.0, t0=10, kappa=0.75, gamma=0.05, init=0.0):
        self.prox_center = prox_center
        self.t0 = t0
        self.kappa = kappa
        self.gamma = gamma
        self._init = init
        self.ub = float("inf")      # upper bound on the value (log step size)
        # State is device-dependent, so it is built by reset(), not here: the
        # caller must reset() before use (as init() does for the adapter role).

    def reset(self, N=None, dtype=None, device=None):
        """Reset the iteration state. With a batch size ``N`` given, seed
        ``prox_center`` at ``init`` over ``(N,)`` for the adapter role."""
        if N is not None:
            self.prox_center = torch.full((N,), float(self._init),
                                          dtype=dtype, device=device)
        self._x_avg = self.prox_center
        self._g_avg = 0.0
        self._t = 0
        self._x_t = self.prox_center
        self._frozen = False

    def step(self, g):
        """Fold subgradient ``g`` (scalar or ``(N,)``) into the average and
        advance one iteration."""
        self._t += 1
        # t0-stabilised running mean of the subgradients
        self._g_avg = (1 - 1 / (self._t + self.t0)) * self._g_avg + g / (
            self._t + self.t0
        )
        # x_t = prox_center - sqrt(t)/gamma * g_avg
        self._x_t = self.prox_center - (self._t ** 0.5) / self.gamma * self._g_avg
        # Bound the state (not just the reported value) so a later downward
        # push from a bad sample acts immediately, with no inflation to unwind.
        if self.ub != float("inf"):
            self._x_t = torch.clamp(self._x_t, max=self.ub)
        # x_avg = (1 - t^-kappa) x_avg + t^-kappa x_t
        weight_t = self._t ** (-self.kappa)
        self._x_avg = (1 - weight_t) * self._x_avg + weight_t * self._x_t

    def get_state(self):
        """Return ``(x_t, x_avg)``, the latest primal point and its average, or
        ``(x_avg, x_avg)`` once frozen."""
        if self._frozen:
            return self._x_avg, self._x_avg
        return self._x_t, self._x_avg

    # ---- adapter role ------------------------------------------------------- #

    def update(self, subgradient):
        """Fold one ``subgradient`` (no-op once frozen)."""
        if not self._frozen:
            self.step(subgradient)

    def set_upper_bound(self, ub):
        """Cap the value (and its running average) at ``ub`` from now on."""
        self.ub = ub

    def finalize(self):
        """Freeze the estimate: :meth:`get_state` now reports ``(x_avg, x_avg)``."""
        self._frozen = True


class Reinforce:
    """Derivative-free minimiser of the Gaussian-smoothed objective

        J(mu) = E_{eps ~ N(0, I)} [ f(mu + sigma * eps) ]

    from noisy evaluations of ``f`` alone, usable as an adapter. Each :meth:`step`
    forms the score-function estimate

        grad J ~ (f_t - b_t) * eps_t / sigma

    with ``b_t`` an EMA baseline, and feeds it to a :class:`DualAveraging` on
    ``mu``. ``f_t`` and the state are ``(N,)``, one entry per problem.

    As an adapter, :meth:`get_state` reports the perturbed point
    ``mu_t + sigma * eps``, which is the next value to try, and the averaged
    ``mu``. Once frozen it reports ``(mu, mu)``.

    Parameters
    ----------
    n : int or None
        Number of problems in the minimiser role, or None when the batch size
        is instead seeded through :meth:`reset`.
    sigma : float
        Radius the objective is smoothed over, and the denominator of the
        gradient estimate. Default 0.1.
    ema_decay : float
        Decay of the baseline ``b_t``. Default 0.2.
    gamma : float
        Gain of the underlying dual averaging. Default 0.05.
    init : float
        Initial value seeded over ``(N,)`` at :meth:`reset` in the adapter role.
        Ignored by the minimiser, which uses ``prox_center`` instead. Default 0.
    """

    def __init__(self, n: int = None, sigma: float = 0.1, ema_decay: float = 0.2,
                 gamma: float = 0.05, init=0.0):
        self.n           = n
        self.sigma       = sigma
        self.ema_decay   = ema_decay
        self.prox_center = torch.zeros(n) if n is not None else None
        self._init       = init
        self._dual       = DualAveraging(gamma=gamma)
        self._g          = None             # EMA baseline, None until first step
        self._frozen     = False
        self.ub          = float("inf")     # upper bound on the value (log step size)

    def _draw_eps(self) -> torch.Tensor:
        """Draw a fresh ``(N,)`` perturbation ``eps ~ N(0, I)`` on
        ``prox_center``'s device and dtype."""
        pc = torch.as_tensor(self.prox_center)
        return torch.randn(self.n, device=pc.device, dtype=pc.dtype)

    def reset(self, N=None, dtype=None, device=None):
        """Reset the estimate and draw the first perturbation. With a batch size
        ``N`` given, seed ``prox_center`` at ``init`` over ``(N,)`` for the
        adapter role."""
        if N is not None:
            self.n = N
            self.prox_center = torch.full((N,), float(self._init),
                                          dtype=dtype, device=device)
        self._dual.prox_center = self.prox_center
        self._dual.reset()
        self._g   = None
        self._eps = self._draw_eps()
        self._frozen = False

    def step(self, f_t: torch.Tensor):
        """Fold the objective value ``f_t`` (``(N,)``) at the current proposal
        into the estimate and draw a fresh perturbation."""
        if self._g is None:
            self._g = f_t.clone()
        else:
            self._g = self.ema_decay * self._g + (1.0 - self.ema_decay) * f_t

        # (f_t - b_t) eps / sigma
        stat = (f_t - self._g) / self.sigma * self._eps
        self._dual.step(stat)

        self._eps = self._draw_eps()

    def get_state(self):
        """Return ``(proposal, mu)``, with ``proposal = mu_t + sigma * eps`` the
        point to evaluate next and ``mu`` the dual-averaged estimate. Both are
        ``mu`` once frozen."""
        x_t, x_avg = self._dual.get_state()
        if self._frozen:
            return x_avg, x_avg
        proposal = x_t + self.sigma * self._eps
        if self.ub != float("inf"):
            proposal = torch.clamp(proposal, max=self.ub)   # never evaluate past the cap
        return proposal, x_avg

    # ---- adapter role ------------------------------------------------------- #

    def set_upper_bound(self, ub):
        """Cap the value (proposal and dual-averaged state) at ``ub``."""
        self.ub = ub
        self._dual.set_upper_bound(ub)

    def update(self, cost):
        """Fold one ``cost`` (no-op once frozen)."""
        if not self._frozen:
            self.step(cost)

    def finalize(self):
        """Freeze the estimate: :meth:`get_state` now reports ``(x_avg, x_avg)``."""
        self._frozen = True
