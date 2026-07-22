"""Value adapters, vectorised over a leading axis so N independent problems are
adapted elementwise in parallel.

Each adapter carries a value and exposes the interface a caller drives:

    reset(N, dtype, device)   size the state to ``(N,)`` at the initial value
    update(signal)            fold one signal, move the estimate
    finalize()                freeze the estimate
    get_state() -> (x, x_avg) current value and its running average; after
                              finalize (or with no adaptation) both are x_avg

They have no notion of what the value means; the caller owns that.

``NoAdaptation`` holds the value fixed.  ``DualAveraging`` moves it toward a
target via Nesterov dual averaging from a stream of subgradients; it is also a
standalone convex minimiser.  ``Reinforce`` minimises a Gaussian-smoothed
objective from noisy evaluations alone, via a score-function estimate driving a
``DualAveraging``.
"""

# Nesterov dual averaging (Nesterov, "Primal-dual subgradient methods for
# convex problems". Hoffman & Gelman, "The No-U-Turn Sampler").

import torch


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

    def update(self, signal):
        """No-op: the value never moves."""
        pass

    def finalize(self):
        """No-op: nothing to freeze."""
        pass

    def get_state(self):
        """Return ``(value, value)`` -- constant for both entries."""
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
    ``init`` over ``(N,)``; :meth:`update` folds one subgradient; :meth:`finalize`
    freezes it, after which :meth:`get_state` reports ``(x_avg, x_avg)``.

    Parameters
    ----------
    prox_center : float or Tensor
        Reference point the primal sequence is pulled toward (minimiser use).
        Default 0.
    t0 : float
        Early-iteration stabiliser.  Default 10.
    kappa : float
        Averaging-weight exponent in (0.5, 1].  Default 0.75.
    gamma : float
        Step scale.  Default 0.05.
    init : float
        Initial value seeded over ``(N,)`` at :meth:`reset` in the adapter role;
        ignored by the minimiser, which uses ``prox_center``.  Default 0.
    """

    def __init__(self, prox_center=0.0, t0=10, kappa=0.75, gamma=0.05, init=0.0):
        self.prox_center = prox_center
        self.t0 = t0
        self.kappa = kappa
        self.gamma = gamma
        self._init = init
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
        # x_avg = (1 - t^-kappa) x_avg + t^-kappa x_t
        weight_t = self._t ** (-self.kappa)
        self._x_avg = (1 - weight_t) * self._x_avg + weight_t * self._x_t

    def get_state(self):
        """Return ``(x_t, x_avg)`` -- the latest primal point and its average,
        or ``(x_avg, x_avg)`` once frozen."""
        if self._frozen:
            return self._x_avg, self._x_avg
        return self._x_t, self._x_avg

    # ---- adapter role ------------------------------------------------------- #

    def update(self, subgradient):
        """Fold one ``subgradient`` (no-op once frozen)."""
        if not self._frozen:
            self.step(subgradient)

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
    ``mu + sigma*eps`` (the next value to try) and ``mu``; once frozen it reports
    ``(mu, mu)``.

    Parameters
    ----------
    n : int or None
        Number of problems (minimiser use); ``None`` when seeded via :meth:`reset`.
    sigma : float
        Smoothing radius and gradient-estimate denominator.  Default 0.1.
    ema_decay : float
        EMA baseline decay.  Default 0.2.
    gamma : float
        Step scale of the underlying dual averaging.  Default 0.05.
    init : float
        Initial value seeded over ``(N,)`` at :meth:`reset` in the adapter role;
        ignored by the minimiser, which uses ``prox_center``.  Default 0.
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
        """Return ``(proposal, mu)`` -- the perturbed point ``x_t + sigma*eps``
        to evaluate next and the dual-averaged estimate ``x_avg`` -- or
        ``(x_avg, x_avg)`` once frozen."""
        x_t, x_avg = self._dual.get_state()
        if self._frozen:
            return x_avg, x_avg
        return x_t + self.sigma * self._eps, x_avg

    # ---- adapter role ------------------------------------------------------- #

    def update(self, cost):
        """Fold one ``cost`` (no-op once frozen)."""
        if not self._frozen:
            self.step(cost)

    def finalize(self):
        """Freeze the estimate: :meth:`get_state` now reports ``(x_avg, x_avg)``."""
        self._frozen = True
