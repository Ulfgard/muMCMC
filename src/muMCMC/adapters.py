"""Stochastic optimizers, vectorised over a leading axis so N independent
problems are solved elementwise in parallel.

``DualAveraging`` minimises a convex objective from a stream of subgradients.
``REINFORCEAdapter`` minimises a Gaussian-smoothed objective from noisy
evaluations alone, via a score-function gradient estimate driving a
``DualAveraging``.
"""

# Nesterov dual averaging (Nesterov, "Primal-dual subgradient methods for
# convex problems". Hoffman & Gelman, "The No-U-Turn Sampler").

import torch


class DualAveraging:
    """Dual-averaging minimiser of a convex objective from a stream of
    subgradients.

    Each :meth:`step` folds a subgradient into the running mean ``g_avg`` and
    updates

        x_t   = prox_center - sqrt(t)/gamma * g_avg
        x_avg = (1 - t^-kappa) x_avg + t^-kappa x_t

    ``prox_center``, the subgradient, and the state are scalars or ``(N,)``
    tensors, updated elementwise.

    Parameters
    ----------
    prox_center : float or Tensor
        Reference point the primal sequence is pulled toward.  May be set after
        construction.  Default 0.
    t0 : float
        Early-iteration stabiliser.  Default 10.
    kappa : float
        Averaging-weight exponent in (0.5, 1].  Default 0.75.
    gamma : float
        Step scale.  Default 0.05.
    """

    def __init__(self, prox_center=0.0, t0=10, kappa=0.75, gamma=0.05):
        self.prox_center = prox_center
        self.t0 = t0
        self.kappa = kappa
        self.gamma = gamma
        self.reset()

    def reset(self):
        """Reset the iteration state to ``prox_center``."""
        self._x_avg = self.prox_center
        self._g_avg = 0.0
        self._t = 0
        self._x_t = self.prox_center

    def step(self, g):
        """Fold subgradient ``g`` (scalar or ``(N,)``) into the average and
        advance one iteration.
        """
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
        """Return ``(x_t, x_avg)``, the latest primal point and its average."""
        return self._x_t, self._x_avg


class REINFORCEAdapter:
    """Derivative-free minimiser of the Gaussian-smoothed objective

        J(mu) = E_{eps ~ N(0, I)} [ f(mu + sigma * eps) ]

    from noisy evaluations of ``f`` alone.  Each :meth:`step` forms the
    score-function estimate

        grad J ≈ (f_t - b_t) * eps_t / sigma

    with ``b_t`` an EMA baseline, and feeds it to a ``DualAveraging`` on ``mu``.
    ``f_t`` and the state are ``(N,)``, one entry per independent problem.

    Parameters
    ----------
    n : int
        Number of independent problems.
    sigma : float
        Smoothing radius and gradient-estimate denominator.  Default 0.1.
    ema_decay : float
        EMA baseline decay.  Default 0.2.
    gamma : float
        Step scale of the underlying dual averaging.  Default 0.05.
    """

    def __init__(self, n: int, sigma: float = 0.1, ema_decay: float = 0.2,
                 gamma: float = 0.05):
        self.n           = n
        self.sigma       = sigma
        self.ema_decay   = ema_decay
        self.prox_center = torch.zeros(n)   # (N,), set externally
        self._dual       = DualAveraging(gamma=gamma)
        self._g          = None             # EMA baseline, None until first step

    def _draw_eps(self) -> torch.Tensor:
        """Draw a fresh ``(N,)`` perturbation ``eps ~ N(0, I)`` on
        ``prox_center``'s device and dtype."""
        pc = torch.as_tensor(self.prox_center)
        return torch.randn(self.n, device=pc.device, dtype=pc.dtype)

    def reset(self):
        """Reset the estimate and draw the first perturbation."""
        self._dual.prox_center = self.prox_center    # (N,)
        self._dual.reset()
        self._g   = None
        self._eps = self._draw_eps()        # (N,)

    def step(self, f_t: torch.Tensor):
        """Fold the objective value ``f_t`` (``(N,)``) at the current proposal
        into the estimate and draw a fresh perturbation.
        """
        if self._g is None:
            self._g = f_t.clone()
        else:
            self._g = self.ema_decay * self._g + (1.0 - self.ema_decay) * f_t

        # (f_t - b_t) eps / sigma
        stat = (f_t - self._g) / self.sigma * self._eps
        self._dual.step(stat)

        self._eps = self._draw_eps()

    def get_state(self):
        """Return ``(proposal, mu)``, both ``(N,)``: the point ``mu + sigma*eps``
        to evaluate next, and the dual-averaged estimate of the optimum.
        """
        x_t, x_avg = self._dual.get_state()
        return (x_t + self.sigma * self._eps, x_avg)
