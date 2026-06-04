"""muMCMC -- batched Riemannian Manifold HMC and NUTS over constrained spaces.

The library samples in an unconstrained space (via a ``space`` object that owns
the transform, prior, and free/fixed split) while the user specifies the model
in constrained coordinates.  ``RMHMC`` is a single-threaded, GPU-batched
Riemannian Manifold HMC (implicit-midpoint integrator); ``NUTS`` wraps Pyro's
NUTS with the same constrained-space reparameterization.  Both share a common
``run_mcmc`` driver and per-chain ``diagnostics`` schema.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .BaseSampler import BaseSampler, PyroSampler
from .RMHMC import RMHMC, RMHMCState
from .NUTS import NUTS
from .spaces import (
    ElementwiseTransform,
    TransformedMetric,
    UnconstrainedSpace,
    UniformBoxSpace,
    transforms,
)
from .adapters import DualAveraging, REINFORCEAdapter

__all__ = [
    "BaseSampler",
    "PyroSampler",
    "RMHMC",
    "RMHMCState",
    "NUTS",
    "ElementwiseTransform",
    "TransformedMetric",
    "UnconstrainedSpace",
    "UniformBoxSpace",
    "transforms",
    "DualAveraging",
    "REINFORCEAdapter",
    "__version__",
]
