"""muMCMC -- batched Riemannian Manifold HMC and NUTS over constrained spaces.

The library samples in an unconstrained space (via a ``space`` object that owns
the transform, prior, and free/fixed split) while the user specifies the model
in constrained coordinates.  ``RMHMC`` is a single-threaded, GPU-batched
Riemannian Manifold HMC (implicit-midpoint integrator); ``HMC`` is the
constant-mass-matrix Euclidean sampler with an explicit leapfrog integrator;
``NUTS`` wraps Pyro's NUTS with the same constrained-space reparameterization.
All share a common ``run_mcmc`` driver and per-chain ``diagnostics`` schema.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .BaseSampler import BaseSampler, PyroSampler
from .RMHMC import RMHMC, RMHMCState
from .HMC import HMC, HMCState
from .NUTS import NUTS
from .SMC import SMC
from .PT import PT
from .spaces import (
    ElementwiseTransform,
    TemperedMetric,
    TemperedPotential,
    TemperedGradient,
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
    "HMC",
    "HMCState",
    "NUTS",
    "SMC",
    "PT",
    "ElementwiseTransform",
    "TemperedMetric",
    "TemperedPotential",
    "TemperedGradient",
    "UnconstrainedSpace",
    "UniformBoxSpace",
    "transforms",
    "DualAveraging",
    "REINFORCEAdapter",
    "__version__",
]
