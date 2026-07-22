"""muMCMC: batched HMC samplers over constrained spaces.

The library samples in an unconstrained space (via a ``space`` object that owns
the transform, prior, and free/fixed split) while the user specifies the model
in constrained coordinates.  ``RMHMC`` is Riemannian Manifold HMC with an
implicit-midpoint integrator.  ``HMC`` is the constant-mass-matrix Euclidean
sampler with an explicit leapfrog integrator.  ``LMC`` is the explicit
Lagrangian (velocity) variant of RMHMC.  ``NUTS`` wraps Pyro's NUTS with the
same constrained-space reparameterization.  All share a common ``run_mcmc``
driver and per-chain ``diagnostics`` schema.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .BaseSampler import BaseSampler, PyroSampler
from .HamiltonianSampler import HamiltonianSampler
from .RMHMC import RMHMC, RMHMCState
from .HMC import HMC, HMCState
from .LMC import LMC, LMCState
from .NUTS import NUTS
from .SMC import SMC
from .PT import PT
from .spaces import (
    ElementwiseTransform,
    TemperedAffine,
    TemperedMetric,
    UnconstrainedSpace,
    UniformBoxSpace,
    transforms,
)
from .adapters import DualAveraging, Reinforce, NoAdaptation

__all__ = [
    "BaseSampler",
    "PyroSampler",
    "HamiltonianSampler",
    "RMHMC",
    "RMHMCState",
    "HMC",
    "HMCState",
    "LMC",
    "LMCState",
    "NUTS",
    "SMC",
    "PT",
    "ElementwiseTransform",
    "TemperedAffine",
    "TemperedMetric",
    "UnconstrainedSpace",
    "UniformBoxSpace",
    "transforms",
    "DualAveraging",
    "Reinforce",
    "NoAdaptation",
    "__version__",
]
