# =========================================================================== #
#                                                                             #
#  The normal chart                                                           #
#                                                                             #
#  A space is a group of named variables carrying a prior. The prior is held  #
#  as a diffeomorphism of the standard normal,                                #
#                                                                             #
#      theta = T(z),   z ~ N(0, I),                                           #
#                                                                             #
#  elementwise and strictly increasing in each coordinate, and every sampler  #
#  runs in z. There are no constraints. A variable ranges over the support of #
#  its own prior, which T maps the whole line onto, so a positive variable is #
#  a log-normal rather than an unbounded one behind a transform.              #
#                                                                             #
#  Two identities follow. With log p(theta) = log phi(z) - log|det J| and     #
#  J = dtheta/dz, the temperature-free potential collapses,                   #
#                                                                             #
#      U_base = -log p(theta) - log|det J| = -log phi(z),                     #
#                                                                             #
#  so the prior leaves the inner loop. And the prior's metric in theta is     #
#  M_theta = J^-T J^-1, whose pushforward Jᵀ M_theta J is the identity. That  #
#  identity is also the exact Hessian of U_base, since the prior read in its  #
#  own chart is N(0, I) rather than something approximated by it, so a scheme #
#  needing a constant prior metric puts no condition on the prior.            #
#                                                                             #
# =========================================================================== #

from ._priors import (
    DistributionSpace,
    LogNormalSpace,
    NormalSpace,
    UnnormalizedSpace,
)
from ._space import Space
from ._tempered import TemperedAffine, TemperedMetric, broadcast_beta
from ._transform import ElementwiseMap, NormalTransform

__all__ = [
    "DistributionSpace",
    "LogNormalSpace",
    "NormalSpace",
    "Space",
    "UnnormalizedSpace",
]
