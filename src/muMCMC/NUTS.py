from typing import Callable
 
from .BaseSampler import PyroSampler
import pyro
import pyro.infer.mcmc
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
 
 
class _RichDiagNUTS(pyro.infer.mcmc.NUTS):
    """Pyro NUTS kernel whose ``diagnostics()`` also returns the adapted
    post-warmup state.

    Pyro adapts a pickled copy of the kernel per worker and aggregates each
    worker's ``diagnostics()`` dict; the default reports only divergences and
    acceptance rate.  This adds:

        step_size:           final adapted step size (post warmup)
        divergences:         post-warmup step indices that diverged
        accept_cnt:          accumulated accept count post warmup
        t:                   total step count (warmup + sampling)
        warmup_steps:        warmup count
        inverse_mass_matrix: adapted IMM at end of warmup (full d-by-d if
                             full_mass else diagonal d-vector)
        inverse_mass_matrix_site_key: site-name tuple pyro stored the IMM under
    """
 
    def diagnostics(self):
        out = super().diagnostics()
        out['step_size'] = float(self.step_size)
        out['divergences_list'] = list(self._divergences)
        out['accept_cnt'] = int(self._accept_cnt)
        out['t'] = int(self._t)
        out['warmup_steps'] = int(self._warmup_steps)
        try:
            imm_dict = self._adapter.mass_matrix_adapter.inverse_mass_matrix
            if imm_dict:
                (key, imm), = imm_dict.items()
                out['inverse_mass_matrix'] = imm.detach().cpu()
                out['inverse_mass_matrix_site_key'] = key
        except (AttributeError, ValueError) as e:
            out['inverse_mass_matrix_error'] = repr(e)
        return out
 
 
class NUTS(PyroSampler):
    """
    No-U-Turn Sampler with automatic constrained-space reparameterization.
 
    Thin wrapper around Pyro's NUTS kernel with the correctly transformed
    potential, see BaseSampler.  The underlying kernel is a ``_RichDiagNUTS``,
    so the adapted step size and inverse mass matrix are available per chain
    via ``sampler.mcmc.diagnostics()``.

    Parameters
    ----------
    potential_fn : callable
        Likelihood-only potential.  Takes a vector in coordinates defined by space.
    space
        Parameter space object.
    adapt_step_size, adapt_mass_matrix, full_mass, target_accept_prob,
    jit_compile : standard NUTS knobs.
    """
 
    def __init__(
        self,
        potential_fn: Callable,
        space,
        *,
        adapt_step_size: bool = True,
        adapt_mass_matrix: bool = True,
        full_mass: bool = True,
        target_accept_prob: float = 0.8,
        jit_compile: bool = False,
    ):
        super().__init__(
            potential_fn,
            space,
            requires_metric=False,
        )
        self._kernel = _RichDiagNUTS(
            potential_fn=self._pyro_potential,
            adapt_step_size=adapt_step_size,
            adapt_mass_matrix=adapt_mass_matrix,
            full_mass=full_mass,
            target_accept_prob=target_accept_prob,
            jit_compile=jit_compile,
        )
 
    @property
    def kernel(self) -> MCMCKernel:
        return self._kernel
