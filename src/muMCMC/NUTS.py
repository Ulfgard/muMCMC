from typing import Callable
 
from .BaseSampler import PyroSampler
import pyro
import pyro.infer.mcmc
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
 
 
class _RichDiagNUTS(pyro.infer.mcmc.NUTS):
    """Pyro NUTS kernel with an extended diagnostics() that exposes the
    full post-warmup state.
 
    Pyro spawns one worker process per chain, pickles a copy of the
    kernel into each, and adapts that copy independently.  The parent
    process's kernel never adapts -- so reading
    sampler._kernel.step_size or .inverse_mass_matrix from the parent
    after run() returns the un-adapted initial values.
 
    The post-warmup state lives in each worker's kernel copy.  Pyro
    collects worker state by calling kernel.diagnostics() in each
    worker and aggregating the returned dicts in the parent under
    self._diagnostics[chain_id].  Default HMC.diagnostics() returns
    only divergences and acceptance rate; we override to also include
    step_size, accept counts, the warmup count, and the adapted
    inverse mass matrix.
 
    Returned dict:
        step_size:           final adapted step size (post warmup)
        divergences:         list of post-warmup step indices that diverged
        accept_cnt:          accumulated accept count post warmup
        t:                   total step count (warmup + sampling)
        warmup_steps:        warmup count
        inverse_mass_matrix: tensor (full d-by-d if full_mass=True,
                             diag d-vector otherwise) of the adapted
                             IMM at end of warmup.
        inverse_mass_matrix_site_key: the site-name tuple under which
                             pyro stored the IMM (kept for traceability).
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
    potential, see BaseSampler.
 
    The underlying pyro kernel is a _RichDiagNUTS subclass that exposes
    the adapted inverse_mass_matrix and step_size in its diagnostics()
    so that per-chain values can be read via sampler.mcmc.diagnostics()
    after run_mcmc().
 
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
