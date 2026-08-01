from typing import Callable
 
from .MCMCSampler import PyroSampler
import pyro
import pyro.infer.mcmc
from pyro.infer.mcmc.mcmc_kernel import MCMCKernel
 
 
class _RichDiagNUTS(pyro.infer.mcmc.NUTS):
    """Pyro NUTS kernel whose ``diagnostics()`` also reports the state warmup
    adapted, which Pyro keeps on the kernel and does not otherwise return.

    The keys it adds are

        step_size                     step size warmup settled on
        divergences_list              sampling step indices that diverged
        accept_cnt                    accepted transitions since warmup ended
        t                             steps taken, warmup and sampling
        warmup_steps                  warmup steps taken
        inverse_mass_matrix           the matrix warmup settled on, of shape
                                      (d, d) under full_mass and (d,) otherwise
        inverse_mass_matrix_site_key  the site name it is stored under

    The last two are absent when Pyro holds no mass matrix, in which case
    ``inverse_mass_matrix_error`` carries the exception that reported it.
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
    """No-U-Turn Sampler running on the free variables.

    The transition is Pyro's NUTS kernel, given the potential
    :class:`~muMCMC.MCMCSampler.MCMCSampler` assembles. The trajectory length
    and the mass matrix are Pyro's to choose, so this class holds no step size
    and no integrator of its own.

    Parameters
    ----------
    potential_fn : callable
        ``potential_fn(theta_full) -> U_lik``, the likelihood potential
        ``-log p(data | theta)`` on the full variable vector.
    space : Space
        Parameter space, giving the prior's normal chart and the free/fixed
        split.
    adapt_step_size : bool
        Adapt the step size during warmup toward ``target_accept_prob``.
    adapt_mass_matrix : bool
        Estimate the inverse mass matrix from the warmup draws.
    full_mass : bool
        Adapt a dense inverse mass matrix rather than a diagonal one. Unused
        when ``adapt_mass_matrix`` is False.
    target_accept_prob : float
        Target Metropolis acceptance probability for the step-size
        adaptation.
    jit_compile : bool
        Let Pyro trace the potential and its gradient with the PyTorch JIT.
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
