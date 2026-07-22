"""Integrator-level tests for the HMC leapfrog.

These check the two structural properties the Metropolis correctness of HMC
rests on: the leapfrog map is (with a momentum flip) an involution
(time-reversible), and it is a near-energy-conserving symplectic map so the
energy error shrinks with the step size.
"""
import torch
from pyro.distributions import Normal

from muMCMC import HMC, UnconstrainedSpace

torch.set_default_dtype(torch.float64)

NAMES = ["a", "b"]
MU = torch.tensor([1.0, -0.5])
SIGMA2 = 1.0


def _space():
    return UnconstrainedSpace(NAMES, priors={n: Normal(0.0, 1.0) for n in NAMES})


def _model(theta):
    return 0.5 * (((theta - MU) ** 2) / SIGMA2).sum(-1)


def _armed(step_size, num_steps, mass_matrix=None):
    """An HMC and its initialized state, so the integrator can be driven
    directly through ``integrate``."""
    s = HMC(_model, _space(), step_size=step_size, num_steps=num_steps,
            mass_matrix=mass_matrix, adapt_step_size=False)
    q0 = torch.tensor([[0.3, -0.2], [0.1, 0.4]])   # two chains
    return s, s.init(q0)


def _integrate(s, state, num_steps):
    for _ in range(num_steps):
        state = s.integrate(state, s.step_size)
    return state


def test_leapfrog_is_reversible():
    torch.manual_seed(0)
    dense = torch.tensor([[2.0, 0.3], [0.3, 1.0]])   # SPD mass matrix
    for mass in (None, dense):
        s, state = _armed(0.15, 12, mass_matrix=mass)
        q0 = state.q
        state.p = s._sample_momentum(q0.shape[0], q0.shape[1], q0.dtype, q0.device)
        p0 = state.p

        fwd = _integrate(s, state, s.num_steps)
        # Flip momentum and integrate the same number of steps: we must return
        # to the start with the momentum flipped back.
        fwd.p = -fwd.p
        back = _integrate(s, fwd, s.num_steps)

        assert torch.allclose(back.q, q0, atol=1e-10)
        assert torch.allclose(back.p, -p0, atol=1e-10)


def test_leapfrog_energy_error_shrinks_with_step_size():
    # Fixed trajectory length T = step_size * num_steps, refined step size:
    # a 2nd-order integrator cuts the endpoint energy error ~4x per halving.
    def max_energy_error(step_size, num_steps):
        s, state = _armed(step_size, num_steps)
        state.p = s._sample_momentum(state.q.shape[0], state.q.shape[1],
                                     state.q.dtype, state.q.device)
        H0 = state.U.value + s._kinetic(state.p)
        end = _integrate(s, state, s.num_steps)
        HL = end.U.value + s._kinetic(end.p)
        return float((HL - H0).abs().max())

    torch.manual_seed(0)
    err_big = max_energy_error(0.2, 15)      # T = 3.0
    torch.manual_seed(0)
    err_small = max_energy_error(0.1, 30)    # T = 3.0

    assert err_small < 0.4 * err_big


def test_identity_mass_kinetic_matches_half_p_squared():
    s, _ = _armed(0.1, 4)
    p = torch.tensor([[1.0, 2.0], [-0.5, 0.3]])
    assert torch.allclose(s._kinetic(p), 0.5 * (p ** 2).sum(-1))
