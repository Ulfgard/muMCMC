"""Tests for the parallel tempering sampler.

PT is an MCMCSampler driven by the inherited ``run_mcmc``; ``num_chains`` is the
number of independent ladders.  Statistical-recovery tests: a conjugate Gaussian
(posterior moments + evidence against the closed form) and -- the point of PT --
a well-separated bimodal target, whose target chain must show both modes because
the hot prior reference feeds them up the ladder through swaps.
"""
import math

import torch
import pytest

from muMCMC import RMHMC, PT, NormalSpace

torch.set_default_dtype(torch.float64)


def gaussian_1d(lam, mu):
    def model(theta):
        U = 0.5 * lam * (theta[..., 0] - mu) ** 2
        G = lam * torch.eye(1, dtype=theta.dtype).expand(*theta.shape[:-1], 1, 1)
        return U, G
    return model


def gaussian_1d_space():
    return NormalSpace(["x"])


def bimodal_1d(m, s):
    def model(theta):
        t = theta[..., 0]
        U = -torch.logsumexp(torch.stack(
            [-0.5 * ((t - m) / s) ** 2, -0.5 * ((t + m) / s) ** 2], dim=-1), dim=-1)
        G = torch.eye(1, dtype=theta.dtype).expand(*theta.shape[:-1], 1, 1)
        return U, G
    return model


def test_pt_swap_is_grad_free_with_grad_requiring_model():
    # A model closure holding a requires_grad parameter would, under the old
    # potential_likelihood recompute, build a retained autograd graph that
    # accumulates into _u_lik_sum. Reading U_lik from the grad-free state
    # avoids it.
    torch.manual_seed(0)
    scale = torch.tensor(3.0, requires_grad=True)

    def model(theta):
        U = 0.5 * scale * (theta[..., 0] - 2.0) ** 2
        G = scale * torch.eye(1, dtype=theta.dtype).expand(*theta.shape[:-1], 1, 1)
        return U, G

    sampler = RMHMC(model, gaussian_1d_space(), step_size=0.3, num_steps=4,
                    adapt_step_size=False)
    pt = PT(sampler, torch.linspace(0.0, 1.0, 4))
    pt.run_mcmc({n: torch.tensor(0.0) for n in pt.space.free_names}, num_samples=20, num_warmup_steps=10,
                num_chains=1, disable_progbar=True)
    assert not pt._u_lik_sum.requires_grad


def test_pt_recovers_gaussian_and_evidence():
    torch.manual_seed(0)
    lam, mu = 3.0, 2.0
    space = gaussian_1d_space()
    sampler = RMHMC(gaussian_1d(lam, mu), space, step_size=0.3, num_steps=6,
                    adapt_step_size=False)
    pt = PT(sampler, torch.linspace(0.0, 1.0, 6))

    x = pt.run_mcmc({n: torch.tensor(0.0) for n in pt.space.free_names}, num_samples=350, num_warmup_steps=100,
                    num_chains=1, disable_progbar=True)["x"]
    assert x.shape == (1, 350)

    post_mean = lam * mu / (1.0 + lam)
    post_var = 1.0 / (1.0 + lam)
    assert float(x.mean()) == pytest.approx(post_mean, abs=0.1)
    assert float(x.var()) == pytest.approx(post_var, rel=0.35)

    diag = pt.diagnostics()
    assert diag["swap_accept_rate"].shape == (5,)
    assert diag["communication_barrier"] > 0.0

    # TI evidence on a coarse linear ladder -- ballpark check
    log_Z = -0.5 * math.log(1.0 + lam) - 0.5 * lam * mu ** 2 / (1.0 + lam)
    assert float(diag["log_evidence"]) == pytest.approx(log_Z, abs=0.35)


def test_pt_multi_ladder_shapes_and_rhat():
    torch.manual_seed(0)
    space = gaussian_1d_space()
    sampler = RMHMC(gaussian_1d(3.0, 2.0), space, step_size=0.3, num_steps=4,
                    adapt_step_size=False)
    pt = PT(sampler, torch.linspace(0.0, 1.0, 4))

    out = pt.run_mcmc({n: torch.tensor(0.0) for n in pt.space.free_names}, num_samples=60, num_warmup_steps=15,
                      num_chains=3, disable_progbar=True)
    assert out["x"].shape == (3, 60)                    # three independent ladders
    assert pt.diagnostics()["swap_accept_rate"].shape == (3,)


def test_pt_recovers_bimodal_with_balanced_mass():
    torch.manual_seed(0)
    m, s = 2.5, 0.5
    space = NormalSpace(["x"], sigma=2.0)
    sampler = RMHMC(bimodal_1d(m, s), space, step_size=0.25, num_steps=6,
                    adapt_step_size=False)
    pt = PT(sampler, torch.linspace(0.0, 1.0, 6))

    x = pt.run_mcmc({n: torch.tensor(0.0) for n in pt.space.free_names}, num_samples=600, num_warmup_steps=150,
                    num_chains=1, disable_progbar=True)["x"]

    near_pos = (x - m).abs() < 1.0
    near_neg = (x + m).abs() < 1.0
    assert float(near_pos.to(torch.float64).mean()) > 0.3
    assert float(near_neg.to(torch.float64).mean()) > 0.3
    frac_pos = float((x > 0).to(torch.float64).mean())
    assert frac_pos == pytest.approx(0.5, abs=0.15)


def test_pt_reports_the_kernel_s_coordinates():
    # PT wraps a kernel that may run in coordinates of its own, so a run has to
    # come back on the variables rather than on the kernel's positions. A chart
    # that is the identity hides the difference, so this one is shifted.
    from muMCMC import ChartRATTLE, NormalSpace
    from muMCMC.ChartRATTLE import LocationScaleChart

    torch.manual_seed(0)
    m, shift = 3, 5.0
    space = NormalSpace(["v"], mu=shift, sigma=1.0)
    chart = LocationScaleChart(
        mean=lambda th: torch.tanh(th).expand(th.shape[0], m),
        cov=lambda th: (torch.exp(0.3 * th[:, 0])[:, None, None]
                        * torch.eye(m, dtype=th.dtype)),
        x=torch.randn(m))
    kernel = ChartRATTLE(chart, space, step_size=0.1, num_steps=6,
                         adapt_step_size=False)
    pt = PT(kernel, torch.tensor([0.5, 1.0]))
    out = pt.run_mcmc({"v": torch.tensor(shift)}, 60, 40, num_chains=4,
                      disable_progbar=True)
    # theta = shift + q, so draws sit near the shift. Reporting q would put them
    # near zero.
    assert abs(float(out["v"].mean()) - shift) < 1.0
