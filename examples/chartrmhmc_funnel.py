"""ChartRMHMC on Neal's funnel, given a noisy observation.

The generative model, in the hierarchical form ChartRMHMC targets:

    η ~ N(0, 1)                           log-scale hyperparameter
    x = e^{σ η / 2} ε,   ε ~ N(0, I_m)    observation at that scale

One x is observed and the target is p(η | x). ChartRMHMC samples the joint
(η, ε) on the manifold {φ_η(ε) = x} and reports η, which is the variable θ.
The funnel is a conditionally-Gaussian layer with μ(η) = 0 and
Σ(η) = e^{σ η} I, so it is a LocationScaleChart built from μ and Σ alone.

η is scalar, so the exact posterior follows from one-dimensional quadrature and
the run is checked against it. Run:  python examples/chartrmhmc_funnel.py
"""
import torch

from muMCMC.ChartRMHMC import ChartRMHMC, LocationScaleChart
from muMCMC.spaces import NormalSpace

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(7)
    sigma, m = 3.0, 6

    # A noisy observation drawn at a true log-scale.
    eta_true = torch.randn(1)
    x = torch.exp(sigma * eta_true / 2) * torch.randn(m)

    # The model is the batched μ(η) and Σ(η), and nothing else.
    mean = lambda eta: torch.zeros(eta.shape[0], m, dtype=eta.dtype)
    cov = lambda eta: torch.exp(sigma * eta[:, 0])[:, None, None] * torch.eye(m, dtype=eta.dtype)
    constraint = LocationScaleChart(mean, cov, x)

    # Exact 1-D posterior by quadrature (η scalar), for reference.
    grid = torch.linspace(-8, 8, 8001)
    log_post = -(0.5 * grid ** 2 + 0.5 * torch.exp(-sigma * grid) * (x * x).sum()
                 + 0.5 * m * sigma * grid)
    w = torch.softmax(log_post, dim=0)
    mean_q = (w * grid).sum()
    sd_q = (w * (grid - mean_q) ** 2).sum().sqrt()

    # A fixed step size, so the run shows the integrator alone. Anderson
    # solves the n = 1 position equation in a few iterations per substep.
    sampler = ChartRMHMC(
        constraint,
        # eta ~ N(0, 1) is the model's own prior, so the space carries it, and
        # its normal chart is where the constant prior block M = I comes from.
        NormalSpace(["log_scale"], mu=0.0, sigma=1.0),
        step_size=0.08, num_steps=12, adapt_step_size=False,
        solver="anderson", fp_tol=1e-9,
    )
    out = sampler.run_mcmc(
        {"log_scale": torch.tensor(0.0)},
        num_samples=300, num_warmup_steps=150,
        num_chains=48, disable_progbar=False,
    )
    v = out["log_scale"].reshape(-1)
    diag = sampler.diagnostics()

    print(f"\ntrue log-scale      {float(eta_true):+.4f}")
    print(f"quadrature          mean {float(mean_q):+.4f}   sd {float(sd_q):.4f}")
    print(f"ChartRMHMC ({v.numel()})   mean {float(v.mean()):+.4f}   sd {float(v.std()):.4f}")
    print(f"acceptance          {float(diag['accept_rate'].mean()):.3f}")
    print(f"divergences         {int(diag['num_divergences'].sum())}")
    print(f"solve iters / step  {float(diag['fp_iters_mean'].mean()):.2f}")

    assert abs(float(v.mean()) - float(mean_q)) < 0.03, "posterior mean off"
    assert abs(float(v.std()) - float(sd_q)) < 0.03, "posterior sd off"
    print("\nrecovered the funnel posterior within tolerance.")


if __name__ == "__main__":
    main()
