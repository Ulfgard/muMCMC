"""ChartRATTLE on Neal's funnel, given a noisy observation.

Generative model, the hierarchical form ChartRATTLE targets:

    η ~ N(0, 1)                       log-scale hyperparameter
    x = e^{σ η / 2} ε,   ε ~ N(0, I_m)    noisy observation at that scale

We observe one x and want p(η | x). ChartRATTLE samples the joint (η, ε)
constrained to {φ_η(ε) = x} and reads η = θ off the constrained chain. The
funnel is a conditionally-Gaussian layer μ(η) = 0, Σ(η) = e^{σ η} I, so it is a
plain LocationScaleChart: the user supplies μ and Σ, nothing else.

η is scalar, so the exact posterior is available by 1-D quadrature and we check
the chain against it. Run:  python examples/chartrattle_funnel.py
"""
import torch

from muMCMC.ChartRATTLE import ChartRATTLE, LocationScaleChart
from muMCMC.spaces import UnconstrainedSpace

torch.set_default_dtype(torch.float64)


def main():
    torch.manual_seed(7)
    sigma, m = 3.0, 6

    # A noisy observation drawn at a true log-scale.
    eta_true = torch.randn(1)
    x = torch.exp(sigma * eta_true / 2) * torch.randn(m)

    # The model as the user writes it: batched μ(η) and Σ(η).
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

    # Fixed step (adaptation is out of scope here). Anderson solves the n = 1
    # position equation in a few iterations per substep.
    sampler = ChartRATTLE(
        constraint,
        # eta ~ N(0, 1) is the model's own prior, so the space carries it. A
        # space with no prior would give ChartRATTLE a flat one.
        UnconstrainedSpace(["log_scale"], priors={
            "log_scale": torch.distributions.Normal(torch.tensor(0.0),
                                                    torch.tensor(1.0))}),
        step_size=0.08, num_steps=12, adapt_step_size=False,
        solver="anderson", fp_tol=1e-9,
    )
    out = sampler.run_mcmc(
        torch.zeros(1), num_samples=300, num_warmup_steps=150,
        num_chains=48, disable_progbar=False,
    )
    v = out["log_scale"].reshape(-1)
    diag = sampler.diagnostics()

    print(f"\ntrue log-scale      {float(eta_true):+.4f}")
    print(f"quadrature          mean {float(mean_q):+.4f}   sd {float(sd_q):.4f}")
    print(f"ChartRATTLE ({v.numel()})   mean {float(v.mean()):+.4f}   sd {float(v.std()):.4f}")
    print(f"acceptance          {float(diag['accept_rate'].mean()):.3f}")
    print(f"divergences         {int(diag['num_divergences'].sum())}")
    print(f"solve iters / step  {float(diag['fp_iters_mean'].mean()):.2f}")

    assert abs(float(v.mean()) - float(mean_q)) < 0.03, "posterior mean off"
    assert abs(float(v.std()) - float(sd_q)) < 0.03, "posterior sd off"
    print("\nrecovered the funnel posterior within tolerance.")


if __name__ == "__main__":
    main()
