# muMCMC

A minimal library implementing HMC variants using pytorch. Samplers are batched
and can thus handle parallel chains naturally.

Currently implemented are  **Riemannian Manifold HMC** and **NUTS** samplers over constrained
parameter spaces, built on PyTorch and Pyro.

You write your model in **constrained** coordinates; the sampler works in an
unconstrained space via a `space` object that owns the transform, the prior,
and the free/fixed parameter split. `RMHMC` is a single-threaded,
GPU-batched Riemannian Manifold HMC with an implicit-midpoint integrator and
derivative-free step-size adaptation. `NUTS` wraps Pyro's NUTS with the same
constrained-space reparameterization. Both share one `run_mcmc` driver and a
common per-chain `diagnostics()` schema.

> **Status:** alpha. The API is still moving. Parallel tempering is on the
> roadmap (the batched driver and per-chain state permutation are already in
> place for it).

## Installation

From source (PyTorch and Pyro are pulled in as dependencies):

```bash
pip install git+https://github.com/Ulfgard/muMCMC.git
```

or, for development:

```bash
git clone https://github.com/Ulfgard/muMCMC.git
cd muMCMC
pip install -e ".[test]"
pytest
```

## Quickstart

### RMHMC

`RMHMC` needs a model returning the likelihood potential `U = -log p(data | theta)`
**and** a symmetric positive-definite metric `G`, both in constrained
coordinates. The prior log-prob and prior metric are added by the `space`.

```python
import torch
from muMCMC import RMHMC, UnconstrainedSpace
from pyro.distributions import Normal

torch.set_default_dtype(torch.float64)   # float64 recommended for the metric solves

names = ["x", "y"]
space = UnconstrainedSpace(names, priors={n: Normal(0.0, 1.0) for n in names})

def model(theta):
    U = 0.5 * (theta ** 2).sum(-1)                                   # -log likelihood
    G = torch.eye(2) + 0.3 * theta[..., :, None] * theta[..., None, :]  # SPD metric
    return U, G

sampler = RMHMC(model, space, step_size=0.3, num_steps=8)
samples = sampler.run_mcmc(
    torch.zeros(2), num_samples=1000, num_warmup_steps=500, num_chains=4,
)

print(samples["x"].shape)                  # (num_chains, num_samples) = (4, 1000)
print(sampler.diagnostics()["accept_rate"])  # per-chain tensor, shape (4,)
```

The implicit-midpoint step is solved per chain by a fixed-point iteration. Two
solvers are available via `solver=`: `"picard"` (default) and `"anderson"`,
which applies Anderson acceleration to the same equation and often converges in
fewer iterations — hence fewer likelihood/metric evaluations — on stiff metrics.
The endpoint is identical up to `fp_tol`, so acceptance and mixing are
unchanged; only solver cost differs. Its history length defaults to `dim(q)`
(the free-parameter dimension), whose per-iteration linear-algebra overhead is
negligible next to a single model evaluation.

`damping=` (β ∈ (0, 1], default `1.0`) under-relaxes either solver as
`(1−β)·z + β·(solver step)`. Because the implicit-midpoint iteration has a
near-imaginary eigenvalue spectrum, β < 1 can converge at step sizes where the
undamped iteration diverges, at the cost of more iterations. It changes only
stability, not the endpoint.

```python
sampler = RMHMC(model, space, step_size=0.3, num_steps=8,
                solver="anderson",            # or anderson_history=<m>
                damping=0.8)                  # β < 1 for extra stability
```

### NUTS

`NUTS` takes only the scalar likelihood potential (no metric):

```python
from muMCMC import NUTS

def logp(theta):
    return 0.5 * (theta ** 2).sum(-1)

nuts = NUTS(logp, space)
samples = nuts.run_mcmc(
    torch.zeros(2), num_samples=1000, num_warmup_steps=500, num_chains=1,
)
```

## Diagnostics

Both samplers expose a common per-chain schema as `(num_chains,)` tensors:

| key               | meaning                              |
| ----------------- | ------------------------------------ |
| `accept_rate`     | post-warmup acceptance rate          |
| `num_divergences` | post-warmup divergence count         |
| `step_size`       | final (adapted) step size            |

`RMHMC.diagnostics()` adds integrator-specific extras as running per-chain
summaries (`(num_chains,)` tensors, not full per-step history, so the
footprint is constant over a run): `delta_H_abs_mean` / `delta_H_abs_max`,
`residual_mean` / `residual_max`, and `fp_iters_mean` / `fp_iters_max`. The
full Pyro detail for `NUTS` (r-hat, n-eff, inverse mass matrix, divergence
indices) remains available via `sampler.mcmc.diagnostics()`.

## Layout

```
src/muMCMC/
    BaseSampler.py   # general base + own batched driver; PyroSampler subclass
    RMHMC.py         # Riemannian Manifold HMC (integrator + sampler)
    NUTS.py          # Pyro NUTS with constrained-space reparameterization
    spaces.py        # transforms, prior/metric pull-back, free/fixed split
    adapters.py      # dual-averaging + derivative-free (REINFORCE) optimizer
```

## License

MIT. See [LICENSE](LICENSE).
