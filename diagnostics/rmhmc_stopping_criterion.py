"""Diagnostic harness for the RMHMC implicit-midpoint stopping criterion.

Goal
----
Test the hypothesis that the current convergence test in
``_implicit_midpoint_step`` -- a single absolute max-norm on the stacked
residual ``r = [r_q, r_p]`` (``RMHMC.py``: ``r.abs().amax(-1) < tol``) -- is
"metric-blind": it mixes the q-block (position units) and the p-block
(momentum units, scaled like sqrt(metric)) and stops when the largest
*absolute* residual drops below ``tol``.  On an anisotropic metric this gives
an effective per-coordinate *relative* tolerance that varies by the full
dynamic range of the metric spectrum, so low-sensitivity directions can be
under-resolved -> low ESS there.

What this harness does (no core code touched)
---------------------------------------------
1. Builds a small anisotropic Gaussian target whose "correct" RMHMC metric is
   the target precision (so the integrator is geometry-matched).
     (a) CONSTANT metric  -> implicit midpoint conserves H *exactly* in
         infinite precision, so any per-coordinate ESS deficit is attributable
         to the finite-tol solve alone (clean control).
     (b) POSITION-DEPENDENT metric (rank-1 update) -> trajectory is actually
         perturbed by under-resolution.
2. Re-implements the Picard solve locally (mirroring ``_implicit_midpoint_step``)
   so it can record, per coordinate and per block, the residual at the moment
   the *current* max-norm criterion declares convergence.  Also records the
   geometry-aware (energy-norm) residual contributions for comparison.
3. Runs short chains, computes per-coordinate ESS, and tabulates
   ESS vs. metric sensitivity vs. per-coordinate residual.

Run: ``python diagnostics/rmhmc_stopping_criterion.py``
"""

import sys
import os

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from muMCMC.RMHMC import _midpoint_map, _hamiltonian
from muMCMC.spaces import UnconstrainedSpace
from muMCMC.RMHMC import RMHMC

torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- #
#  Targets: anisotropic Gaussian, metric = precision                          #
# --------------------------------------------------------------------------- #

def make_lambdas(d, dynamic_range):
    """Log-spaced metric eigenvalues from 1 down to 1/dynamic_range.

    Small lambda  = low curvature = wide posterior = LOW sensitivity.
    Large lambda  = high curvature = narrow posterior = HIGH sensitivity.
    """
    return torch.logspace(0, -torch.log10(torch.tensor(float(dynamic_range))),
                          steps=d)


def model_const(lambdas):
    """U = 1/2 z^T diag(lambda) z ; metric = diag(lambda) (constant)."""
    Lam = torch.diag(lambdas)

    def model_fn(theta):
        U = 0.5 * (lambdas * theta**2).sum(-1)
        return U, Lam.expand(*theta.shape[:-1], *Lam.shape)
    return model_fn


def model_posdep(lambdas, alpha=0.1):
    """Same potential, but a genuinely position-dependent SPD metric:
    G(z) = diag(lambda) + alpha * (z z^T) elementwise-scaled by sqrt(lambda)
    so the rank-1 part stays commensurate with each coordinate's scale.
    """
    Lam = torch.diag(lambdas)
    s = lambdas.sqrt()

    def model_fn(theta):
        U = 0.5 * (lambdas * theta**2).sum(-1)
        v = (s * theta)
        rank1 = alpha * v[..., :, None] * v[..., None, :]
        G = Lam.expand(*theta.shape[:-1], *Lam.shape) + rank1
        return U, G
    return model_fn


# --------------------------------------------------------------------------- #
#  Instrumented Picard solve (mirror of _implicit_midpoint_step)              #
# --------------------------------------------------------------------------- #

def instrumented_step(q, p, eps, evaluate_model, max_iter, tol, metric_at):
    """One implicit-midpoint step, returning the endpoint AND per-coordinate
    residual diagnostics at the convergence point.

    metric_at(q_mid) -> TransformedMetric, used to weight the residual into the
    energy norm for comparison (q-block in G-norm, p-block in G^{-1}-norm).
    Single chain (N=1) for clarity of the diagnostic.
    """
    d = q.shape[-1]

    def blocks(z):
        F_q, F_p = _midpoint_map(q, p, z[..., :d], z[..., d:], eps, evaluate_model)
        r = z - torch.cat([F_q, F_p], dim=-1)
        return r

    z_k = torch.cat([q, p], dim=-1)
    r_k = blocks(z_k)
    conv_iter = max_iter
    for i in range(1, max_iter + 1):
        z_k = z_k - r_k
        r_k = blocks(z_k)
        if float(r_k.abs().amax(-1)) < tol:
            conv_iter = i
            break

    # residual at the declared convergence point
    r_q = r_k[..., :d]
    r_p = r_k[..., d:]

    q_mid = (0.5 * (q + z_k[..., :d]))
    m = metric_at(q_mid)
    # energy-norm per-coordinate contributions:
    #   ||r_q||^2_G = r_q . (G r_q),   ||r_p||^2_{G^-1} = r_p . (G^-1 r_p)
    Gr_q = m.metric_times_vec(r_q)
    Ginv_r_p = m.inv_metric_times_vec(r_p)
    e_q = (r_q * Gr_q)          # per-coord contribution to G-norm^2
    e_p = (r_p * Ginv_r_p)      # per-coord contribution to G^-1-norm^2

    return z_k[..., :d], z_k[..., d:], conv_iter, {
        "abs_r_q": r_q.abs().squeeze(0),
        "abs_r_p": r_p.abs().squeeze(0),
        "energy_q": e_q.squeeze(0),
        "energy_p": e_p.squeeze(0),
    }


# --------------------------------------------------------------------------- #
#  ESS (Geyer initial-positive-sequence, per coordinate, averaged over chains)#
# --------------------------------------------------------------------------- #

def ess_per_coordinate(samples):
    """samples: (chains, T, d). Returns (d,) ESS averaged over chains."""
    C, T, d = samples.shape
    out = torch.zeros(d)
    for c in range(C):
        x = samples[c]                          # (T, d)
        x = x - x.mean(0, keepdim=True)
        var = (x**2).mean(0)                    # (d,)
        ess = torch.zeros(d)
        for j in range(d):
            if float(var[j]) <= 0:
                ess[j] = T
                continue
            xc = x[:, j]
            # autocorrelation via direct sums up to T//2, Geyer truncation
            rho_sum = 0.0
            prev_pair = None
            t = 1
            while t < T - 1:
                rho_t = float((xc[:-t] * xc[t:]).mean() / var[j])
                rho_t1 = float((xc[:-(t + 1)] * xc[(t + 1):]).mean() / var[j])
                pair = rho_t + rho_t1
                if pair < 0:
                    break
                rho_sum += pair
                t += 2
            ess[j] = T / (1.0 + 2.0 * rho_sum)
        out += ess
    return out / C


# --------------------------------------------------------------------------- #
#  Driver: short batched run that also captures per-coordinate residuals      #
# --------------------------------------------------------------------------- #

def run_case(name, model_fn, lambdas, *, tol, num_steps, eps, n_chains,
             n_samples, n_warmup, seed=0):
    d = lambdas.numel()
    space = UnconstrainedSpace([f"x{i}" for i in range(d)])
    sampler = RMHMC(model_fn, space, step_size=eps, num_steps=num_steps,
                    adapt_step_size=False, fp_tol=tol, fp_max_iter=200)

    torch.manual_seed(seed)
    z0 = torch.zeros(n_chains, d)
    s = sampler.init(z0)
    sampler.end_warmup()  # no adaptation; freeze

    metric_at = lambda qm: sampler.evaluate_model(qm)[1]

    collected = []
    # per-coordinate residual accumulators (single chain 0, instrumented)
    res_abs_q = torch.zeros(d); res_abs_p = torch.zeros(d)
    res_e_q = torch.zeros(d);   res_e_p = torch.zeros(d)
    n_rec = 0

    total = n_warmup + n_samples
    for it in range(total):
        s = sampler.step(s)
        if it >= n_warmup:
            collected.append(s.q.clone())
            # instrument one extra leapfrog on chain 0 to read residual structure
            q0 = s.q[:1]; p0 = s.metric.select(
                torch.ones(n_chains, dtype=torch.bool), s.metric).sample_momentum()[:1]
            eps_t = sampler.step_size[:1]
            _, _, _, diag = instrumented_step(
                q0, p0, eps_t, sampler.evaluate_model, 200, tol, metric_at)
            res_abs_q += diag["abs_r_q"]; res_abs_p += diag["abs_r_p"]
            res_e_q += diag["energy_q"];  res_e_p += diag["energy_p"]
            n_rec += 1

    samples = torch.stack(collected, 0).transpose(0, 1)  # (chains, T, d)
    ess = ess_per_coordinate(samples)

    res_abs_q /= n_rec; res_abs_p /= n_rec
    res_e_q /= n_rec;   res_e_p /= n_rec

    print(f"\n=== {name}  (tol={tol:.0e}, eps={eps}, L={num_steps}) ===")
    diag = sampler.diagnostics()
    print(f"accept_rate (mean): {float(diag['accept_rate'].float().mean()):.3f}")
    print(f"{'coord':>5} {'lambda':>10} {'std(post)':>10} {'ESS':>8} {'ESS%':>6} "
          f"{'|r_q|':>10} {'|r_p|':>10} {'r_q^2_G':>10} {'r_p^2_Ginv':>11}")
    post_std = lambdas.rsqrt()
    for j in range(d):
        print(f"{j:>5} {float(lambdas[j]):>10.2e} {float(post_std[j]):>10.2e} "
              f"{float(ess[j]):>8.1f} {100*float(ess[j])/n_samples:>5.0f}% "
              f"{float(res_abs_q[j]):>10.2e} {float(res_abs_p[j]):>10.2e} "
              f"{float(res_e_q[j]):>10.2e} {float(res_e_p[j]):>11.2e}")
    return ess


if __name__ == "__main__":
    d = 5
    dynamic_range = 1e6
    lambdas = make_lambdas(d, dynamic_range)
    common = dict(num_steps=10, eps=0.3, n_chains=4,
                  n_samples=2000, n_warmup=200)

    print("Metric eigenvalues (lambda):", [f"{float(x):.1e}" for x in lambdas])
    print("Low lambda = low sensitivity = wide posterior = the suspect directions.")

    # Control: constant metric (integrator exact in infinite precision)
    run_case("CONSTANT metric, loose tol", model_const(lambdas),
             lambdas, tol=1e-4, **common)
    run_case("CONSTANT metric, tight tol", model_const(lambdas),
             lambdas, tol=1e-12, **common)

    # Position-dependent metric
    run_case("POS-DEP metric, loose tol", model_posdep(lambdas),
             lambdas, tol=1e-4, **common)
    run_case("POS-DEP metric, tight tol", model_posdep(lambdas),
             lambdas, tol=1e-12, **common)
