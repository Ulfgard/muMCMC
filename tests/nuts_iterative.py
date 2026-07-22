"""Reference NUTS, iterative single-chain variant.

Same target and simplifications as ``nuts_reference`` (identity mass, fixed step
size, one chain, a bare ``potential_fn``), but the recursion is replaced by the
explicit doubling loop that the batched kernel will use -- recursion cannot
vectorise across chains, so the iterative form is the real stepping stone.

Two things differ from the recursive oracle, both deliberate:

* **Generalized (momentum-sum) U-turn.** A subtree spanning momenta ``r_i`` is
  turning when ``(Σ r_i)·r_left ≤ 0`` or ``(Σ r_i)·r_right ≤ 0`` (Betancourt
  2017, generalized criterion). This is what the checkpoint/batched form needs;
  the oracle's ``(q⁺−q⁻)·r`` form is equivalent in spirit but not identical, so
  the two trace different trajectories while sampling the same target.
* **Whole subtrees are built before their U-turn is tested**, matching how the
  batched kernel advances all chains in lockstep. Turns are found by brute force
  over every balanced sub-span (the O(depth) checkpoint scheme is a later,
  separately-validated optimisation).

The multinomial trajectory sampling (uniform merge within a subtree, biased
progressive merge of each new subtree against the running tree) is identical to
the oracle, so the two must agree on the sampled distribution.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from nuts_reference import (
    _value_and_grad, _leapfrog, _hamiltonian, _rand, _StepInfo,
)


# --------------------------------------------------------------------------- #
#  Generalized U-turn                                                         #
# --------------------------------------------------------------------------- #

def generalized_turn(rho: torch.Tensor, r_left: torch.Tensor,
                     r_right: torch.Tensor) -> bool:
    """Generalized no-U-turn test: turning when the span's summed momentum
    ``rho`` opposes the momentum at either end (identity mass, so velocity =
    momentum)."""
    return bool((rho @ r_left <= 0) or (rho @ r_right <= 0))


def _subtree_turns(ps_spatial) -> bool:
    """True if any balanced (dyadic) sub-span of the subtree U-turns. ``ps_spatial``
    is the subtree's leaf momenta in trajectory order (minus -> plus)."""
    n = len(ps_spatial)
    span = 2
    while span <= n:
        for start in range(0, n, span):
            block = ps_spatial[start:start + span]
            rho = torch.stack(block).sum(0)
            if generalized_turn(rho, block[0], block[-1]):
                return True
        span *= 2
    return False


# --------------------------------------------------------------------------- #
#  Subtree build (iterative)                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class _Sub:
    """A built subtree. ``far_*`` is the outermost leaf (the new frontier on the
    extended side); ``q_prop`` / ``logw`` the multinomial proposal and log-weight;
    ``rho`` the summed leaf momenta; ``turns`` a within-subtree U-turn; ``diverged``
    an energy blow-up; ``sum_alpha`` / ``n_alpha`` the acceptance statistic."""
    far_q: torch.Tensor
    far_p: torch.Tensor
    far_grad: torch.Tensor
    q_prop: torch.Tensor
    logw: torch.Tensor
    rho: torch.Tensor
    turns: bool
    diverged: bool
    sum_alpha: float
    n_alpha: int


def _build_subtree(potential_fn, q, p, grad, H0, v, depth, eps,
                   max_delta_H, gen) -> _Sub:
    """Build ``2**depth`` leaves by leapfrogging in direction ``v`` from
    ``(q, p, grad)``, stopping early on divergence."""
    n_leaves = 2 ** depth
    cur_q, cur_p, cur_grad = q, p, grad
    created = []                                   # (q, p, grad) in creation order
    logw = None
    q_prop = None
    rho = None
    sum_alpha, n_alpha = 0.0, 0

    for _ in range(n_leaves):
        cur_q, cur_p, cur_grad, U = _leapfrog(potential_fn, cur_q, cur_p, cur_grad, v * eps)
        H = _hamiltonian(U, cur_p)
        delta = H - H0
        if (not torch.isfinite(H)) or bool(delta > max_delta_H):
            far = created[-1] if created else (cur_q, cur_p, cur_grad)
            return _Sub(far[0], far[1], far[2], q_prop, logw, rho,
                        turns=True, diverged=True,
                        sum_alpha=sum_alpha, n_alpha=n_alpha)

        lw = -H
        logw = lw if logw is None else torch.logaddexp(logw, lw)
        # Uniform (by weight) progressive proposal within the subtree.
        if q_prop is None or math.log(_rand(gen)) < float(lw - logw):
            q_prop = cur_q
        rho = cur_p.clone() if rho is None else rho + cur_p
        sum_alpha += float(torch.exp(torch.clamp(-delta, max=0.0)))
        n_alpha += 1
        created.append((cur_q, cur_p, cur_grad))

    # Spatial (minus -> plus) order: creation order for v = +1, reversed for -1.
    spatial = created if v == 1 else list(reversed(created))
    ps_spatial = [pp for (_, pp, _) in spatial]
    far = created[-1]                              # frontier leaf on the extended side
    return _Sub(far[0], far[1], far[2], q_prop, logw, rho,
                turns=_subtree_turns(ps_spatial), diverged=False,
                sum_alpha=sum_alpha, n_alpha=n_alpha)


# --------------------------------------------------------------------------- #
#  Transition + driver                                                        #
# --------------------------------------------------------------------------- #

def nuts_step(potential_fn, q, eps, *, max_tree_depth, max_delta_H, gen):
    """One iterative NUTS transition from ``q``. Returns ``(q_next, _StepInfo)``."""
    U0, grad0 = _value_and_grad(potential_fn, q)
    p0 = torch.randn(q.shape, generator=gen, dtype=q.dtype, device=q.device)
    H0 = _hamiltonian(U0, p0)

    q_minus = q_plus = q
    p_minus = p_plus = p0
    grad_minus = grad_plus = grad0

    q_sample = q
    logw = -H0
    rho_total = p0.clone()                         # summed momentum over the whole tree
    keep = True
    depth = 0
    diverged = False
    sum_alpha, n_alpha = 0.0, 0

    while keep and depth < max_tree_depth:
        v = -1 if _rand(gen) < 0.5 else 1
        if v == -1:
            sub = _build_subtree(potential_fn, q_minus, p_minus, grad_minus,
                                 H0, v, depth, eps, max_delta_H, gen)
            q_minus, p_minus, grad_minus = sub.far_q, sub.far_p, sub.far_grad
        else:
            sub = _build_subtree(potential_fn, q_plus, p_plus, grad_plus,
                                 H0, v, depth, eps, max_delta_H, gen)
            q_plus, p_plus, grad_plus = sub.far_q, sub.far_p, sub.far_grad

        sum_alpha += sub.sum_alpha
        n_alpha += sub.n_alpha

        if sub.diverged:
            diverged = True
            break

        # Biased progressive merge of the new subtree against the running tree.
        if not sub.turns and math.log(_rand(gen)) < min(0.0, float(sub.logw - logw)):
            q_sample = sub.q_prop
        logw = torch.logaddexp(logw, sub.logw)
        rho_total = rho_total + sub.rho

        depth += 1
        global_turn = generalized_turn(rho_total, p_minus, p_plus)
        keep = (not sub.turns) and (not global_turn)

    return q_sample, _StepInfo(tree_depth=depth, diverged=diverged,
                               accept_stat=sum_alpha / max(n_alpha, 1))


def nuts_sample(potential_fn, q_init, num_samples, *, step_size, num_warmup=0,
                max_tree_depth=10, max_delta_H=1000.0, seed=0) -> dict:
    """Iterative-kernel counterpart of ``nuts_reference.nuts_sample`` -- same
    return schema (``samples`` / ``tree_depth`` / ``accept_stat`` /
    ``num_divergences``)."""
    gen = torch.Generator(device=q_init.device).manual_seed(seed)
    q = q_init.clone()

    samples, depths, accepts = [], [], []
    divergences = 0
    for it in range(num_warmup + num_samples):
        q, info = nuts_step(potential_fn, q, step_size,
                            max_tree_depth=max_tree_depth,
                            max_delta_H=max_delta_H, gen=gen)
        if it >= num_warmup:
            samples.append(q.clone())
            depths.append(info.tree_depth)
            accepts.append(info.accept_stat)
            divergences += int(info.diverged)

    return {
        "samples": torch.stack(samples),
        "tree_depth": torch.tensor(depths),
        "accept_stat": torch.tensor(accepts),
        "num_divergences": divergences,
    }
