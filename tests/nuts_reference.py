"""Reference NUTS: single-chain, recursive, deliberately un-optimised.

This is a correctness oracle, not library code. It samples ``q ~ exp(-U(q))``
for a plain potential ``U = -log p`` (the same convention as the rest of
muMCMC), taking the potential as a bare callable on a ``(d,)`` tensor -- no
space, no tempering, no batch axis, no adapter. The point is to have an
implementation whose control flow maps line-for-line onto the No-U-Turn
pseudocode (Hoffman & Gelman 2014; multinomial variant of Betancourt 2017), so
the eventual batched/iterative kernel can be validated against it.

Deliberate simplifications:

* identity mass matrix (momentum ``p ~ N(0, I)``, kinetic ``½ pᵀp``);
* fixed step size (no dual averaging);
* recursion, not the iterative checkpoint stack;
* one chain, so no per-chain termination masking.

The transition uses multinomial trajectory sampling: every state on the tree
carries weight ``exp(-H)``, the proposal is drawn proportional to those weights
(uniform merges inside a subtree, biased progressive merge of each new subtree
against the running tree), and expansion stops on a generalized U-turn or a
divergence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch


# --------------------------------------------------------------------------- #
#  Leapfrog + energy (identity mass)                                          #
# --------------------------------------------------------------------------- #

def _value_and_grad(potential_fn: Callable, q: torch.Tensor):
    """Return ``(U, dU/dq)`` at ``q`` via autograd, both detached."""
    q = q.detach().requires_grad_(True)
    with torch.enable_grad():
        U = potential_fn(q)
        (grad,) = torch.autograd.grad(U, q)
    return U.detach(), grad.detach()


def _leapfrog(potential_fn: Callable, q, p, grad, eps: float):
    """One leapfrog step of size ``eps`` (identity mass, so ``dq/dt = p``).
    Returns ``(q', p', grad', U')``."""
    p = p - 0.5 * eps * grad
    q = q + eps * p
    U, grad = _value_and_grad(potential_fn, q)
    p = p - 0.5 * eps * grad
    return q, p, grad, U


def _hamiltonian(U: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``H = U + ½ pᵀp``."""
    return U + 0.5 * (p * p).sum()


def _no_uturn(q_minus, q_plus, p_minus, p_plus) -> bool:
    """Generalized (identity-mass) no-U-turn test on a subtree's endpoints:
    both ``(q⁺−q⁻)·p⁻`` and ``(q⁺−q⁻)·p⁺`` non-negative."""
    dq = q_plus - q_minus
    return bool((dq @ p_minus >= 0) and (dq @ p_plus >= 0))


# --------------------------------------------------------------------------- #
#  Recursive tree                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class _Subtree:
    """One built subtree. Endpoints ``minus``/``plus`` are its extremes in
    trajectory order; ``q_prop`` is the multinomially-drawn proposal; ``logw``
    the log of its summed weight ``Σ exp(-H)``; ``keep`` is the continue flag
    (False on U-turn or divergence within); ``diverged`` flags an energy blow-up;
    ``sum_alpha``/``n_alpha`` accumulate the Metropolis acceptance statistic for
    step-size adaptation."""
    q_minus: torch.Tensor
    p_minus: torch.Tensor
    grad_minus: torch.Tensor
    q_plus: torch.Tensor
    p_plus: torch.Tensor
    grad_plus: torch.Tensor
    q_prop: torch.Tensor
    logw: torch.Tensor
    keep: bool
    diverged: bool
    sum_alpha: float
    n_alpha: int


def _build_tree(potential_fn, q, p, grad, H0, v, depth, eps,
                max_delta_H, gen) -> _Subtree:
    """Recursively build a balanced subtree of ``2**depth`` leapfrog steps in
    direction ``v ∈ {-1, +1}`` from ``(q, p, grad)``."""
    if depth == 0:
        # Base case: a single leapfrog in direction v.
        q1, p1, grad1, U1 = _leapfrog(potential_fn, q, p, grad, v * eps)
        H1 = _hamiltonian(U1, p1)
        delta = H1 - H0
        diverged = (not torch.isfinite(H1)) or bool(delta > max_delta_H)
        logw = -H1
        alpha = 0.0 if diverged else float(torch.exp(torch.clamp(-delta, max=0.0)))
        return _Subtree(q1, p1, grad1, q1, p1, grad1, q1, logw,
                        keep=not diverged, diverged=diverged,
                        sum_alpha=alpha, n_alpha=1)

    # Recurse: first half, then (if still alive) the outer half.
    left = _build_tree(potential_fn, q, p, grad, H0, v, depth - 1, eps,
                       max_delta_H, gen)
    if not left.keep:
        return left

    if v == -1:
        outer = _build_tree(potential_fn, left.q_minus, left.p_minus,
                            left.grad_minus, H0, v, depth - 1, eps, max_delta_H, gen)
        q_minus, p_minus, grad_minus = outer.q_minus, outer.p_minus, outer.grad_minus
        q_plus, p_plus, grad_plus = left.q_plus, left.p_plus, left.grad_plus
    else:
        outer = _build_tree(potential_fn, left.q_plus, left.p_plus,
                            left.grad_plus, H0, v, depth - 1, eps, max_delta_H, gen)
        q_minus, p_minus, grad_minus = left.q_minus, left.p_minus, left.grad_minus
        q_plus, p_plus, grad_plus = outer.q_plus, outer.p_plus, outer.grad_plus

    # Merge the two halves' proposals uniformly by weight: pick the outer half
    # with probability w_outer / (w_left + w_outer).
    logw = torch.logaddexp(left.logw, outer.logw)
    if math.log(_rand(gen)) < float(outer.logw - logw):
        q_prop = outer.q_prop
    else:
        q_prop = left.q_prop

    keep = (outer.keep
            and _no_uturn(q_minus, q_plus, p_minus, p_plus))
    return _Subtree(q_minus, p_minus, grad_minus, q_plus, p_plus, grad_plus,
                    q_prop, logw, keep=keep, diverged=left.diverged or outer.diverged,
                    sum_alpha=left.sum_alpha + outer.sum_alpha,
                    n_alpha=left.n_alpha + outer.n_alpha)


# --------------------------------------------------------------------------- #
#  Transition + driver                                                        #
# --------------------------------------------------------------------------- #

def _rand(gen) -> float:
    """Scalar uniform in [0, 1) from ``gen``."""
    return float(torch.rand((), generator=gen))


@dataclass
class _StepInfo:
    tree_depth: int
    diverged: bool
    accept_stat: float


def nuts_step(potential_fn, q, eps, *, max_tree_depth, max_delta_H, gen):
    """One NUTS transition from ``q``. Returns ``(q_next, _StepInfo)``."""
    U0, grad0 = _value_and_grad(potential_fn, q)
    p0 = torch.randn(q.shape, generator=gen, dtype=q.dtype, device=q.device)
    H0 = _hamiltonian(U0, p0)

    q_minus = q_plus = q
    p_minus = p_plus = p0
    grad_minus = grad_plus = grad0

    q_sample = q
    logw = -H0                       # running tree weight includes the start state
    keep = True
    depth = 0
    diverged = False
    sum_alpha, n_alpha = 0.0, 0

    while keep and depth < max_tree_depth:
        v = -1 if _rand(gen) < 0.5 else 1
        if v == -1:
            sub = _build_tree(potential_fn, q_minus, p_minus, grad_minus, H0,
                              v, depth, eps, max_delta_H, gen)
            q_minus, p_minus, grad_minus = sub.q_minus, sub.p_minus, sub.grad_minus
        else:
            sub = _build_tree(potential_fn, q_plus, p_plus, grad_plus, H0,
                              v, depth, eps, max_delta_H, gen)
            q_plus, p_plus, grad_plus = sub.q_plus, sub.p_plus, sub.grad_plus

        sum_alpha += sub.sum_alpha
        n_alpha += sub.n_alpha
        diverged = diverged or sub.diverged

        # Biased progressive merge: accept the new subtree's proposal with
        # probability min(1, w_sub / w_running). Only from a still-valid subtree.
        if sub.keep and math.log(_rand(gen)) < min(0.0, float(sub.logw - logw)):
            q_sample = sub.q_prop
        logw = torch.logaddexp(logw, sub.logw)

        depth += 1
        keep = sub.keep and _no_uturn(q_minus, q_plus, p_minus, p_plus)

    return q_sample, _StepInfo(tree_depth=depth, diverged=diverged,
                               accept_stat=sum_alpha / max(n_alpha, 1))


def nuts_sample(potential_fn: Callable, q_init: torch.Tensor, num_samples: int, *,
                step_size: float, num_warmup: int = 0, max_tree_depth: int = 10,
                max_delta_H: float = 1000.0, seed: int = 0) -> dict:
    """Draw ``num_samples`` post-warmup samples of ``q ~ exp(-U)`` with a fixed
    step size.

    Returns a dict with ``samples`` ``(num_samples, d)`` and the per-step
    ``tree_depth`` / ``accept_stat`` ``(num_samples,)`` plus an integer
    ``num_divergences``.
    """
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
