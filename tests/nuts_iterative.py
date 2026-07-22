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
* **Leaves are streamed through an O(depth) checkpoint turn detector**
  (:class:`_TurnChecker`) rather than stored and scanned. It maintains a stack of
  completed balanced subtrees and tests each subtree's generalized U-turn the
  moment it completes -- the scheme the batched kernel will use (with the stack
  replaced by fixed level-indexed arrays). The brute-force ``_subtree_turns`` is
  retained as the oracle it is validated against.

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
    is the subtree's leaf momenta in trajectory order (minus -> plus).

    Reference (brute-force, O(2**depth) memory) turn detector. The streaming
    :class:`_TurnChecker` below reproduces its decisions in O(depth) memory and is
    validated against it; this is kept as that oracle."""
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


def _trailing_zeros(m: int) -> int:
    """Number of trailing zero bits of ``m >= 1`` (the 2-adic valuation), i.e.
    how many balanced subtrees complete when leaf ``m - 1`` is added."""
    return (m & -m).bit_length() - 1


class _TurnChecker:
    """Streaming generalized-U-turn detector -- the O(depth) checkpoint scheme.

    Leaves are fed in creation order via :meth:`add`. Completed balanced subtrees
    are kept on a stack, each as ``[p_left, p_right, rho]`` (endpoint momenta and
    summed momentum). Adding leaf ``n`` completes ``ν₂(n+1)`` subtrees -- pop that
    many, merging each with the running node and testing its U-turn -- then push
    the result. The stack holds at most ``depth + 1`` nodes, so memory is
    O(depth·d) rather than O(2**depth·d).

    ``generalized_turn`` is symmetric in its two endpoints, so feeding leaves in
    creation order (rather than spatial order) gives the same decision; the batched
    kernel will feed leaves the same way with the stack replaced by fixed
    level-indexed arrays.
    """

    def __init__(self):
        self._stack = []          # each: [p_left, p_right, rho]
        self._n = 0               # leaves added so far

    def add(self, p: torch.Tensor) -> bool:
        """Fold in one leaf momentum; return True if a completing subtree U-turns."""
        node_left, node_right, node_rho = p, p, p
        turned = False
        for _ in range(_trailing_zeros(self._n + 1)):
            left_left, _left_right, left_rho = self._stack.pop()
            node_left, node_rho = left_left, left_rho + node_rho
            if generalized_turn(node_rho, node_left, node_right):
                turned = True
        self._stack.append([node_left, node_right, node_rho])
        self._n += 1
        return turned


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
    """Build up to ``2**depth`` leaves by leapfrogging in direction ``v`` from
    ``(q, p, grad)``, stopping early on a divergence or a within-subtree U-turn.
    Turns are detected incrementally by :class:`_TurnChecker`, so no per-leaf
    history is stored."""
    n_leaves = 2 ** depth
    cur_q, cur_p, cur_grad = q, p, grad
    checker = _TurnChecker()
    far_q = far_p = far_grad = None                # frontier leaf on the extended side
    logw = None
    q_prop = None
    rho = None
    turns = False
    sum_alpha, n_alpha = 0.0, 0

    for _ in range(n_leaves):
        cur_q, cur_p, cur_grad, U = _leapfrog(potential_fn, cur_q, cur_p, cur_grad, v * eps)
        H = _hamiltonian(U, cur_p)
        delta = H - H0
        if (not torch.isfinite(H)) or bool(delta > max_delta_H):
            if far_q is None:
                far_q, far_p, far_grad = cur_q, cur_p, cur_grad
            return _Sub(far_q, far_p, far_grad, q_prop, logw, rho,
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
        far_q, far_p, far_grad = cur_q, cur_p, cur_grad

        if checker.add(cur_p):                     # a completing subtree U-turned
            turns = True
            break

    return _Sub(far_q, far_p, far_grad, q_prop, logw, rho,
                turns=turns, diverged=False,
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
