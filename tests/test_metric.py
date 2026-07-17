"""Contract tests for the metric pipeline: ``space.push_forward_metric`` and the
tempering-aware evaluation objects ``TemperedMetric`` / ``TemperedPotential``.

``push_forward_metric`` restricts a constrained-space metric to the free block
and scales by the diagonal Jacobian ``dθ/dz`` (elementwise transforms, so the
free block of the push-forward is the push-forward of the free block).
``TemperedMetric`` and ``TemperedPotential`` assemble the metric and potential
affinely in an inverse temperature; ``beta`` is slot-bound, so a moved
configuration is retempered to its slot's temperature by ``reorder``/``select``
alone.
"""
import torch

from muMCMC.spaces import (
    TemperedMetric,
    TemperedPotential,
    UnconstrainedSpace,
    UniformBoxSpace,
)

torch.set_default_dtype(torch.float64)

ATOL = 1e-9


def _rand_spd(n, d):
    A = torch.randn(n, d, d)
    return A @ A.transpose(-2, -1) + d * torch.eye(d)


def _matvec(M, v):
    return (M @ v[..., None])[..., 0]


def _identity_space(d):
    return UnconstrainedSpace([f"x{i}" for i in range(d)])


# --------------------------------------------------------------------------- #
#  push_forward_metric                                                        #
# --------------------------------------------------------------------------- #

def test_push_forward_identity_is_free_block():
    torch.manual_seed(0)
    n, d = 4, 3
    s = _identity_space(d)
    G = _rand_spd(n, d)
    z = torch.randn(n, d)
    A = s.push_forward_metric(G, s.map_to_constrained_vector(z))
    assert torch.allclose(A, G, atol=ATOL)          # identity J, no fixed


def test_push_forward_projects_out_fixed():
    torch.manual_seed(1)
    n = 4
    s = UnconstrainedSpace(["a", "b", "c"], fixed={"c": 0.0})   # free = a, b
    G = _rand_spd(n, 3)
    z_free = torch.randn(n, 2)
    A = s.push_forward_metric(G, s.map_to_constrained_vector(z_free))
    assert torch.allclose(A, G[:, :2, :2], atol=ATOL)


def test_push_forward_box_scales_by_jacobian():
    torch.manual_seed(2)
    n, d = 4, 2
    s = UniformBoxSpace({"x": (-1.0, 1.0), "y": (0.0, 4.0)}, ["x", "y"], device="cpu")
    G = _rand_spd(n, d)
    z = torch.randn(n, d)
    theta_map = s.map_to_constrained_vector(z)
    A = s.push_forward_metric(G, theta_map)
    dJ = theta_map.jacobian_diag
    assert torch.allclose(A, dJ[..., :, None] * G * dJ[..., None, :], atol=ATOL)


# --------------------------------------------------------------------------- #
#  TemperedMetric: matrix ops vs. dense reference                             #
# --------------------------------------------------------------------------- #

def test_metric_ops_match_dense():
    torch.manual_seed(3)
    n, d = 4, 3
    A_lik, A_prior = _rand_spd(n, d), _rand_spd(n, d)
    beta = torch.tensor([0.0, 0.3, 0.7, 1.0])
    m = TemperedMetric(A_lik, A_prior, beta)

    G = beta.reshape(-1, 1, 1) * A_lik + A_prior
    v = torch.randn(n, d)
    assert torch.allclose(m.inv_metric_times_vec(v), _matvec(torch.linalg.inv(G), v), atol=ATOL)
    assert torch.allclose(m.log_det_metric(), torch.logdet(G), atol=ATOL)


def test_metric_no_prior():
    torch.manual_seed(4)
    n, d = 3, 2
    A_lik = _rand_spd(n, d)
    m = TemperedMetric(A_lik, None, 1.0)
    v = torch.randn(n, d)
    assert torch.allclose(m.inv_metric_times_vec(v), _matvec(torch.linalg.inv(A_lik), v), atol=ATOL)


def test_sample_momentum_covariance():
    torch.manual_seed(5)
    d = 2
    A_lik = _rand_spd(1, d)
    m = TemperedMetric(A_lik, None, 1.0)
    draws = torch.stack([m.sample_momentum()[0] for _ in range(40000)])
    cov = torch.cov(draws.T)
    assert not draws.requires_grad
    assert torch.allclose(cov, A_lik[0], atol=0.1)


# --------------------------------------------------------------------------- #
#  Retempering: beta slot-bound under reorder / select                        #
# --------------------------------------------------------------------------- #

def test_metric_reorder_retempers():
    torch.manual_seed(6)
    n, d = 4, 2
    A_lik, A_prior = _rand_spd(n, d), _rand_spd(n, d)
    beta = torch.tensor([0.0, 0.3, 0.7, 1.0])
    m = TemperedMetric(A_lik, A_prior, beta)

    perm = torch.tensor([3, 1, 0, 2])
    r = m.reorder(perm)
    # beta stays; the A pieces move -> slot i factorizes beta[i]*A_lik[perm[i]] + A_prior[perm[i]]
    G_ref = beta.reshape(-1, 1, 1) * A_lik[perm] + A_prior[perm]
    assert torch.allclose(r.L, torch.linalg.cholesky(G_ref), atol=ATOL)


def test_metric_select_keeps_temperature():
    torch.manual_seed(7)
    n, d = 3, 2
    beta = torch.tensor([0.2, 0.5, 0.9])
    a = TemperedMetric(_rand_spd(n, d), _rand_spd(n, d), beta)
    b = TemperedMetric(_rand_spd(n, d), _rand_spd(n, d), beta)
    mask = torch.tensor([True, False, True])
    c = a.select(mask, b)
    v = torch.randn(n, d)
    ref = torch.where(mask[:, None], a.inv_metric_times_vec(v), b.inv_metric_times_vec(v))
    assert torch.allclose(c.inv_metric_times_vec(v), ref, atol=ATOL)


# --------------------------------------------------------------------------- #
#  TemperedPotential                                                          #
# --------------------------------------------------------------------------- #

def test_potential_value():
    torch.manual_seed(8)
    U_lik, U_base = torch.randn(4), torch.randn(4)
    beta = torch.tensor([0.0, 0.3, 0.7, 1.0])
    assert torch.allclose(TemperedPotential(U_lik, U_base, beta).value, beta * U_lik + U_base)


def test_potential_reorder_retempers():
    torch.manual_seed(9)
    U_lik, U_base = torch.randn(4), torch.randn(4)
    beta = torch.tensor([0.0, 0.3, 0.7, 1.0])
    pot = TemperedPotential(U_lik, U_base, beta)
    perm = torch.tensor([2, 0, 3, 1])
    r = pot.reorder(perm)
    assert torch.allclose(r.value, beta * U_lik[perm] + U_base[perm])
    assert r.beta is beta


def test_potential_select_shares_temperature():
    torch.manual_seed(10)
    beta = torch.tensor([0.2, 0.5, 0.9])
    a = TemperedPotential(torch.randn(3), torch.randn(3), beta)
    b = TemperedPotential(torch.randn(3), torch.randn(3), beta)
    mask = torch.tensor([True, False, True])
    c = a.select(mask, b)
    assert torch.allclose(c.value, torch.where(mask, a.value, b.value))


# --------------------------------------------------------------------------- #
#  evaluate_model assembles both objects                                      #
# --------------------------------------------------------------------------- #

def test_evaluate_model_assembles_potential_and_metric():
    torch.manual_seed(11)
    from muMCMC.RMHMC import RMHMC

    def model(theta):
        U = 0.5 * (theta ** 2).sum(-1)
        n = theta.shape[-1]
        G = torch.eye(n, dtype=theta.dtype) + 0.2 * theta[..., :, None] * theta[..., None, :]
        return U, G

    space = UnconstrainedSpace(
        ["x0", "x1", "x2"],
        prior_metric_fn=lambda theta: torch.eye(3, dtype=theta.dtype).expand(
            *theta.shape[:-1], 3, 3),
    )
    s = RMHMC(model, space, adapt_step_size=False)
    z = torch.randn(5, 3)
    beta = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])

    potential, metric = s.evaluate_model(z, beta)
    # no prior log-prob and identity Jacobian, so U_base = 0 and U = beta * U_lik.
    U_lik = 0.5 * (z ** 2).sum(-1)
    assert torch.allclose(potential.value, beta * U_lik, atol=ATOL)
    # metric at beta: G_u = beta*(I + theta theta^T) + I
    v = torch.randn(5, 3)
    G = beta.reshape(-1, 1, 1) * (torch.eye(3) + 0.2 * z[:, :, None] * z[:, None, :]) + torch.eye(3)
    assert torch.allclose(metric.inv_metric_times_vec(v), _matvec(torch.linalg.inv(G), v), atol=ATOL)
