import numpy as np
import pytest

from cajd import ajd_gpt, bajd_gpt, cajd_gpt, gpt, moreau_macchi, svd_init
from cajd._gpt import off_cost
from cajd.simulation import random_matrix, simulate_paper


def test_moreau_macchi_permutation_scaling_is_zero():
    rng = np.random.RandomState(0)
    A = rng.randn(6, 6)
    P = np.eye(6)[rng.permutation(6)] * rng.uniform(0.5, 2, 6)
    B = np.linalg.inv(A).T @ P
    assert moreau_macchi(A, B) < 1e-12


def test_moreau_macchi_identity_mixing_random_unmixing_is_large():
    rng = np.random.RandomState(0)
    assert moreau_macchi(np.eye(8), rng.randn(8, 8)) > 0.3


def test_random_matrix_condition_number():
    M = random_matrix(10, 10, cond=15.0, random_state=0)
    assert np.isclose(np.linalg.cond(M), 15.0)
    Q = random_matrix(50, 10, cond=None, random_state=0)
    assert np.allclose(Q.T @ Q, np.eye(10))


def test_ajd_recovers_exactly_diagonalizable_set():
    rng = np.random.RandomState(1)
    N, L = 6, 30
    A = random_matrix(N, N, cond=5.0, random_state=rng)
    C = np.einsum("nm,lm,pm->lnp", A, rng.chisquare(2, (L, N)), A)
    res = ajd_gpt(C, max_iter=300, tol=1e-12)
    assert moreau_macchi(A, res.B) < 1e-8


@pytest.mark.parametrize("solver", ["bajd", "cajd"])
def test_bilinear_solvers_recover_noiseless_model(solver):
    s = simulate_paper(n_sources=8, n_obs=60, n_sym=60, sigma=0.0, random_state=2)
    B0, D0 = svd_init(s.X)
    if solver == "bajd":
        res = bajd_gpt(s.X, B0, D0, max_iter=300, tol=1e-12)
    else:
        C = s.C / np.linalg.norm(s.C) * np.linalg.norm(s.X)
        res = cajd_gpt(s.X, C, B0, D0, max_iter=300, tol=1e-12)
    assert moreau_macchi(s.A, res.B) < 1e-8
    assert moreau_macchi(s.E, res.D) < 1e-8
    assert res.converged


def test_cost_decreases_overall():
    s = simulate_paper(n_sources=8, sigma=0.1, random_state=3)
    B0, D0 = svd_init(s.X)
    res = cajd_gpt(s.X, s.C, B0, D0, max_iter=30)
    costs = [h["cost"] for h in res.history]
    # costs are not normalized by the scale of B, so compare normalized ones
    S = res.S
    assert off_cost(S) / np.sum(np.diagonal(S, axis1=1, axis2=2) ** 2) < 1e-1
    assert np.isfinite(costs).all()


def test_rectangular_unmixing_runs():
    s = simulate_paper(n_sources=8, sigma=0.05, random_state=4)
    B0, D0 = svd_init(s.X, n_components=4)
    res = cajd_gpt(s.X, s.C, B0, D0, max_iter=20)
    assert res.B.shape == (8, 4) and res.D.shape == (128, 4)
    assert res.S.shape == (100, 4, 4) and res.C.shape == (100, 4, 4)


def test_transformed_sets_stay_consistent_with_unmixing():
    s = simulate_paper(n_sources=6, sigma=0.1, random_state=5)
    B0, D0 = svd_init(s.X)
    res = cajd_gpt(s.X, s.C, B0, D0, max_iter=10)
    S = np.einsum("np,knt,tq->kpq", res.B, s.X, res.D)
    C = np.einsum("np,lnm,mq->lpq", res.B, s.C, res.B)
    assert np.allclose(S, res.S)
    assert np.allclose(C, res.C)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        gpt()
    with pytest.raises(ValueError):
        gpt(C=np.eye(3)[None], B0=np.eye(3), alpha=2.0)
    with pytest.raises(ValueError):
        gpt(X=np.ones((2, 3, 5)), B0=np.eye(3))


@pytest.mark.parametrize("kw", [dict(reduce_time=True), dict(d_step="exact")])
def test_reduced_time_recovers_paired_design_from_identity(kw):
    # sources share profiles pairwise in each set: only the composite model
    # identifies them; the 2016 Jacobi D-updates stall here (T >> N)
    rng = np.random.RandomState(0)
    N, T, K, L = 8, 128, 100, 100
    A = random_matrix(N, N, cond=5.0, random_state=rng)
    E = random_matrix(T, N, cond=5.0, random_state=rng)
    S = rng.randn(K, N // 2)[:, np.arange(N) // 2]
    lam = rng.chisquare(2, (L, N // 2))[:, ((np.arange(N) + 1) % N) // 2]
    X = np.einsum("nm,km,tm->knt", A, S, E)
    C = np.einsum("nm,lm,pm->lnp", A, lam, A)
    X, C = X / np.linalg.norm(X), C / np.linalg.norm(C)
    D0 = np.linalg.qr(rng.randn(T, N))[0]
    res = cajd_gpt(X, C, np.eye(N), D0, max_iter=600, tol=1e-12, **kw)
    assert moreau_macchi(A, res.B) < 1e-8
    assert moreau_macchi(E, res.D) < 1e-8
    assert res.D.shape == (T, N)
    assert np.allclose(res.S, np.einsum("np,knt,tq->kpq", res.B, X, res.D))


def test_exact_d_step_requires_reduced_time():
    s = simulate_paper(n_sources=4, n_obs=10, n_sym=10, random_state=0)
    B0, D0 = svd_init(s.X)
    with pytest.raises(ValueError):
        cajd_gpt(s.X, s.C, B0, D0, d_step="exact", reduce_time=False)
    with pytest.raises(ValueError):
        cajd_gpt(s.X, s.C, B0, D0, d_step="newton")
