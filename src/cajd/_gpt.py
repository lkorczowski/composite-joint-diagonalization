"""Gauss Planar Transformation (GPT) solvers for AJD, BAJD and CAJD.

All three solvers minimize an off-diagonality criterion with Jacobi-like
sweeps of non-orthogonal planar updates on the columns of the unmixing
matrices (Korczowski et al., EUSIPCO 2016, Sec. II-C):

    b_i <- b_i + beta  * b_j
    d_i <- d_i + gamma * d_j

with beta and gamma the closed-form minimizers of the (i, j) and (j, i)
off-diagonal terms. Conventions:

    C : (L, N, N)   symmetric target matrices,  C_l ~ A diag A^T
    X : (K, N, T)   observation matrices,       X_k ~ A diag E^T
    B : (N, P)      spatial unmixing,   B^T C_l B and B^T X_k D ~ diagonal
    D : (T, P)      temporal unmixing
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GPTResult:
    """Output of a GPT solver.

    Attributes
    ----------
    B : ndarray, shape (N, P)
        Spatial unmixing matrix.
    D : ndarray, shape (T, P) or None
        Temporal unmixing matrix (None for AJD).
    S : ndarray, shape (K, P, P) or None
        Transformed observations ``B.T @ X_k @ D``.
    C : ndarray, shape (L, P, P) or None
        Transformed symmetric matrices ``B.T @ C_l @ B``.
    n_iter : int
        Number of sweeps performed.
    converged : bool
        Whether the tolerance was reached before ``max_iter``.
    history : list of dict
        One entry per sweep (index 0 = initial state) with ``conv`` and
        ``cost`` and, if a callback is given, its return value.
    """

    B: np.ndarray
    D: np.ndarray = None
    S: np.ndarray = None
    C: np.ndarray = None
    n_iter: int = 0
    converged: bool = False
    history: list = field(default_factory=list)


def off_cost(M):
    """Sum of squared off-diagonal entries of a stack of square matrices."""
    M = np.asarray(M)
    return float(np.sum(M**2) - np.sum(np.diagonal(M, axis1=-2, axis2=-1) ** 2))


def _normalize_columns(B, M_rows, M_cols=()):
    """Rescale the columns of B to unit norm and propagate to transformed sets.

    The GPT updates only depend on ratios, so rescaling does not change the
    path of the algorithm; it only prevents column norms from drifting.
    """
    norms = np.linalg.norm(B, axis=0)
    norms[norms == 0] = 1.0
    B /= norms
    for M in M_rows:  # B^T appears on the left: rows
        M /= norms[None, :, None]
    for M in M_cols:  # B appears on the right: columns
        M /= norms[None, None, :]
    return B


def gpt(X=None, C=None, B0=None, D0=None, alpha=0.5, max_iter=100, tol=1e-9,
        normalize=True, callback=None):
    """Composite approximate joint diagonalization by GPT.

    Minimizes

        alpha * sum_l ||off(B^T C_l B)||^2 + (1 - alpha) * sum_k ||off(B^T X_k D)||^2

    ``alpha = 1`` (or ``X=None``) gives AJD, ``alpha = 0`` (or ``C=None``)
    gives bilinear AJD (BAJD).

    Parameters
    ----------
    X : ndarray, shape (K, N, T) or None
        Observation matrices (bilinear set).
    C : ndarray, shape (L, N, N) or None
        Symmetric matrices (linear set).
    B0 : ndarray, shape (N, P)
        Initial spatial unmixing matrix. P <= N sources are estimated.
    D0 : ndarray, shape (T, P) or None
        Initial temporal unmixing matrix. Required when X is given.
    alpha : float in [0, 1]
        Weight of the linear (symmetric) term.
    max_iter : int
        Maximum number of sweeps.
    tol : float
        Stop when the RMS of the sweep's beta/gamma falls below ``tol``.
    normalize : bool
        Rescale the columns of B and D to unit norm after each sweep.
    callback : callable or None
        ``callback(B, D)`` called after each sweep; its return value is
        stored in ``history[it]['callback']``.

    Returns
    -------
    result : GPTResult
    """
    if X is None and C is None:
        raise ValueError("At least one of X or C must be given.")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
    if X is None:
        alpha = 1.0
    if C is None:
        alpha = 0.0
    use_x = X is not None and alpha < 1.0
    use_c = C is not None and alpha > 0.0

    B = np.array(B0, dtype=float, copy=True)
    P = B.shape[1]
    D = None
    S = Ct = None
    if use_x:
        if D0 is None:
            raise ValueError("D0 is required when X is used.")
        D = np.array(D0, dtype=float, copy=True)
        S = np.einsum("np,knt,tq->kpq", B, np.asarray(X, float), D)
    if use_c:
        Ct = np.einsum("np,lnm,mq->lpq", B, np.asarray(C, float), B)

    # Weights from the paper: symmetric matrices count both (i,j) and (j,i).
    wx = 1.0 - alpha
    wc = 2.0 * alpha

    def cost():
        val = 0.0
        if use_x:
            val += wx * off_cost(S)
        if use_c:
            val += alpha * off_cost(Ct)
        return val

    def record(it, conv):
        entry = {"it": it, "conv": conv, "cost": cost()}
        if callback is not None:
            entry["callback"] = callback(B, D)
        history.append(entry)

    history = []
    record(0, np.nan)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        acc = 0.0
        for i in range(P):
            for j in range(P):
                if i == j:
                    continue
                num = 0.0
                den = 0.0
                gamma = 0.0
                if use_x:
                    sjj = S[:, j, j]
                    num += wx * np.dot(S[:, i, j], sjj)
                    den_x = np.dot(sjj, sjj)
                    den += wx * den_x
                    if den_x > 0:
                        gamma = -np.dot(S[:, j, i], sjj) / den_x
                if use_c:
                    cjj = Ct[:, j, j]
                    num += wc * np.dot(Ct[:, i, j], cjj)
                    den += wc * np.dot(cjj, cjj)
                beta = -num / den if den > 0 else 0.0

                B[:, i] += beta * B[:, j]
                if use_x:
                    D[:, i] += gamma * D[:, j]
                    S[:, i, :] += beta * S[:, j, :]
                    S[:, :, i] += gamma * S[:, :, j]
                if use_c:
                    # row then column update gives the exact congruence,
                    # including C_ii + 2 beta C_ij + beta^2 C_jj
                    Ct[:, i, :] += beta * Ct[:, j, :]
                    Ct[:, :, i] += beta * Ct[:, :, j]
                acc += beta**2 + gamma**2

        if normalize:
            rows = [S] if use_x else []
            both = [Ct] if use_c else []
            _normalize_columns(B, rows + both, both)
            if use_x:
                _normalize_columns(D, [], [S])

        conv = np.sqrt(acc / max(2 * P * (P - 1), 1))
        record(it, conv)
        if conv < tol:
            converged = True
            break

    return GPTResult(B=B, D=D, S=S, C=Ct, n_iter=it, converged=converged,
                     history=history)


def ajd_gpt(C, B0=None, **kwargs):
    """Non-orthogonal AJD of symmetric matrices C by GPT (cost eq. 6)."""
    C = np.asarray(C, float)
    if B0 is None:
        B0 = np.eye(C.shape[1])
    return gpt(X=None, C=C, B0=B0, alpha=1.0, **kwargs)


def bajd_gpt(X, B0, D0, **kwargs):
    """Bilinear AJD of observation matrices X by GPT (cost eq. 7)."""
    return gpt(X=X, C=None, B0=B0, D0=D0, alpha=0.0, **kwargs)


def cajd_gpt(X, C, B0, D0, alpha=0.5, **kwargs):
    """Composite AJD of X (bilinear) and C (linear) by GPT (cost eq. 8)."""
    return gpt(X=X, C=C, B0=B0, D0=D0, alpha=alpha, **kwargs)


def svd_init(X, n_components=None):
    """Initial B0, D0 from the SVD of the average observation (paper eq. 9-10).

    Parameters
    ----------
    X : ndarray, shape (K, N, T)
    n_components : int or None
        Number of sources P (default N).

    Returns
    -------
    B0 : ndarray, shape (N, P)
    D0 : ndarray, shape (T, P)
    """
    X = np.asarray(X, float)
    N = X.shape[1]
    P = N if n_components is None else n_components
    U, _, Vt = np.linalg.svd(X.mean(axis=0), full_matrices=False)
    return U[:, :P].copy(), Vt[:P].T.copy()
