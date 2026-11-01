"""Separation and ERP-estimation quality metrics."""

import numpy as np


def moreau_macchi(A, B):
    """Moreau-Macchi index of ``H = B.T @ A`` (paper eq. 11).

    0 means perfect separation up to permutation and scaling; values near 1
    mean no separation.

    Parameters
    ----------
    A : ndarray, shape (N, N)
        True mixing matrix (or true temporal mixing E).
    B : ndarray, shape (N, N)
        Estimated unmixing matrix (or estimated temporal unmixing D).
    """
    H = np.abs(np.asarray(B).T @ np.asarray(A))
    n = H.shape[0]
    rows = np.sum(H.sum(axis=1) / H.max(axis=1) - 1)
    cols = np.sum(H.sum(axis=0) / H.max(axis=0) - 1)
    return float((rows + cols) / (2 * n * (n - 1)))


def snr_db(estimate, truth, axis=None):
    """Output SNR in dB: ``10 log10(||truth||^2 / ||estimate - truth||^2)``."""
    estimate = np.asarray(estimate, float)
    truth = np.asarray(truth, float)
    sig = np.sum(truth**2, axis=axis)
    err = np.sum((estimate - truth) ** 2, axis=axis)
    return 10 * np.log10(sig / err)


def correlation(estimate, truth, axis=-1):
    """Pearson correlation along ``axis`` (e.g. per channel waveform)."""
    a = np.asarray(estimate, float)
    b = np.asarray(truth, float)
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    num = np.sum(a * b, axis=axis)
    den = np.sqrt(np.sum(a**2, axis=axis) * np.sum(b**2, axis=axis))
    return num / den
