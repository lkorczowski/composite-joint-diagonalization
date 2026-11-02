"""Single-trial latency estimation (Congedo, Korczowski et al., 2016, ACSTP).

Each trial is compared with the leave-one-out average of its class, both
spatially filtered; its latency is the shift that maximizes their
cross-correlation, taking among the peaks above 66 % of the maximum the one
closest to zero. Averages are recomputed from the realigned trials and the
estimation is repeated until the latencies stop changing.
"""

import numpy as np


def shift_epochs(X, shifts):
    """Delay each epoch by ``shifts[k]`` samples (zero padded).

    ``out[k, :, t] = X[k, :, t - shifts[k]]``; a negative shift advances.
    """
    X = np.asarray(X, float)
    out = np.zeros_like(X)
    T = X.shape[-1]
    for k, s in enumerate(np.asarray(shifts, int)):
        if s >= 0:
            out[k, ..., s:] = X[k, ..., :T - s]
        else:
            out[k, ..., :T + s] = X[k, ..., -s:]
    return out


def _xcorr(template, Y, max_shift):
    """c[k, j] = <template, Y_k advanced by s_j> for s_j in [-max_shift, max_shift]."""
    T = Y.shape[-1]
    c = np.empty((len(Y), 2 * max_shift + 1))
    for j, s in enumerate(range(-max_shift, max_shift + 1)):
        # Y_k advanced by s: Y_k[t + s] aligned with template[t]
        if s >= 0:
            c[:, j] = np.einsum("pt,kpt->k", template[..., :T - s], Y[..., s:])
        else:
            c[:, j] = np.einsum("pt,kpt->k", template[..., -s:], Y[..., :T + s])
    return c


def _pick_peak(c, max_shift, ratio=0.66):
    """Shift of the peak closest to zero among those above ``ratio`` of the best."""
    c = c - c.min()
    peaks = [j for j in range(1, len(c) - 1) if c[j] >= c[j - 1] and c[j] > c[j + 1]]
    if not peaks:
        return 0
    peaks = np.array(peaks)
    peaks = peaks[c[peaks] >= ratio * c[peaks].max()]
    return int(peaks[np.argmin(np.abs(peaks - max_shift))] - max_shift)


def estimate_latencies(X, filters, max_shift, latencies=None, template=None,
                       max_iter=10):
    """Estimate single-trial latencies of one class of epochs.

    Parameters
    ----------
    X : ndarray, shape (K, N, T)
        Epochs of a single class, as recorded.
    filters : ndarray, shape (N, P)
        Spatial filters applied before correlating.
    max_shift : int
        Largest latency searched, in samples.
    latencies : ndarray, shape (K,) or None
        Initial latencies (default zero).
    template : ndarray, shape (N, T) or None
        Fixed reference waveform (e.g. a training average). If None, each
        trial is compared with the leave-one-out average of the realigned
        trials and the procedure is iterated (ACSTP).
    max_iter : int
        Maximum number of realignments when ``template`` is None.

    Returns
    -------
    latencies : ndarray of int, shape (K,)
        Estimated delay of each trial: ``shift_epochs(X, -latencies)`` aligns
        them.
    """
    K = len(X)
    lat = np.zeros(K, int) if latencies is None else np.asarray(latencies, int).copy()
    Y = np.einsum("np,knt->kpt", filters, X)
    if template is not None:
        c = _xcorr(filters.T @ template, Y, max_shift)
        return np.array([_pick_peak(ck, max_shift) for ck in c])
    for _ in range(max_iter):
        Ya = shift_epochs(Y, -lat)
        total = Ya.sum(axis=0)
        new = np.empty(K, int)
        for k in range(K):
            loo = (total - Ya[k]) / max(K - 1, 1)
            new[k] = _pick_peak(_xcorr(loo, Y[k:k + 1], max_shift)[0], max_shift)
        change = np.mean(np.abs(new - lat))
        lat = new
        # ACSTP stopping rule: on average less than max_shift / K samples moved
        if change < max_shift / K:
            break
    return lat
