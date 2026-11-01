"""Target sets for joint diagonalization: co-spectra, covariances, averages."""

import numpy as np
from sklearn.utils import check_random_state


def cospectra(X, sfreq, freqs=None, fres=0.5, window="hann"):
    """Bartlett co-spectral matrices averaged over epochs.

    For each frequency f the co-spectrum is the real part of the
    cross-spectral matrix, averaged over epochs (Bartlett's method):
    ``C_f = mean_k Re(Z_k(f) Z_k(f)^H)``.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz.
    freqs : array-like or None
        Frequencies in Hz to keep (nearest FFT bins). Default: 1 to 20 Hz
        in steps of ``fres``, as in the paper.
    fres : float
        Frequency resolution in Hz; the FFT is zero-padded accordingly.
    window : {'hann', None}
        Taper applied to each epoch.

    Returns
    -------
    C : ndarray, shape (n_freqs, n_channels, n_channels)
    freqs : ndarray, shape (n_freqs,)
        Frequencies actually used.
    """
    X = np.asarray(X, float)
    n_times = X.shape[-1]
    if freqs is None:
        freqs = np.arange(1.0, 20.0 + fres / 2, fres)
    freqs = np.asarray(freqs, float)
    n_fft = max(int(round(sfreq / fres)), n_times)
    taper = np.hanning(n_times) if window == "hann" else np.ones(n_times)
    Xc = X - X.mean(axis=-1, keepdims=True)
    Z = np.fft.rfft(Xc * taper, n=n_fft, axis=-1)  # (K, N, F)
    fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sfreq)
    idx = np.unique([np.argmin(np.abs(fft_freqs - f)) for f in freqs])
    Zf = Z[:, :, idx]  # (K, N, F')
    C = np.einsum("knf,kmf->fnm", Zf, Zf.conj()).real / X.shape[0]
    return C, fft_freqs[idx]


def covariances(X):
    """Sample covariance matrix of each epoch, shape (n_epochs, N, N)."""
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=-1, keepdims=True)
    return np.einsum("knt,kmt->knm", Xc, Xc) / (X.shape[-1] - 1)


def bootstrap_averages(X, y=None, n_averages=20, average_size=None,
                       random_state=None):
    """Ensemble averages of random subsets of epochs, per class.

    Averaging boosts the evoked (bilinear) part relative to the ongoing EEG.
    The paper used 40 averages of 50 TARGET epochs and 10 averages of 50
    NON-TARGET epochs.

    Parameters
    ----------
    X : ndarray, shape (n_epochs, n_channels, n_times)
    y : array-like, shape (n_epochs,) or None
        Class labels. If None, all epochs form one class.
    n_averages : int or dict
        Number of averages per class, or ``{label: n}``.
    average_size : int or None
        Epochs per average. Default: half the epochs of the class, capped
        at 50.
    random_state : int, RandomState or None

    Returns
    -------
    Xbar : ndarray, shape (n_total_averages, n_channels, n_times)
    ybar : ndarray, shape (n_total_averages,)
    """
    X = np.asarray(X, float)
    rng = check_random_state(random_state)
    y = np.zeros(len(X), int) if y is None else np.asarray(y)
    Xbar, ybar = [], []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        n = n_averages.get(label, 0) if isinstance(n_averages, dict) else n_averages
        size = average_size or max(1, min(50, len(idx) // 2))
        size = min(size, len(idx))
        for _ in range(n):
            pick = rng.choice(idx, size=size, replace=False)
            Xbar.append(X[pick].mean(axis=0))
            ybar.append(label)
    return np.array(Xbar), np.array(ybar)
