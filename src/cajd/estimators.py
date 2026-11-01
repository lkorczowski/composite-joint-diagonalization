"""scikit-learn compatible estimators (usable in MNE, MOABB, pyRiemann pipelines).

Inputs follow the MNE / MOABB convention: ``X`` is an array of epochs of
shape ``(n_epochs, n_channels, n_times)`` or an ``mne.Epochs`` object.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

from ._gpt import gpt, svd_init
from .stats import bootstrap_averages, covariances, cospectra


def _get_data(X):
    """Return (array, sfreq or None) from an array or an mne.Epochs."""
    if hasattr(X, "get_data") and hasattr(X, "info"):
        return np.asarray(X.get_data(), float), float(X.info["sfreq"])
    X = np.asarray(X, float)
    if X.ndim != 3:
        raise ValueError(
            f"X must have shape (n_epochs, n_channels, n_times), got {X.shape}.")
    return X, None


class CAJD(TransformerMixin, BaseEstimator):
    """Composite Approximate Joint Diagonalization for ERP source separation.

    Jointly diagonalizes a **bilinear** set (ensemble averages or epochs,
    ``X_k ~ A S_k E^T``, evoked activity) and a **linear** set (co-spectra
    or covariances, ``C_l ~ A Lambda_l A^T``, ongoing activity) with the same
    spatial unmixing B (Korczowski et al., EUSIPCO 2016). ``alpha=0`` gives
    bilinear AJD (BAJD) and ``alpha=1`` gives classic AJD.

    Parameters
    ----------
    n_components : int or None
        Number of sources P (default: n_channels).
    alpha : float in [0, 1]
        Weight of the linear term. Both sets are normalized to unit total
        energy first, so 0.5 weighs them equally.
    bilinear : {'averages', 'epochs'}
        Bilinear set: bootstrapped ensemble averages (default, as in the
        paper) or the raw epochs.
    linear : {'cospectra', 'covariances', 'cospectra+evoked'}
        Linear set: co-spectra of the epochs (paper), per-epoch
        covariances, or co-spectra plus covariances of the averages.
    sfreq : float or None
        Sampling frequency, required for co-spectra unless X is mne.Epochs.
    freqs : array-like or None
        Co-spectra frequencies (default 1-20 Hz every 0.5 Hz).
    n_averages : int or dict
        Bootstrapped averages per class.
    average_size : int or None
        Epochs per average (default: half the class, at most 50).
    init : {'svd', 'identity'}
        Initialization of B and D (paper Sec. II-D).
    max_iter, tol : GPT stopping rules.
    random_state : int, RandomState or None

    Attributes
    ----------
    filters_ : ndarray, shape (n_channels, P)
        Spatial unmixing B; sources are ``filters_.T @ X_k``.
    patterns_ : ndarray, shape (n_channels, P)
        Spatial patterns (columns = scalp maps), ``pinv(B.T)``.
    temporal_filters_ : ndarray, shape (n_times, P)
        Temporal unmixing D.
    temporal_patterns_ : ndarray, shape (P, n_times)
        Temporal patterns (rows = source waveforms), ``pinv(D)``.
    evoked_power_ : ndarray, shape (P,)
        Power of each source's bilinear back-projection on the bilinear set;
        components are sorted by decreasing ``evoked_power_``.
    n_iter_ : int
    history_ : list of dict
    """

    def __init__(self, n_components=None, alpha=0.5, bilinear="averages",
                 linear="cospectra", sfreq=None, freqs=None, n_averages=20,
                 average_size=None, init="svd", max_iter=100, tol=1e-9,
                 random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.bilinear = bilinear
        self.linear = linear
        self.sfreq = sfreq
        self.freqs = freqs
        self.n_averages = n_averages
        self.average_size = average_size
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ------------------------------------------------------------------ sets
    def _bilinear_set(self, X, y, rng):
        if self.bilinear == "averages":
            Xb, _ = bootstrap_averages(X, y, self.n_averages, self.average_size, rng)
            return Xb
        if self.bilinear == "epochs":
            return X
        raise ValueError(f"Unknown bilinear set {self.bilinear!r}.")

    def _linear_set(self, X, Xb, sfreq):
        if self.alpha == 0:
            return None
        if self.linear == "covariances":
            return covariances(X)
        if self.linear not in ("cospectra", "cospectra+evoked"):
            raise ValueError(f"Unknown linear set {self.linear!r}.")
        if sfreq is None:
            raise ValueError("sfreq is required for co-spectra.")
        C, _ = cospectra(X, sfreq, self.freqs)
        if self.linear == "cospectra+evoked":
            C = np.concatenate([C, covariances(Xb)])
        return C

    # ------------------------------------------------------------------- fit
    def fit(self, X, y=None):
        """Estimate spatial and temporal unmixing matrices from epochs."""
        X, sfreq_data = _get_data(X)
        sfreq = self.sfreq if self.sfreq is not None else sfreq_data
        rng = check_random_state(self.random_state)
        n_channels, n_times = X.shape[1:]
        P = self.n_components or n_channels

        Xb = self._bilinear_set(X, y, rng)
        C = self._linear_set(X, Xb, sfreq)
        # unit total energy per set, so that alpha is a relative weight
        Xb_n = Xb / np.linalg.norm(Xb)
        C_n = None if C is None else C / np.linalg.norm(C)

        if self.init == "svd":
            B0, D0 = svd_init(Xb, P)
        elif self.init == "identity":
            B0 = np.eye(n_channels)[:, :P]
            D0 = np.linalg.qr(rng.randn(n_times, P))[0]
        else:
            raise ValueError(f"Unknown init {self.init!r}.")

        if self.alpha == 1:
            res = gpt(X=None, C=C_n, B0=B0, max_iter=self.max_iter, tol=self.tol)
            B = res.B
            # AJD does not identify the temporal side: take the average
            # filtered waveforms as temporal patterns, so B^T Xbar D = I
            D = np.linalg.pinv(np.einsum("np,knt->pt", B, Xb) / len(Xb))
        else:
            res = gpt(X=Xb_n, C=C_n, B0=B0, D0=D0, alpha=self.alpha,
                      max_iter=self.max_iter, tol=self.tol)
            B, D = res.B, res.D

        # sort components by evoked power on the bilinear set
        patterns = np.linalg.pinv(B.T)
        tpatterns = np.linalg.pinv(D)
        amps = np.einsum("np,knt,tp->kp", B, Xb, D)
        power = np.mean(amps**2, axis=0) * np.sum(patterns**2, axis=0) * np.sum(
            tpatterns**2, axis=1)
        order = np.argsort(power)[::-1]

        self.filters_ = B[:, order]
        self.temporal_filters_ = D[:, order]
        self.patterns_ = patterns[:, order]
        self.temporal_patterns_ = tpatterns[order]
        self.evoked_power_ = power[order]
        self.n_iter_ = res.n_iter
        self.history_ = res.history
        return self

    # ------------------------------------------------------------- transform
    def transform(self, X):
        """Spatially filtered sources ``B.T @ X_k``, shape (n_epochs, P, n_times)."""
        X, _ = _get_data(X)
        return np.einsum("np,knt->kpt", self.filters_, X)

    def amplitudes(self, X):
        """Single-trial source amplitudes ``diag(B.T X_k D)``, shape (n_epochs, P)."""
        X, _ = _get_data(X)
        return np.einsum("np,knt,tp->kp", self.filters_, X, self.temporal_filters_)

    def denoise(self, X, components=None, mode="bilinear"):
        """Back-project selected sources to the sensors.

        Parameters
        ----------
        components : int, array-like or None
            Component indices (default: the first one, highest evoked power).
        mode : {'bilinear', 'spatial'}
            'bilinear' keeps only the fixed waveform scaled by the
            single-trial amplitude (paper eq. 13); 'spatial' keeps the full
            filtered time course of each source (xDAWN-like).

        Returns
        -------
        X_hat : ndarray, shape (n_epochs, n_channels, n_times)
        """
        X, _ = _get_data(X)
        comps = np.atleast_1d(0 if components is None else components)
        Ap = self.patterns_[:, comps]
        if mode == "bilinear":
            a = self.amplitudes(X)[:, comps]
            return np.einsum("nc,kc,ct->knt", Ap, a, self.temporal_patterns_[comps])
        if mode == "spatial":
            s = self.transform(X)[:, comps]
            return np.einsum("nc,kct->knt", Ap, s)
        raise ValueError(f"Unknown mode {mode!r}.")


class BAJD(CAJD):
    """Bilinear AJD: CAJD with ``alpha=0`` (only the bilinear set)."""

    def __init__(self, n_components=None, bilinear="averages", n_averages=20,
                 average_size=None, init="svd", max_iter=100, tol=1e-9,
                 random_state=None):
        super().__init__(n_components=n_components, alpha=0.0,
                         bilinear=bilinear, n_averages=n_averages,
                         average_size=average_size, init=init,
                         max_iter=max_iter, tol=tol, random_state=random_state)


class AJD(CAJD):
    """Classic non-orthogonal AJD: CAJD with ``alpha=1`` (only the linear set)."""

    def __init__(self, n_components=None, linear="cospectra+evoked", sfreq=None,
                 freqs=None, n_averages=20, average_size=None, init="svd",
                 max_iter=100, tol=1e-9, random_state=None):
        super().__init__(n_components=n_components, alpha=1.0, linear=linear,
                         sfreq=sfreq, freqs=freqs, n_averages=n_averages,
                         average_size=average_size, init=init,
                         max_iter=max_iter, tol=tol, random_state=random_state)
