"""scikit-learn compatible estimators (usable in MNE, MOABB, pyRiemann pipelines).

Inputs follow the MNE / MOABB convention: ``X`` is an array of epochs of
shape ``(n_epochs, n_channels, n_times)`` or an ``mne.Epochs`` object.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state

from ._gpt import gpt, svd_init
from .latency import estimate_latencies, shift_epochs
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
    reduce_time : bool
        Solve the temporal unmixing D in the N-dimensional temporal subspace
        of the bilinear set instead of all T samples (default). This removes
        the T - N directions the model cannot use, converges much faster and
        is more accurate under noise; False reproduces the 2016 algorithm.
    max_shift : int
        Largest single-trial latency, in samples, corrected as in ACSTP
        (Congedo et al., 2016): trials are realigned on the leave-one-out
        average of their class, filtered by the leading components, and the
        model is refitted on the realigned epochs. 0 (default) disables it.
    latency_components : int
        Number of leading components used to estimate latencies.
    latency_classes : list or None
        Classes whose latencies are estimated (default: the largest label,
        i.e. targets). Other epochs are left as they are.
    latency_refits : int
        Alternations between latency estimation and refitting the model.
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
    latencies_ : ndarray of int, shape (n_epochs,)
        Estimated latency of each training epoch (zeros if ``max_shift=0``).
    template_ : ndarray, shape (n_channels, n_times)
        Realigned average of the latency classes, used to estimate the
        latencies of new epochs.
    """

    def __init__(self, n_components=None, alpha=0.5, bilinear="averages",
                 linear="cospectra", sfreq=None, freqs=None, n_averages=20,
                 average_size=None, init="svd", max_iter=100, tol=1e-9,
                 reduce_time=True, max_shift=0, latency_components=1,
                 latency_classes=None, latency_refits=2, random_state=None):
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
        self.reduce_time = reduce_time
        self.max_shift = max_shift
        self.latency_components = latency_components
        self.latency_classes = latency_classes
        self.latency_refits = latency_refits
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
        lat = np.zeros(len(X), int)
        self._fit_unmixing(X, y, sfreq, rng)
        if self.max_shift:
            labels = np.zeros(len(X), int) if y is None else np.asarray(y)
            classes = (self.latency_classes if self.latency_classes is not None
                       else [labels.max()])
            for _ in range(self.latency_refits):
                Bl = self.filters_[:, :self.latency_components]
                for c in classes:
                    m = labels == c
                    lat[m] = estimate_latencies(X[m], Bl, self.max_shift, lat[m])
                self._fit_unmixing(shift_epochs(X, -lat), y, sfreq, rng)
            m = np.isin(labels, classes)
            self.template_ = shift_epochs(X[m], -lat[m]).mean(axis=0)
        else:
            self.template_ = X.mean(axis=0) if y is None else X[y == np.max(y)].mean(axis=0)
        self.latencies_ = lat
        return self

    def _fit_unmixing(self, X, y, sfreq, rng):
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
                      max_iter=self.max_iter, tol=self.tol,
                      reduce_time=self.reduce_time)
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

    # ------------------------------------------------------------- transform
    def transform(self, X):
        """Spatially filtered sources ``B.T @ X_k``, shape (n_epochs, P, n_times)."""
        X, _ = _get_data(X)
        return np.einsum("np,knt->kpt", self.filters_, X)

    def estimate_latencies(self, X):
        """Latency of each epoch relative to ``template_`` (zeros if ``max_shift=0``)."""
        X, _ = _get_data(X)
        if not self.max_shift:
            return np.zeros(len(X), int)
        return estimate_latencies(X, self.filters_[:, :self.latency_components],
                                  self.max_shift, template=self.template_)

    def _latencies(self, X, latencies):
        if latencies is None:
            return self.estimate_latencies(X)
        return np.asarray(latencies, int)

    def amplitudes(self, X, latencies=None):
        """Single-trial source amplitudes ``diag(B.T X_k D)``, shape (n_epochs, P).

        With ``max_shift > 0`` the epochs are first realigned on their
        estimated (or given) latencies.
        """
        X, _ = _get_data(X)
        if self.max_shift or latencies is not None:
            X = shift_epochs(X, -self._latencies(X, latencies))
        return np.einsum("np,knt,tp->kp", self.filters_, X, self.temporal_filters_)

    def denoise(self, X, components=None, mode="bilinear", latencies=None):
        """Back-project selected sources to the sensors.

        Parameters
        ----------
        components : int, array-like or None
            Component indices (default: the first one, highest evoked power).
        mode : {'bilinear', 'spatial'}
            'bilinear' keeps only the fixed waveform scaled by the
            single-trial amplitude (paper eq. 13); 'spatial' keeps the full
            filtered time course of each source (xDAWN-like).
        latencies : array-like of int or None
            Single-trial latencies for the bilinear reconstruction, which is
            then delayed by them. Default: estimated if ``max_shift > 0``.

        Returns
        -------
        X_hat : ndarray, shape (n_epochs, n_channels, n_times)
        """
        X, _ = _get_data(X)
        comps = np.atleast_1d(0 if components is None else components)
        Ap = self.patterns_[:, comps]
        if mode == "bilinear":
            lat = (self._latencies(X, latencies) if self.max_shift or latencies is not None
                   else np.zeros(len(X), int))
            a = self.amplitudes(X, lat)[:, comps]
            Xh = np.einsum("nc,kc,ct->knt", Ap, a, self.temporal_patterns_[comps])
            return shift_epochs(Xh, lat) if lat.any() else Xh
        if mode == "spatial":
            s = self.transform(X)[:, comps]
            return np.einsum("nc,kct->knt", Ap, s)
        raise ValueError(f"Unknown mode {mode!r}.")


class BAJD(CAJD):
    """Bilinear AJD: CAJD with ``alpha=0`` (only the bilinear set)."""

    def __init__(self, n_components=None, bilinear="averages", n_averages=20,
                 average_size=None, init="svd", max_iter=100, tol=1e-9,
                 reduce_time=True, max_shift=0, latency_components=1,
                 latency_classes=None, latency_refits=2, random_state=None):
        super().__init__(n_components=n_components, alpha=0.0,
                         bilinear=bilinear, n_averages=n_averages,
                         average_size=average_size, init=init,
                         max_iter=max_iter, tol=tol, reduce_time=reduce_time,
                         max_shift=max_shift, latency_components=latency_components,
                         latency_classes=latency_classes,
                         latency_refits=latency_refits, random_state=random_state)


class AJD(CAJD):
    """Classic non-orthogonal AJD: CAJD with ``alpha=1`` (only the linear set)."""

    def __init__(self, n_components=None, linear="cospectra+evoked", sfreq=None,
                 freqs=None, n_averages=20, average_size=None, init="svd",
                 max_iter=100, tol=1e-9, max_shift=0, latency_components=1,
                 latency_classes=None, latency_refits=2, random_state=None):
        super().__init__(n_components=n_components, alpha=1.0, linear=linear,
                         sfreq=sfreq, freqs=freqs, n_averages=n_averages,
                         average_size=average_size, init=init,
                         max_iter=max_iter, tol=tol, max_shift=max_shift,
                         latency_components=latency_components,
                         latency_classes=latency_classes,
                         latency_refits=latency_refits, random_state=random_state)
