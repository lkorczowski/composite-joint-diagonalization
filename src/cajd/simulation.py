"""Synthetic data generators.

``simulate_paper`` reproduces the matrix-level simulation of the EUSIPCO 2016
paper (Sec. III-A). ``simulate_erp_eeg`` generates more realistic EEG
epochs: ERP sources with a fixed waveform (bilinear part) embedded in
ongoing oscillatory and 1/f background sources (linear part), all mixed by
the same spatial matrix A.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter
from sklearn.utils import check_random_state


def random_matrix(n_rows, n_cols, cond=None, random_state=None):
    """Random matrix with a prescribed condition number.

    Parameters
    ----------
    n_rows, n_cols : int
    cond : None, float or (low, high)
        None gives orthonormal columns. A float sets the condition number;
        a tuple draws it uniformly in ``[low, high)``.
    """
    rng = check_random_state(random_state)
    k = min(n_rows, n_cols)
    U, _ = np.linalg.qr(rng.randn(n_rows, k))
    V, _ = np.linalg.qr(rng.randn(n_cols, k))
    if cond is None:
        return U @ V.T
    c = rng.uniform(*cond) if np.ndim(cond) else float(cond)
    s = np.exp(rng.uniform(0, np.log(c), size=k))
    s[0], s[-1] = 1.0, c  # pin the extremes so cond == c exactly
    return U @ np.diag(s) @ V.T


@dataclass
class PaperSimulation:
    X: np.ndarray  # (K, N, T)
    C: np.ndarray  # (L, N, N)
    A: np.ndarray  # (N, N)
    E: np.ndarray  # (T, N)


def simulate_paper(n_sources=16, n_times=128, n_obs=100, n_sym=100,
                   cond_A=(1, 20), cond_E=(1, 20), sigma=0.1,
                   random_state=None):
    """Matrix-level simulation of the paper (eq. 12 and the R_x(l) model).

    ``X_k = A S_k E^T + sigma N_k`` with S_k diagonal N(0, 1), and
    ``R_l = A Lambda_l A^T + sigma^2 / 2 (N_l + N_l^T)`` with Lambda_l diagonal
    chi-squared(2). Condition numbers are set exactly by construction (the
    paper used rejection sampling).
    """
    rng = check_random_state(random_state)
    N, T = n_sources, n_times
    A = random_matrix(N, N, cond_A, rng)
    E = random_matrix(T, N, cond_E, rng)
    S = rng.randn(n_obs, N)
    X = np.einsum("nm,km,tm->knt", A, S, E) + sigma * rng.randn(n_obs, N, T)
    lam = rng.chisquare(2, size=(n_sym, N))
    noise = rng.randn(n_sym, N, N)
    C = np.einsum("nm,lm,pm->lnp", A, lam, A) + sigma**2 / 2 * (
        noise + noise.transpose(0, 2, 1))
    return PaperSimulation(X=X, C=C, A=A, E=E)


@dataclass
class SimulatedERP:
    """Simulated EEG epochs with known evoked ground truth.

    Attributes
    ----------
    X : (n_epochs, n_channels, n_times) observed epochs
    y : (n_epochs,) 1 = target (ERP present), 0 = non-target
    evoked : (n_epochs, n_channels, n_times) true evoked part of each epoch
    A : (n_channels, n_channels) spatial mixing (first ``n_erp`` columns are ERP sources)
    waveforms : (n_erp, n_times) ERP source waveforms
    amplitudes : (n_epochs, n_erp) single-trial ERP amplitudes
    latencies : (n_epochs,) single-trial latency shift in samples
    sfreq : sampling frequency
    """

    X: np.ndarray
    y: np.ndarray
    evoked: np.ndarray
    A: np.ndarray
    waveforms: np.ndarray
    amplitudes: np.ndarray
    latencies: np.ndarray
    sfreq: float
    n_erp: int


def erp_waveform(times, rng, n_peaks=None):
    """Random ERP-like waveform: sum of Gaussian deflections, unit peak."""
    n_peaks = n_peaks or rng.randint(1, 4)
    w = np.zeros_like(times)
    for _ in range(n_peaks):
        lat = rng.uniform(0.1, 0.6)
        width = rng.uniform(0.02, 0.08)
        w += rng.choice([-1, 1]) * rng.uniform(0.5, 1) * np.exp(
            -0.5 * ((times - lat) / width) ** 2)
    return w / np.max(np.abs(w))


def background_source(n_samples, sfreq, rng, kind):
    """Unit-variance ongoing source: 'oscillation' (AR(2) resonator) or 'pink'."""
    white = rng.randn(n_samples)
    if kind == "oscillation":
        f0 = rng.choice([rng.uniform(4, 8), rng.uniform(8, 13), rng.uniform(13, 30)])
        r = rng.uniform(0.95, 0.99)
        a = [1, -2 * r * np.cos(2 * np.pi * f0 / sfreq), r**2]
        x = lfilter([1], a, white)
    else:  # 1/f noise by spectral shaping
        F = np.fft.rfft(white)
        f = np.fft.rfftfreq(n_samples, 1 / sfreq)
        f[0] = f[1]
        x = np.fft.irfft(F / np.sqrt(f), n=n_samples)
    x -= x.mean()
    return x / x.std()


def simulate_erp_eeg(n_channels=16, n_times=128, n_epochs=200, sfreq=128.0,
                     n_erp=2, snr_db=-10.0, target_ratio=0.5, amplitude_sd=0.3,
                     jitter_ms=0.0, cond_A=(1, 20), sensor_noise=0.05,
                     shared_amplitude=False, background="mixed",
                     random_state=None):
    """Simulate P300-like epochs: ERP sources in ongoing EEG, shared mixing A.

    Each epoch is ``X_k = A_erp diag(a_k) W + A_bg Z_k + noise`` where W holds
    ``n_erp`` fixed waveforms (bilinear part, present on target epochs only)
    and Z_k are ``n_channels - n_erp`` ongoing background sources (alpha /
    theta / beta oscillations and 1/f noise) cut from continuous signals
    (linear part).

    Parameters
    ----------
    snr_db : float
        Input SNR on target epochs: power of the evoked part over power of
        the background (including sensor noise), in dB.
    amplitude_sd : float
        Standard deviation of single-trial amplitudes around 1.
    jitter_ms : float
        Standard deviation of single-trial latency shifts (breaks the
        bilinear model when > 0).
    sensor_noise : float
        Standard deviation of white sensor noise relative to the background.
    shared_amplitude : bool
        If True, all ERP sources share the same single-trial amplitude, so
        the bilinear model alone cannot separate them (only their subspace).
    background : {'mixed', 'oscillation', 'pink'}
        Kind of background sources: alternating oscillations and 1/f noise,
        or only one kind. Sources of the same 1/f kind share a spectrum, so
        co-spectra cannot separate them.
    """
    rng = check_random_state(random_state)
    N, T, K = n_channels, n_times, n_epochs
    times = np.arange(T) / sfreq
    A = random_matrix(N, N, cond_A, rng)
    A /= np.linalg.norm(A, axis=0)

    W = np.array([erp_waveform(times, rng) for _ in range(n_erp)])
    y = (rng.rand(K) < target_ratio).astype(int)
    amps = 1 + amplitude_sd * rng.randn(K, 1 if shared_amplitude else n_erp)
    amps = np.broadcast_to(amps, (K, n_erp)) * y[:, None]
    lat = np.round(rng.randn(K) * jitter_ms * sfreq / 1000).astype(int)

    src = np.empty((K, n_erp, T))
    for k in range(K):
        src[k] = amps[k][:, None] * np.roll(W, lat[k], axis=1)
    evoked = np.einsum("ne,ket->knt", A[:, :n_erp], src)

    n_bg = N - n_erp
    if background == "mixed":
        kinds = ["oscillation" if i % 2 == 0 else "pink" for i in range(n_bg)]
    elif background in ("oscillation", "pink"):
        kinds = [background] * n_bg
    else:
        raise ValueError(f"Unknown background {background!r}.")
    cont = np.array([background_source(K * T, sfreq, rng, kd) for kd in kinds])
    Z = cont.reshape(n_bg, K, T).transpose(1, 0, 2)
    background = np.einsum("nb,kbt->knt", A[:, n_erp:], Z)
    background += sensor_noise * rng.randn(K, N, T)

    p_bg = np.mean(background**2)
    p_ev = np.mean(evoked[y == 1] ** 2) if y.any() else 1.0
    gain = np.sqrt(p_bg / p_ev * 10 ** (snr_db / 10))
    evoked *= gain
    amps = amps * gain
    return SimulatedERP(X=evoked + background, y=y, evoked=evoked, A=A,
                        waveforms=W, amplitudes=amps, latencies=lat,
                        sfreq=sfreq, n_erp=n_erp)
