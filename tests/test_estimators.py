import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from cajd import AJD, BAJD, CAJD, bootstrap_averages, cospectra, covariances
from cajd.metrics import correlation, snr_db
from cajd.simulation import simulate_erp_eeg


@pytest.fixture(scope="module")
def sim():
    return simulate_erp_eeg(n_channels=8, n_epochs=300, snr_db=-5, random_state=0)


def test_cospectra_shape_and_symmetry(sim):
    C, freqs = cospectra(sim.X, sim.sfreq)
    assert C.shape == (len(freqs), 8, 8)
    assert np.allclose(C, C.transpose(0, 2, 1))
    assert freqs[0] == pytest.approx(1.0) and freqs[-1] == pytest.approx(20.0)


def test_covariances_match_numpy(sim):
    C = covariances(sim.X[:3])
    assert np.allclose(C[1], np.cov(sim.X[1]))


def test_bootstrap_averages_per_class(sim):
    Xb, yb = bootstrap_averages(sim.X, sim.y, n_averages={1: 7, 0: 3},
                                average_size=10, random_state=0)
    assert Xb.shape == (10, 8, sim.X.shape[-1])
    assert (yb == 1).sum() == 7


@pytest.mark.parametrize("est", [CAJD(sfreq=128.0), BAJD(), AJD(sfreq=128.0)])
def test_fit_attributes_and_shapes(sim, est):
    est = clone(est).set_params(random_state=0)
    est.fit(sim.X, sim.y)
    N, T = sim.X.shape[1:]
    assert est.filters_.shape == (N, N)
    assert est.temporal_filters_.shape == (T, N)
    assert est.patterns_.shape == (N, N)
    assert est.temporal_patterns_.shape == (N, T)
    assert np.all(np.diff(est.evoked_power_) <= 1e-12)
    assert est.transform(sim.X).shape == sim.X.shape
    assert est.amplitudes(sim.X).shape == (len(sim.X), N)
    assert est.denoise(sim.X, [0, 1]).shape == sim.X.shape


def test_get_params_and_clone():
    est = CAJD(alpha=0.3, sfreq=256.0)
    assert clone(est).get_params()["alpha"] == 0.3
    assert "alpha" not in BAJD().get_params()
    assert "linear" in AJD().get_params()


def test_cajd_denoising_improves_snr(sim):
    est = CAJD(sfreq=sim.sfreq, random_state=0).fit(sim.X, sim.y)
    target = sim.y == 1
    Xh = est.denoise(sim.X[target], components=list(range(sim.n_erp)))
    assert snr_db(Xh, sim.evoked[target]) > snr_db(sim.X[target], sim.evoked[target]) + 5
    mean_erp = Xh.mean(0)
    assert np.median(correlation(mean_erp, sim.evoked[target].mean(0))) > 0.9


def test_works_in_sklearn_pipeline(sim):
    flat = FunctionTransformer(lambda a: a.reshape(len(a), -1))
    pipe = make_pipeline(CAJD(sfreq=sim.sfreq, n_components=2, random_state=0),
                         flat, LogisticRegression(max_iter=500))
    scores = cross_val_score(pipe, sim.X, sim.y, cv=3, scoring="roc_auc")
    assert scores.mean() > 0.8


def test_requires_sfreq_for_cospectra(sim):
    with pytest.raises(ValueError, match="sfreq"):
        CAJD().fit(sim.X, sim.y)


def test_accepts_mne_epochs(sim):
    mne = pytest.importorskip("mne")
    info = mne.create_info(8, sim.sfreq, "eeg")
    epochs = mne.EpochsArray(sim.X, info, verbose=False)
    est = CAJD(random_state=0).fit(epochs, sim.y)  # sfreq read from info
    assert est.transform(epochs).shape == sim.X.shape
