"""Is the ERP waveform preserved? Single-trial means and spread per method.

For one simulated subject and the electrode with the largest ERP, each panel
shows on the held-out target trials:
  * the noiseless ERP: mean (black) and +/- 1 SD across trials (grey band),
  * the plain ensemble average of the noisy trials (thin grey),
  * the mean of the denoised trials (colour) and their +/- 1 SD (band).
Rows are conditions (input SNR, overcomplete background, latency jitter),
columns are methods; CAJD+lat uses the ACSTP latency correction.

Usage: python experiments/erp_shape.py [seed]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pyriemann.spatialfilters import Xdawn

from cajd import AJD, BAJD, CAJD
from cajd.metrics import snr_db
from cajd.simulation import simulate_erp_eeg

OUT = Path(__file__).parent / "results"
N_EPOCHS, N_ERP, SFREQ = 400, 2, 128.0
CONDITIONS = [
    ("-5 dB", dict(snr_db=-5)),
    ("-15 dB", dict(snr_db=-15)),
    ("-5 dB, 128 background sources", dict(snr_db=-5, n_background=128)),
    ("-5 dB, jitter +/-5 samples", dict(snr_db=-5, jitter_samples=5)),
]
METHODS = ["xDAWN", "AJD-cov", "BAJD", "CAJD-cov", "CAJD-cov spatial", "CAJD-cov+lat"]
COLORS = dict(zip(METHODS, ["#c2410c", "#7c3aed", "#0891b2", "#1d4ed8", "#15803d", "#be185d"]))


def denoised(sim, tr, te, seed):
    comps = list(range(N_ERP))
    out = {}
    xd = Xdawn(nfilter=N_ERP, classes=[1]).fit(sim.X[tr], sim.y[tr])
    W, Pt = xd.filters_[:N_ERP], xd.patterns_[:N_ERP]
    out["xDAWN"] = np.einsum("cn,kct->knt", Pt, np.einsum("cn,knt->kct", W, sim.X[te]))
    common = dict(random_state=seed, max_iter=100)
    for name, est in {"AJD-cov": AJD(linear="covariances", **common),
                      "BAJD": BAJD(**common),
                      "CAJD-cov": CAJD(linear="covariances", **common),
                      "CAJD-cov+lat": CAJD(linear="covariances", max_shift=8, **common)}.items():
        est.fit(sim.X[tr], sim.y[tr])
        out[name] = est.denoise(sim.X[te], comps, "bilinear")
        if name == "CAJD-cov":
            out["CAJD-cov spatial"] = est.denoise(sim.X[te], comps, "spatial")
    return out


def main(seed=0):
    OUT.mkdir(exist_ok=True)
    t = np.arange(128) / SFREQ * 1000
    fig, axes = plt.subplots(len(CONDITIONS), len(METHODS), figsize=(17, 10),
                             sharex=True, sharey="row")
    for i, (label, kw) in enumerate(CONDITIONS):
        sim = simulate_erp_eeg(n_channels=16, n_epochs=N_EPOCHS, sfreq=SFREQ, n_erp=N_ERP,
                               random_state=seed, **kw)
        perm = np.random.RandomState(seed).permutation(N_EPOCHS)
        tr, te = perm[: N_EPOCHS // 2], perm[N_EPOCHS // 2:]
        te = te[sim.y[te] == 1]
        truth = sim.evoked[te]
        ch = int(np.argmax(np.mean(truth**2, axis=(0, 2))))
        est = denoised(sim, tr, te, seed)
        scale = np.abs(truth[:, ch].mean(0)).max()
        for j, m in enumerate(METHODS):
            ax = axes[i, j]
            tm, ts = truth[:, ch].mean(0) / scale, truth[:, ch].std(0) / scale
            ax.fill_between(t, tm - ts, tm + ts, color="0.85", lw=0)
            ax.plot(t, sim.X[te, ch].mean(0) / scale, color="0.55", lw=0.7)
            dm, ds = est[m][:, ch].mean(0) / scale, est[m][:, ch].std(0) / scale
            ax.fill_between(t, dm - ds, dm + ds, color=COLORS[m], alpha=0.2, lw=0)
            ax.plot(t, tm, color="k", lw=1.6)
            ax.plot(t, dm, color=COLORS[m], lw=1.3)
            r = np.corrcoef(dm, tm)[0, 1]
            ax.text(0.02, 0.96, f"SNR {snr_db(est[m], truth):.1f} dB\nr(mean) {r:.3f}",
                    transform=ax.transAxes, va="top", fontsize=7.5)
            if i == 0:
                ax.set_title(m, fontsize=10, color=COLORS[m])
            if j == 0:
                ax.set_ylabel(f"{label}\nchannel {ch}", fontsize=8.5)
            if i == len(CONDITIONS) - 1:
                ax.set_xlabel("time (ms)")
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Black: noiseless ERP mean (grey band +/-1 SD across trials). "
                 "Thin grey: ensemble average of the noisy trials. "
                 "Colour: mean of denoised trials (band +/-1 SD). "
                 "Amplitude normalized to the true peak.", fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "erp_shape.png", dpi=130)
    print("saved", OUT / "erp_shape.png")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
