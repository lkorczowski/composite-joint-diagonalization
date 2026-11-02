"""Real EEG: Brain Invaders 2015a (MOABB BI2015a, 32 electrodes, P300).

Per subject: epochs 0-1 s, 1-20 Hz, 128 Hz. Models are fitted on half of the
epochs; the other half is denoised. There is no ground truth, so we report
on the held-out target epochs:
  * fidelity: correlation (all channels) between the mean of the denoised
    trials and the plain average of *all* target epochs of the subject;
  * single-trial SNR: power of the target mean over the mean power of the
    single-trial deviations from it, in dB (higher = trials closer to their
    average, i.e. less residual noise);
  * classification AUC (target vs non-target) of the denoised trials, by
    their correlation with the training target average.

Figures:
  results/bi2015a_erp.png   mean +/- 1 SD of target trials at Cz/Pz, raw vs
                            each method, one row per subject;
  results/bi2015a_maps.png  spatial patterns of the two leading components
                            per method, one block per subject.

Usage: python experiments/real_bi2015a.py [subjects...]   (default 1-5)
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from moabb.datasets import BI2015a
from moabb.paradigms import P300
from pyriemann.spatialfilters import Xdawn
from sklearn.metrics import roc_auc_score

from cajd import AJD, BAJD, CAJD

mne.set_log_level("WARNING")
OUT = Path(__file__).parent / "results"
N_COMP = 2
METHODS = ["raw", "xDAWN", "AJD-cov", "BAJD", "CAJD-cov", "CAJD-cov spatial", "CAJD-cov+lat"]
COLORS = dict(zip(METHODS, ["0.4", "#c2410c", "#7c3aed", "#0891b2", "#1d4ed8", "#15803d",
                            "#be185d"]))


def load(subject):
    epochs, labels, _ = P300(fmin=1, fmax=20, resample=128).get_data(
        BI2015a(), [subject], return_epochs=True)
    X = epochs.get_data(copy=True)[..., :128] * 1e6  # microvolts, 0-1 s
    y = (np.asarray(labels) == "Target").astype(int)
    return X, y, epochs.info


def fit_all(X, y, tr, seed):
    """Return {method: (denoise function, spatial patterns or None)}."""
    comps = list(range(N_COMP))
    out = {"raw": (lambda Z: Z, None)}
    xd = Xdawn(nfilter=N_COMP, classes=[1]).fit(X[tr], y[tr])
    W, Pt = xd.filters_[:N_COMP], xd.patterns_[:N_COMP]
    out["xDAWN"] = (lambda Z: np.einsum("cn,kct->knt", Pt, np.einsum("cn,knt->kct", W, Z)),
                    Pt.T)
    common = dict(random_state=seed, max_iter=200)
    models = {"AJD-cov": AJD(linear="covariances", **common), "BAJD": BAJD(**common),
              "CAJD-cov": CAJD(linear="covariances", **common),
              "CAJD-cov+lat": CAJD(linear="covariances", max_shift=8, **common)}
    for name, est in models.items():
        est.fit(X[tr], y[tr])
        out[name] = (lambda Z, e=est: e.denoise(Z, comps, "bilinear"), est.patterns_[:, comps])
        if name == "CAJD-cov":
            out["CAJD-cov spatial"] = (lambda Z, e=est: e.denoise(Z, comps, "spatial"),
                                       est.patterns_[:, comps])
    return out


def single_trial_snr(Z):
    m = Z.mean(0)
    return 10 * np.log10(np.sum(m**2) / np.mean(np.sum((Z - m) ** 2, axis=(1, 2))))


def main(subjects):
    OUT.mkdir(exist_ok=True)
    t = np.arange(128) / 128 * 1000
    rows, maps = [], {}
    fig, axes = plt.subplots(len(subjects), len(METHODS), figsize=(18, 2.4 * len(subjects)),
                             sharex=True, squeeze=False)
    for i, s in enumerate(subjects):
        X, y, info = load(s)
        rng = np.random.RandomState(s)
        tr = np.zeros(len(y), bool)
        for c in (0, 1):
            idx = np.flatnonzero(y == c)
            tr[rng.choice(idx, len(idx) // 2, replace=False)] = True
        te = ~tr
        ref = X[y == 1].mean(0)
        models = fit_all(X, y, tr, s)
        ch = info["ch_names"].index("Cz") if "Cz" in info["ch_names"] else 0
        if "Pz" in info["ch_names"]:
            ch = info["ch_names"].index("Pz")
        Zte = {m: f(X[te]) for m, (f, _) in models.items()}
        template = X[tr & (y == 1)].mean(0)
        for m, Z in Zte.items():
            tgt = Z[y[te] == 1]
            score = np.einsum("knt,nt->k", Z, template)
            rows.append(dict(subject=s, method=m,
                             fidelity=float(np.corrcoef(tgt.mean(0).ravel(), ref.ravel())[0, 1]),
                             st_snr=float(single_trial_snr(tgt)),
                             auc=float(roc_auc_score(y[te], score)),
                             n_targets=int(len(tgt))))
            ax = axes[i, METHODS.index(m)]
            raw = X[te][y[te] == 1][:, ch]
            ax.fill_between(t, raw.mean(0) - raw.std(0), raw.mean(0) + raw.std(0),
                            color="0.88", lw=0)
            ax.plot(t, raw.mean(0), color="k", lw=1.2)
            dm, ds = tgt[:, ch].mean(0), tgt[:, ch].std(0)
            if m != "raw":
                ax.fill_between(t, dm - ds, dm + ds, color=COLORS[m], alpha=0.25, lw=0)
                ax.plot(t, dm, color=COLORS[m], lw=1.2)
            r = rows[-1]
            ax.text(0.02, 0.97, f"ST-SNR {r['st_snr']:.1f} dB\nAUC {r['auc']:.2f}",
                    transform=ax.transAxes, va="top", fontsize=7)
            if i == 0:
                ax.set_title(m, fontsize=10, color=COLORS[m])
            if METHODS.index(m) == 0:
                ax.set_ylabel(f"S{s} {info['ch_names'][ch]} (µV)", fontsize=8.5)
            ax.tick_params(labelsize=7)
            ax.spines[["top", "right"]].set_visible(False)
        maps[s] = (info, {m: p for m, (_, p) in models.items() if p is not None
                          and m != "CAJD-cov spatial"})
        print("subject", s, flush=True)
    for ax in axes[-1]:
        ax.set_xlabel("time (ms)")
    fig.suptitle("Held-out target trials. Black + grey band: plain average +/- 1 SD of the raw "
                 "trials. Colour: mean +/- 1 SD of the denoised trials (2 components).",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT / "bi2015a_erp.png", dpi=120)

    names = list(next(iter(maps.values()))[1])
    fig, axes = plt.subplots(len(subjects), len(names) * N_COMP,
                             figsize=(2.0 * len(names) * N_COMP, 2.1 * len(subjects)),
                             squeeze=False)
    for i, s in enumerate(subjects):
        info, pats = maps[s]
        info = info.copy()
        info.set_montage("standard_1020", on_missing="ignore")
        for j, m in enumerate(names):
            for c in range(N_COMP):
                ax = axes[i, j * N_COMP + c]
                p = pats[m][:, c]
                p = p / np.abs(p).max() * np.sign(p[np.argmax(np.abs(p))])
                mne.viz.plot_topomap(p, info, axes=ax, show=False, contours=0, sensors=False)
                if i == 0:
                    ax.set_title(f"{m}\ncomp {c + 1}", fontsize=8.5, color=COLORS[m])
            axes[i, 0].set_ylabel(f"S{s}", fontsize=9)
            axes[i, 0].yaxis.set_visible(True)
    fig.tight_layout()
    fig.savefig(OUT / "bi2015a_maps.png", dpi=120)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bi2015a.csv", index=False)
    summary = df.groupby("method")[["fidelity", "st_snr", "auc"]].median().loc[METHODS]
    summary.round(3).to_csv(OUT / "bi2015a_summary.csv")
    print(df.pivot(index="subject", columns="method", values="st_snr")[METHODS].round(1))
    print(summary.round(3))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [1, 2, 3, 4, 5])
