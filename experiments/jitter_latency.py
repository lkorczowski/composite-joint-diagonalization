"""Latency jitter with and without the ACSTP latency correction.

Same setting as robustness.py (16 electrodes, 2 ERP sources, 400 epochs,
fit on half, evaluated on held-out targets), jitter uniform in [-J, J]
samples. '+lat' methods estimate single-trial latencies (max_shift=8) at
fit time and on the evaluated trials.

Usage: python experiments/jitter_latency.py [n_seeds]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.spatialfilters import Xdawn

from cajd import BAJD, CAJD
from cajd.metrics import snr_db
from cajd.simulation import simulate_erp_eeg

OUT = Path(__file__).parent / "results"
N_EPOCHS, N_ERP = 400, 2


def run(seed, snr_in, jitter):
    sim = simulate_erp_eeg(n_channels=16, n_epochs=N_EPOCHS, n_erp=N_ERP, snr_db=snr_in,
                           jitter_samples=jitter, random_state=seed)
    perm = np.random.RandomState(seed).permutation(N_EPOCHS)
    tr, te = perm[: N_EPOCHS // 2], perm[N_EPOCHS // 2:]
    tgt = te[sim.y[te] == 1]
    truth, comps = sim.evoked[tgt], list(range(N_ERP))
    out = {}
    xd = Xdawn(nfilter=N_ERP, classes=[1]).fit(sim.X[tr], sim.y[tr])
    W, Pt = xd.filters_[:N_ERP], xd.patterns_[:N_ERP]
    out["xDAWN"] = np.einsum("cn,kct->knt", Pt, np.einsum("cn,knt->kct", W, sim.X[tgt]))
    common = dict(random_state=seed, max_iter=100)
    models = {"BAJD": BAJD(**common), "BAJD+lat": BAJD(max_shift=8, **common),
              "CAJD-cov": CAJD(linear="covariances", **common),
              "CAJD-cov+lat": CAJD(linear="covariances", max_shift=8, **common)}
    lat_err = {}
    for name, est in models.items():
        est.fit(sim.X[tr], sim.y[tr])
        out[name] = est.denoise(sim.X[tgt], comps, "bilinear")
        if name.endswith("+lat"):
            e = est.estimate_latencies(sim.X[tgt]) - sim.latencies[tgt]
            lat_err[name] = float(np.mean(np.abs(e - np.median(e))))
    return [dict(method=m, snr=float(snr_db(Xh, truth)), lat_mae=lat_err.get(m, np.nan))
            for m, Xh in out.items()]


def main(n_seeds=10):
    rows = []
    for snr_in in [-15, -5]:
        for jitter in range(6):
            for seed in range(n_seeds):
                for r in run(seed, snr_in, jitter):
                    rows.append(dict(r, snr_in=snr_in, jitter=jitter, seed=seed))
            print(snr_in, jitter, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "jitter_latency.csv", index=False)
    summary = df.groupby(["snr_in", "jitter", "method"])[["snr", "lat_mae"]].median()
    summary = summary.unstack("method").round(2)
    summary.to_csv(OUT / "jitter_latency_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
