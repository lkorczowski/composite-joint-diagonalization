"""Robustness of ERP estimation to three violations of the ideal setting.

1. Overcomplete background: 14 (square) to 256 background sources for 16
   electrodes, so no square unmixing can separate everything.
2. Latency jitter: each trial shifted by a uniform integer in [-J, J]
   samples (J = 0..5 at 128 Hz, i.e. up to +/-39 ms).
3. Label noise: 0 to 25 % of the training labels flipped (target <->
   non-target); evaluation always uses the true labels.

16 electrodes, 2 ERP sources, 400 epochs, input SNR -15 and -5 dB, fit on
half the epochs, evaluated on the held-out true targets.

Usage: python experiments/robustness.py [n_seeds]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.spatialfilters import Xdawn

from cajd import AJD, BAJD, CAJD
from cajd.metrics import snr_db
from cajd.simulation import simulate_erp_eeg

OUT = Path(__file__).parent / "results"
N_EPOCHS, N_ERP, SFREQ = 400, 2, 128.0


def pattern_error(true, est):
    t = true / np.linalg.norm(true, axis=0)
    e = est / np.linalg.norm(est, axis=0)
    return float(1 - np.abs(t.T @ e).max(axis=1).mean())


def subspace_error(true, est):
    q1, q2 = np.linalg.qr(true)[0], np.linalg.qr(est)[0]
    return float(1 - np.linalg.svd(q1.T @ q2, compute_uv=False).mean())


def run(seed, snr_in, n_background=None, jitter_samples=0, label_noise=0.0):
    sim = simulate_erp_eeg(n_channels=16, n_epochs=N_EPOCHS, sfreq=SFREQ, n_erp=N_ERP,
                           snr_db=snr_in, n_background=n_background,
                           jitter_samples=jitter_samples, random_state=seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(N_EPOCHS)
    tr, te = perm[: N_EPOCHS // 2], perm[N_EPOCHS // 2:]
    y_tr = sim.y[tr].copy()
    flip = rng.rand(len(tr)) < label_noise
    y_tr[flip] = 1 - y_tr[flip]
    tgt = te[sim.y[te] == 1]
    truth, A_erp, comps = sim.evoked[tgt], sim.A[:, :N_ERP], list(range(N_ERP))

    out = {}
    xd = Xdawn(nfilter=N_ERP, classes=[1]).fit(sim.X[tr], y_tr)
    W, Pt = xd.filters_[:N_ERP], xd.patterns_[:N_ERP]
    out["xDAWN"] = (Pt.T, np.einsum("cn,kct->knt", Pt, np.einsum("cn,knt->kct", W, sim.X[tgt])))
    common = dict(random_state=seed, max_iter=100)
    models = {"AJD-cov": AJD(linear="covariances", **common), "BAJD": BAJD(**common),
              "CAJD-cov": CAJD(linear="covariances", **common)}
    for name, est in models.items():
        est.fit(sim.X[tr], y_tr)
        out[name] = (est.patterns_[:, comps], est.denoise(sim.X[tgt], comps, "bilinear"))
        if name == "CAJD-cov":
            out["CAJD-cov-spatial"] = (est.patterns_[:, comps],
                                       est.denoise(sim.X[tgt], comps, "spatial"))
    return [dict(method=m, snr=float(snr_db(Xh, truth)), pattern_err=pattern_error(A_erp, P),
                 subspace_err=subspace_error(A_erp, P)) for m, (P, Xh) in out.items()]


def main(n_seeds=10):
    OUT.mkdir(exist_ok=True)
    tests = ([("n_background", v) for v in [14, 32, 64, 128, 256]]
             + [("jitter_samples", v) for v in [0, 1, 2, 3, 4, 5]]
             + [("label_noise", v) for v in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]])
    rows = []
    for snr_in in [-15, -5]:
        for factor, value in tests:
            for seed in range(n_seeds):
                for r in run(seed, snr_in, **{factor: value}):
                    rows.append(dict(r, factor=factor, value=value, snr_in=snr_in, seed=seed))
            print(snr_in, factor, value, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "robustness.csv", index=False)
    summary = df.groupby(["snr_in", "factor", "value", "method"])[
        ["snr", "pattern_err", "subspace_err"]].median().unstack("method").round(3)
    summary.to_csv(OUT / "robustness_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
