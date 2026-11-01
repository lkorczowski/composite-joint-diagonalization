"""Hypothesis test on synthetic EEG: random ERPs inside random ongoing EEG.

Does mining the linear (ongoing EEG) and bilinear (ERP) structure jointly
give better single-trial ERP estimates than either model alone?

For each condition (input SNR, number of epochs, latency jitter) and seed:
  * simulate P300-like epochs (``cajd.simulation.simulate_erp_eeg``),
  * fit every method on a training half, evaluate on the held-out half,
  * reconstruct the evoked part of each held-out TARGET epoch from the
    ``n_erp`` leading components and compare with the ground truth.

Methods: no filtering (raw epoch), ensemble average (the same mean waveform
for every trial), xDAWN (pyRiemann), AJD, BAJD, CAJD (bilinear
back-projection, eq. 13), each with co-spectra or per-epoch covariances as
the linear set, and "-spatial" variants (filtered time courses).

Metrics on held-out target epochs:
  * ``snr``: output SNR in dB of the single-trial evoked estimate,
  * ``corr``: correlation between the estimated and true average ERP,
  * ``mm_erp``: subspace error of the recovered ERP patterns (1 - mean
    cosine between estimated and true ERP scalp maps, after matching).

Usage: python experiments/synthetic_snr.py [n_seeds]
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.spatialfilters import Xdawn

from cajd import AJD, BAJD, CAJD
from cajd.metrics import correlation, snr_db
from cajd.simulation import simulate_erp_eeg

OUT = Path(__file__).parent / "results"
SFREQ = 128.0


def pattern_error(true_patterns, est_patterns):
    """1 - mean |cos| between each true pattern and its best estimated match."""
    t = true_patterns / np.linalg.norm(true_patterns, axis=0)
    e = est_patterns / np.linalg.norm(est_patterns, axis=0)
    cos = np.abs(t.T @ e)
    return float(1 - cos.max(axis=1).mean())


def evaluate(name, Xh, sim, test, patterns=None):
    tgt = test[sim.y[test] == 1]
    truth = sim.evoked[tgt]
    row = {
        "method": name,
        "snr": float(snr_db(Xh, truth)),
        "corr": float(np.mean(correlation(Xh.mean(0), truth.mean(0)))),
    }
    if patterns is not None:
        row["pattern_err"] = pattern_error(sim.A[:, : sim.n_erp], patterns)
    return row


def run_condition(snr_in, n_epochs, jitter, seed, n_channels=16, n_erp=2):
    sim = simulate_erp_eeg(n_channels=n_channels, n_epochs=n_epochs, sfreq=SFREQ,
                           n_erp=n_erp, snr_db=snr_in, jitter_ms=jitter,
                           random_state=seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_epochs)
    train, test = perm[: n_epochs // 2], perm[n_epochs // 2:]
    tgt = test[sim.y[test] == 1]
    comps = list(range(n_erp))
    rows = [evaluate("raw", sim.X[tgt], sim, test),
            evaluate("average", np.broadcast_to(
                sim.X[train][sim.y[train] == 1].mean(0), sim.evoked[tgt].shape),
                sim, test)]

    xd = Xdawn(nfilter=n_erp, classes=[1]).fit(sim.X[train], sim.y[train])
    W, Pt = xd.filters_[:n_erp], xd.patterns_[:n_erp]
    rows.append(evaluate("xDAWN", np.einsum("cn,kct->knt", Pt,
                         np.einsum("cn,knt->kct", W, sim.X[tgt])), sim, test, Pt.T))

    common = dict(random_state=seed, max_iter=50)
    # linear set: co-spectra (paper) or per-epoch covariances (non-stationarity)
    models = {
        "AJD-cosp": AJD(sfreq=SFREQ, linear="cospectra+evoked", **common),
        "AJD-cov": AJD(linear="covariances", **common),
        "BAJD": BAJD(**common),
        "CAJD-cosp": CAJD(sfreq=SFREQ, alpha=0.5, linear="cospectra", **common),
        "CAJD-cov": CAJD(alpha=0.5, linear="covariances", **common),
    }
    for name, est in models.items():
        est.fit(sim.X[train], sim.y[train])
        rows.append(evaluate(name, est.denoise(sim.X[tgt], comps, "bilinear"),
                             sim, test, est.patterns_[:, comps]))
        if name.startswith(("CAJD", "BAJD")):
            rows.append(evaluate(name + "-spatial",
                                 est.denoise(sim.X[tgt], comps, "spatial"),
                                 sim, test))
    for r in rows:
        r.update(snr_in=snr_in, n_epochs=n_epochs, jitter_ms=jitter, seed=seed)
    return rows


def main(n_seeds=10):
    OUT.mkdir(exist_ok=True)
    grid = list(itertools.product([-20, -15, -10, -5, 0], [100, 200, 400], [0, 25],
                                  range(n_seeds)))
    rows = []
    for i, (snr_in, n_ep, jit, seed) in enumerate(grid):
        rows += run_condition(snr_in, n_ep, jit, seed)
        if i % 20 == 0:
            print(f"{i}/{len(grid)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "synthetic_snr.csv", index=False)
    summary = (df.groupby(["jitter_ms", "n_epochs", "snr_in", "method"])[
        ["snr", "corr", "pattern_err"]].median().round(3))
    summary.to_csv(OUT / "synthetic_snr_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
