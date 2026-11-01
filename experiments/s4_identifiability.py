"""S4: when neither the linear nor the bilinear structure identifies the sources.

Part A (matrix level). Sources come in pairs that share the same profile:
  * bilinear set X_k = A S_k E^T: sources (0,1), (2,3), ... share their
    trial amplitudes, so BAJD can only recover each pair's subspace;
  * linear set C_l = A Lambda_l A^T: sources (7,0), (1,2), ... share their
    powers, so AJD can only recover those pairs' subspaces.
The concatenated profiles are all distinct, so the composite model is
identifiable. A 'generic' design (all profiles distinct) is the control.

Part B (EEG-like). An ERP is present in every epoch (no target / non-target
on-off contrast), the ERP sources share their single-trial amplitude, and the
background is oscillatory. Per-epoch covariances and the bilinear model each
see the ERP sources as one block; co-spectra can tell them apart only through
their different waveforms' spectra.

Usage: python experiments/s4_identifiability.py [n_runs]
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyriemann.spatialfilters import Xdawn

from cajd import AJD, BAJD, CAJD, ajd_gpt, bajd_gpt, cajd_gpt, moreau_macchi
from cajd.metrics import snr_db
from cajd.simulation import random_matrix, simulate_erp_eeg

OUT = Path(__file__).parent / "results"


# --------------------------------------------------------------- part A
def simulate_pairs(design, sigma, N=8, T=128, K=100, L=100, random_state=None):
    rng = np.random.RandomState(random_state)
    A = random_matrix(N, N, (1, 20), rng)
    E = random_matrix(T, N, (1, 20), rng)
    if design == "generic":
        S = rng.randn(K, N)
        lam = rng.chisquare(2, (L, N))
    else:
        S = rng.randn(K, N // 2)[:, np.arange(N) // 2]
        lam = rng.chisquare(2, (L, N // 2))[:, ((np.arange(N) + 1) % N) // 2]
    X = np.einsum("nm,km,tm->knt", A, S, E) + sigma * rng.randn(K, N, T)
    noise = rng.randn(L, N, N)
    C = np.einsum("nm,lm,pm->lnp", A, lam, A) + sigma**2 / 2 * (noise + noise.transpose(0, 2, 1))
    return X, C, A, E


def part_a(n_runs):
    rows = []
    for design in ["generic", "paired"]:
        for sigma in [0.0, 0.01, 0.05, 0.1]:
            for seed in range(n_runs):
                X, C, A, E = simulate_pairs(design, sigma, random_state=seed)
                rng = np.random.RandomState(seed + 10_000)
                B0, D0 = np.eye(8), np.linalg.qr(rng.randn(128, 8))[0]
                Xn, Cn = X / np.linalg.norm(X), C / np.linalg.norm(C)
                kw = dict(max_iter=200, tol=1e-12)
                res = {"AJD": ajd_gpt(Cn, B0=B0, **kw),
                       "BAJD": bajd_gpt(Xn, B0, D0, **kw),
                       "CAJD": cajd_gpt(Xn, Cn, B0, D0, alpha=0.5, **kw)}
                for m, r in res.items():
                    rows.append(dict(design=design, sigma=sigma, seed=seed, method=m,
                                     mm_B=moreau_macchi(A, r.B)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "s4a_pairs.csv", index=False)
    return df


# --------------------------------------------------------------- part B
def pattern_error(true, est):
    t = true / np.linalg.norm(true, axis=0)
    e = est / np.linalg.norm(est, axis=0)
    return float(1 - np.abs(t.T @ e).max(axis=1).mean())


def subspace_error(true, est):
    q1, q2 = np.linalg.qr(true)[0], np.linalg.qr(est)[0]
    return float(1 - np.linalg.svd(q1.T @ q2, compute_uv=False).mean())


def part_b(n_runs):
    sfreq, n_erp = 128.0, 2
    rows = []
    for snr_in in [-15, -10, -5, 0]:
        for seed in range(n_runs):
            sim = simulate_erp_eeg(n_channels=16, n_epochs=200, sfreq=sfreq, n_erp=n_erp,
                                   snr_db=snr_in, target_ratio=1.0, shared_amplitude=True,
                                   background="oscillation", random_state=seed)
            perm = np.random.RandomState(seed).permutation(200)
            tr, te = perm[:100], perm[100:]
            truth, A_erp = sim.evoked[te], sim.A[:, :n_erp]
            comps = list(range(n_erp))
            ests = {}
            xd = Xdawn(nfilter=n_erp, classes=[1]).fit(sim.X[tr], sim.y[tr])
            W, Pt = xd.filters_[:n_erp], xd.patterns_[:n_erp]
            ests["xDAWN"] = (Pt.T, np.einsum("cn,kct->knt", Pt,
                                             np.einsum("cn,knt->kct", W, sim.X[te])))
            common = dict(random_state=seed, max_iter=100)
            models = {
                "AJD-cosp": AJD(sfreq=sfreq, linear="cospectra", **common),
                "AJD-cov": AJD(linear="covariances", **common),
                "BAJD": BAJD(**common),
                "CAJD-cosp": CAJD(sfreq=sfreq, linear="cospectra", **common),
                "CAJD-cov": CAJD(linear="covariances", **common),
            }
            for name, est in models.items():
                est.fit(sim.X[tr], sim.y[tr])
                ests[name] = (est.patterns_[:, comps],
                              est.denoise(sim.X[te], comps, "spatial"))
            for name, (pat, Xh) in ests.items():
                rows.append(dict(snr_in=snr_in, seed=seed, method=name,
                                 pattern_err=pattern_error(A_erp, pat),
                                 subspace_err=subspace_error(A_erp, pat),
                                 snr=float(snr_db(Xh, truth))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "s4b_eeg.csv", index=False)
    return df


def main(n_runs=20):
    OUT.mkdir(exist_ok=True)
    a = part_a(n_runs)
    ta = a.groupby(["design", "sigma", "method"]).mm_B.median().unstack("method")
    print("Part A: median Moreau-Macchi index on B\n", ta.to_string(float_format="%.2e"))
    b = part_b(max(n_runs // 2, 5))
    tb = b.groupby(["snr_in", "method"])[["pattern_err", "subspace_err", "snr"]].median()
    print("\nPart B: medians\n", tb.unstack("method").round(3).to_string())
    with open(OUT / "s4_summary.md", "w") as f:
        f.write("## Part A: median Moreau-Macchi index on B\n\n")
        f.write(ta.to_markdown(floatfmt=".2e") + "\n\n")
        f.write("## Part B: medians\n\n")
        f.write(tb.unstack("method").round(3).to_markdown() + "\n")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
