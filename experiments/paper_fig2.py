"""Reproduce the simulation of the EUSIPCO 2016 paper (Fig. 2, Table I).

K = 100 bilinear matrices X_k (N=16, T=128), L = 100 symmetric matrices R_l,
non-orthogonal A and E with condition number in [1, 20), noise sigma in
{0, 0.1}. AJD runs on {X_k X_k^T, R_l}, BAJD on {X_k}, CAJD on both.
B is initialized with the identity (as in the paper's simulation) and D
with a random orthonormal matrix.

Usage: python experiments/paper_fig2.py [n_runs]
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cajd import ajd_gpt, bajd_gpt, cajd_gpt, moreau_macchi
from cajd.simulation import simulate_paper

OUT = Path(__file__).parent / "results"
N_SWEEPS = 40


def run(seed, sigma):
    s = simulate_paper(n_sources=16, sigma=sigma, random_state=seed)
    rng = np.random.RandomState(seed + 10_000)
    B0 = np.eye(16)
    D0 = np.linalg.qr(rng.randn(128, 16))[0]
    cb = lambda B, D: moreau_macchi(s.A, B)  # noqa: E731
    covs = np.einsum("knt,kmt->knm", s.X, s.X)
    # put both sets on the same scale (the paper's alpha = 0.5 setting)
    C_n = s.C / np.linalg.norm(s.C)
    X_n = s.X / np.linalg.norm(s.X)
    kw = dict(max_iter=N_SWEEPS, tol=0, callback=cb)
    out = {}
    out["AJD"] = ajd_gpt(np.concatenate([covs / np.linalg.norm(covs), C_n]), B0=B0, **kw)
    out["BAJD"] = bajd_gpt(X_n, B0, D0, **kw)
    out["CAJD"] = cajd_gpt(X_n, C_n, B0, D0, alpha=0.5, **kw)
    curves = {k: [h["callback"] for h in r.history] for k, r in out.items()}
    final_E = {k: moreau_macchi(s.E, r.D) for k, r in out.items() if r.D is not None}
    return curves, final_E


def main(n_runs=100):
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharex=True)
    colors = {"AJD": "#d62728", "CAJD": "#1f77b4", "BAJD": "#bcbd22"}
    lines = ["| sigma | method | I_MM(B) median | 10% | 90% | I_MM(D) median |",
             "|---|---|---|---|---|---|"]
    for ax, sigma in zip(axes, [0.0, 0.1]):
        res = [run(seed, sigma) for seed in range(n_runs)]
        for m in ["AJD", "CAJD", "BAJD"]:
            curves = np.array([r[0][m] for r in res])
            ax.plot(np.log10(np.median(curves, 0)), "o-", ms=3, color=colors[m],
                    label=f"{m} (median)")
            last = curves[:, -1]
            q = np.quantile(last, [0.5, 0.1, 0.9])
            dmed = np.median([r[1][m] for r in res]) if m != "AJD" else np.nan
            lines.append(f"| {sigma} | {m} | {q[0]:.2e} | {q[1]:.2e} | {q[2]:.2e} | {dmed:.2e} |")
        ax.set_title(f"sigma = {sigma}  ({n_runs} runs)")
        ax.set_xlabel("number of sweeps")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("log10 Moreau-Macchi index on B")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(OUT / "paper_fig2.png", dpi=150)
    table = "\n".join(lines)
    (OUT / "paper_fig2.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
