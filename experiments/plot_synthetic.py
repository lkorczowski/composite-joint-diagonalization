"""Figure for experiments/synthetic_snr.py: output SNR and pattern error vs. input SNR."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RES = Path(__file__).parent / "results"
METHODS = {"average": "#7f7f7f", "xDAWN": "#2ca02c", "AJD-cov": "#d62728",
           "BAJD": "#bcbd22", "CAJD-cov": "#1f77b4", "CAJD-cov-spatial": "#17becf"}

df = pd.read_csv(RES / "synthetic_snr.csv")
df = df[df.n_epochs == 200]
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True)
for col, jit in enumerate([0, 25]):
    d = df[df.jitter_ms == jit]
    for row, metric in enumerate(["snr", "pattern_err"]):
        ax = axes[row, col]
        for m, c in METHODS.items():
            g = d[d.method == m].groupby("snr_in")[metric]
            if g.median().isna().all():
                continue
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            ax.plot(med.index, med, "o-", color=c, ms=4, label=m)
            ax.fill_between(med.index, lo, hi, color=c, alpha=0.12)
        ax.grid(alpha=0.3)
        if row == 0:
            ax.set_title(f"200 epochs, latency jitter {jit} ms")
        ax.set_xlabel("input SNR (dB)")
    axes[0, 0].set_ylabel("single-trial output SNR (dB)")
    axes[1, 0].set_ylabel("ERP pattern error (0 = perfect)")
axes[0, 0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(RES / "synthetic_snr.png", dpi=150)
