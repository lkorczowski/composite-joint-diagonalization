"""Very noisy regime: does CAJD/AJD beat xDAWN when the ERP is deeply buried?

Same protocol and methods as synthetic_snr.py (fit on half, evaluate on the
held-out target epochs), restricted to input SNR -30..-15 dB and 100..800
epochs, no jitter.

Usage: python experiments/low_snr.py [n_seeds]
"""

import itertools
import sys

import pandas as pd

from synthetic_snr import OUT, run_condition


def main(n_seeds=10):
    OUT.mkdir(exist_ok=True)
    rows = []
    grid = list(itertools.product([-30, -25, -20, -15], [100, 200, 400, 800], range(n_seeds)))
    for i, (snr_in, n_ep, seed) in enumerate(grid):
        rows += run_condition(snr_in, n_ep, 0, seed)
        if i % 20 == 0:
            print(f"{i}/{len(grid)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "low_snr.csv", index=False)
    print(df.groupby(["n_epochs", "snr_in", "method"])[["snr", "pattern_err"]]
          .median().unstack("method").round(2).to_string())


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
