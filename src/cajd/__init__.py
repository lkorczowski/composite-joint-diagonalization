"""Composite Approximate Joint Diagonalization (CAJD) for EEG / ERP source separation."""

from ._gpt import GPTResult, ajd_gpt, bajd_gpt, cajd_gpt, gpt, svd_init
from .estimators import AJD, BAJD, CAJD
from .latency import estimate_latencies, shift_epochs
from .metrics import correlation, moreau_macchi, snr_db
from .stats import bootstrap_averages, cospectra, covariances

__version__ = "0.1.0.dev0"

__all__ = [
    "AJD", "BAJD", "CAJD", "GPTResult", "ajd_gpt", "bajd_gpt", "cajd_gpt",
    "gpt", "svd_init", "moreau_macchi", "snr_db", "correlation",
    "bootstrap_averages", "cospectra", "covariances", "estimate_latencies",
    "shift_epochs",
]
