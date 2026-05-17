"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sygnal_clustering.config import DROP_COLUMNS, N_FEATURES, N_SAMPLES


def load_waveforms(path: Path | str) -> np.ndarray:
    """Load Run200 waveform matrix (n_samples, 500)."""
    path = Path(path)
    df = pd.read_csv(path, sep=" ", header=None, skipinitialspace=True)
    df = df.drop(columns=DROP_COLUMNS, errors="ignore")
    if df.shape[1] != N_FEATURES:
        msg = f"Expected {N_FEATURES} features, got {df.shape[1]}"
        raise ValueError(msg)
    x = df.to_numpy(dtype=np.float64)
    if x.shape[0] != N_SAMPLES:
        msg = f"Expected {N_SAMPLES} rows, got {x.shape[0]}"
        raise ValueError(msg)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite values in waveform matrix")
    return x
