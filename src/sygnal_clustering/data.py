"""Data loading utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sygnal_clustering.config import DATA_PATH, DROP_COLUMNS, N_FEATURES, N_SAMPLES, REPO_ROOT


def resolve_data_path() -> Path:
    """Find Run200_Wave_0_1.txt (local repo, Colab /content, cwd)."""
    candidates = [
        DATA_PATH,
        REPO_ROOT / "Run200_Wave_0_1.txt",
        Path("Run200_Wave_0_1.txt"),
        Path("/content/Run200_Wave_0_1.txt"),
        Path("/content/sygnal_types/Run200_Wave_0_1.txt"),
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    msg = "Run200_Wave_0_1.txt not found in any standard path"
    raise FileNotFoundError(msg)


def load_raw_dataframe(path: Path | str | None = None) -> pd.DataFrame:
    """Load raw Run200 CSV before column drop."""
    path = Path(path) if path is not None else resolve_data_path()
    return pd.read_csv(path, sep=" ", header=None, skipinitialspace=True)


def load_metadata_column(column: int, path: Path | str | None = None) -> np.ndarray:
    """Load a single metadata column from the raw CSV (e.g. col2 for method A)."""
    df = load_raw_dataframe(path)
    if column < 0 or column >= df.shape[1]:
        msg = f"Column {column} out of range for {df.shape[1]} columns"
        raise ValueError(msg)
    return df[column].to_numpy(dtype=np.float64)


def load_waveforms(path: Path | str | None = None) -> np.ndarray:
    """Load Run200 waveform matrix (n_samples, 500)."""
    path = Path(path) if path is not None else resolve_data_path()
    df = load_raw_dataframe(path)
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
