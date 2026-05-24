"""Shared I/O helpers for submissions and experiment reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def write_submission(labels: np.ndarray, path: Path) -> Path:
    """Write Kaggle submission CSV (index, cluster)."""
    df = pd.DataFrame({"index": np.arange(len(labels)), "cluster": labels.astype(int)})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_json_report(data: dict, path: Path) -> Path:
    """Write experiment report JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path
