"""v3: three Kaggle control submissions (A/B/C)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from sygnal_clustering.config import DATA_PATH, RANDOM_STATE, REPO_ROOT
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.metrics import cluster_sizes, clustering_scores
from sygnal_clustering.models import remap_labels_physics
from sygnal_clustering.signal_extraction import extract_description_features

ARTIFACTS_V3_DIR = REPO_ROOT / "artifacts_v3"
SUBMISSION3_PATH = REPO_ROOT / "submission3.csv"
SUBMISSION3A_PATH = REPO_ROOT / "submission3a.csv"
SUBMISSION3B_PATH = REPO_ROOT / "submission3b.csv"
SUBMISSION3C_PATH = REPO_ROOT / "submission3c.csv"


def _load_raw_col2(path: Path | None = None) -> np.ndarray:
    path = path or DATA_PATH
    df = pd.read_csv(path, sep=" ", header=None, skipinitialspace=True)
    return df[2].to_numpy(dtype=np.float64)


def method_a_meta_col2_tercile(path: Path | None = None) -> np.ndarray:
    """Hypothesis A: class structure follows metadata column 2 tertiles."""
    col2 = _load_raw_col2(path)
    thresholds = np.quantile(col2, [1.0 / 3.0, 2.0 / 3.0])
    return np.digitize(col2, thresholds).astype(np.int64)


def method_b_description_gmm3(
    x: np.ndarray | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Hypothesis B: Description features + GMM-3."""
    if x is None:
        x = load_waveforms(DATA_PATH)
    features = extract_description_features(x)
    z = QuantileTransformer(
        output_distribution="normal",
        random_state=random_state,
    ).fit_transform(features[:, :3])
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=20,
        random_state=random_state,
    )
    raw = gmm.fit_predict(z)
    labels = remap_labels_physics(raw, features)
    return labels, features


def method_c_gmm2_low_confidence(
    x: np.ndarray | None = None,
    uncertain_fraction: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Hypothesis C: GMM-2 + top uncertain_fraction → cluster 2."""
    if x is None:
        x = load_waveforms(DATA_PATH)
    features = extract_description_features(x)
    z = RobustScaler().fit_transform(features[:, :4])
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=20,
        random_state=random_state,
    )
    gmm.fit(z)
    proba = gmm.predict_proba(z)
    labels = gmm.predict(z).astype(np.int64)
    uncertainty = 1.0 - proba.max(axis=1)
    n_unc = max(1, int(len(labels) * uncertain_fraction))
    unc_idx = np.argsort(uncertainty)[-n_unc:]
    labels[unc_idx] = 2

    out = np.zeros(len(labels), dtype=np.int64)
    particle = labels != 2
    m0 = particle & (labels == 0)
    m1 = particle & (labels == 1)
    if m0.any() and m1.any():
        if features[m0, 1].mean() <= features[m1, 1].mean():
            out[m0], out[m1] = 0, 1
        else:
            out[m0], out[m1] = 1, 0
    else:
        out[particle] = labels[particle]
    out[unc_idx] = 2
    return out, features


def labels_to_submission(labels: np.ndarray, path: Path) -> Path:
    df = pd.DataFrame({"index": np.arange(len(labels)), "cluster": labels.astype(int)})
    df.to_csv(path, index=False)
    return path


def metrics_for_labels(labels: np.ndarray, features: np.ndarray, method: str) -> dict:
    z = RobustScaler().fit_transform(features[:, :3])
    scores = clustering_scores(z, labels, random_state=RANDOM_STATE)
    sizes = cluster_sizes(labels)
    counts = np.bincount(labels, minlength=3).astype(float)
    fr = counts / counts.sum()
    return {
        **scores,
        **sizes,
        "fraction_0": float(fr[0]),
        "fraction_1": float(fr[1]),
        "fraction_2": float(fr[2]),
        "max_cluster_fraction": float(fr.max()),
        "method": method,
    }


def run_all_v3(
    x: np.ndarray | None = None,
    path: Path | None = None,
) -> dict:
    """Generate submission3a/b/c and submission3 (=3b)."""
    path = path or DATA_PATH
    if x is None:
        x = load_waveforms(path)

    lab_a = method_a_meta_col2_tercile(path)
    fe_a = extract_description_features(x)  # for metrics only
    lab_b, fe_b = method_b_description_gmm3(x)
    lab_c, fe_c = method_c_gmm2_low_confidence(x)

    labels_to_submission(lab_a, SUBMISSION3A_PATH)
    labels_to_submission(lab_b, SUBMISSION3B_PATH)
    labels_to_submission(lab_c, SUBMISSION3C_PATH)
    shutil.copy(SUBMISSION3C_PATH, SUBMISSION3_PATH)

    report = {
        "A_meta_col2": metrics_for_labels(lab_a, fe_a, "meta_col2_tercile"),
        "B_description_gmm3": metrics_for_labels(lab_b, fe_b, "description_gmm3"),
        "C_gmm2_unc5": metrics_for_labels(lab_c, fe_c, "gmm2_uncertain_5pct"),
        "paths": {
            "submission3": str(SUBMISSION3_PATH),
            "submission3a": str(SUBMISSION3A_PATH),
            "submission3b": str(SUBMISSION3B_PATH),
            "submission3c": str(SUBMISSION3C_PATH),
        },
        "recommended": "C",
        "kaggle_scores": {
            "submission_v1": 0.36568,
            "submission2": 0.34447,
            "submission3a": 0.29426,
            "submission3b": 0.36666,
            "submission3c": 0.44571,
        },
    }
    ARTIFACTS_V3_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_V3_DIR / "experiment_report_v3.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
