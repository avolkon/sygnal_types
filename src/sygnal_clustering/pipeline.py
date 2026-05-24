"""Production pipeline (v3): method C + Kaggle control experiments A/B/C."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from sygnal_clustering.config import (
    ARTIFACTS_V3_DIR,
    DATA_PATH,
    RANDOM_STATE,
    SUBMISSION3_PATH,
    SUBMISSION3A_PATH,
    SUBMISSION3B_PATH,
    SUBMISSION3C_PATH,
    SUBMISSION_PATH,
)
from sygnal_clustering.data import load_metadata_column, load_waveforms
from sygnal_clustering.io import save_json_report, write_submission
from sygnal_clustering.metrics import cluster_sizes, clustering_scores
from sygnal_clustering.models import fit_gmm_quantile_3, remap_labels_physics
from sygnal_clustering.signal_extraction import extract_description_features


def method_a_meta_col2_tercile(path: Path | str | None = None) -> np.ndarray:
    """Hypothesis A: class structure follows metadata column 2 tertiles."""
    col2 = load_metadata_column(2, path)
    thresholds = np.quantile(col2, [1.0 / 3.0, 2.0 / 3.0])
    return np.digitize(col2, thresholds).astype(np.int64)


def method_b_description_gmm3(
    x: np.ndarray | None = None,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Hypothesis B: Description features + GMM-3."""
    if x is None:
        x = load_waveforms()
    features = extract_description_features(x)
    labels = fit_gmm_quantile_3(features, random_state=random_state)
    return labels, features


def method_c_gmm2_low_confidence(
    x: np.ndarray | None = None,
    uncertain_fraction: float = 0.05,
    random_state: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    """Hypothesis C: GMM-2 + top uncertain_fraction → cluster 2 (production)."""
    if x is None:
        x = load_waveforms()
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
    return remap_labels_physics(labels, features), features


def labels_to_submission(labels: np.ndarray, path: Path) -> Path:
    """Alias for write_submission (backward compatibility)."""
    return write_submission(labels, path)


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
    path: Path | str | None = None,
) -> dict:
    """Generate submission3a/b/c; submission3 and submission.csv are copies of 3c."""
    data_path = path or DATA_PATH
    if x is None:
        x = load_waveforms(data_path)

    lab_a = method_a_meta_col2_tercile(data_path)
    fe_a = extract_description_features(x)
    lab_b, fe_b = method_b_description_gmm3(x)
    lab_c, fe_c = method_c_gmm2_low_confidence(x)

    write_submission(lab_a, SUBMISSION3A_PATH)
    write_submission(lab_b, SUBMISSION3B_PATH)
    write_submission(lab_c, SUBMISSION3C_PATH)
    shutil.copy(SUBMISSION3C_PATH, SUBMISSION3_PATH)
    shutil.copy(SUBMISSION3C_PATH, SUBMISSION_PATH)

    report = {
        "A_meta_col2": metrics_for_labels(lab_a, fe_a, "meta_col2_tercile"),
        "B_description_gmm3": metrics_for_labels(lab_b, fe_b, "description_gmm3"),
        "C_gmm2_unc5": metrics_for_labels(lab_c, fe_c, "gmm2_uncertain_5pct"),
        "paths": {
            "submission": str(SUBMISSION_PATH),
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
    save_json_report(report, ARTIFACTS_V3_DIR / "experiment_report_v3.json")
    return report
