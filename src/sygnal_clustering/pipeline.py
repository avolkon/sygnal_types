"""End-to-end clustering pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

from sygnal_clustering.config import (
    ARTIFACTS_DIR,
    DATA_PATH,
    RANDOM_STATE,
    SUBMISSION_PATH,
)
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.features import (
    build_clustering_matrix,
    extract_domain_features,
    first_pc_amplitude_charge,
)
from sygnal_clustering.metrics import cluster_sizes, clustering_scores
from sygnal_clustering.models import (
    TwoStageConfig,
    fit_gmm_three,
    fit_two_stage_gmm,
    remap_labels_physics,
)


class SygnalClusteringPipeline:
    """Production pipeline: two-stage GMM + physics remap."""

    method_name = "two_stage_gmm_robust"

    def __init__(
        self,
        psd_offset: int = 3,
        psd_short_len: int = 30,
        confidence_threshold: float = 0.72,
        isolation_contamination: float = 0.08,
        pca_components: int = 25,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.psd_offset = psd_offset
        self.psd_short_len = psd_short_len
        self.confidence_threshold = confidence_threshold
        self.isolation_contamination = isolation_contamination
        self.pca_components = pca_components
        self.random_state = random_state
        self.gmm_: GaussianMixture | None = None
        self.labels_: np.ndarray | None = None
        self.features_: np.ndarray | None = None
        self.z_full_: np.ndarray | None = None

    def fit_predict(self, x: np.ndarray | None = None) -> np.ndarray:
        if x is None:
            x = load_waveforms(DATA_PATH)
        features = extract_domain_features(
            x, psd_offset=self.psd_offset, psd_short_len=self.psd_short_len
        )
        pc1 = first_pc_amplitude_charge(features, random_state=self.random_state)
        z_full = build_clustering_matrix(
            x, features, pc1, pca_components=self.pca_components, random_state=self.random_state
        )
        z_binary = RobustScaler().fit_transform(features[:, :4])  # peak, charge, psd, tail

        config = TwoStageConfig(
            confidence_threshold=self.confidence_threshold,
            isolation_contamination=self.isolation_contamination,
        )
        labels, gmm = fit_two_stage_gmm(
            z_binary, z_full, config=config, random_state=self.random_state
        )
        labels = remap_labels_physics(labels, features)

        self.gmm_ = gmm
        self.labels_ = labels
        self.features_ = features
        self.z_full_ = z_full
        return labels

    def metrics(self) -> dict:
        if self.labels_ is None or self.z_full_ is None:
            raise RuntimeError("Call fit_predict first")
        scores = clustering_scores(self.z_full_, self.labels_, random_state=self.random_state)
        sizes = cluster_sizes(self.labels_)
        return {**scores, **sizes, "method": self.method_name}

    def save_submission(self, path: Path | None = None) -> Path:
        if self.labels_ is None:
            raise RuntimeError("Call fit_predict first")
        path = path or SUBMISSION_PATH
        df = pd.DataFrame({"index": np.arange(len(self.labels_)), "cluster": self.labels_})
        df.to_csv(path, index=False)
        return path

    def save_artifacts(self, directory: Path | None = None) -> Path:
        directory = directory or ARTIFACTS_DIR
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "pipeline.joblib")
        if self.labels_ is not None:
            with open(directory / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(self.metrics(), f, indent=2)
        return directory

    @classmethod
    def load(cls, path: Path | None = None) -> SygnalClusteringPipeline:
        path = path or ARTIFACTS_DIR / "pipeline.joblib"
        return joblib.load(path)


def compare_methods(x: np.ndarray, random_state: int = RANDOM_STATE) -> list[dict]:
    """Compare two-stage vs GMM-3 for experiment notebook."""
    features = extract_domain_features(x)
    pc1 = first_pc_amplitude_charge(features, random_state=random_state)
    z_full = build_clustering_matrix(x, features, pc1, random_state=random_state)

    results: list[dict] = []

    for conf in (0.68, 0.72, 0.76):
        pipe = SygnalClusteringPipeline(confidence_threshold=conf, random_state=random_state)
        pipe.fit_predict(x)
        m = pipe.metrics()
        m["confidence_threshold"] = conf
        m["type"] = "two_stage"
        results.append(m)

    labels3, _ = fit_gmm_three(z_full, random_state=random_state)
    labels3 = remap_labels_physics(labels3, features)
    m3 = clustering_scores(z_full, labels3, random_state=random_state)
    m3.update(cluster_sizes(labels3))
    m3["type"] = "gmm3_end_to_end"
    results.append(m3)

    return results
