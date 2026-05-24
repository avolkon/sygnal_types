"""Legacy v2 pipeline: balanced GMM quantile."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import QuantileTransformer, RobustScaler

from sygnal_clustering.config import (
    ARTIFACTS_V2_DIR,
    RANDOM_STATE,
    SUBMISSION2_PATH,
)
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.io import write_submission
from sygnal_clustering.legacy.features import extract_domain_features, first_pc_amplitude_charge
from sygnal_clustering.metrics import cluster_sizes, clustering_scores
from sygnal_clustering.models import fit_gmm_quantile_3


class SygnalClusteringPipelineV2:
    """Archived v2: GMM-3 on quantile-normalized features + physics remap."""

    method_name = "balanced_gmm_quantile_psd"

    def __init__(
        self,
        psd_offset: int = 3,
        psd_short_len: int = 30,
        anomaly_quantile: float = 0.10,
        use_gmm_primary: bool = True,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.psd_offset = psd_offset
        self.psd_short_len = psd_short_len
        self.anomaly_quantile = anomaly_quantile
        self.use_gmm_primary = use_gmm_primary
        self.random_state = random_state
        self.labels_: np.ndarray | None = None
        self.features_: np.ndarray | None = None
        self.z_cluster_: np.ndarray | None = None
        self.balance_fractions_: dict[str, float] | None = None

    def _fit_pc1_psd_tails(self, features: np.ndarray) -> np.ndarray:
        pc1 = first_pc_amplitude_charge(features, random_state=self.random_state)
        psd = features[:, 2]
        labels = (pc1 > np.median(pc1)).astype(np.int64)

        q = self.anomaly_quantile / 2.0
        lo, hi = np.quantile(psd, [q, 1.0 - q])
        anomaly = (psd < lo) | (psd > hi)
        labels[anomaly] = 2

        particle = ~anomaly
        m0 = particle & (labels == 0)
        m1 = particle & (labels == 1)
        if m0.any() and m1.any() and features[m0, 1].mean() > features[m1, 1].mean():
            labels[m0], labels[m1] = 1, 0
        return labels

    def fit_predict(self, x: np.ndarray | None = None) -> np.ndarray:
        if x is None:
            x = load_waveforms()
        features = extract_domain_features(
            x, psd_offset=self.psd_offset, psd_short_len=self.psd_short_len
        )
        if self.use_gmm_primary:
            labels = fit_gmm_quantile_3(features, random_state=self.random_state)
            self.z_cluster_ = QuantileTransformer(
                output_distribution="normal", random_state=self.random_state
            ).fit_transform(features[:, :3])
        else:
            labels = self._fit_pc1_psd_tails(features)
            self.z_cluster_ = RobustScaler().fit_transform(features[:, :3])

        self.features_ = features
        self.labels_ = labels
        counts = np.bincount(labels, minlength=3).astype(float)
        total = counts.sum()
        self.balance_fractions_ = {
            f"fraction_{i}": float(counts[i] / total) for i in range(3)
        }
        return labels

    def metrics(self) -> dict:
        if self.labels_ is None or self.z_cluster_ is None:
            raise RuntimeError("Call fit_predict first")
        scores = clustering_scores(self.z_cluster_, self.labels_, random_state=self.random_state)
        sizes = cluster_sizes(self.labels_)
        balance = self.balance_fractions_ or {}
        max_frac = max(balance.values()) if balance else float("nan")
        return {
            **scores,
            **sizes,
            **balance,
            "max_cluster_fraction": max_frac,
            "method": self.method_name,
        }

    def save_submission(self, path: Path | None = None) -> Path:
        if self.labels_ is None:
            raise RuntimeError("Call fit_predict first")
        return write_submission(self.labels_, path or SUBMISSION2_PATH)

    def save_artifacts(self, directory: Path | None = None) -> Path:
        directory = directory or ARTIFACTS_V2_DIR
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / "pipeline_v2.joblib")
        if self.labels_ is not None:
            with open(directory / "metrics.json", "w", encoding="utf-8") as f:
                json.dump(self.metrics(), f, indent=2)
        return directory

    @classmethod
    def load(cls, path: Path | None = None) -> SygnalClusteringPipelineV2:
        path = path or ARTIFACTS_V2_DIR / "pipeline_v2.joblib"
        return joblib.load(path)


def compare_v2_methods(x: np.ndarray, random_state: int = RANDOM_STATE) -> list[dict]:
    """Compare GMM-quantile vs PC1+PSD tails for archived experiments."""
    results: list[dict] = []
    for use_gmm, name in [(True, "balanced_gmm_quantile_psd"), (False, "pc1_psd_tails")]:
        pipe = SygnalClusteringPipelineV2(use_gmm_primary=use_gmm, random_state=random_state)
        pipe.fit_predict(x)
        m = pipe.metrics()
        m["variant"] = name
        results.append(m)
    return results
