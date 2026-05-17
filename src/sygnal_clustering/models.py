"""Clustering model implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture


@dataclass
class TwoStageConfig:
    confidence_threshold: float = 0.72
    isolation_contamination: float = 0.08
    use_isolation_forest: bool = True


def fit_two_stage_gmm(
    z_binary: np.ndarray,
    z_full: np.ndarray,
    config: TwoStageConfig | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, GaussianMixture]:
    """GMM k=2 on physical features; uncertain + outliers → cluster 2."""
    config = config or TwoStageConfig()
    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        n_init=15,
        random_state=random_state,
    )
    gmm.fit(z_binary)
    proba = gmm.predict_proba(z_binary)
    labels = gmm.predict(z_binary).astype(np.int64)
    max_proba = proba.max(axis=1)
    uncertain = max_proba < config.confidence_threshold

    if config.use_isolation_forest:
        iso = IsolationForest(
            contamination=config.isolation_contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        outlier = iso.fit_predict(z_full) == -1
    else:
        outlier = np.zeros(len(labels), dtype=bool)

    labels[uncertain | outlier] = 2
    return labels, gmm


def fit_gmm_three(z: np.ndarray, random_state: int = 42) -> tuple[np.ndarray, GaussianMixture]:
    gmm = GaussianMixture(
        n_components=3,
        covariance_type="full",
        n_init=15,
        random_state=random_state,
    )
    labels = gmm.fit_predict(z)
    return labels.astype(np.int64), gmm


def remap_labels_physics(
    labels: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    """Remap to TZ order: 0,1 = particle types (by charge), 2 = anomalies."""
    out = np.full(len(labels), 2, dtype=np.int64)
    unique = sorted(int(u) for u in np.unique(labels))
    if len(unique) == 3 and 2 not in unique:
        # End-to-end k=3: smallest cluster → anomalies
        sizes = [(u, int(np.sum(labels == u))) for u in unique]
        sizes.sort(key=lambda t: t[1])
        anomaly_id = sizes[0][0]
        particle_ids = [sizes[1][0], sizes[2][0]]
    else:
        anomaly_id = 2
        particle_ids = [u for u in unique if u != 2][:2]

    if len(particle_ids) < 2:
        return labels.astype(np.int64)

    stats = []
    for u in particle_ids:
        mask = labels == u
        stats.append((u, features[mask, 1].mean()))
    stats.sort(key=lambda t: t[1])
    mapping = {stats[0][0]: 0, stats[1][0]: 1, anomaly_id: 2}
    for old, new in mapping.items():
        out[labels == old] = new
    return out
