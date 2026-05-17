"""Clustering quality metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def clustering_scores(
    z: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
    sample_size: int = 8000,
) -> dict[str, float]:
    """Internal validation metrics for unsupervised clustering."""
    n_clusters = len(np.unique(labels))
    if n_clusters < 2:
        return {
            "silhouette": float("nan"),
            "calinski_harabasz": float("nan"),
            "davies_bouldin": float("nan"),
        }
    n = min(sample_size, len(labels))
    return {
        "silhouette": float(
            silhouette_score(z, labels, sample_size=n, random_state=random_state)
        ),
        "calinski_harabasz": float(calinski_harabasz_score(z, labels)),
        "davies_bouldin": float(davies_bouldin_score(z, labels)),
    }


def cluster_sizes(labels: np.ndarray, n_clusters: int = 3) -> dict[str, int]:
    counts = np.bincount(labels, minlength=n_clusters)
    return {f"cluster_{i}": int(counts[i]) for i in range(n_clusters)}


def agreement_rate(a: np.ndarray, b: np.ndarray) -> float:
    """Fraction of identical labels (before permutation)."""
    return float(np.mean(a == b))


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    return float(adjusted_rand_score(a, b))
