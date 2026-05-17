"""Domain feature extraction (PSD, amplitude, charge)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler


def extract_domain_features(
    x: np.ndarray,
    psd_offset: int = 3,
    psd_short_len: int = 30,
    decay_fraction: float = 0.40,
) -> np.ndarray:
    """Physics-motivated features from 500-bin waveforms."""
    peak = x.max(axis=1)
    charge = x.sum(axis=1)
    argmax = x.argmax(axis=1)
    n = x.shape[1]

    short = np.zeros(len(x), dtype=np.float64)
    long_gate = np.zeros(len(x), dtype=np.float64)
    decay_time = np.zeros(len(x), dtype=np.float64)

    for i in range(len(x)):
        p = int(argmax[i])
        s0 = min(n - 1, p + psd_offset)
        s1 = min(n, s0 + psd_short_len)
        short[i] = x[i, s0:s1].sum()
        long_gate[i] = charge[i]
        thr = peak[i] * (1.0 - decay_fraction)
        tail = x[i, p:]
        below = np.where(tail <= thr)[0]
        decay_time[i] = float(below[0]) if len(below) else float(n - p)

    psd = short / (long_gate + 1e-9)
    tail_ratio = np.array([x[i, int(argmax[i]) :].sum() for i in range(len(x))]) / (charge + 1e-9)
    amp_area = peak * charge
    baseline = x[:, :20].mean(axis=1)
    snr = peak / (baseline + 1e-9)

    return np.column_stack(
        [
            peak,
            charge,
            psd,
            tail_ratio,
            amp_area,
            snr,
            decay_time,
            argmax.astype(np.float64),
        ]
    )


def first_pc_amplitude_charge(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    """First principal component in (charge, peak) space — Description discriminator."""
    charge = features[:, 1:2]
    peak = features[:, 0:1]
    z = RobustScaler().fit_transform(np.hstack([charge, peak]))
    return PCA(n_components=1, random_state=random_state).fit_transform(z).ravel()


def build_clustering_matrix(
    x: np.ndarray,
    features: np.ndarray,
    pc1: np.ndarray,
    pca_components: int = 25,
    random_state: int = 42,
) -> np.ndarray:
    """Combined feature matrix for clustering."""
    scaler = RobustScaler()
    f_scaled = scaler.fit_transform(features)
    pca = PCA(n_components=pca_components, random_state=random_state)
    x_reduced = pca.fit_transform(RobustScaler().fit_transform(x))
    pc1_col = pc1.reshape(-1, 1)
    return np.hstack([f_scaled, pc1_col, x_reduced])
