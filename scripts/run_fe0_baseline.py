"""FE-0: freeze prep constants + baseline submission (legacy GMM-2 + 5% uncertain)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.signal_extraction import (  # noqa: E402
    BASELINE_BINS,
    DECAY_FRAC,
    EPS,
    N_SIGMA,
    VALLEY_RATIO_MAX,
    build_qc_flags,
    calibrate_psd_windows,
    extract_prep_features,
    valley_ratio,
)

DATA_NAME = "Run200_Wave_0_1.txt"
N_SAMPLES = 23_479
DROP_COLS = [0, 1, 2, 3, 504]
POLARITY = "negative"
RANDOM_STATE = 42
UNCERTAIN_FRAC = 0.05
MAX_GRID_POINTS = 64
CALIB_SUBSAMPLE = 5000
PSD_OFFSET_GRID = [1, 2, 3, 5, 8, 12, 16, 24]
PSD_SHORT_GRID = [10, 15, 20, 30, 40, 50, 60, 80]

OUT_DIR = ROOT / "submissions" / "fe0_baseline"
REVIEW_PATH = ROOT / "Разработка" / "Ревью" / "0808_FE0_baseline.md"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_X() -> tuple[np.ndarray, Path]:
    candidates = [
        ROOT / "data" / DATA_NAME,
        Path("data") / DATA_NAME,
        Path("..") / "data" / DATA_NAME,
    ]
    for path in candidates:
        if path.exists():
            raw = pd.read_csv(path, sep=" ", header=None, skipinitialspace=True)
            X = raw.drop(columns=DROP_COLS, errors="ignore").to_numpy(dtype=np.float64)
            if X.shape != (N_SAMPLES, 500):
                raise ValueError(f"Unexpected X shape {X.shape}")
            return X, path.resolve()
    raise FileNotFoundError(f"{DATA_NAME} not found")


def remap_labels_physics(labels: np.ndarray, feat: np.ndarray) -> np.ndarray:
    """Class 2 = anomalies; 0/1 by mean charge (legacy notebook)."""
    out = np.full(len(labels), 2, dtype=np.int64)
    unique = sorted(int(u) for u in np.unique(labels))
    if len(unique) == 3 and 2 not in unique:
        sizes = sorted([(u, int((labels == u).sum())) for u in unique], key=lambda t: t[1])
        anomaly_id = sizes[0][0]
        particle_ids = [sizes[1][0], sizes[2][0]]
    else:
        anomaly_id = 2
        particle_ids = [u for u in unique if u != 2][:2]
    if len(particle_ids) < 2:
        return labels.astype(np.int64)
    stats = sorted([(u, feat[labels == u, 1].mean()) for u in particle_ids], key=lambda t: t[1])
    mapping = {stats[0][0]: 0, stats[1][0]: 1, anomaly_id: 2}
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def fractions(labels: np.ndarray) -> dict[str, float]:
    counts = np.bincount(labels.astype(int), minlength=3)
    fr = counts / counts.sum()
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def main() -> None:
    X, data_path = load_X()
    digest = sha256_file(data_path)
    print(f"data: {data_path}")
    print(f"sha256: {digest}")

    # rough prep for ROI (default windows), then calibrate
    prep0 = extract_prep_features(X, polarity=POLARITY, baseline_bins=BASELINE_BINS, n_sigma=N_SIGMA)
    off_best, short_best, vr_cal, info_cal = calibrate_psd_windows(
        prep0.x0,
        prep0.i_peak,
        prep0.i_end_roi,
        prep0.noise_std,
        offsets=PSD_OFFSET_GRID,
        shorts=PSD_SHORT_GRID,
        max_grid_points=MAX_GRID_POINTS,
        subsample=CALIB_SUBSAMPLE,
        random_state=RANDOM_STATE,
        eps=EPS,
    )
    prep = extract_prep_features(
        X,
        polarity=POLARITY,
        baseline_bins=BASELINE_BINS,
        n_sigma=N_SIGMA,
        decay_frac=DECAY_FRAC,
        psd_offset=off_best,
        psd_short=short_best,
        eps=EPS,
    )
    vr_psd, info_psd = valley_ratio(prep.psd, eps=EPS)
    vr_decay, info_decay = valley_ratio(prep.decay_time, eps=EPS)
    gate_ok = (vr_psd < VALLEY_RATIO_MAX) or (vr_decay < VALLEY_RATIO_MAX)
    qc = build_qc_flags(prep)

    # live feature contract (for freeze / FE-1)
    feat_live = np.column_stack(
        [prep.psd, prep.decay_time, prep.charge_roi, prep.tail_ratio]
    )
    X_clust = RobustScaler().fit_transform(feat_live)
    from sklearn.decomposition import PCA

    evr = PCA(n_components=min(3, X_clust.shape[1]), random_state=RANDOM_STATE).fit(X_clust).explained_variance_ratio_

    # baseline = notebook model-5 recipe on prep features[:, :4]
    features = np.column_stack(
        [prep.peak_above, prep.charge_roi, prep.psd, prep.decay_time, prep.snr]
    )
    z5 = RobustScaler().fit_transform(features[:, :4])
    gmm2 = GaussianMixture(
        n_components=2, covariance_type="full", n_init=20, random_state=RANDOM_STATE
    )
    gmm2.fit(z5)
    proba = gmm2.predict_proba(z5)
    labels = gmm2.predict(z5).astype(np.int64)
    uncertainty = 1.0 - proba.max(axis=1)
    n_unc = max(1, int(len(labels) * UNCERTAIN_FRAC))
    labels[np.argsort(uncertainty)[-n_unc:]] = 2
    labels = remap_labels_physics(labels, features)
    fr = fractions(labels)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"index": np.arange(N_SAMPLES), "cluster": labels.astype(int)})
    sub_path = OUT_DIR / "submission.csv"
    sub.to_csv(sub_path, index=False)

    freeze = {
        "stage": "FE-0",
        "data_path": str(data_path),
        "sha256": digest,
        "n_samples": N_SAMPLES,
        "POLARITY": POLARITY,
        "PSD_OFFSET": int(off_best),
        "PSD_SHORT": int(short_best),
        "valley_ratio_psd": float(vr_psd),
        "valley_ratio_decay": float(vr_decay),
        "GATE_OK": bool(gate_ok),
        "X_clust_cols": ["psd_calibrated", "decay_time", "charge_roi", "tail_ratio"],
        "pca_evr_live": [float(x) for x in evr],
        "candidate_class2_rate": float(qc["candidate_class2"].mean()),
        "baseline_model": "GMM-2 + uncertain_fraction=0.05 on features[:,:4]=peak,charge,psd,decay",
        "fractions": fr,
        "counts": {str(i): int((labels == i).sum()) for i in range(3)},
        "historical_kaggle_note": "pre-prep notebook reported accuracy 0.45968 with same recipe on old features",
        "submission": str(sub_path.relative_to(ROOT)),
    }
    (OUT_DIR / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")

    REVIEW_PATH.write_text(
        f"""# FE-0 baseline (EPIC-FE-0808)

| Поле | Значение |
|---|---|
| **Дата** | 09.08.2026 |
| **Submission** | `{sub_path.relative_to(ROOT).as_posix()}` |
| **Модель** | legacy notebook §9: GMM-2 + `uncertain_fraction=0.05` на prep `features[:, :4]` |
| **Цель этапа** | число «до файнтюна» + freeze prep |

## Freeze prep

| Константа | Значение |
|---|---|
| DATA | `{DATA_NAME}` |
| SHA256 | `{digest}` |
| POLARITY | `{POLARITY}` |
| PSD windows | offset={off_best}, short={short_best} |
| valley_ratio PSD / decay | {vr_psd:.4f} / {vr_decay:.4f} |
| GATE_OK | {gate_ok} |
| X_clust | psd, decay_time, charge_roi, tail_ratio |
| PCA EVR (live) | {np.round(evr, 4).tolist()} |
| candidate_class2 (QC OR) | {qc["candidate_class2"].mean():.4f} |

## Baseline labels

| Класс | Count | Fraction |
|---|---:|---:|
| 0 | {(labels == 0).sum()} | {fr["f0"]} |
| 1 | {(labels == 1).sum()} | {fr["f1"]} |
| 2 | {(labels == 2).sum()} | {fr["f2"]} |

## Запрос к заказчику (апрув / LB)

1. Залить `{sub_path.relative_to(ROOT).as_posix()}` на Kaggle.
2. Прислать **accuracy / место**.
3. Исторический ориентир в ноутбуке (до prep): **0.45968**.

После числа — продолжаем FE-1/FE-2 (live `X_clust`, без квоты 5%).
""",
        encoding="utf-8",
    )

    print(json.dumps(freeze, indent=2))
    print(f"wrote {sub_path}")
    print(f"wrote {REVIEW_PATH}")


if __name__ == "__main__":
    main()
