"""Companion runner for avo_sygnal_types_8: same champion pipeline + submission assert.

Writes:
  - notebooks/submission.csv
  - submissions/notebook8/submission.csv
  - submissions/notebook8/metrics_capture.json (refresh)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.signal_extraction import EPS, extract_prep_features, valley_ratio  # noqa: E402

DROP = [0, 1, 2, 3, 504]
N = 23_479
RS = 42
OFFSET, SHORT, DELTA = 4, 42, 0.003
Q_LO = 0.07
LB = 0.89109
OUT = ROOT / "submissions" / "notebook8"
FREEZE = ROOT / "submissions" / "psd_remainder14" / "P14_2b_qlo_0070" / "submission.csv"


def fractions(lab: np.ndarray) -> dict[str, float]:
    fr = np.bincount(lab.astype(int), minlength=3) / len(lab)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def metrics(z: np.ndarray, lab: np.ndarray) -> dict[str, float]:
    n = min(8000, len(lab))
    return {
        "silhouette": float(silhouette_score(z, lab, sample_size=n, random_state=RS)),
        "calinski_harabasz": float(calinski_harabasz_score(z, lab)),
        "davies_bouldin": float(davies_bouldin_score(z, lab)),
    }


def labels_lo_only(psd: np.ndarray, *, delta: float = DELTA, q_lo: float = Q_LO) -> tuple[np.ndarray, dict]:
    vr, info = valley_ratio(psd, eps=EPS)
    thr = float(info["valley"]) + float(delta)
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    qv = float(np.quantile(psd[np.isfinite(psd)], q_lo))
    lab = lab.copy()
    lab[psd < qv] = 2
    return lab, {"valley_ratio": float(vr), "thr": thr, "valley": float(info["valley"]), "qlo_value": qv}


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    psd = prep.psd
    X_clust = RobustScaler().fit_transform(np.column_stack([prep.peak_above, prep.charge_roi, prep.psd]))

    # champion
    lab4, meta4 = labels_lo_only(psd)
    ref = pd.read_csv(FREEZE)["cluster"].to_numpy()
    diff = int((lab4 != ref).sum())
    assert diff == 0, diff

    # models for compare snapshot
    lab1 = KMeans(n_clusters=3, n_init=15, random_state=RS).fit_predict(X_clust)
    # simple remap by size+charge
    from collections import Counter

    sizes = Counter(lab1.tolist())
    anomaly = min(sizes, key=sizes.get)
    parts = [u for u in sizes if u != anomaly]
    means = sorted(parts, key=lambda u: float(prep.charge_roi[lab1 == u].mean()))
    mapping = {means[0]: 0, means[1]: 1, anomaly: 2}
    lab1 = np.array([mapping[int(x)] for x in lab1], dtype=np.int64)

    gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=20, random_state=RS)
    gmm.fit(psd.reshape(-1, 1))
    proba = gmm.predict_proba(psd.reshape(-1, 1))
    lab2 = gmm.predict(psd.reshape(-1, 1)).astype(np.int64)
    if psd[lab2 == 0].mean() > psd[lab2 == 1].mean():
        lab2 = 1 - lab2
    unc = 1.0 - proba.max(axis=1)
    lab2 = lab2.copy()
    lab2[np.argsort(unc)[-max(1, int(0.05 * N)) :]] = 2

    vr, info = valley_ratio(psd, eps=EPS)
    thr = float(info["valley"]) + DELTA
    base01 = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[base01 == 0].mean() > psd[base01 == 1].mean():
        base01 = 1 - base01
    qlo_s, qhi_s = np.quantile(psd, [0.015, 0.985])
    lab3 = base01.copy()
    lab3[(psd < qlo_s) | (psd > qhi_s)] = 2

    lab5 = lab4.copy()
    near = [i for i in np.argsort(np.abs(psd - thr)) if lab4[i] < 2][:200]
    lab5[near] = 1 - lab5[near]

    compare = {
        "1_KMeans": {**metrics(X_clust, lab1), **fractions(lab1)},
        "2_GMM2_unc5": {**metrics(X_clust, lab2), **fractions(lab2)},
        "3_symmetric": {**metrics(X_clust, lab3), **fractions(lab3), "LB_ref": 0.85838},
        "4_lo_only_CHAMP": {**metrics(X_clust, lab4), **fractions(lab4), "LB_ref": LB, **meta4},
        "5_softflip": {**metrics(X_clust, lab5), **fractions(lab5), "diff_vs_champ": 200},
    }

    tune_q = []
    for q in (0.05, 0.06, 0.07, 0.08):
        lab, meta = labels_lo_only(psd, q_lo=q)
        tune_q.append({"q_lo": q, **fractions(lab), "diff_vs_q07": int((lab != lab4).sum()), **meta})

    tune_d = []
    for d in (0.001, 0.003, 0.005):
        lab, meta = labels_lo_only(psd, delta=d)
        tune_d.append({"delta": d, **fractions(lab), "diff_vs_d003": int((lab != lab4).sum()), **meta})

    OUT.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"index": np.arange(N), "cluster": lab4.astype(int)})
    sub.to_csv(OUT / "submission.csv", index=False)
    (ROOT / "notebooks").mkdir(parents=True, exist_ok=True)
    sub.to_csv(ROOT / "notebooks" / "submission.csv", index=False)

    payload = {
        "lb": LB,
        "diff_freeze": diff,
        "fractions": fractions(lab4),
        "counts": {f"n{i}": int(c) for i, c in enumerate(np.bincount(lab4, minlength=3))},
        "evr": PCA(random_state=RS).fit(X_clust).explained_variance_ratio_.tolist(),
        "compare": compare,
        "tune_q": tune_q,
        "tune_d": tune_d,
        "meta_champ": meta4,
    }
    (OUT / "metrics_capture.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("OK diff_freeze=", diff, "fractions=", fractions(lab4))
    print("wrote", OUT / "submission.csv")
    print("wrote", ROOT / "notebooks" / "submission.csv")


if __name__ == "__main__":
    main()
