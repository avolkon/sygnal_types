"""Soft-zone labeling around PSD threshold (champion windows 4,42)."""

from __future__ import annotations

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
    build_qc_flags,
    extract_prep_features,
    valley_ratio,
)

DROP_COLS = [0, 1, 2, 3, 504]
N_SAMPLES = 23_479
DELTA = 0.003
OFFSET, SHORT = 4, 42
OUT = ROOT / "submissions" / "fe_soft"


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP_COLS, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(
        X,
        polarity="negative",
        baseline_bins=BASELINE_BINS,
        n_sigma=N_SIGMA,
        decay_frac=DECAY_FRAC,
        psd_offset=OFFSET,
        psd_short=SHORT,
        eps=EPS,
    )
    _, info = valley_ratio(prep.psd, eps=EPS)
    valley = float(info.get("valley", np.median(prep.psd)))
    thr = valley + DELTA
    qc = build_qc_flags(prep)["candidate_class2"]
    psd, decay, tail = prep.psd, prep.decay_time, prep.tail_ratio

    base = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[base == 0].mean() > psd[base == 1].mean():
        base = 1 - base

    dec0 = float(np.median(decay[base == 0]))
    dec1 = float(np.median(decay[base == 1]))
    tail0 = float(np.median(tail[base == 0]))
    tail1 = float(np.median(tail[base == 1]))
    print(f"thr={thr:.4f} valley={valley:.4f} dec0/1={dec0:.3f}/{dec1:.3f}")

    OUT.mkdir(parents=True, exist_ok=True)

    def save(name: str, lab: np.ndarray, meta: dict) -> None:
        out = lab.copy()
        out[qc] = 2
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N_SAMPLES), "cluster": out.astype(int)}).to_csv(
            d / "submission.csv", index=False
        )
        fr = np.bincount(out, minlength=3) / N_SAMPLES
        meta = {
            **meta,
            "fractions": {f"f{i}": round(float(fr[i]), 4) for i in range(3)},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(name, meta["fractions"], f"soft={meta.get('soft_rate', 0):.4f}", meta.get("note", ""))

    # baseline champion copy
    save("SOFT_BASE", base.copy(), {"soft_rate": 0.0, "note": "champion hard thr, no soft", "half": 0})

    halves = [0.01, 0.015, 0.02, 0.03, 0.04]
    for half in halves:
        tag = f"{half:.3f}".replace(".", "")
        soft = (psd >= thr - half) & (psd <= thr + half) & (~qc)
        soft_rate = float(soft.mean())

        lab = base.copy()
        d0 = np.abs(decay - dec0)
        d1 = np.abs(decay - dec1)
        lab[soft] = np.where(d0[soft] <= d1[soft], 0, 1)
        save(
            f"S_decay_h{tag}",
            lab,
            {"half": half, "soft_rate": soft_rate, "note": "soft: nearest median decay"},
        )

        lab = base.copy()
        t0 = np.abs(tail - tail0)
        t1 = np.abs(tail - tail1)
        lab[soft] = np.where(t0[soft] <= t1[soft], 0, 1)
        save(
            f"S_tail_h{tag}",
            lab,
            {"half": half, "soft_rate": soft_rate, "note": "soft: nearest median tail"},
        )

        lab = base.copy()
        lab[soft] = 2
        save(
            f"S_to2_h{tag}",
            lab,
            {"half": half, "soft_rate": soft_rate, "note": "soft -> class2"},
        )

    # GMM on decay+tail inside soft h=0.02
    half = 0.02
    soft = (psd >= thr - half) & (psd <= thr + half) & (~qc)
    feat = np.column_stack([decay, tail])
    z = RobustScaler().fit_transform(feat[soft])
    gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=20, random_state=42)
    gmm.fit(z)
    raw_lab = gmm.predict(z)
    comp_dec = [float(decay[soft][raw_lab == u].mean()) for u in (0, 1)]
    mapping = {
        u: (0 if abs(comp_dec[u] - dec0) <= abs(comp_dec[u] - dec1) else 1) for u in (0, 1)
    }
    lab = base.copy()
    lab[soft] = np.array([mapping[int(x)] for x in raw_lab], dtype=np.int64)
    save(
        "S_gmm_dt_h0020",
        lab,
        {
            "half": half,
            "soft_rate": float(soft.mean()),
            "note": "soft: GMM(decay,tail)",
            "mapping": mapping,
            "comp_dec": comp_dec,
        },
    )

    # decay midpoint rule
    mid = 0.5 * (dec0 + dec1)
    lab = base.copy()
    if dec0 <= dec1:
        lab[soft] = np.where(decay[soft] <= mid, 0, 1)
    else:
        lab[soft] = np.where(decay[soft] <= mid, 1, 0)
    save(
        "S_decay_mid_h0020",
        lab,
        {"half": 0.02, "soft_rate": float(soft.mean()), "note": f"soft: decay vs mid={mid:.3f}"},
    )

    # how many labels change vs base
    base_qc = base.copy()
    base_qc[qc] = 2
    print("\nDiff vs champion:")
    for p in sorted(OUT.glob("S_*/submission.csv")):
        c = pd.read_csv(p)["cluster"].to_numpy()
        diff = int((c != base_qc).sum())
        print(f"  {p.parent.name}: diff={diff}")

    review = ROOT / "Разработка" / "Ревью" / "0808_FE_softzone.md"
    review.write_text(
        f"""# FE: soft-зона на границе PSD

База: окна **(4,42)**, thr = valley+{DELTA} (= {thr:.4f}), class2=QC. Чемпион LB **0.85570**.

## Идея

Далеко от порога — как чемпион. В полосе `|PSD - thr| ≤ half` решаем по **decay/tail**.

## Первый upload

`submissions/fe_soft/S_decay_h0020/submission.csv`  
(soft half=0.02, nearest median decay)

Ориентир: **> 0.85570**.

Далее: `S_gmm_dt_h0020`, `S_tail_h0020`, `S_decay_h0015`, `S_to2_h0020`.
""",
        encoding="utf-8",
    )
    print("first:", (OUT / "S_decay_h0020" / "submission.csv").resolve())
    print("wrote", review)


if __name__ == "__main__":
    main()
