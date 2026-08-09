"""Tune PSD-outlier quantiles for class2 (champion windows 4,42)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.signal_extraction import (  # noqa: E402
    extract_prep_features,
    valley_ratio,
)

DROP = [0, 1, 2, 3, 504]
N = 23_479
OUT = ROOT / "submissions" / "fe_qc"


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=4, psd_short=42)
    _, info = valley_ratio(prep.psd)
    thr = float(info["valley"]) + 0.003
    psd = prep.psd
    base = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[base == 0].mean() > psd[base == 1].mean():
        base = 1 - base

    OUT.mkdir(parents=True, exist_ok=True)

    def save(name: str, mask: np.ndarray, note: str) -> None:
        lab = base.copy()
        lab[mask] = 2
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
            d / "submission.csv", index=False
        )
        fr = np.bincount(lab, minlength=3) / N
        meta = {
            "note": note,
            "rate": float(mask.mean()),
            "fractions": {f"f{i}": round(float(fr[i]), 4) for i in range(3)},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(name, meta["fractions"], f"rate={meta['rate']:.4f}", note)

    pairs = [
        (0.005, 0.995, "OUT_q005_995"),
        (0.01, 0.99, "OUT_q01_99"),  # current champion recipe
        (0.015, 0.985, "OUT_q015_985"),
        (0.02, 0.98, "OUT_q02_98"),
        (0.03, 0.97, "OUT_q03_97"),
        (0.05, 0.95, "OUT_q05_95"),
        (0.01, 0.98, "OUT_q01_98"),
        (0.02, 0.99, "OUT_q02_99"),
        (0.005, 0.99, "OUT_q005_99"),
        (0.01, 0.995, "OUT_q01_995"),
    ]
    for lo, hi, name in pairs:
        qlo, qhi = np.quantile(psd, [lo, hi])
        mask = (psd < qlo) | (psd > qhi)
        save(name, mask, f"outlier quantiles ({lo},{hi}) bounds=({qlo:.4f},{qhi:.4f})")

    # one-sided: only high PSD tail / only low
    q01, q99 = np.quantile(psd, [0.01, 0.99])
    save("OUT_hi99", psd > q99, "only high PSD >q99")
    save("OUT_lo01", psd < q01, "only low PSD <q01")

    review = ROOT / "Разработка" / "Ревью" / "0808_FE_qc_ablation.md"
    review.write_text(
        """# FE B1: абляция QC + тюнинг outlier

База: окна **(4,42)**, thr=valley+0.003.

## LB

| Вариант | Score | Статус |
|---|---:|---|
| **QC_only_psd_outlier (q01–q99)** | **0.85744** | **чемпион** |
| QC_NONE | 0.85574 | |
| QC_ALL | 0.85570 | |
| QC_only_multi_peak | 0.85514 | вредит |

## Следующий upload

`submissions/fe_qc/OUT_q02_98/submission.csv` (чуть шире outlier-зона)

Ориентир: **> 0.85744**. Затем `OUT_q015_985`, `OUT_q03_97`, `OUT_hi99`.
""",
        encoding="utf-8",
    )
    print("FIRST", (OUT / "OUT_q02_98" / "submission.csv").resolve())


if __name__ == "__main__":
    main()
