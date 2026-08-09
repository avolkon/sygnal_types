"""Tune low-only class2 quantile on frozen PSD 0/1 (new champ P14_2 = 0.87001)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.signal_extraction import EPS, extract_prep_features, valley_ratio  # noqa: E402

DROP = [0, 1, 2, 3, 504]
N = 23_479
OFFSET, SHORT = 4, 42
DELTA = 0.003
CHAMP_SCORE = 0.87001
BASE_QLO = 0.015
QLOS = [0.005, 0.01, 0.012, 0.015, 0.018, 0.02, 0.025, 0.03]
OUT = ROOT / "submissions" / "psd_remainder14"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_psd_qlo_tune.md"


def fractions(lab: np.ndarray) -> dict[str, float]:
    fr = np.bincount(lab.astype(int), minlength=3) / len(lab)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    psd = prep.psd
    vr, info = valley_ratio(psd, eps=EPS)
    thr = float(info["valley"]) + DELTA

    base01 = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[base01 == 0].mean() > psd[base01 == 1].mean():
        base01 = 1 - base01

    base_path = OUT / "P14_2_c2_lo_only" / "submission.csv"
    base = pd.read_csv(base_path).cluster.to_numpy()

    rows = []
    OUT.mkdir(parents=True, exist_ok=True)
    for q in QLOS:
        qlo = float(np.quantile(psd[np.isfinite(psd)], q))
        lab = base01.copy()
        lab[psd < qlo] = 2
        name = f"P14_2b_qlo_{str(q).replace('.', '')}"
        # normalize name: 0.015 -> qlo_0015
        tag = f"{int(round(q * 1000)):04d}"
        name = f"P14_2b_qlo_{tag}"
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(d / "submission.csv", index=False)
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
            d / f"submission{name}.csv", index=False
        )
        meta = {
            "note": f"lo-only class2 at q={q}; hi tail never rejected",
            "q_lo": q,
            "qlo_value": qlo,
            "thr": thr,
            "fractions": fractions(lab),
            "diff_vs_P14_2": int((lab != base).sum()),
            "n_class2": int((lab == 2).sum()),
            "champion_score_ref": CHAMP_SCORE,
            "base_q_lo": BASE_QLO,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append({"name": name, **meta})
        print(name, meta["fractions"], f"diff_vs_P14_2={meta['diff_vs_P14_2']}", f"q={q}")

    # upload order: neighbors of 0.015 first
    order = [
        "P14_2b_qlo_0020",
        "P14_2b_qlo_0010",
        "P14_2b_qlo_0025",
        "P14_2b_qlo_0012",
        "P14_2b_qlo_0018",
        "P14_2b_qlo_0030",
        "P14_2b_qlo_0005",
    ]
    (OUT / "qlo_tune_summary.json").write_text(
        json.dumps(
            {
                "new_champion_ref": CHAMP_SCORE,
                "base": "P14_2_c2_lo_only",
                "upload_order": order,
                "candidates": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    REVIEW.write_text(
        f"""# FE: tune q_lo on asymmetric class2 (base LB **{CHAMP_SCORE}**)

Контракт заморожен: PSD (4,42), valley+0.003, **hi-хвост никогда не в 2**.

| name | q_lo | n_class2 | diff vs P14_2 |
|---|---:|---:|---:|
"""
        + "\n".join(
            f"| `{r['name']}` | {r['q_lo']} | {r['n_class2']} | {r['diff_vs_P14_2']} |" for r in rows
        )
        + f"""

## Upload order

1. `P14_2b_qlo_0020` (чуть шире отказ)
2. `P14_2b_qlo_0010` (чуть уже)
3. Дальше по росту / симптомам.

Ориентир: **> {CHAMP_SCORE}**. База `q=0.015` уже залита как P14_2.
""",
        encoding="utf-8",
    )
    print("FIRST", (OUT / "P14_2b_qlo_0020" / "submission.csv").resolve())


if __name__ == "__main__":
    main()
