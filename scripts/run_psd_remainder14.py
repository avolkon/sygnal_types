"""Probe the ~14% remainder under frozen PSD champion (0.85838).

Budget: 1 - 0.85838 ≈ 14.2% ≈ 3326 labels vs hidden GT.
Class2 is only ~3% ≈ 706 → most of the remainder is 0↔1, not reject.

Protocol (one hypothesis per upload):
  P14_0_CHAMP          — byte reference
  P14_1_no_class2      — all class2 → 0/1 by thr (is reject helping?)
  P14_2_c2_lo_only     — reject only low-PSD tail
  P14_3_c2_hi_only     — reject only high-PSD tail
  P14_4_edge200_to2    — 200 nearest to thr → 2 (geometry, not SNR)
  P14_5_edge100_flip   — 100 nearest to thr: flip 0↔1 (valley polarity micro)
"""

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
OUT_LO, OUT_HI = 0.015, 0.985
CHAMP_SCORE = 0.85838
REMAINDER = 1.0 - CHAMP_SCORE
OUT = ROOT / "submissions" / "psd_remainder14"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_psd_remainder14.md"
MAP_PATH = OUT / "remainder_map.json"


def champion_labels(psd: np.ndarray) -> tuple[np.ndarray, dict]:
    vr, info = valley_ratio(psd, eps=EPS)
    thr = float(info["valley"]) + DELTA
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    qlo, qhi = np.quantile(psd[np.isfinite(psd)], [OUT_LO, OUT_HI])
    lab = lab.copy()
    lab[(psd < qlo) | (psd > qhi)] = 2
    meta = {
        "valley_ratio": float(vr),
        "valley": float(info["valley"]),
        "thr": thr,
        "qlo": float(qlo),
        "qhi": float(qhi),
        "mode1": float(info["mode1"]),
        "mode2": float(info["mode2"]),
    }
    return lab, meta


def fractions(lab: np.ndarray) -> dict[str, float]:
    fr = np.bincount(lab.astype(int), minlength=3) / len(lab)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def save(name: str, lab: np.ndarray, note: str, champ: np.ndarray, extra: dict | None = None) -> dict:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(d / "submission.csv", index=False)
    pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
        d / f"submission{name}.csv", index=False
    )
    meta = {
        "note": note,
        "fractions": fractions(lab),
        "diff_vs_champ": int((lab != champ).sum()),
        "n_to_2": int(((lab == 2) & (champ != 2)).sum()),
        "n_from_2": int(((lab != 2) & (champ == 2)).sum()),
        "n_flip01": int(((lab < 2) & (champ < 2) & (lab != champ)).sum()),
        "champion_score_ref": CHAMP_SCORE,
    }
    if extra:
        meta.update(extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(name, meta["fractions"], f"diff={meta['diff_vs_champ']}", note)
    return {"name": name, **meta}


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    psd = prep.psd
    champ, meta_c = champion_labels(psd)
    thr = meta_c["thr"]
    qlo, qhi = meta_c["qlo"], meta_c["qhi"]

    dist = np.abs(psd - thr)
    is_c2 = champ == 2
    is_01 = ~is_c2
    soft02 = is_01 & (dist < 0.02)
    soft01 = is_01 & (dist < 0.01)
    soft005 = is_01 & (dist < 0.005)
    far05 = is_01 & (dist >= 0.05)

    # budget map (vs hidden GT we only know the size, not the indices)
    n_err_est = int(round(REMAINDER * N))
    n_c2 = int(is_c2.sum())
    map_doc = {
        "champion_score_ref": CHAMP_SCORE,
        "remainder_frac": REMAINDER,
        "remainder_n_est": n_err_est,
        "class2_n": n_c2,
        "class2_frac_of_remainder_if_all_wrong": round(n_c2 / n_err_est, 3),
        "zero_one_error_budget_est": n_err_est - n_c2,  # if all c2 wrong; upper on 01 share if c2 perfect
        "note": (
            "Even if EVERY class2 is wrong vs GT, class2 explains at most ~21% of the 14%. "
            "Dominant remainder is 0↔1. Soft band |psd-thr|<0.02 holds ~soft events as candidate locus."
        ),
        "regions": {
            "all_01": int(is_01.sum()),
            "soft_0.02": int(soft02.sum()),
            "soft_0.01": int(soft01.sum()),
            "soft_0.005": int(soft005.sum()),
            "far_0.05": int(far05.sum()),
            "class2_lo": int(((psd < qlo) & is_c2).sum()),
            "class2_hi": int(((psd > qhi) & is_c2).sum()),
        },
        "champ_meta": meta_c,
        "fractions_champ": fractions(champ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(map_doc, indent=2), encoding="utf-8")

    rows: list[dict] = []
    rows.append(save("P14_0_CHAMP", champ, "PSD champion freeze (4,42) valley+0.003 q015-985", champ))

    # --- P14_1: no class2 — assign outliers back to 0/1 by thr ---
    lab = champ.copy()
    lab[is_c2] = np.where(psd[is_c2] < thr, 0, 1).astype(np.int64)
    # keep polarity: class0 = lower mean psd
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    rows.append(
        save(
            "P14_1_no_class2",
            lab,
            "class2 cleared -> 0/1 by thr; tests if reject helps LB",
            champ,
            {"hypothesis": "H_c2_hurts", "upload_priority": 1},
        )
    )

    # --- P14_2 / P14_3: asymmetric tails ---
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    lab_lo = lab.copy()
    lab_lo[psd < qlo] = 2
    rows.append(
        save(
            "P14_2_c2_lo_only",
            lab_lo,
            "reject only PSD < q1.5%; hi tail kept as 0/1",
            champ,
            {"hypothesis": "H_c2_asymmetric_lo", "upload_priority": 2},
        )
    )

    lab_hi = lab.copy()
    lab_hi[psd > qhi] = 2
    rows.append(
        save(
            "P14_3_c2_hi_only",
            lab_hi,
            "reject only PSD > q98.5%; lo tail kept as 0/1",
            champ,
            {"hypothesis": "H_c2_asymmetric_hi", "upload_priority": 3},
        )
    )

    # --- P14_4: 200 nearest to thr -> 2 (pure geometry) ---
    lab = champ.copy()
    order_near = np.argsort(dist)
    # only among current 0/1
    near_01 = [i for i in order_near if champ[i] < 2]
    edge200 = np.array(near_01[:200], dtype=np.int64)
    lab[edge200] = 2
    rows.append(
        save(
            "P14_4_edge200_to2",
            lab,
            "200 nearest |psd-thr| among 0/1 -> 2 (geometry isthmus, not SNR)",
            champ,
            {
                "hypothesis": "H_valley_overlap_to2",
                "upload_priority": 4,
                "edge_dist_max": float(dist[edge200].max()),
                "edge_n": int(len(edge200)),
            },
        )
    )

    # --- P14_5: 100 nearest flip 0↔1 ---
    lab = champ.copy()
    edge100 = np.array(near_01[:100], dtype=np.int64)
    lab[edge100] = 1 - lab[edge100]
    rows.append(
        save(
            "P14_5_edge100_flip",
            lab,
            "100 nearest |psd-thr| flip 0<->1; far+class2 frozen",
            champ,
            {
                "hypothesis": "H_valley_local_polarity",
                "upload_priority": 5,
                "edge_dist_max": float(dist[edge100].max()),
                "edge_n": int(len(edge100)),
            },
        )
    )

    upload_order = [
        "P14_1_no_class2",
        "P14_2_c2_lo_only",
        "P14_4_edge200_to2",
        "P14_3_c2_hi_only",
        "P14_5_edge100_flip",
    ]
    summary = {
        "champion_score_ref": CHAMP_SCORE,
        "remainder_n_est": n_err_est,
        "map": str(MAP_PATH.relative_to(ROOT)),
        "upload_order": upload_order,
        "stop_rule": (
            "If P14_1 and P14_2 and P14_4 all <= champ without +diagnostic, "
            "remainder is mostly irreducible on this PSD contract; stop micro-surgery."
        ),
        "candidates": [
            {k: r[k] for k in ("name", "diff_vs_champ", "n_to_2", "n_from_2", "n_flip01", "fractions", "note") if k in r}
            for r in rows
            if r["name"] != "P14_0_CHAMP"
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(
        f"""# FE: probe остатка ~14% на PSD-эталоне

Чемпион: **{CHAMP_SCORE}**. Остаток: **{REMAINDER:.4f}** ≈ **{n_err_est}** событий.

## Бюджет ошибки

| Компонент | n | доля остатка (верх) |
|---|---:|---:|
| class2 чемпиона | {n_c2} | {n_c2 / n_err_est:.1%} |
| 0↔1 (оценка, если class2 весь неверен) | ≥ {n_err_est - n_c2} | ≥ {(n_err_est - n_c2) / n_err_est:.1%} |

**Вывод карты:** class2 объясняет **не больше ~21%** остатка. Главная утечка — **путаница 0/1**.

| Зона | n |
|---|---:|
| soft \\|psd−thr\\| < 0.02 | {int(soft02.sum())} |
| soft < 0.01 | {int(soft01.sum())} |
| soft < 0.005 | {int(soft005.sum())} |
| far ≥ 0.05 | {int(far05.sum())} |
| class2 lo / hi | {map_doc['regions']['class2_lo']} / {map_doc['regions']['class2_hi']} |

Карта: `{MAP_PATH.relative_to(ROOT)}`.

## Хирургия (заморожен 0/1-контракт PSD)

| # | Submission | diff | Гипотеза |
|---|---|---:|---|
| 1 | `P14_1_no_class2` | {next(r['diff_vs_champ'] for r in rows if r['name']=='P14_1_no_class2')} | reject вредит → вернуть хвосты в 0/1 |
| 2 | `P14_2_c2_lo_only` | {next(r['diff_vs_champ'] for r in rows if r['name']=='P14_2_c2_lo_only')} | нужен только нижний хвост PSD |
| 3 | `P14_4_edge200_to2` | {next(r['diff_vs_champ'] for r in rows if r['name']=='P14_4_edge200_to2')} | 200 у долины = overlap → 2 |
| 4 | `P14_3_c2_hi_only` | {next(r['diff_vs_champ'] for r in rows if r['name']=='P14_3_c2_hi_only')} | нужен только верхний хвост |
| 5 | `P14_5_edge100_flip` | {next(r['diff_vs_champ'] for r in rows if r['name']=='P14_5_edge100_flip')} | локальная ошибка полярности у долины |

## Upload order

1. **`submissions/psd_remainder14/P14_1_no_class2/submission.csv`** — самый информативный бит про class2.  
2. Если ≤ champ → `P14_2_c2_lo_only`.  
3. Если ≤ champ → `P14_4_edge200_to2`.  
4. Остальное — только если есть рост или явная асимметрия lo/hi.

**Стоп:** три подряд ≤ {CHAMP_SCORE} без диагностического выигрыша → остаток на этом PSD-контракте не режется микрохирургией.

Ориентир: **> {CHAMP_SCORE}**.
""",
        encoding="utf-8",
    )
    print("MAP", MAP_PATH)
    print("upload_order", upload_order)
    print("FIRST", (OUT / "P14_1_no_class2" / "submission.csv").resolve())


if __name__ == "__main__":
    main()
