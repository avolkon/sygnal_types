"""Angel isthmus: map PSD overlap band + rare probes only inside it.

Outside soft band: frozen champion labels (PSD 4/42, valley+0.003, q015-985).
Inside soft band: visibility / shape rules — not global retune.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.signal_extraction import (  # noqa: E402
    EPS,
    extract_prep_features,
    valley_ratio,
)

DROP = [0, 1, 2, 3, 504]
N = 23_479
OUT = ROOT / "submissions" / "angel_isthmus"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_angel_isthmus.md"
MAP_PATH = OUT / "isthmus_map.json"

OFFSET, SHORT = 4, 42
DELTA = 0.003
OUT_LO, OUT_HI = 0.015, 0.985
CHAMP_SCORE = 0.85838
HALF = 0.02  # soft half-width around thr


def champion_labels(psd: np.ndarray) -> tuple[np.ndarray, float, dict]:
    vr, info = valley_ratio(psd, eps=EPS)
    thr = float(info["valley"]) + DELTA
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    qlo, qhi = np.quantile(psd[np.isfinite(psd)], [OUT_LO, OUT_HI])
    lab = lab.copy()
    lab[(psd < qlo) | (psd > qhi)] = 2
    return lab, thr, {"valley_ratio": float(vr), "valley": float(info["valley"]), "thr": thr, "qlo": float(qlo), "qhi": float(qhi)}


def summarize(mask: np.ndarray, prep, lab: np.ndarray) -> dict:
    m = mask
    if m.sum() == 0:
        return {"n": 0}
    return {
        "n": int(m.sum()),
        "psd_med": float(np.nanmedian(prep.psd[m])),
        "decay_med": float(np.nanmedian(prep.decay_time[m])),
        "charge_med": float(np.nanmedian(prep.charge_roi[m])),
        "snr_med": float(np.nanmedian(prep.snr[m])),
        "peak_med": float(np.nanmedian(prep.peak_above[m])),
        "roi_len_med": float(np.nanmedian(prep.i_end_roi[m] - prep.i_peak[m] + 1)),
        "f0": float(((lab[m] == 0).sum()) / m.sum()),
        "f1": float(((lab[m] == 1).sum()) / m.sum()),
        "f2": float(((lab[m] == 2).sum()) / m.sum()),
    }


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    champ, thr, meta_c = champion_labels(prep.psd)

    soft = np.abs(prep.psd - thr) < HALF
    # exclude already-class2 from soft surgery (keep outlier policy)
    soft_01 = soft & (champ < 2)
    roi_len = (prep.i_end_roi - prep.i_peak + 1).astype(np.float64)
    snr = prep.snr
    peak = prep.peak_above

    # visibility: weak = low SNR inside soft
    snr_q25 = float(np.nanquantile(snr[soft_01], 0.25)) if soft_01.any() else float("nan")
    snr_q50 = float(np.nanmedian(snr[soft_01])) if soft_01.any() else float("nan")
    weak = soft_01 & (snr <= snr_q25)
    mid_soft = soft_01 & (snr > snr_q25)

    # normalized tail shape proxy: mean of x_norm in bins [peak+8 : peak+24] / peak
    # (cheap, no new formula family)
    x0 = prep.x0
    i_peak = prep.i_peak
    tail_shape = np.empty(N, dtype=np.float64)
    for i in range(N):
        p = int(i_peak[i])
        pk = float(peak[i]) + EPS
        a, b = min(p + 8, x0.shape[1] - 1), min(p + 24, x0.shape[1])
        if b <= a:
            tail_shape[i] = 0.0
        else:
            tail_shape[i] = float(x0[i, a:b].mean() / pk)

    map_doc = {
        "champion_score_ref": CHAMP_SCORE,
        "thr": thr,
        "half": HALF,
        "soft_n": int(soft.sum()),
        "soft_01_n": int(soft_01.sum()),
        "snr_q25_in_soft": snr_q25,
        "snr_q50_in_soft": snr_q50,
        "regions": {
            "all": summarize(np.ones(N, dtype=bool), prep, champ),
            "soft": summarize(soft, prep, champ),
            "soft_lab0": summarize(soft_01 & (champ == 0), prep, champ),
            "soft_lab1": summarize(soft_01 & (champ == 1), prep, champ),
            "soft_weak_snr_q25": summarize(weak, prep, champ),
            "far": summarize(np.abs(prep.psd - thr) >= 0.05, prep, champ),
        },
        "note": (
            "soft_lab0 (fast@edge) is dimmer than soft_lab1; decay medians collide; "
            "probe only mutates soft_01."
        ),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(map_doc, indent=2), encoding="utf-8")

    rows: list[dict] = []

    def save(name: str, lab: np.ndarray, note: str, extra: dict | None = None) -> Path:
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        path = d / "submission.csv"
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(path, index=False)
        fr = np.bincount(lab, minlength=3) / N
        meta = {
            "note": note,
            "diff_vs_champ": int((lab != champ).sum()),
            "fractions": {f"f{i}": round(float(fr[i]), 4) for i in range(3)},
            "mutated_in_soft": int(((lab != champ) & soft).sum()),
        }
        if extra:
            meta.update(extra)
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append({"name": name, **meta})
        print(name, meta["fractions"], f"diff={meta['diff_vs_champ']}", f"soft_mut={meta['mutated_in_soft']}", note)
        return path

    save("ISTH_CHAMP", champ, "champion copy")

    # Probe 1: weak soft visibility -> class 2 (don't force 0/1)
    lab = champ.copy()
    lab[weak] = 2
    path_first = save(
        "ISTH_weak_to2",
        lab,
        f"soft&snr<=q25({snr_q25:.2f}) -> 2; else champ",
        {"snr_q25": snr_q25},
    )

    # Probe 2: weak soft currently labeled slow(1) -> flip to fast(0)
    # hypothesis: noise inflated PSD of dim events
    lab = champ.copy()
    flip = weak & (champ == 1)
    lab[flip] = 0
    save("ISTH_weak1_to0", lab, "soft weak & label1 -> 0 (noise-inflated PSD hyp)")

    # Probe 3: weak soft -> 2, and among mid_soft re-split by tail_shape median
    lab = champ.copy()
    lab[weak] = 2
    if mid_soft.sum() > 50:
        med_ts = float(np.median(tail_shape[mid_soft]))
        # higher residual tail -> slow(1), lower -> fast(0); align to champ majority
        tmp = np.where(tail_shape < med_ts, 0, 1).astype(np.int64)
        # align polarity with champ on mid_soft
        if (tmp[mid_soft] == champ[mid_soft]).mean() < 0.5:
            tmp = 1 - tmp
        lab[mid_soft] = tmp[mid_soft]
        save(
            "ISTH_weak2_mid_tailshape",
            lab,
            "weak->2; mid-soft re-label by norm tail[8:24] median split",
            {"tail_med": med_ts},
        )

    # Probe 4: soft only — ROI length split (local), keep far labels
    lab = champ.copy()
    if soft_01.sum() > 50:
        med_roi = float(np.median(roi_len[soft_01]))
        tmp = np.where(roi_len >= med_roi, 0, 1).astype(np.int64)  # longer ROI -> fast globally
        if (tmp[soft_01] == champ[soft_01]).mean() < 0.5:
            tmp = 1 - tmp
        lab[soft_01] = tmp[soft_01]
        # restore class2 outliers globally
        qlo, qhi = meta_c["qlo"], meta_c["qhi"]
        lab[(prep.psd < qlo) | (prep.psd > qhi)] = 2
        save("ISTH_soft_roi_split", lab, f"soft_01 re-split by roi_len med={med_roi:.1f}; far=champ")

    # Probe 5: narrower isthmus half=0.01, only weak->2
    soft_n = np.abs(prep.psd - thr) < 0.01
    weak_n = soft_n & (champ < 2) & (snr <= float(np.nanquantile(snr[soft_n & (champ < 2)], 0.25)))
    lab = champ.copy()
    lab[weak_n] = 2
    save("ISTH_narrow_weak_to2", lab, "half=0.01 soft; snr q25 -> 2")

    order = [
        "ISTH_weak_to2",
        "ISTH_narrow_weak_to2",
        "ISTH_weak1_to0",
        "ISTH_weak2_mid_tailshape",
        "ISTH_soft_roi_split",
    ]
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "champion_score_ref": CHAMP_SCORE,
                "upload_order": order,
                "map": str(MAP_PATH.relative_to(ROOT)),
                "candidates": [
                    {k: r[k] for k in ("name", "diff_vs_champ", "mutated_in_soft", "fractions", "note")}
                    for r in rows
                    if r["name"] != "ISTH_CHAMP"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    first = next(r for r in rows if r["name"] == "ISTH_weak_to2")
    s0 = map_doc["regions"]["soft_lab0"]
    s1 = map_doc["regions"]["soft_lab1"]
    REVIEW.write_text(
        f"""# FE angel: карта перешейка + probe (ветка `angel`)

Чемпион снаружи не трогаем: **{CHAMP_SCORE}**, окна (4,42), valley+{DELTA}, q015–985.

Перешеек: `|PSD − thr| < {HALF}` (thr={thr:.4f}).

## Карта

| Регион | n | snr_med | peak_med | charge_med | roi_len_med |
|---|---:|---:|---:|---:|---:|
| soft & label0 (быстрые у края) | {s0["n"]} | {s0["snr_med"]:.1f} | {s0["peak_med"]:.0f} | {s0["charge_med"]:.0f} | {s0["roi_len_med"]:.0f} |
| soft & label1 (медленные у края) | {s1["n"]} | {s1["snr_med"]:.1f} | {s1["peak_med"]:.0f} | {s1["charge_med"]:.0f} | {s1["roi_len_med"]:.0f} |
| soft weak SNR≤q25 | {map_doc["regions"]["soft_weak_snr_q25"]["n"]} | {map_doc["regions"]["soft_weak_snr_q25"]["snr_med"]:.1f} | — | — | — |

Полный JSON: `{MAP_PATH.relative_to(ROOT)}`.

**Вывод карты:** у края быстрые — тусклые; медленные — ярче. `decay` в перешейке бесполезен. Гипотеза probe: тусклые soft — плохая видимость формы → не насиловать 0/1.

## Первый upload

`{path_first.relative_to(ROOT)}`

| Поле | Значение |
|---|---|
| правило | soft & SNR≤q25 → **класс 2**; иначе champ |
| diff vs champ | **{first["diff_vs_champ"]}** |
| mutated in soft | {first["mutated_in_soft"]} |
| fractions | {first["fractions"]} |

Ориентир: **> {CHAMP_SCORE}**. Если ≤ — `ISTH_narrow_weak_to2`, затем stop isthmus.
""",
        encoding="utf-8",
    )
    print("MAP", MAP_PATH)
    print("FIRST", path_first.resolve())
    print("upload_order", order)


if __name__ == "__main__":
    main()
