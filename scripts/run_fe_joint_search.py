"""Joint-ish search: windows + thr delta + outlier rate under one recipe.

Cannot score LB locally — emit diverse candidates near champion for upload.
Champion ref: (4,42), delta=+0.003, outlier q015-985 → 0.85838.
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
    extract_prep_features,
    valley_ratio,
)

DROP = [0, 1, 2, 3, 504]
N = 23_479
OUT = ROOT / "submissions" / "fe_joint"


def labels_for(psd: np.ndarray, valley: float, delta: float, out_lo: float, out_hi: float) -> np.ndarray:
    thr = valley + delta
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    qlo, qhi = np.quantile(psd, [out_lo, out_hi])
    lab = lab.copy()
    lab[(psd < qlo) | (psd > qhi)] = 2
    return lab


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)

    # champion reference labels
    prep_c = extract_prep_features(X, polarity="negative", psd_offset=4, psd_short=42)
    _, info_c = valley_ratio(prep_c.psd)
    valley_c = float(info_c["valley"])
    champ = labels_for(prep_c.psd, valley_c, 0.003, 0.015, 0.985)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def save(name: str, lab: np.ndarray, meta: dict) -> None:
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
            d / "submission.csv", index=False
        )
        fr = np.bincount(lab, minlength=3) / N
        diff = int((lab != champ).sum())
        meta = {
            **meta,
            "diff_vs_champ": diff,
            "fractions": {f"f{i}": round(float(fr[i]), 4) for i in range(3)},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append({"name": name, **meta})
        print(name, meta["fractions"], f"diff={diff}", meta.get("note", ""))

    save(
        "J_CHAMP",
        champ,
        {"note": "champion copy", "offset": 4, "short": 42, "delta": 0.003, "out": [0.015, 0.985]},
    )

    # 1) window grid with FIXED champion class2/delta recipe
    window_grid = [
        (4, 42),
        (4, 40),
        (4, 41),
        (4, 43),
        (4, 44),
        (4, 38),
        (3, 42),
        (3, 40),
        (3, 44),
        (5, 42),
        (5, 40),
        (5, 44),
        (2, 42),
        (6, 42),
        (4, 36),
        (4, 48),
        (4, 50),
        (3, 38),
        (5, 38),
        (4, 46),
    ]
    for off, short in window_grid:
        if (off, short) == (4, 42):
            continue
        prep = extract_prep_features(X, polarity="negative", psd_offset=off, psd_short=short)
        _, info = valley_ratio(prep.psd)
        valley = float(info.get("valley", np.median(prep.psd)))
        lab = labels_for(prep.psd, valley, 0.003, 0.015, 0.985)
        vr = float(valley_ratio(prep.psd)[0])
        save(
            f"J_W_{off}_{short}",
            lab,
            {
                "note": "windows retune under champ outlier/delta",
                "offset": off,
                "short": short,
                "delta": 0.003,
                "out": [0.015, 0.985],
                "valley_ratio": vr,
                "valley": valley,
            },
        )

    # 2) on champ windows: fine delta × fine outlier
    for delta in [0.001, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005]:
        for out_lo, out_hi, tag in [
            (0.012, 0.988, "012_988"),
            (0.015, 0.985, "015_985"),
            (0.017, 0.983, "017_983"),
            (0.02, 0.98, "02_98"),
        ]:
            if abs(delta - 0.003) < 1e-12 and (out_lo, out_hi) == (0.015, 0.985):
                continue
            lab = labels_for(prep_c.psd, valley_c, delta, out_lo, out_hi)
            name = f"J_D{str(delta).replace('.', '')}_O{tag}"
            save(
                name,
                lab,
                {
                    "note": "delta×outlier on (4,42)",
                    "offset": 4,
                    "short": 42,
                    "delta": delta,
                    "out": [out_lo, out_hi],
                },
            )

    # 3) absolute thr grid (ignore valley) + champ outlier
    for thr in np.arange(0.62, 0.66, 0.002):
        lab = np.where(prep_c.psd < thr, 0, 1).astype(np.int64)
        if prep_c.psd[lab == 0].mean() > prep_c.psd[lab == 1].mean():
            lab = 1 - lab
        qlo, qhi = np.quantile(prep_c.psd, [0.015, 0.985])
        lab = lab.copy()
        lab[(prep_c.psd < qlo) | (prep_c.psd > qhi)] = 2
        save(
            f"J_THR_{int(round(thr * 1000))}",
            lab,
            {"note": f"absolute thr={thr:.3f}", "offset": 4, "short": 42, "thr": float(thr)},
        )

    # 4) alternate PSD: short/long (diagnostic formula) with same pipeline shape
    short_gate = prep_c.short
    long_gate = prep_c.charge_roi
    psd_alt = short_gate / (long_gate + 1e-9)
    # bimodal? use median split + outlier
    thr_alt = float(np.median(psd_alt))
    lab = np.where(psd_alt < thr_alt, 0, 1).astype(np.int64)
    # map: lower short/long often = more tail = class? keep low->0 convention on THIS feature
    qlo, qhi = np.quantile(psd_alt, [0.015, 0.985])
    lab = lab.copy()
    lab[(psd_alt < qlo) | (psd_alt > qhi)] = 2
    save(
        "J_ALT_shortlong_med",
        lab,
        {"note": "ALT formula short/long median split + q015-985", "formula": "short/long"},
    )

    # rank upload order: prefer moderate diff 80..1200, closer windows first
    cand = [r for r in rows if r["name"] != "J_CHAMP"]
    cand.sort(
        key=lambda r: (
            0 if 80 <= r["diff_vs_champ"] <= 1200 else 1,
            abs(r.get("offset", 4) - 4) + abs(r.get("short", 42) - 42) / 10
            if "offset" in r
            else 9,
            -r["diff_vs_champ"] if 80 <= r["diff_vs_champ"] <= 1200 else r["diff_vs_champ"],
        )
    )

    summary = {
        "champion_score_ref": 0.85838,
        "gap_to_088": 0.02162,
        "gap_to_089": 0.03162,
        "upload_order": [r["name"] for r in cand[:12]],
        "n_candidates": len(cand),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    review = ROOT / "Разработка" / "Ревью" / "0808_FE_path_88.md"
    lines = [
        "# Путь к 0.88–0.89 (оценка + план)",
        "",
        "## Вижу ли возможность?",
        "",
        "**Да, ограниченно.** Нужно +0.022…0.032 к **0.85838** ≈ 500–750 верных перестановок меток.",
        "",
        "| Оценка | Комментарий |",
        "|---|---|",
        "| **0.86–0.87** | реалистично при удачном joint retune окон под новый class2 |",
        "| **0.88** | возможен, если окна+thr+outlier ещё не в глобальном максимуме |",
        "| **0.89** | труднее; нужен новый систематический эффект |",
        "| **0.90–0.91** | маловероятно на текущей одной оси PSD без новой физики |",
        "",
        "Почему ещё есть шанс на 0.88: окна (4,42) искали при **старом** QC; class2 теперь **q015–985** — сетку окон надо пересчитать.",
        "",
        "Что уже выжато / вредит: soft decay/tail, асимметрия outlier, B2 wider-nonsat, multi_peak.",
        "",
        "## Первый upload",
        "",
        f"`submissions/fe_joint/{summary['upload_order'][0]}/submission.csv`",
        "",
        "Ориентир: **> 0.85838**. Топ очереди:",
        "",
    ]
    for i, name in enumerate(summary["upload_order"][:8], 1):
        meta = next(r for r in rows if r["name"] == name)
        lines.append(f"{i}. `{name}` — diff={meta['diff_vs_champ']} — {meta.get('note','')}")
    review.write_text("\n".join(lines), encoding="utf-8")
    print("upload_order", summary["upload_order"][:8])
    print("FIRST", (OUT / summary["upload_order"][0] / "submission.csv").resolve())


if __name__ == "__main__":
    main()
