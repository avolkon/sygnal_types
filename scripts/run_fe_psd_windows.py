"""FE: PSD window search for LB — valley+DELTA + QC (champion recipe)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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

DATA_NAME = "Run200_Wave_0_1.txt"
N_SAMPLES = 23_479
DROP_COLS = [0, 1, 2, 3, 504]
POLARITY = "negative"
DELTA = 0.003  # champion offset from valley
OUT = ROOT / "submissions" / "fe_windows"

# Curated grid: champion + neighbors + diverse probes (keep upload budget small)
WINDOW_GRID = [
    (5, 50),  # champion baseline windows
    (5, 40),
    (5, 60),
    (5, 30),
    (5, 80),
    (3, 50),
    (8, 50),
    (2, 50),
    (12, 50),
    (3, 40),
    (8, 40),
    (5, 20),
    (1, 50),
    (8, 60),
    (3, 60),
]


def load_X() -> np.ndarray:
    raw = pd.read_csv(ROOT / "data" / DATA_NAME, sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP_COLS, errors="ignore").to_numpy(dtype=np.float64)
    assert X.shape == (N_SAMPLES, 500)
    return X


def labels_from_prep(prep) -> tuple[np.ndarray, dict]:
    vr, info = valley_ratio(prep.psd, eps=EPS)
    valley = float(info.get("valley", np.median(prep.psd)))
    thr = valley + DELTA
    lab = np.where(prep.psd < thr, 0, 1).astype(np.int64)
    if prep.psd[lab == 0].mean() > prep.psd[lab == 1].mean():
        lab = 1 - lab
    qc = build_qc_flags(prep)
    lab = lab.copy()
    lab[qc["candidate_class2"]] = 2
    fr = np.bincount(lab, minlength=3) / len(lab)
    meta = {
        "valley_ratio": float(vr),
        "valley": valley,
        "thr": thr,
        "delta": DELTA,
        "GATE_OK": bool(vr < 0.7),
        "fractions": {f"f{i}": round(float(fr[i]), 4) for i in range(3)},
        "n_modes": int(info.get("n_modes", -1)),
    }
    return lab, meta


def main() -> None:
    X = load_X()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for off, short in WINDOW_GRID:
        name = f"W_{off}_{short}"
        prep = extract_prep_features(
            X,
            polarity=POLARITY,
            baseline_bins=BASELINE_BINS,
            n_sigma=N_SIGMA,
            decay_frac=DECAY_FRAC,
            psd_offset=off,
            psd_short=short,
            eps=EPS,
        )
        lab, meta = labels_from_prep(prep)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N_SAMPLES), "cluster": lab.astype(int)}).to_csv(
            d / "submission.csv", index=False
        )
        meta.update({"offset": off, "short": short, "name": name})
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append(meta)
        print(name, meta["fractions"], f"vr={meta['valley_ratio']:.4f}", f"thr={meta['thr']:.4f}")

    # upload order: prefer good valley_ratio but explore neighbors of (5,50) first
    order = sorted(
        rows,
        key=lambda r: (
            0 if (r["offset"], r["short"]) == (5, 50) else 1,
            abs(r["offset"] - 5) + abs(r["short"] - 50) / 10,
            r["valley_ratio"],
        ),
    )
    summary = {
        "recipe": "thr = valley + 0.003; class2 = QC; remap low-PSD -> 0",
        "champion_ref_score": 0.85182,
        "champion_windows": [5, 50],
        "upload_order": [r["name"] for r in order if (r["offset"], r["short"]) != (5, 50)],
        "all": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    review = ROOT / "Разработка" / "Ревью" / "0808_FE_psd_windows.md"
    lines = [
        "# FE: поиск окон PSD под LB",
        "",
        f"Рецепт: `thr = valley + {DELTA}` + QC. Референс чемпиона: **(5,50) → 0.85182**.",
        "",
        "## Upload order (после (5,50))",
        "",
        "| # | Папка | offset | short | valley_ratio | f2 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(order, 1):
        if (r["offset"], r["short"]) == (5, 50):
            continue
        lines.append(
            f"| {i} | `submissions/fe_windows/{r['name']}/` | {r['offset']} | {r['short']} | {r['valley_ratio']:.4f} | {r['fractions']['f2']} |"
        )
    lines += [
        "",
        "## Развилка",
        "",
        "Заливать по порядку; прислать score. Если окно > 0.85182 — новое чемпионское; "
        "если все ≤ — стоп оси окон, обсуждаем QC.",
        "",
        f"**Первый upload:** `submissions/fe_windows/{summary['upload_order'][0]}/submission.csv`",
    ]
    review.write_text("\n".join(lines), encoding="utf-8")
    print("first:", summary["upload_order"][0])
    print("wrote", review)


if __name__ == "__main__":
    main()
