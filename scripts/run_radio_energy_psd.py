"""H-radio: energy-dependent PSD threshold (branch `radio`).

Freeze champion physics:
  windows (4,42), PSD=(L-S)/L via extract_prep_features,
  class2 = PSD outside global [q1.5%, q98.5%].

Only change: thr = valley_bin + DELTA inside energy bins
(peak_above or charge_roi quantiles), not one global valley.
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
OUT = ROOT / "submissions" / "radio_energy"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_radio_energy_psd.md"

OFFSET, SHORT = 4, 42
DELTA = 0.003
OUT_LO, OUT_HI = 0.015, 0.985
CHAMP_SCORE = 0.85838
MIN_BIN = 400


def labels_from_thr(psd: np.ndarray, thr: np.ndarray) -> np.ndarray:
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    m0 = psd[lab == 0]
    m1 = psd[lab == 1]
    if len(m0) and len(m1) and m0.mean() > m1.mean():
        lab = 1 - lab
    return lab


def apply_class2(lab: np.ndarray, psd: np.ndarray) -> np.ndarray:
    out = lab.copy()
    qlo, qhi = np.quantile(psd[np.isfinite(psd)], [OUT_LO, OUT_HI])
    out[(psd < qlo) | (psd > qhi)] = 2
    return out


def bin_thresholds(psd: np.ndarray, energy: np.ndarray, n_bins: int, delta: float, global_thr: float) -> tuple[np.ndarray, list[dict]]:
    """Per-quantile energy bin: thr_i = valley_i + delta (fallback global_thr)."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(energy[np.isfinite(energy)], qs)
    # unique edges (flat energy tails)
    edges = np.unique(edges)
    if len(edges) < 3:
        thr = np.full(N, global_thr, dtype=np.float64)
        return thr, [{"bin": 0, "n": N, "thr": global_thr, "fallback": "global_flat_energy"}]

    # digitize: bin 1..len(edges)-1
    bins = np.digitize(energy, edges[1:-1], right=False)
    thr = np.empty(N, dtype=np.float64)
    meta_bins: list[dict] = []
    for b in range(len(edges) - 1):
        mask = bins == b
        n_b = int(mask.sum())
        info_b: dict = {"bin": b, "n": n_b, "e_lo": float(edges[b]), "e_hi": float(edges[b + 1])}
        if n_b < MIN_BIN:
            thr[mask] = global_thr
            info_b.update({"thr": global_thr, "fallback": "too_small", "valley_ratio": None})
        else:
            vr, info = valley_ratio(psd[mask], eps=EPS)
            valley = info.get("valley")
            if not np.isfinite(vr) or valley is None or info.get("n_modes", 0) < 2:
                thr[mask] = global_thr
                info_b.update(
                    {
                        "thr": global_thr,
                        "fallback": "unimodal_or_fail",
                        "valley_ratio": float(vr) if np.isfinite(vr) else None,
                    }
                )
            else:
                t = float(valley) + delta
                thr[mask] = t
                info_b.update({"thr": t, "valley": float(valley), "valley_ratio": float(vr), "fallback": None})
        meta_bins.append(info_b)
    return thr, meta_bins


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    psd = prep.psd

    vr_g, info_g = valley_ratio(psd, eps=EPS)
    global_thr = float(info_g["valley"]) + DELTA
    champ = apply_class2(labels_from_thr(psd, np.full(N, global_thr)), psd)

    OUT.mkdir(parents=True, exist_ok=True)
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
            "global_valley_ratio": float(vr_g),
            "global_thr": global_thr,
        }
        if extra:
            meta.update(extra)
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append({"name": name, **meta})
        print(name, meta["fractions"], f"diff={meta['diff_vs_champ']}", note)
        return path

    save("RAD_CHAMP", champ, "champion copy: global valley+delta + q015-985")

    configs = [
        ("RAD_peak_q4_valley", prep.peak_above, 4, "peak_above quartiles, thr=valley_b+delta"),
        ("RAD_peak_q3_valley", prep.peak_above, 3, "peak_above tertiles, thr=valley_b+delta"),
        ("RAD_charge_q4_valley", prep.charge_roi, 4, "charge_roi quartiles, thr=valley_b+delta"),
        ("RAD_charge_q3_valley", prep.charge_roi, 3, "charge_roi tertiles, thr=valley_b+delta"),
    ]
    # small delta variants on primary axis only
    for dname, delta in [("RAD_peak_q4_d000", 0.0), ("RAD_peak_q4_d005", 0.005), ("RAD_peak_q4_dm003", -0.003)]:
        configs.append((dname, prep.peak_above, 4, f"peak_above q4, delta={delta}"))

    first_path: Path | None = None
    for name, energy, n_bins, note in configs:
        delta = DELTA
        if name.endswith("_d000"):
            delta = 0.0
        elif name.endswith("_d005"):
            delta = 0.005
        elif name.endswith("_dm003"):
            delta = -0.003
        thr, bin_meta = bin_thresholds(psd, energy, n_bins, delta, global_thr)
        lab = apply_class2(labels_from_thr(psd, thr), psd)
        path = save(name, lab, note, {"bins": bin_meta, "n_bins_req": n_bins, "delta": delta})
        if name == "RAD_peak_q4_valley":
            first_path = path

    assert first_path is not None
    order = [
        "RAD_peak_q4_valley",
        "RAD_charge_q4_valley",
        "RAD_peak_q3_valley",
        "RAD_peak_q4_d005",
        "RAD_peak_q4_d000",
    ]
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "champion_score_ref": CHAMP_SCORE,
                "frozen": {"offset": OFFSET, "short": SHORT, "delta_default": DELTA, "outlier": [OUT_LO, OUT_HI]},
                "upload_order": order,
                "candidates": [{k: r[k] for k in ("name", "diff_vs_champ", "fractions", "note") if k in r} for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    peak_meta = next(r for r in rows if r["name"] == "RAD_peak_q4_valley")
    REVIEW.write_text(
        f"""# FE radio: энергозависимый PSD-cut (ветка `radio`)

Чемпион: **{CHAMP_SCORE}** — глобальный valley+{DELTA}, окна ({OFFSET},{SHORT}), class2 q015–985.

Гипотеза H-radio: порог 0/1 = f(энергия), не одна вертикаль на PSD.
Заморожено: формула `(L−S)/L`, окна, class2. Меняется только per-bin thr.

## Первый upload

`{first_path.relative_to(ROOT)}`

| Поле | Значение |
|---|---|
| энергия | `peak_above`, 4 квартиля |
| thr | valley_bin + {DELTA} (fallback = global) |
| diff vs champ | **{peak_meta["diff_vs_champ"]}** |
| fractions | {peak_meta["fractions"]} |

Ориентир: **> {CHAMP_SCORE}**. Если ≤ — `RAD_charge_q4_valley`, затем stop H-radio.

## LB

| Вариант | Score |
|---|---:|
| champ | **{CHAMP_SCORE}** |
| `RAD_peak_q4_valley` | *(ожидает upload)* |
""",
        encoding="utf-8",
    )
    print("FIRST", first_path.resolve())
    print("upload_order", order)


if __name__ == "__main__":
    main()
