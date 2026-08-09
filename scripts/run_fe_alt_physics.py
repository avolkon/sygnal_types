"""Expensive axis B: alternate PSD formulas + adaptive ROI / n_sigma / baseline.

Label recipe after computing psd:
  thr = valley(psd) + DELTA; low->0; outlier q015-985 -> 2
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
    BASELINE_BINS,
    EPS,
    apply_polarity,
    baseline_noise,
    extract_prep_features,
    to_x0,
    valley_ratio,
)

DROP = [0, 1, 2, 3, 504]
N = 23_479
OUT = ROOT / "submissions" / "fe_alt"
DELTA = 0.003
OUT_LO, OUT_HI = 0.015, 0.985
OFFSET, SHORT = 4, 42


def finalize(psd: np.ndarray, delta: float = DELTA) -> tuple[np.ndarray, dict]:
    vr, info = valley_ratio(psd, eps=EPS)
    valley = float(info.get("valley", np.median(psd)))
    thr = valley + delta
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if np.isfinite(psd[lab == 0]).any() and psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    qlo, qhi = np.quantile(psd[np.isfinite(psd)], [OUT_LO, OUT_HI])
    lab = lab.copy()
    lab[(psd < qlo) | (psd > qhi)] = 2
    meta = {"valley_ratio": float(vr), "valley": valley, "thr": thr, "n_modes": int(info.get("n_modes", -1))}
    return lab, meta


def psd_on_roi(x0: np.ndarray, i_peak: np.ndarray, i_end: np.ndarray, offset: int, short: int) -> np.ndarray:
    n = len(x0)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        p, e = int(i_peak[i]), int(i_end[i])
        roi = x0[i, p : e + 1]
        if len(roi) == 0:
            out[i] = 0.0
            continue
        long = float(roi.sum())
        s0 = min(len(roi) - 1, max(0, offset))
        s1 = min(len(roi), s0 + short)
        short_c = float(roi[s0:s1].sum())
        out[i] = (long - short_c) / (long + EPS)
    return out


def roi_end_nsigma(wave: np.ndarray, i_peak: int, noise_std: float, n_sigma: float) -> int:
    thr = n_sigma * float(noise_std)
    n = len(wave)
    for j in range(i_peak, n):
        if wave[j] <= thr:
            return j
    return n - 1


def roi_end_frac(wave: np.ndarray, i_peak: int, peak: float, frac: float) -> int:
    # end when below frac * peak
    thr = frac * peak
    n = len(wave)
    for j in range(i_peak, n):
        if wave[j] <= thr:
            return j
    return n - 1


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)

    # champion labels for diff
    prep_c = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    champ, _ = finalize(prep_c.psd)

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []

    def save(name: str, lab: np.ndarray, note: str, extra: dict | None = None) -> None:
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
            d / "submission.csv", index=False
        )
        fr = np.bincount(lab, minlength=3) / N
        diff = int((lab != champ).sum())
        meta = {"note": note, "diff_vs_champ": diff, "fractions": {f"f{i}": round(float(fr[i]), 4) for i in range(3)}}
        if extra:
            meta.update(extra)
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append({"name": name, **meta})
        print(name, meta["fractions"], f"diff={diff}", note)

    save("ALT_CHAMP", champ, "champion copy")

    # --- Formula variants on champion ROI/windows ---
    long = prep_c.charge_roi
    short_c = prep_c.short
    formulas = {
        "ALT_f_tail_over_long": (long - short_c) / (long + EPS),  # = champ psd
        "ALT_f_short_over_long": short_c / (long + EPS),
        "ALT_f_diff_over_sum": (long - short_c) / (long + short_c + EPS),
        "ALT_f_short_only_norm": short_c / (np.maximum(prep_c.peak_above, EPS)),
        "ALT_f_long_norm_peak": long / (np.maximum(prep_c.peak_above, EPS)),
        "ALT_f_log_tail": np.log1p(np.maximum(long - short_c, 0.0)) - np.log1p(np.maximum(long, 0.0)),
    }
    for name, psd in formulas.items():
        if name == "ALT_f_tail_over_long":
            continue  # identical to champ
        lab, meta = finalize(psd)
        save(name, lab, f"formula variant", meta)

    # --- n_sigma ROI variants ---
    x_pos = apply_polarity(X, "negative")
    baseline, noise = baseline_noise(x_pos, BASELINE_BINS)
    x0 = to_x0(x_pos, baseline)
    i_peak = np.argmax(x0, axis=1)

    for n_sigma in [2.0, 2.5, 3.5, 4.0, 5.0]:
        i_end = np.array(
            [roi_end_nsigma(x0[i], int(i_peak[i]), float(noise[i]), n_sigma) for i in range(N)],
            dtype=np.int64,
        )
        psd = psd_on_roi(x0, i_peak, i_end, OFFSET, SHORT)
        lab, meta = finalize(psd)
        save(f"ALT_nsig_{str(n_sigma).replace('.', '')}", lab, f"ROI n_sigma={n_sigma}", meta)

    # --- decay-fraction ROI as long gate ---
    peak = x0[np.arange(N), i_peak]
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5]:
        i_end = np.array(
            [roi_end_frac(x0[i], int(i_peak[i]), float(peak[i]), frac) for i in range(N)],
            dtype=np.int64,
        )
        psd = psd_on_roi(x0, i_peak, i_end, OFFSET, SHORT)
        lab, meta = finalize(psd)
        save(f"ALT_roifrac_{int(frac*100)}", lab, f"ROI end at {frac}*peak", meta)

    # --- fixed length ROI from peak ---
    for length in [40, 60, 80, 100, 120]:
        i_end = np.minimum(i_peak + length - 1, x0.shape[1] - 1)
        psd = psd_on_roi(x0, i_peak, i_end, OFFSET, SHORT)
        lab, meta = finalize(psd)
        save(f"ALT_fixlen_{length}", lab, f"fixed ROI length={length}", meta)

    # --- baseline bins ---
    for bb in [30, 40, 60, 80]:
        prep = extract_prep_features(
            X, polarity="negative", baseline_bins=bb, psd_offset=OFFSET, psd_short=SHORT
        )
        lab, meta = finalize(prep.psd)
        save(f"ALT_basebins_{bb}", lab, f"BASELINE_BINS={bb}", meta)

    # --- windows with n_sigma=4 (combo) ---
    i_end4 = np.array(
        [roi_end_nsigma(x0[i], int(i_peak[i]), float(noise[i]), 4.0) for i in range(N)],
        dtype=np.int64,
    )
    for off, short in [(4, 42), (4, 50), (3, 42), (5, 42), (4, 30), (6, 40)]:
        psd = psd_on_roi(x0, i_peak, i_end4, off, short)
        lab, meta = finalize(psd)
        save(f"ALT_ns4_W_{off}_{short}", lab, f"n_sigma=4 windows=({off},{short})", meta)

    # --- PSD on x_norm ---
    x_norm = x0 / (peak[:, None] + EPS)
    i_end = prep_c.i_end_roi
    psd = psd_on_roi(x_norm, i_peak, i_end, OFFSET, SHORT)
    lab, meta = finalize(psd)
    save("ALT_xnorm_psd", lab, "PSD computed on x_norm", meta)

    # upload order: moderate diff first, prefer formula/nsigma novelty
    cand = [r for r in rows if r["name"] != "ALT_CHAMP" and 100 <= r["diff_vs_champ"] <= 8000]
    cand.sort(key=lambda r: (0 if r["name"].startswith("ALT_f_") or "nsig" in r["name"] or "roifrac" in r["name"] else 1, -r["diff_vs_champ"]))
    if not cand:
        cand = [r for r in rows if r["name"] != "ALT_CHAMP"]
        cand.sort(key=lambda r: -r["diff_vs_champ"])

    order = [r["name"] for r in cand[:10]]
    (OUT / "summary.json").write_text(
        json.dumps({"upload_order": order, "champion_ref": 0.85838}, indent=2), encoding="utf-8"
    )

    # Prefer first: short/long formula (classic alt), then nsig 4, then roifrac 20
    preferred = [
        "ALT_f_short_over_long",
        "ALT_f_diff_over_sum",
        "ALT_nsig_40",
        "ALT_nsig_25",
        "ALT_roifrac_20",
        "ALT_fixlen_80",
        "ALT_xnorm_psd",
        "ALT_ns4_W_4_50",
    ]
    first = next((p for p in preferred if (OUT / p / "submission.csv").exists()), order[0])

    review = ROOT / "Разработка" / "Ревью" / "0808_FE_alt_physics.md"
    review.write_text(
        f"""# FE ось B: альтернативная физика PSD / ROI

Чемпион-референс: **0.85838** (3σ ROI, (long−short)/long, окна 4/42, q015–985).

## Гипотезы
1. Другая формула PSD (`short/long`, `(L−S)/(L+S)`, …)
2. Другой ROI (`n_sigma≠3`, спад до доли пика, фиксированная длина)
3. PSD на `x_norm`, другие baseline bins

Без гарантии 0.88.

## Первый upload

`submissions/fe_alt/{first}/submission.csv`

Ориентир: **> 0.85838**.
""",
        encoding="utf-8",
    )
    print("FIRST", (OUT / first / "submission.csv").resolve())
    print("preferred_available", [p for p in preferred if (OUT / p).exists()])


if __name__ == "__main__":
    main()
