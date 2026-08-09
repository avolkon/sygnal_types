"""B2: stratified rules on champion PSD recipe (4,42 / +0.003 / outlier q015-985)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.signal_extraction import (  # noqa: E402
    build_qc_flags,
    extract_prep_features,
    valley_ratio,
)

DROP = [0, 1, 2, 3, 504]
N = 23_479
OUT = ROOT / "submissions" / "fe_b2"
OFFSET, SHORT = 4, 42
DELTA = 0.003
OUT_LO, OUT_HI = 0.015, 0.985


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    qc = build_qc_flags(prep)
    _, info = valley_ratio(prep.psd)
    valley = float(info["valley"])
    thr = valley + DELTA
    psd = prep.psd

    base = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[base == 0].mean() > psd[base == 1].mean():
        base = 1 - base

    qlo, qhi = np.quantile(psd, [OUT_LO, OUT_HI])
    outlier = (psd < qlo) | (psd > qhi)

    def apply_outlier(lab: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        out = lab.copy()
        m = outlier if mask is None else (outlier & mask)
        out[m] = 2
        return out

    champ = apply_outlier(base)
    OUT.mkdir(parents=True, exist_ok=True)

    def save(name: str, lab: np.ndarray, note: str) -> None:
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
            d / "submission.csv", index=False
        )
        fr = np.bincount(lab, minlength=3) / N
        diff = int((lab != champ).sum())
        meta = {
            "note": note,
            "diff_vs_champ": diff,
            "fractions": {f"f{i}": round(float(fr[i]), 4) for i in range(3)},
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(name, meta["fractions"], f"diff={diff}", note)

    save("B2_CHAMP", champ, "reference champion copy")

    sat = qc["qc_saturated"]
    print(f"saturated rate={sat.mean():.4f} n={sat.sum()} nonsat={(~sat).sum()}")

    # --- A: different delta on saturated vs non-saturated ---
    for d_sat, d_ns, name in [
        (0.003, 0.0, "B2_delta_sat003_ns000"),
        (0.003, 0.005, "B2_delta_sat003_ns005"),
        (0.0, 0.003, "B2_delta_sat000_ns003"),
        (0.005, 0.003, "B2_delta_sat005_ns003"),
        (0.003, -0.003, "B2_delta_sat003_nsm003"),
    ]:
        thr_s = valley + d_sat
        thr_n = valley + d_ns
        lab = np.empty(N, dtype=np.int64)
        lab[sat] = np.where(psd[sat] < thr_s, 0, 1)
        lab[~sat] = np.where(psd[~sat] < thr_n, 0, 1)
        if psd[lab == 0].mean() > psd[lab == 1].mean():
            lab = 1 - lab
        save(name, apply_outlier(lab), f"thr sat={thr_s:.4f} nonsat={thr_n:.4f}")

    # --- B: nonsat keep hard labels, no outlier; sat = champion ---
    lab = champ.copy()
    lab[~sat] = base[~sat]  # remove outlier class2 for nonsat
    save("B2_nonsat_no_outlier", lab, "nonsat: pure thr; sat: champ+outlier")

    # --- C: nonsat -> class2 (rare stratum as anomalies) ---
    lab = base.copy()
    lab = apply_outlier(lab)
    lab[~sat] = 2
    save("B2_nonsat_to2", lab, "all nonsaturated -> class2")

    # --- D: stratify by SNR quartiles: different delta per quartile ---
    snr = prep.snr
    qs = np.quantile(snr, [0.25, 0.5, 0.75])
    bins = np.digitize(snr, qs)  # 0..3
    for deltas, name in [
        ([0.0, 0.003, 0.003, 0.005], "B2_snr_delta_v1"),
        ([0.005, 0.003, 0.003, 0.0], "B2_snr_delta_v2"),
        ([0.003, 0.003, 0.003, 0.003], "B2_snr_delta_flat"),  # = champ before outlier
    ]:
        lab = np.empty(N, dtype=np.int64)
        for b, dlt in enumerate(deltas):
            m = bins == b
            t = valley + dlt
            lab[m] = np.where(psd[m] < t, 0, 1)
        if psd[lab == 0].mean() > psd[lab == 1].mean():
            lab = 1 - lab
        save(name, apply_outlier(lab), f"SNR-quartile deltas={deltas}")

    # --- E: stratify by roi_length (short vs long) ---
    roi = prep.roi_length
    med_roi = float(np.median(roi))
    short_roi = roi <= med_roi
    for d_short, d_long, name in [
        (0.0, 0.003, "B2_roi_ds000_dl003"),
        (0.003, 0.0, "B2_roi_ds003_dl000"),
        (0.005, 0.003, "B2_roi_ds005_dl003"),
        (0.003, 0.005, "B2_roi_ds003_dl005"),
    ]:
        lab = np.empty(N, dtype=np.int64)
        lab[short_roi] = np.where(psd[short_roi] < valley + d_short, 0, 1)
        lab[~short_roi] = np.where(psd[~short_roi] < valley + d_long, 0, 1)
        if psd[lab == 0].mean() > psd[lab == 1].mean():
            lab = 1 - lab
        save(name, apply_outlier(lab), f"roi split med={med_roi:.1f}")

    # --- F: peak_above low vs high (amp strata) ---
    amp = prep.peak_above
    med_amp = float(np.median(amp))
    low_amp = amp <= med_amp
    for d_lo, d_hi, name in [
        (0.0, 0.003, "B2_amp_lo000_hi003"),
        (0.003, 0.0, "B2_amp_lo003_hi000"),
        (0.005, 0.003, "B2_amp_lo005_hi003"),
        (0.003, 0.005, "B2_amp_lo003_hi005"),
    ]:
        lab = np.empty(N, dtype=np.int64)
        lab[low_amp] = np.where(psd[low_amp] < valley + d_lo, 0, 1)
        lab[~low_amp] = np.where(psd[~low_amp] < valley + d_hi, 0, 1)
        if psd[lab == 0].mean() > psd[lab == 1].mean():
            lab = 1 - lab
        save(name, apply_outlier(lab), f"amp split med={med_amp:.1f}")

    # --- G: outlier only inside saturated; nonsat never class2 ---
    lab = base.copy()
    lab[outlier & sat] = 2
    save("B2_outlier_only_sat", lab, "outlier->2 only if saturated")

    # --- H: different outlier width per sat/nonsat ---
    lab = base.copy()
    for mask, lo, hi in [(sat, 0.015, 0.985), (~sat, 0.05, 0.95)]:
        ql, qh = np.quantile(psd[mask], [lo, hi]) if mask.sum() > 50 else (qlo, qhi)
        lab[mask & ((psd < ql) | (psd > qh))] = 2
    save("B2_outlier_wider_nonsat", lab, "nonsat outlier 5-95; sat 1.5-98.5")

    review = ROOT / "Разработка" / "Ревью" / "0808_FE_b2_strata.md"
    review.write_text(
        f"""# FE B2: стратификация

База-чемпион: окна (4,42), thr=valley+{DELTA}, class2=PSD∉[q{OUT_LO},q{OUT_HI}] → **0.85838**.

`qc_saturated` ≈ {sat.mean():.1%} выборки — страта «nonsat» редкая (~{(~sat).mean():.1%}).

## Первый upload

`submissions/fe_b2/B2_nonsat_no_outlier/submission.csv`

Ориентир: **> 0.85838**.

Далее: `B2_roi_ds000_dl003`, `B2_amp_lo000_hi003`, `B2_delta_sat003_ns000`, `B2_outlier_only_sat`.
""",
        encoding="utf-8",
    )
    print("FIRST", (OUT / "B2_nonsat_no_outlier" / "submission.csv").resolve())
    print("wrote", review)


if __name__ == "__main__":
    main()
