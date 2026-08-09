"""LIGHT MVP: M0 freeze, LIGHT-1 psd_front, LIGHT-2 tau_eff + honesty.

Protocol: Разработка/0808_Сценарий_light_угасание.md
  LIGHT-0  M0 champion byte-copy reference
  LIGHT-1  psd_front on (4,42), thr=valley+0.003, class2=q1.5–q98.5
  LIGHT-2  tau_eff after 40% + GMM-log valley + honesty→2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.light_decay import (  # noqa: E402
    FRONT_ALPHA,
    extract_light_features,
    labels_from_axis,
    labels_from_tau_eff,
)
from sygnal_clustering.signal_extraction import extract_prep_features, valley_ratio  # noqa: E402

DROP = [0, 1, 2, 3, 504]
N = 23_479
OFFSET, SHORT = 4, 42
DELTA = 0.003
OUT_LO, OUT_HI = 0.015, 0.985
CHAMP_SCORE = 0.85838
OUT = ROOT / "submissions" / "light"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_light_mvp.md"
REVIEW2 = ROOT / "Разработка" / "Ревью" / "0808_FE_light_2_tau.md"


def fractions(lab: np.ndarray) -> dict[str, float]:
    fr = np.bincount(lab.astype(int), minlength=3) / len(lab)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def save_sub(name: str, lab: np.ndarray, note: str, meta_extra: dict | None = None) -> Path:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "submission.csv"
    pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(path, index=False)
    # named copy for upload archive habit
    named = d / f"submission{name}.csv"
    pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(named, index=False)
    meta = {"note": note, "fractions": fractions(lab), "n": int(len(lab))}
    if meta_extra:
        meta.update(meta_extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(name, meta["fractions"], note)
    return path


def main() -> None:
    data_path = ROOT / "data" / "Run200_Wave_0_1.txt"
    raw = pd.read_csv(data_path, sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    if X.shape != (N, 500):
        raise ValueError(f"Unexpected X shape {X.shape}")

    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    light = extract_light_features(
        X,
        polarity="negative",
        psd_offset=OFFSET,
        psd_short=SHORT,
        alpha=FRONT_ALPHA,
    )

    # --- LIGHT-0: M0 champion (psd_argmax) ---
    lab0, meta0 = labels_from_axis(light.psd_argmax, delta=DELTA, out_lo=OUT_LO, out_hi=OUT_HI)
    # cross-check vs prep.psd labels
    lab0_prep, meta0_prep = labels_from_axis(prep.psd, delta=DELTA, out_lo=OUT_LO, out_hi=OUT_HI)
    if not np.array_equal(lab0, lab0_prep):
        raise ValueError("LIGHT-0 labels diverge from prep.psd champion path")

    OUT.mkdir(parents=True, exist_ok=True)
    save_sub(
        "LIGHT_0_M0",
        lab0,
        "M0 champion freeze: PSD argmax (4,42), valley+0.003, q015-985",
        {
            "step": "LIGHT-0",
            "axis": "psd_argmax",
            "champion_score_ref": CHAMP_SCORE,
            "window": [OFFSET, SHORT],
            **meta0,
            "diff_vs_prep_psd": 0,
        },
    )

    # --- LIGHT-1: psd_front, same policy ---
    lab1, meta1 = labels_from_axis(light.psd_front, delta=DELTA, out_lo=OUT_LO, out_hi=OUT_HI)
    diff = int((lab1 != lab0).sum())
    corr = float(np.corrcoef(light.psd_argmax, light.psd_front)[0, 1])
    n_no_front = int(light.qc_no_front.sum())
    n_bad_fit = int(light.qc_bad_decay_fit.sum())
    # diagnostic only — LIGHT-1 does NOT force honesty→2 (that is LIGHT-2)
    save_sub(
        "LIGHT_1_psd_front",
        lab1,
        "LIGHT-1: (L-S)/L windows from front t0 (alpha=0.2), same class2 policy",
        {
            "step": "LIGHT-1",
            "axis": "psd_front",
            "alpha": FRONT_ALPHA,
            "champion_score_ref": CHAMP_SCORE,
            "window": [OFFSET, SHORT],
            "diff_vs_M0": diff,
            "corr_psd_front_argmax": corr,
            "n_qc_no_front": n_no_front,
            "n_qc_bad_decay_fit": n_bad_fit,
            "tau_eff_finite_frac": float(np.isfinite(light.tau_eff).mean()),
            "fit_r2_median": float(np.nanmedian(light.fit_r2)),
            **meta1,
            "m0_valley_ratio": meta0["valley_ratio"],
            "m0_thr": meta0["thr"],
        },
    )

    # --- LIGHT-2: tau_eff + honesty→2 ---
    # Raw hist valley on tau is poisoned by a high-tau bump (false mode ~35).
    honest = ~light.qc_bad_decay_fit
    tau_h = light.tau_eff[honest]
    lo95, hi95 = np.quantile(tau_h, [0.01, 0.95])
    vr_clip, info_clip = valley_ratio(tau_h[(tau_h >= lo95) & (tau_h <= hi95)])
    vr_raw, info_raw = valley_ratio(light.tau_eff[np.isfinite(light.tau_eff)])

    lab2, meta2 = labels_from_tau_eff(
        light.tau_eff,
        honesty_mask=honest,
        out_lo=OUT_LO,
        out_hi=OUT_HI,
    )
    diff2 = int((lab2 != lab0).sum())
    both = (lab2 < 2) & (lab0 < 2)
    agree_m0 = float((lab2[both] == lab0[both]).mean()) if both.any() else float("nan")
    corr_tau_psd = float(
        np.corrcoef(light.tau_eff[honest], light.psd_argmax[honest])[0, 1]
    )
    save_sub(
        "LIGHT_2_tau_eff",
        lab2,
        "LIGHT-2: tau_eff after 40% (GMM-log valley) + honesty->2 + thin tau tails",
        {
            "step": "LIGHT-2",
            "axis": "tau_eff",
            "champion_score_ref": CHAMP_SCORE,
            "diff_vs_M0": diff2,
            "agree_M0_on_01": agree_m0,
            "corr_tau_psd_argmax": corr_tau_psd,
            "fit_r2_median": float(np.nanmedian(light.fit_r2)),
            "tau_eff_finite_frac": float(np.isfinite(light.tau_eff).mean()),
            "valley_raw": {
                "valley_ratio": float(vr_raw),
                "valley": info_raw.get("valley"),
                "mode1": info_raw.get("mode1"),
                "mode2": info_raw.get("mode2"),
                "note": "poisoned by high-tau bump — not used for split",
            },
            "valley_clip_p01_p95": {
                "valley_ratio": float(vr_clip),
                "valley": info_clip.get("valley"),
                "mode1": info_clip.get("mode1"),
                "mode2": info_clip.get("mode2"),
            },
            **meta2,
        },
    )

    review = {
        "title": "LIGHT MVP (0/1/2)",
        "champion_score_ref": CHAMP_SCORE,
        "LIGHT_0": {"fractions": fractions(lab0), **meta0},
        "LIGHT_1": {
            "fractions": fractions(lab1),
            "diff_vs_M0": diff,
            "corr_psd_front_argmax": corr,
            "n_qc_no_front": n_no_front,
            **meta1,
        },
        "LIGHT_2": {
            "fractions": fractions(lab2),
            "diff_vs_M0": diff2,
            "agree_M0_on_01": agree_m0,
            **meta2,
        },
        "upload_hint": "Upload LIGHT_2_tau_eff/submission.csv as upload #2 (tau hypothesis).",
        "next": "LIGHT-3 rho two-exp if tau axis is not worse diagnostically",
    }
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(
        "# FE LIGHT MVP\n\n"
        f"- Champ ref: **{CHAMP_SCORE}**\n"
        f"- LIGHT-0 M0 fractions: `{fractions(lab0)}` valley_ratio={meta0['valley_ratio']:.4f} thr={meta0['thr']:.6f}\n"
        f"- LIGHT-1 psd_front: `{fractions(lab1)}` valley_ratio={meta1['valley_ratio']:.4f} thr={meta1['thr']:.6f}\n"
        f"- LIGHT-2 tau_eff: `{fractions(lab2)}` thr={meta2['thr']:.4f} agree_M0={agree_m0:.4f}\n"
        f"- L1 diff vs M0: **{diff}**; L2 diff vs M0: **{diff2}**\n\n"
        f"```json\n{json.dumps(review, indent=2)}\n```\n",
        encoding="utf-8",
    )
    REVIEW2.write_text(
        "# FE LIGHT-2: tau_eff + honesty\n\n"
        f"- Champ ref: **{CHAMP_SCORE}**\n"
        f"- Axis: honest single-exp `tau_eff` after 40% amplitude drop (not discrete decay=3)\n"
        f"- Split: GMM-2 equal-responsibility on `log(tau)` "
        f"(means ≈ {meta2['means_lin'][0]:.2f} / {meta2['means_lin'][1]:.2f}, thr={meta2['thr']:.4f})\n"
        f"- Why not raw valley: mode2≈{info_raw.get('mode2')} is a heavy-tail bump; "
        f"valley_ratio_raw={vr_raw:.3f} is a false gate pass\n"
        f"- Clip diagnostic valley (p01–p95): {info_clip.get('valley')} "
        f"(vr={vr_clip:.3f})\n"
        f"- Class2: honesty (`qc_bad_decay_fit`) ∪ thin tau tails q[{OUT_LO},{OUT_HI}] "
        f"— not SNR bulk\n"
        f"- Polarity: large tau → 0 (LB-aligned; tau anti-corr PSD={corr_tau_psd:.3f})\n"
        f"- Fractions: `{fractions(lab2)}`; diff vs M0: **{diff2}**; "
        f"agree on 0/1: **{agree_m0:.4f}**\n"
        f"- honesty→2: {meta2['n_honesty_class2']}; tail→2: {meta2['n_tail_class2']}\n\n"
        f"Upload: `submissions/light/LIGHT_2_tau_eff/submission.csv`\n\n"
        f"```json\n{json.dumps({'LIGHT_2': review['LIGHT_2']}, indent=2)}\n```\n",
        encoding="utf-8",
    )
    print("review ->", REVIEW)
    print("review2 ->", REVIEW2)
    print("done.")


if __name__ == "__main__":
    main()
