"""H_READ pipeline: freeze PSD type; calibrate unreadable-tail class2.

Base success: asymmetric lo-only PSD LB 0.89109.
This run 'fine-tunes' reject by tail readability z_tail=(L-S)/sqrt(L) and SNR
inside the low-PSD door — never rejecting the high-PSD side.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.readability import (  # noqa: E402
    CHAMP_SCORE,
    Q_LO_CHAMP,
    champ_lo_only,
    extract_read_features,
    labels_h_read,
)

DROP = [0, 1, 2, 3, 504]
N = 23_479
OUT = ROOT / "submissions" / "h_read"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_h_read.md"


def fractions(lab: np.ndarray) -> dict[str, float]:
    fr = np.bincount(lab.astype(int), minlength=3) / len(lab)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def save(name: str, lab: np.ndarray, base: np.ndarray, note: str, extra: dict) -> dict:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(d / "submission.csv", index=False)
    pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
        d / f"submission{name}.csv", index=False
    )
    meta = {
        "note": note,
        "fractions": fractions(lab),
        "diff_vs_champ": int((lab != base).sum()),
        "n_to_2": int(((lab == 2) & (base != 2)).sum()),
        "n_from_2": int(((lab != 2) & (base == 2)).sum()),
        "n_flip01": int(((lab < 2) & (base < 2) & (lab != base)).sum()),
        "champion_score_ref": CHAMP_SCORE,
        **extra,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(name, meta["fractions"], f"diff={meta['diff_vs_champ']}", note.encode("ascii", "replace").decode())
    return {"name": name, "fractions": meta["fractions"], "diff_vs_champ": meta["diff_vs_champ"], "note": note}


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    feat = extract_read_features(X)
    base = champ_lo_only(feat, q_lo=Q_LO_CHAMP)

    freeze = ROOT / "submissions" / "psd_remainder14" / "P14_2b_qlo_0070" / "submission.csv"
    if freeze.exists():
        ref = pd.read_csv(freeze).cluster.to_numpy()
        print("base vs freeze diff", int((base != ref).sum()))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(save("READ_0_CHAMP", base, base, "frozen lo-only q7% reference", {"q_lo": Q_LO_CHAMP}))

    # --- precision family: wider door, reject only if tail unreadable ---
    for q_lo, z_pct, tag in [
        (0.10, 50, "q10_z50"),
        (0.10, 40, "q10_z40"),
        (0.10, 30, "q10_z30"),
        (0.12, 40, "q12_z40"),
        (0.08, 50, "q08_z50"),
    ]:
        door = feat.psd < np.quantile(feat.psd, q_lo)
        z_cut = float(np.quantile(feat.z_tail[door], z_pct / 100.0)) if door.any() else 0.0
        lab, meta = labels_h_read(feat, q_lo=q_lo, z_cut=z_cut, s_cut=None)
        rows.append(
            save(
                f"READ_prec_{tag}",
                lab,
                base,
                f"door q={q_lo}; class2 if z_tail < p{z_pct} inside door",
                {**meta, "z_pct_in_door": z_pct},
            )
        )

    # --- rescue family: same door q7%, keep in 2 only weakest z_tail fraction ---
    door07 = feat.psd < np.quantile(feat.psd, Q_LO_CHAMP)
    for keep_pct, tag in [(40, "k40"), (50, "k50"), (60, "k60"), (70, "k70")]:
        # keep_pct = percent of door that remains class2 (weakest z)
        z_cut = float(np.quantile(feat.z_tail[door07], keep_pct / 100.0))
        lab, meta = labels_h_read(feat, q_lo=Q_LO_CHAMP, z_cut=z_cut)
        rows.append(
            save(
                f"READ_rescue07_{tag}",
                lab,
                base,
                f"q7% door; only weakest {keep_pct}% by z_tail -> 2 (rescue rest to 01)",
                {**meta, "keep_pct_of_door": keep_pct},
            )
        )

    # --- snr inside door ---
    for q_lo, s_pct, tag in [(0.10, 40, "q10_s40"), (0.07, 50, "q07_s50"), (0.10, 50, "q10_s50")]:
        door = feat.psd < np.quantile(feat.psd, q_lo)
        s_cut = float(np.quantile(feat.snr[door], s_pct / 100.0))
        lab, meta = labels_h_read(feat, q_lo=q_lo, z_cut=None, s_cut=s_cut)
        rows.append(
            save(
                f"READ_snr_{tag}",
                lab,
                base,
                f"door q={q_lo}; class2 if snr < p{s_pct} inside door",
                {**meta, "s_pct_in_door": s_pct},
            )
        )

    # --- joint z AND snr (stricter unreadability) ---
    q_lo = 0.10
    door = feat.psd < np.quantile(feat.psd, q_lo)
    z_cut = float(np.quantile(feat.z_tail[door], 0.45))
    s_cut = float(np.quantile(feat.snr[door], 0.45))
    # AND semantics: labels_h_read uses OR for weak flags — build AND manually
    from sygnal_clustering.readability import labels_type_psd

    lab = labels_type_psd(feat.psd, feat.thr)
    and_mask = door & (feat.z_tail < z_cut) & (feat.snr < s_cut)
    lab[and_mask] = 2
    # hard floor very low psd
    lab[feat.psd < np.quantile(feat.psd, 0.03)] = 2
    rows.append(
        save(
            "READ_joint_q10_z45_s45_floor03",
            lab,
            base,
            "door q10%; class2 if z AND snr weak; floor q3%",
            {
                "q_lo": q_lo,
                "z_cut": z_cut,
                "s_cut": s_cut,
                "q_lo_floor": 0.03,
                "n_class2": int((lab == 2).sum()),
            },
        )
    )

    upload = [
        "READ_prec_q10_z40",
        "READ_rescue07_k50",
        "READ_joint_q10_z45_s45_floor03",
        "READ_snr_q10_s40",
        "READ_prec_q10_z30",
    ]
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "hypothesis": "H_READ",
                "champion_score_ref": CHAMP_SCORE,
                "upload_order": upload,
                "candidates": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    table = "\n".join(
        f"| `{r['name']}` | {r['fractions']} | {r['diff_vs_champ']} |" for r in rows
    )
    REVIEW.write_text(
        f"""# FE H_READ: readability class2 on frozen PSD type

Hypothesis (consolidated): type=PSD; class2=unreadable tail inside low-PSD door via
`z_tail=(L-S)/sqrt(L)` and/or SNR. Hi-PSD never rejected.

Champ ref: **{CHAMP_SCORE}**. Audit: `0808_Вычитка_консилиум_unreadable.md`.

## Candidates

| name | fractions | diff |
|---|---|---:|
{table}

## Upload order

1. `READ_prec_q10_z40` — wider door, reject only weak z_tail  
2. `READ_rescue07_k50` — same q7% door, rescue strongest half by z_tail  
3. `READ_joint_q10_z45_s45_floor03` — joint readability + hard floor  

Ориентир: **> {CHAMP_SCORE}**.
""",
        encoding="utf-8",
    )
    print("FIRST", OUT / "READ_prec_q10_z40" / "submission.csv")


if __name__ == "__main__":
    main()
