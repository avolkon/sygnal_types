"""One-shot probe: absolute class2 threshold psd < c (lo-only), frozen type cut.

Champ: (4,42) valley+0.003, class2 = psd < q7% -> 0.89109.
Replace quantile door with absolute c near q7 value.
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
CHAMP = 0.89109
OUT = ROOT / "submissions" / "lo_abs_c"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_lo_abs_c.md"


def fractions(lab: np.ndarray) -> dict[str, float]:
    fr = np.bincount(lab.astype(int), minlength=3) / len(lab)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def c_tag(c: float) -> str:
    return f"c{c:.4f}".replace(".", "p")


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=4, psd_short=42)
    psd = prep.psd
    vr, info = valley_ratio(psd, eps=EPS)
    thr = float(info["valley"]) + 0.003
    lab01 = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab01 == 0].mean() > psd[lab01 == 1].mean():
        lab01 = 1 - lab01
    q07 = float(np.quantile(psd, 0.07))
    base = lab01.copy()
    base[psd < q07] = 2

    freeze = ROOT / "submissions" / "psd_remainder14" / "P14_2b_qlo_0070" / "submission.csv"
    if freeze.exists():
        ref = pd.read_csv(freeze).cluster.to_numpy()
        print("base vs freeze", int((base != ref).sum()), "q07", q07, "thr", thr)

    OUT.mkdir(parents=True, exist_ok=True)
    # reference copy
    d0 = OUT / "LOABS_q07_ref"
    d0.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": np.arange(N), "cluster": base.astype(int)}).to_csv(d0 / "submission.csv", index=False)

    cs = sorted(
        {
            round(q07 + d, 4)
            for d in (-0.04, -0.03, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03)
        }
        | {0.50, 0.52, 0.54, 0.55, 0.56}
    )
    # keep c well below type thr so class0 is not wiped
    cs = [c for c in cs if c < thr - 0.02]

    rows = []
    for c in cs:
        lab = lab01.copy()
        lab[psd < c] = 2
        name = f"LOABS_{c_tag(c)}"
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(d / "submission.csv", index=False)
        pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
            d / f"submission{name}.csv", index=False
        )
        meta = {
            "c": c,
            "q07_value": q07,
            "thr": thr,
            "fractions": fractions(lab),
            "diff_vs_champ": int((lab != base).sum()),
            "n_class2": int((lab == 2).sum()),
            "champion_score_ref": CHAMP,
            "note": f"abs class2: psd < {c}; type (4,42) valley+0.003; no hi-tail",
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append({"name": name, **meta})
        print(name, meta["fractions"], "diff", meta["diff_vs_champ"])

    # upload: nearest c below/above q07 by f2, then absolute round numbers near q07
    below = sorted([r for r in rows if r["c"] < q07 - 1e-9], key=lambda r: -r["c"])
    above = sorted([r for r in rows if r["c"] > q07 + 1e-9], key=lambda r: r["c"])
    upload = []
    for seq in (below[:2], above[:2], sorted(rows, key=lambda r: abs(r["fractions"]["f2"] - 0.07))[:3]):
        for r in seq:
            if r["name"] not in upload and abs(r["c"] - q07) > 1e-9:
                upload.append(r["name"])

    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "q07_value": q07,
                "thr": thr,
                "upload_order": upload[:6],
                "first": upload[0] if upload else None,
                "candidates": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    REVIEW.write_text(
        f"""# FE: absolute lo-reject `psd < c` (one-shot family)

Champ quantile door: q7% ≈ **{q07:.4f}** → LB **{CHAMP}**.  
Probe: same type cut, class2 = `psd < c` (absolute).

## Upload order

1. `{upload[0] if upload else "n/a"}`  
2. then: {", ".join(f"`{n}`" for n in upload[1:5])}

Ориентир: **> {CHAMP}**. Если ≤ — окончательный stop на контракте lo-reject.
""",
        encoding="utf-8",
    )
    print("UPLOAD", upload[:6])
    if upload:
        print("FIRST", (OUT / upload[0] / "submission.csv").resolve())


if __name__ == "__main__":
    main()
