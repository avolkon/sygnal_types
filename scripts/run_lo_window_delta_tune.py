"""A/B: retune PSD (offset, short, delta) under frozen lo-only q7% class2.

Never reject high-PSD tail. Base champ: (4,42) valley+0.003 q_lo=7% -> 0.89109.
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
Q_LO = 0.07
CHAMP_SCORE = 0.89109
CHAMP_OFF, CHAMP_SHORT, CHAMP_DELTA = 4, 42, 0.003
OUT = ROOT / "submissions" / "lo_window_delta"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_lo_window_delta.md"

# coarse grid around champion — keep upload budget sane
OFFSETS = [2, 3, 4, 5, 6, 8]
SHORTS = [30, 36, 40, 42, 44, 48, 50]
DELTAS = [0.0, 0.001, 0.002, 0.003, 0.004, 0.005]


def fractions(lab: np.ndarray) -> dict[str, float]:
    fr = np.bincount(lab.astype(int), minlength=3) / len(lab)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def labels_lo_only(psd: np.ndarray, *, delta: float) -> tuple[np.ndarray, dict]:
    vr, info = valley_ratio(psd, eps=EPS)
    if not np.isfinite(vr) or "valley" not in info:
        return np.full(len(psd), 2, dtype=np.int64), {"valley_ratio": float(vr), "ok": False}
    thr = float(info["valley"]) + float(delta)
    lab = np.where(psd < thr, 0, 1).astype(np.int64)
    if psd[lab == 0].mean() > psd[lab == 1].mean():
        lab = 1 - lab
    qlo = float(np.quantile(psd[np.isfinite(psd)], Q_LO))
    lab = lab.copy()
    lab[psd < qlo] = 2
    return lab, {
        "ok": True,
        "valley_ratio": float(vr),
        "valley": float(info["valley"]),
        "thr": thr,
        "qlo_value": qlo,
        "mode1": info.get("mode1"),
        "mode2": info.get("mode2"),
    }


def save(name: str, lab: np.ndarray, base: np.ndarray, note: str, extra: dict) -> Path:
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
        "champion_score_ref": CHAMP_SCORE,
        **extra,
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return d / "submission.csv"


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)

    # champion base
    prep_c = extract_prep_features(X, polarity="negative", psd_offset=CHAMP_OFF, psd_short=CHAMP_SHORT)
    base, meta_c = labels_lo_only(prep_c.psd, delta=CHAMP_DELTA)
    freeze = ROOT / "submissions" / "psd_remainder14" / "P14_2b_qlo_0070" / "submission.csv"
    if freeze.exists():
        ref = pd.read_csv(freeze).cluster.to_numpy()
        print("champ vs freeze diff", int((base != ref).sum()))

    OUT.mkdir(parents=True, exist_ok=True)
    save(
        "LOWD_CHAMP_4_42_d003",
        base,
        base,
        "reference (4,42) d=0.003 lo-q7%",
        {"offset": CHAMP_OFF, "short": CHAMP_SHORT, "delta": CHAMP_DELTA, **meta_c},
    )

    # cache prep per (off, short)
    cache: dict[tuple[int, int], np.ndarray] = {}
    rows = []

    def get_psd(off: int, sh: int) -> np.ndarray:
        key = (off, sh)
        if key not in cache:
            prep = extract_prep_features(X, polarity="negative", psd_offset=off, psd_short=sh)
            cache[key] = prep.psd
            print("cached", key, "valley_ratio_probe...", end=" ")
            vr, _ = valley_ratio(prep.psd, eps=EPS)
            print(f"vr={vr:.4f}")
        return cache[key]

    # Phase 1: window grid at champ delta
    for off in OFFSETS:
        for sh in SHORTS:
            if off == CHAMP_OFF and sh == CHAMP_SHORT:
                continue
            psd = get_psd(off, sh)
            lab, meta = labels_lo_only(psd, delta=CHAMP_DELTA)
            if not meta.get("ok", True):
                continue
            name = f"LOWD_W_{off}_{sh}_d003"
            save(name, lab, base, f"windows ({off},{sh}) d=0.003 lo-q7%", {"offset": off, "short": sh, "delta": CHAMP_DELTA, **meta})
            rows.append(
                {
                    "name": name,
                    "kind": "window",
                    "offset": off,
                    "short": sh,
                    "delta": CHAMP_DELTA,
                    "valley_ratio": meta["valley_ratio"],
                    "diff_vs_champ": int((lab != base).sum()),
                    "fractions": fractions(lab),
                }
            )

    # Phase 2: delta grid at champ windows
    for d in DELTAS:
        if d == CHAMP_DELTA:
            continue
        lab, meta = labels_lo_only(prep_c.psd, delta=d)
        tag = f"d{str(d).replace('.', '').replace('0', '', 1)}" if d != 0 else "d000"
        # cleaner tag
        tag = f"d{int(round(d * 1000)):03d}"
        name = f"LOWD_W_4_42_{tag}"
        save(name, lab, base, f"windows (4,42) delta={d} lo-q7%", {"offset": 4, "short": 42, "delta": d, **meta})
        rows.append(
            {
                "name": name,
                "kind": "delta",
                "offset": 4,
                "short": 42,
                "delta": d,
                "valley_ratio": meta["valley_ratio"],
                "diff_vs_champ": int((lab != base).sum()),
                "fractions": fractions(lab),
            }
        )

    # Rank by valley_ratio (lower better) among windows; pick top upload candidates by vr + moderate diff
    win_rows = [r for r in rows if r["kind"] == "window"]
    win_rows.sort(key=lambda r: (r["valley_ratio"], r["diff_vs_champ"]))
    # also joint: best windows x best deltas (small set)
    joint = []
    top_wins = win_rows[:5]
    for w in top_wins:
        psd = get_psd(w["offset"], w["short"])
        for d in [0.002, 0.003, 0.004]:
            if w["offset"] == 4 and w["short"] == 42 and d == 0.003:
                continue
            lab, meta = labels_lo_only(psd, delta=d)
            name = f"LOWD_W_{w['offset']}_{w['short']}_d{int(round(d*1000)):03d}"
            if (OUT / name).exists() and name != f"LOWD_W_{w['offset']}_{w['short']}_d003":
                # may already exist from phase1 only for d003
                pass
            save(
                name,
                lab,
                base,
                f"joint ({w['offset']},{w['short']}) d={d} lo-q7%",
                {"offset": w["offset"], "short": w["short"], "delta": d, **meta},
            )
            rec = {
                "name": name,
                "kind": "joint",
                "offset": w["offset"],
                "short": w["short"],
                "delta": d,
                "valley_ratio": meta["valley_ratio"],
                "diff_vs_champ": int((lab != base).sum()),
                "fractions": fractions(lab),
            }
            joint.append(rec)
            rows.append(rec)

    # Upload order: best valley_ratio windows first, then delta neighbors, then joint
    upload = [r["name"] for r in win_rows[:6]]
    upload += [r["name"] for r in rows if r["kind"] == "delta"]
    # unique preserve order
    seen = set()
    upload_u = []
    for n in upload + [r["name"] for r in sorted(joint, key=lambda r: r["valley_ratio"])[:4]]:
        if n not in seen:
            seen.add(n)
            upload_u.append(n)

    summary = {
        "champion_score_ref": CHAMP_SCORE,
        "policy": "lo-only q7%; never hi-tail class2",
        "champ_window": [CHAMP_OFF, CHAMP_SHORT],
        "champ_delta": CHAMP_DELTA,
        "best_valley_ratio_windows": win_rows[:10],
        "upload_order": upload_u[:12],
        "n_candidates": len(rows) + 1,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # review table top windows
    top_tbl = "\n".join(
        f"| `{r['name']}` | ({r['offset']},{r['short']}) | {r['valley_ratio']:.4f} | {r['diff_vs_champ']} | {r['fractions']} |"
        for r in win_rows[:12]
    )
    delta_tbl = "\n".join(
        f"| `{r['name']}` | {r['delta']} | {r['diff_vs_champ']} | {r['fractions']} |"
        for r in rows
        if r["kind"] == "delta"
    )
    REVIEW.write_text(
        f"""# FE: retune (offset, short, δ) under lo-only q7%

Champ: **{CHAMP_SCORE}** — windows (4,42), δ=0.003, class2=`psd<q7%` only.

## Top windows by valley_ratio (δ=0.003 fixed)

| name | window | valley_ratio | diff | fractions |
|---|---|---:|---:|---|
| `LOWD_CHAMP_4_42_d003` | (4,42) | {meta_c['valley_ratio']:.4f} | 0 | {fractions(base)} |
{top_tbl}

## Delta sweep at (4,42)

| name | δ | diff | fractions |
|---|---:|---:|---|
{delta_tbl}

## Upload order

1. Best-VR window ≠ champ (if VR better and diff>0)  
2. Neighbor δ at (4,42)  
3. Joint top-window × δ  

Ориентир: **> {CHAMP_SCORE}**. Stop family if 2× ≤ champ.

```json
{json.dumps({"upload_order": upload_u[:12], "best_windows": win_rows[:5]}, indent=2)}
```
""",
        encoding="utf-8",
    )
    print("UPLOAD", upload_u[:8])
    print("BEST_VR", win_rows[0] if win_rows else None)
    print("OUT", OUT)


if __name__ == "__main__":
    main()
