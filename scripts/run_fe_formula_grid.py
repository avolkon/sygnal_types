"""Coarse PSD formula grid + local proxy ranking for rare LB uploads.

Hypothesis: organizers' gate may align with an alternate PSD definition
(e.g. (L-S)/(L+S)) rather than micro-tuning of (L-S)/L.

Exports only top-ranked candidates; full grid stays in grid_summary.json.
Champion ref: windows (4,42), thr=valley+0.003, outlier q015-985 -> LB 0.85838.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
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
OUT = ROOT / "submissions" / "fe_formula_grid"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_formula_grid.md"

DELTA = 0.003
OUT_LO, OUT_HI = 0.015, 0.985
OFFSET, SHORT = 4, 42
CHAMP_SCORE = 0.85838
MAX_EXPORT = 8  # keep upload budget small


@dataclass(frozen=True)
class Recipe:
    offset: int
    short: int
    delta: float = DELTA


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
    meta = {
        "valley_ratio": float(vr),
        "valley": valley,
        "thr": thr,
        "n_modes": int(info.get("n_modes", -1)),
    }
    return lab, meta


def psd_on_roi(
    x0: np.ndarray,
    i_peak: np.ndarray,
    i_end: np.ndarray,
    offset: int,
    short: int,
    formula: str,
    peak: np.ndarray | None = None,
) -> np.ndarray:
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
        tail = long - short_c
        pk = float(peak[i]) if peak is not None else float(np.max(roi))

        if formula == "tail_over_long":
            out[i] = tail / (long + EPS)
        elif formula == "diff_over_sum":
            out[i] = tail / (long + short_c + EPS)
        elif formula == "short_over_long":
            out[i] = short_c / (long + EPS)
        elif formula == "short_over_sum":
            out[i] = short_c / (long + short_c + EPS)
        elif formula == "tail_over_peak":
            out[i] = tail / (pk + EPS)
        elif formula == "short_over_peak":
            out[i] = short_c / (pk + EPS)
        elif formula == "log_tail_over_long":
            out[i] = np.log1p(max(tail, 0.0)) - np.log1p(max(long, 0.0))
        elif formula == "log_diff_over_sum":
            out[i] = np.log1p(max(tail, 0.0)) - np.log1p(max(long + short_c, 0.0))
        elif formula == "sqrt_tail_over_long":
            out[i] = np.sqrt(max(tail, 0.0)) / (np.sqrt(max(long, 0.0)) + EPS)
        elif formula == "tail_minus_short_over_long":
            out[i] = (tail - short_c) / (long + EPS)
        elif formula == "long_minus_2short_over_long":
            out[i] = (long - 2.0 * short_c) / (long + EPS)
        else:
            raise ValueError(formula)
    return out


def proxy_score(row: dict, champ_diff: int) -> float:
    """Higher = more interesting for upload. No LB available locally."""
    vr = row["valley_ratio"]
    diff = row["diff_vs_champ"]
    sep = row["mode_sep"]

    # bimodal PSD preferred; penalize flat / noisy histograms
    vr_term = max(0.0, 0.35 - vr) * 4.0

    # moderate relabeling: enough signal, not random rewrite
    if diff < 20:
        diff_term = -0.5
    elif diff <= 250:
        diff_term = 1.0
    elif diff <= 1200:
        diff_term = 0.6
    elif diff <= 4000:
        diff_term = 0.2
    else:
        diff_term = -0.3

    sep_term = min(sep, 0.25) * 2.0
    formula_bonus = 0.15 if row["formula"] in {"diff_over_sum", "log_diff_over_sum", "short_over_sum"} else 0.0
    return vr_term + diff_term + sep_term + formula_bonus


def mode_sep(psd: np.ndarray, lab: np.ndarray) -> float:
    m0 = psd[(lab == 0) & np.isfinite(psd)]
    m1 = psd[(lab == 1) & np.isfinite(psd)]
    if len(m0) < 10 or len(m1) < 10:
        return 0.0
    return float(abs(m1.mean() - m0.mean()) / (m0.std() + m1.std() + EPS))


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)

    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)
    champ, champ_meta = finalize(prep.psd)
    peak = prep.peak_above

    formulas = [
        "tail_over_long",  # champion baseline
        "diff_over_sum",
        "short_over_long",
        "short_over_sum",
        "tail_over_peak",
        "short_over_peak",
        "log_tail_over_long",
        "log_diff_over_sum",
        "sqrt_tail_over_long",
        "tail_minus_short_over_long",
        "long_minus_2short_over_long",
    ]
    recipes = [
        Recipe(4, 42),
        Recipe(4, 40),
        Recipe(4, 44),
        Recipe(3, 42),
        Recipe(5, 42),
        Recipe(4, 50),
    ]

    rows: list[dict] = []
    for recipe in recipes:
        psd = psd_on_roi(
            prep.x0,
            prep.i_peak,
            prep.i_end_roi,
            recipe.offset,
            recipe.short,
            "tail_over_long",
            peak=peak,
        )
        for formula in formulas:
            if formula == "tail_over_long" and recipe.offset == OFFSET and recipe.short == SHORT:
                psd_f = prep.psd
            else:
                psd_f = psd_on_roi(
                    prep.x0,
                    prep.i_peak,
                    prep.i_end_roi,
                    recipe.offset,
                    recipe.short,
                    formula,
                    peak=peak,
                )
            lab, meta = finalize(psd_f, delta=recipe.delta)
            diff = int((lab != champ).sum())
            sep = mode_sep(psd_f, lab)
            name = f"FG_W{recipe.offset}_{recipe.short}_{formula}"
            row = {
                "name": name,
                "formula": formula,
                "offset": recipe.offset,
                "short": recipe.short,
                "delta": recipe.delta,
                "diff_vs_champ": diff,
                "mode_sep": sep,
                "fractions": {f"f{i}": round(float(np.bincount(lab, minlength=3)[i] / N), 4) for i in range(3)},
                **meta,
            }
            row["proxy_score"] = round(proxy_score(row, diff), 4)
            rows.append({"row": row, "labels": lab})

    rows.sort(key=lambda item: item["row"]["proxy_score"], reverse=True)

    OUT.mkdir(parents=True, exist_ok=True)
    export_dir = OUT / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    seen_formula: set[str] = set()
    for item in rows:
        row = item["row"]
        if row["name"].endswith("_tail_over_long") and row["offset"] == OFFSET and row["short"] == SHORT:
            continue
        if row["formula"] in seen_formula and row["formula"] != "diff_over_sum":
            continue
        if len(exported) >= MAX_EXPORT:
            break
        d = export_dir / row["name"]
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N), "cluster": item["labels"].astype(int)}).to_csv(
            d / "submission.csv", index=False
        )
        (d / "meta.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        exported.append(row["name"])
        seen_formula.add(row["formula"])

    # ensure diff_over_sum on champion windows is always first export
    priority = f"FG_W{OFFSET}_{SHORT}_diff_over_sum"
    if priority not in exported:
        item = next(x for x in rows if x["row"]["name"] == priority)
        d = export_dir / priority
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"index": np.arange(N), "cluster": item["labels"].astype(int)}).to_csv(
            d / "submission.csv", index=False
        )
        (d / "meta.json").write_text(json.dumps(item["row"], indent=2), encoding="utf-8")
        exported = [priority] + [n for n in exported if n != priority][: MAX_EXPORT - 1]

    grid_summary = {
        "champion_score_ref": CHAMP_SCORE,
        "champion_recipe": {"offset": OFFSET, "short": SHORT, "delta": DELTA, "outlier": [OUT_LO, OUT_HI]},
        "n_grid": len(rows),
        "export_order": exported,
        "top20": [item["row"] for item in rows[:20]],
    }
    (OUT / "grid_summary.json").write_text(json.dumps(grid_summary, indent=2), encoding="utf-8")

    first = exported[0]
    first_path = export_dir / first / "submission.csv"
    alt_diff = next(x["row"] for x in rows if x["row"]["name"] == priority)

    REVIEW.write_text(
        f"""# FE formula grid: поиск «их» gate (ветка `exp/alt-formula-gate`)

Чемпион main: **{CHAMP_SCORE}** — `(L−S)/L`, окна ({OFFSET},{SHORT}), valley+{DELTA}, q015–985.

## Стратегия
- Крупная локальная сетка формул × окон (без массовых upload).
- Локальный proxy: `valley_ratio`, разделение мод, умеренный `diff_vs_champ`.
- На Kaggle — только top-{MAX_EXPORT} из `submissions/fe_formula_grid/export/`.

## Первый upload (приоритет)

`{first_path.relative_to(ROOT)}`

| Поле | Значение |
|---|---|
| formula | `{alt_diff["formula"]}` |
| windows | ({alt_diff["offset"]}, {alt_diff["short"]}) |
| diff vs champ | {alt_diff["diff_vs_champ"]} |
| valley_ratio | {alt_diff["valley_ratio"]:.4f} |
| proxy_score | {alt_diff["proxy_score"]} |

Ориентир: **> {CHAMP_SCORE}**. Если ≤ — следующий из `export_order` в `grid_summary.json`.

## LB

| Вариант | Score |
|---|---:|
| champ (main) | **{CHAMP_SCORE}** |
| ALT_f_diff_over_sum (старый alt) | *(ожидает upload)* |
| `{first}` | *(ожидает upload)* |
""",
        encoding="utf-8",
    )

    print(json.dumps({"export_order": exported, "first": str(first_path)}, indent=2))
    print(f"wrote {OUT / 'grid_summary.json'}")
    print(f"wrote {REVIEW}")


if __name__ == "__main__":
    main()
