"""Build all crazy residual pipelines on branch `crazy`.

Base: asymmetric lo-only PSD champ (LB 0.89109).
One hypothesis = one submission folder under submissions/crazy/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.crazy_hypotheses import (  # noqa: E402
    CHAMP_SCORE,
    PIPELINE_REGISTRY,
    hyp_A3b_rolling_baseline,
    make_context,
)

DROP = [0, 1, 2, 3, 504]
N = 23_479
OUT = ROOT / "submissions" / "crazy"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_crazy_pipelines.md"
SCENARIO = ROOT / "Разработка" / "0808_Сценарий_crazy.md"


def fractions(lab: np.ndarray) -> dict[str, float]:
    fr = np.bincount(lab.astype(int), minlength=3) / len(lab)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def save(name: str, lab: np.ndarray, note: str, base: np.ndarray, extra: dict) -> dict:
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(d / "submission.csv", index=False)
    pd.DataFrame({"index": np.arange(N), "cluster": lab.astype(int)}).to_csv(
        d / f"submission{name}.csv", index=False
    )
    meta = {
        "hypothesis": name,
        "note": note,
        "fractions": fractions(lab),
        "diff_vs_base_089109": int((lab != base).sum()),
        "n_to_2": int(((lab == 2) & (base != 2)).sum()),
        "n_from_2": int(((lab != 2) & (base == 2)).sum()),
        "n_flip01": int(((lab < 2) & (base < 2) & (lab != base)).sum()),
        "champion_score_ref": CHAMP_SCORE,
        **extra,
    }
    # drop non-json blocks detail if huge — keep blocks summary count
    if "blocks" in meta and isinstance(meta["blocks"], list):
        meta["n_blocks"] = len(meta["blocks"])
        meta["blocks"] = meta["blocks"][:3] + [{"truncated": True, "total": len(meta["blocks"])}]
    (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    safe_note = note.encode("ascii", "replace").decode("ascii")
    print(name, meta["fractions"], f"diff={meta['diff_vs_base_089109']}", safe_note)
    return {"name": name, **{k: meta[k] for k in ("fractions", "diff_vs_base_089109", "note")}}


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    if X.shape != (N, 500):
        raise ValueError(X.shape)

    ctx = make_context(X)
    base = ctx.base
    # verify against freeze file if present
    freeze_csv = ROOT / "submissions" / "psd_remainder14" / "P14_2b_qlo_0070" / "submission.csv"
    if freeze_csv.exists():
        ref = pd.read_csv(freeze_csv).cluster.to_numpy()
        if not np.array_equal(base, ref):
            # allow tiny mismatch only via print warning — still proceed with recomputed base
            print("WARN base vs freeze diff", int((base != ref).sum()))
        else:
            print("base == P14_2b_qlo_0070 freeze OK")

    OUT.mkdir(parents=True, exist_ok=True)
    save("CRAZY_0_BASE", base, "freeze asymmetric lo-only q7% (0.89109)", base, {"rule": "base"})

    rows = []
    for name, fn in PIPELINE_REGISTRY.items():
        lab, extra = fn(ctx)
        rows.append(save(name, lab, extra.get("rule", name), base, extra))

    # A3b needs raw
    lab, extra = hyp_A3b_rolling_baseline(X, window=500)
    rows.append(save("A3b_rolling_baseline", lab, extra.get("rule", "rolling"), base, extra))

    upload_order = [
        "A0_disagree_decay",
        "A2b_nonproportional",
        "A4b_decay_extremes",
        "A1_isthmus_flip",
        "A0b_disagree_pca",
        "A2_multipeak",
        "A6c_q0_mirror",
        "A3_index_blocks",
        "A3b_rolling_baseline",
        "A5b_balance_ritual",
        "A6b_seed_text",
    ]
    summary = {
        "branch": "crazy",
        "champion_score_ref": CHAMP_SCORE,
        "upload_order": upload_order,
        "stop_rule": "two consecutive <= 0.89109 without diagnostic win -> stop that family",
        "candidates": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    table = "\n".join(
        f"| `{r['name']}` | {r['fractions']} | {r['diff_vs_base_089109']} | {r['note']} |" for r in rows
    )
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(
        f"""# FE crazy: residual puzzle pipelines

Branch: **`crazy`**. Base LB **{CHAMP_SCORE}** (`P14_2b_qlo_0070` contract).

Каждая гипотеза = отдельный пайплайн / submission. Одна заливка — одна гипотеза.

## Сводка

| name | fractions | diff vs base | rule |
|---|---|---:|---|
{table}

## Upload order (абсурд → риск)

1. `A0_disagree_decay` — PSD∧decay несогласие → 2  
2. `A2b_nonproportional` — нарушение amp∝charge → 2  
3. `A4b_decay_extremes` — дискретный decay как ключ 2  
4. `A1_isthmus_flip` — flip только в перешейке  
5. далее по `summary.json`

Ориентир: **> {CHAMP_SCORE}**.  
Сценарий: `0808_Сценарий_crazy.md`.
""",
        encoding="utf-8",
    )

    SCENARIO.write_text(
        f"""# Сценарий CRAZY: головоломка остатка ~11% (0808)

| Поле | Значение |
|---|---|
| **Ветка** | `crazy` (fork от `main`) |
| **База** | asymmetric lo-only PSD, LB **{CHAMP_SCORE}** |
| **Роль** | puzzle / non-linear residual (не матчасть PSD) |
| **Код** | `src/sygnal_clustering/crazy_hypotheses.py` |
| **Runner** | `scripts/run_crazy_pipelines.py` |

## Мандат

Традиционная логика вывела на плато ~0.89–0.90.  
Остаток — головоломка Description; каждая гипотеза — отдельный пайплайн.

## Пайплайны

| ID | Идея | Папка |
|---|---|---|
| CRAZY_0 | freeze base | `CRAZY_0_BASE` |
| A0 | PSD vs decay disagree → 2 | `A0_disagree_decay` |
| A0b | PSD vs PCA disagree → 2 | `A0b_disagree_pca` |
| A1 | isthmus flip 0↔1 | `A1_isthmus_flip` |
| A2 | multi-peak → 2 | `A2_multipeak` |
| A2b | non-proportional amp–charge → 2 | `A2b_nonproportional` |
| A3 | index blocks own valley | `A3_index_blocks` |
| A3b | rolling pedestal | `A3b_rolling_baseline` |
| A4b | decay extremes → 2 | `A4b_decay_extremes` |
| A5b | ritual f2≈10% | `A5b_balance_ritual` |
| A6b | seed 27052019 punch 10%→2 | `A6b_seed_text` |
| A6c | mirror in peak quartile 0..3 | `A6c_q*_mirror` |

## Правила

- Не смешивать две гипотезы в одном upload.  
- База 0/1+lo class2 не откатывать без LB-роста.  
- Стоп семейства: 2× ≤ {CHAMP_SCORE}.

## Reproduce

```bash
git checkout crazy
python scripts/run_crazy_pipelines.py
```
""",
        encoding="utf-8",
    )
    print("OUT", OUT)
    print("FIRST", OUT / "A0_disagree_decay" / "submission.csv")


if __name__ == "__main__":
    main()
