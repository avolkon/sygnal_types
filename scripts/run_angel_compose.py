"""Angel branch: Description composition (PSD + decay + PCA vote).

Not a claim of private Kaggle labels. Reconstructs the organizers' documented
recipe from Разработка/Подготовка/Description.txt §2.3.4 / §3.3.5:
  majority vote of (PSD split, decay-time split, PC1 of peak×charge).

Champion physics frozen for PSD voter: windows (4,42), valley+0.003.
Class-2 policies vary (quantile tails vs vote disagreement).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.signal_extraction import (  # noqa: E402
    EPS,
    extract_prep_features,
    valley_ratio,
)

DROP = [0, 1, 2, 3, 504]
N = 23_479
OUT = ROOT / "submissions" / "angel_compose"
REVIEW = ROOT / "Разработка" / "Ревью" / "0808_FE_angel_compose.md"

OFFSET, SHORT = 4, 42
DELTA = 0.003
OUT_LO, OUT_HI = 0.015, 0.985
CHAMP_SCORE = 0.85838


def split_1d(values: np.ndarray, delta: float = 0.0) -> tuple[np.ndarray, dict]:
    vr, info = valley_ratio(values, eps=EPS)
    valley = float(info.get("valley", np.median(values[np.isfinite(values)])))
    thr = valley + delta
    lab = np.where(values < thr, 0, 1).astype(np.int64)
    return lab, {"valley_ratio": float(vr), "valley": valley, "thr": thr, "n_modes": int(info.get("n_modes", -1))}


def align_to_ref(lab: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Flip 0/1 so lab agrees with ref on majority of non-ref-2 points."""
    out = lab.copy()
    m = ref < 2
    if m.sum() == 0:
        return out
    agree = (out[m] == ref[m]).mean()
    if agree < 0.5:
        out = 1 - out
    return out


def apply_outlier_class2(lab: np.ndarray, psd: np.ndarray) -> np.ndarray:
    out = lab.copy()
    qlo, qhi = np.quantile(psd[np.isfinite(psd)], [OUT_LO, OUT_HI])
    out[(psd < qlo) | (psd > qhi)] = 2
    return out


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "Run200_Wave_0_1.txt", sep=" ", header=None, skipinitialspace=True)
    X = raw.drop(columns=DROP, errors="ignore").to_numpy(dtype=np.float64)
    prep = extract_prep_features(X, polarity="negative", psd_offset=OFFSET, psd_short=SHORT)

    # --- three voters (±1) as in Description ---
    lab_psd, meta_psd = split_1d(prep.psd, DELTA)
    # polarity: lower mean psd -> 0 (champion convention)
    if prep.psd[lab_psd == 0].mean() > prep.psd[lab_psd == 1].mean():
        lab_psd = 1 - lab_psd

    lab_decay, meta_decay = split_1d(prep.decay_time, 0.0)
    lab_decay = align_to_ref(lab_decay, lab_psd)

    xy = np.column_stack([prep.peak_above, prep.charge_roi])
    xy = (xy - np.nanmedian(xy, axis=0)) / (np.nanstd(xy, axis=0) + EPS)
    pc1 = PCA(n_components=1, random_state=42).fit_transform(xy).ravel()
    lab_pca, meta_pca = split_1d(pc1, 0.0)
    lab_pca = align_to_ref(lab_pca, lab_psd)

    votes = np.column_stack([lab_psd, lab_decay, lab_pca])  # 0/1
    vote_sum = votes.sum(axis=1)  # 0..3
    maj = (vote_sum >= 2).astype(np.int64)  # majority -> 1
    n_agree = np.maximum(vote_sum, 3 - vote_sum)  # 2 or 3
    disagree = n_agree < 3

    champ = apply_outlier_class2(lab_psd.copy(), prep.psd)

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
            "disagree_rate": float(disagree.mean()),
        }
        if extra:
            meta.update(extra)
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        rows.append({"name": name, **{k: meta[k] for k in ("diff_vs_champ", "fractions", "note", "disagree_rate")}})
        print(name, meta["fractions"], f"diff={meta['diff_vs_champ']}", note)
        return path

    save("ANGEL_CHAMP", champ, "PSD-only champion reference")

    # A: Description majority + champion class2 tails
    lab_a = apply_outlier_class2(maj.copy(), prep.psd)
    path_a = save(
        "ANGEL_compose_maj",
        lab_a,
        "Description vote PSD+decay+PCA majority; class2=q015-985",
        {"voters": {"psd": meta_psd, "decay": meta_decay, "pca": meta_pca}},
    )

    # B: disagreement -> 2 (emergent third class from composition)
    lab_b = maj.copy()
    lab_b[disagree] = 2
    save("ANGEL_compose_disagree2", lab_b, "majority 0/1; non-unanimous vote -> 2")

    # C: disagreement OR outlier -> 2
    lab_c = maj.copy()
    lab_c[disagree] = 2
    lab_c = apply_outlier_class2(lab_c, prep.psd)
    # keep disagree as 2 even if not outlier
    lab_c[disagree] = 2
    save("ANGEL_compose_disagree2_or_out", lab_c, "disagree->2 OR PSD outlier q015-985")

    # D: weighted — PSD double vote (2 of 4), closer to proven voter
    vote_w = lab_psd + lab_psd + lab_decay + lab_pca  # 0..4
    maj_w = (vote_w >= 2).astype(np.int64)  # tie-break toward 1 if sum==2; use >=2.5? use >= 2 means 2/4 can be 1
    # better: >= 3 for majority of weighted 4
    maj_w = (vote_w >= 3).astype(np.int64)
    lab_d = apply_outlier_class2(maj_w, prep.psd)
    save("ANGEL_compose_psd2x", lab_d, "weighted vote: PSD counted twice; class2=q015-985")

    order = [
        "ANGEL_compose_maj",
        "ANGEL_compose_psd2x",
        "ANGEL_compose_disagree2",
        "ANGEL_compose_disagree2_or_out",
    ]
    (OUT / "summary.json").write_text(
        json.dumps({"champion_score_ref": CHAMP_SCORE, "upload_order": order, "candidates": rows}, indent=2),
        encoding="utf-8",
    )

    first = next(r for r in rows if r["name"] == "ANGEL_compose_maj")
    REVIEW.write_text(
        f"""# FE angel: композиция Description (ветка `angel`)

## Честная рамка

Приватный алгоритм Kaggle и скрытые метки **недоступны**.  
«100%» обещать нельзя. Этот submission — реконструкция **открытого** метода из  
`Разработка/Подготовка/Description.txt`: голосование PSD + время высвечивания + PC1(peak, charge).

Чемпион-референс: **{CHAMP_SCORE}** (только PSD).

## Первый upload

`{path_a.relative_to(ROOT)}`

| Поле | Значение |
|---|---|
| метод | majority(PSD, decay, PCA) |
| PSD | окна (4,42), valley+{DELTA} |
| class2 | q015–985 (как у чемпиона) |
| diff vs champ | **{first["diff_vs_champ"]}** |
| fractions | {first["fractions"]} |
| disagree_rate | {first["disagree_rate"]:.4f} |

Ориентир: **> {CHAMP_SCORE}**.

## LB

| Вариант | Score |
|---|---:|
| champ PSD | **{CHAMP_SCORE}** |
| `ANGEL_compose_maj` | *(ожидает upload)* |
""",
        encoding="utf-8",
    )
    print("FIRST", path_a.resolve())


if __name__ == "__main__":
    main()
