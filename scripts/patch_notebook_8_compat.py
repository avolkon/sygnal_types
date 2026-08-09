# -*- coding: utf-8 -*-
import json
from pathlib import Path

nb_path = Path("notebooks/avo_sygnal_types_8.ipynb")
nb = json.loads(nb_path.read_text(encoding="utf-8"))


def set_src(i: int, text: str) -> None:
    nb["cells"][i]["source"] = text.splitlines(keepends=True)


def get(i: int) -> str:
    return "".join(nb["cells"][i].get("source", []))


for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "markdown":
        continue
    s = get(i)
    if "upload" in s.lower() or "Загрузите датасет" in s or "выбора файла" in s:
        set_src(
            i,
            "Данные: положите `Run200_Wave_0_1.txt` в `data/` (allowlist D10: "
            "`REPO_ROOT/data`, `/content/data`, `/content/sygnal_types/data`). "
            "Интерактивный upload **запрещён** — Run All не зависает.\n",
        )

S3 = r'''# @title § Preprocessing S3 — QC, feature contract, scaler (D6–D7, D11–D12)
"""QC-флаги без квоты; default признаки; RobustScaler только на живых непрерывных."""
qc = build_qc_flags(prep)
qc_df = pd.DataFrame({k: v.astype(np.int8) for k, v in qc.items()})
print("QC flag rates:")
print(qc_df.mean().sort_values(ascending=False).round(4))

feat_df = pd.DataFrame({
    "psd_calibrated": prep.psd,
    "decay_time": prep.decay_time,
    "charge_roi": prep.charge_roi,
    "tail_ratio": prep.tail_ratio,
})
default_cols = ["psd_calibrated", "decay_time", "charge_roi", "tail_ratio"]
variances = feat_df[default_cols].var()
print("feature variances:\n", variances)
live_cols = [c for c in default_cols if variances[c] > 1e-12]
if not live_cols:
    raise ValueError("No live features after variance filter")

feat_live = feat_df[live_cols].to_numpy(dtype=np.float64)
if not np.isfinite(feat_live).all():
    raise ValueError("Non-finite feature matrix (D11)")

print(feat_df.describe().round(4))
display(Markdown("**Что кормим моделям (default):** " + ", ".join(live_cols)))

scaler = RobustScaler()
X_clust = scaler.fit_transform(feat_live)

# aliases for legacy plots (§4–§5); not in X_clust
feat_df["psd"] = feat_df["psd_calibrated"]
feat_df["charge"] = feat_df["charge_roi"]
feat_df["peak"] = prep.peak_above
feat_df["snr"] = prep.snr

# legacy features matrix for remap_labels_physics (col1 = charge)
features = np.column_stack([
    prep.peak_above,
    prep.charge_roi,
    prep.psd,
    prep.decay_time,
    prep.snr,
])
feat_names = ["peak_above", "charge_roi", "psd", "decay_time", "snr"]

pca = PCA(n_components=min(3, X_clust.shape[1]), random_state=RANDOM_STATE)
pca.fit(X_clust)
print("PCA EVR on live prep features:", np.round(pca.explained_variance_ratio_, 4))
if pca.explained_variance_ratio_[0] > 0.99:
    print("RISK: PC1 dominates — check feature diversity (not a blocker if PSD gate OK)")
'''

for i, c in enumerate(nb["cells"]):
    if "§ Preprocessing S3" in get(i):
        set_src(i, S3)
        break

for i, c in enumerate(nb["cells"]):
    s = get(i)
    if "PSD vs charge" in s and "ax.scatter" in s:
        set_src(i, s.replace("PSD (short/total)", "PSD (long-short)/long"))

for i, c in enumerate(nb["cells"]):
    s = get(i)
    if "pca_full = PCA(random_state=RANDOM_STATE)" in s and "range(1, 4)" in s:
        s = s.replace(
            'axes[0].bar(range(1, 4), evr, color="steelblue", edgecolor="white")',
            "n_comp = len(evr)\n"
            'axes[0].bar(range(1, n_comp + 1), evr, color="steelblue", edgecolor="white")',
        )
        s = s.replace(
            'axes[1].plot(range(1, 4), cum, "o-", color="coral")',
            'axes[1].plot(range(1, n_comp + 1), cum, "o-", color="coral")',
        )
        set_src(i, s)

for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "markdown" and "Признаки импульса" in get(i):
        set_src(
            i,
            """## Признаки после prep

Считаются на `x_pos` / `x0` с 3σ ROI:

| Признак | Смысл |
|---|---|
| **psd_calibrated** | `(long−short)/long` по ROI, окна после калибровки |
| **decay_time** | спад до `(1−0.4)·peak_above` на `x0` |
| **charge_roi** | сумма по 3σ ROI |
| **tail_ratio** | хвост ROI / charge |

`peak` / старый SNR **не** в default `X_clust`. Gate: см. `GATE_OK` и `valley_ratio`.
""",
        )
        break

nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("patched ok")
