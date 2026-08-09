# -*- coding: utf-8 -*-
"""Rebuild avo_sygnal_types_8.ipynb with EPIC-PREP-0808 preprocessing section."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "avo_sygnal_types_8.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def src_of(cell: dict) -> str:
    return "".join(cell.get("source", []))


def clear_outputs(nb: dict) -> None:
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"] = []
            c["execution_count"] = None


LOAD_CELL = r'''# @title §2. Загрузка данных (D10 allowlist — без upload)
"""Чтение Run200_Wave_0_1.txt только из разрешённых корней (EPIC-PREP-0808 D10)."""
from pathlib import Path

DATA_NAME = "Run200_Wave_0_1.txt"
cwd = Path.cwd().resolve()

# REPO_ROOT: не подниматься из /content в / (баг Colab: parent(/content)=/)
if (cwd / "src").exists():
    REPO_ROOT = cwd
elif (cwd.parent / "src").exists():
    REPO_ROOT = cwd.parent
elif cwd == Path("/content") or str(cwd).startswith("/content/"):
    REPO_ROOT = Path("/content")
elif cwd.name == "notebooks":
    REPO_ROOT = cwd.parent
else:
    REPO_ROOT = cwd

ALLOWED_ROOTS = [
    (REPO_ROOT / "data").resolve(),
    Path("/content/data").resolve(),
    Path("/content/sygnal_types/data").resolve(),
    Path("/content").resolve(),  # Colab: файл рядом с ноутбуком
]

candidates = []
seen = set()
for root in ALLOWED_ROOTS:
    p = root / DATA_NAME
    key = str(p)
    if key not in seen:
        seen.add(key)
        candidates.append(p)

print(f"cwd={cwd}")
print(f"REPO_ROOT={REPO_ROOT}")


def _is_allowed(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    if resolved.name != DATA_NAME:
        return False
    for root in ALLOWED_ROOTS:
        try:
            if resolved.is_relative_to(root):
                return True
        except AttributeError:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
    return False


raw_df = None
used_path = None
for p in candidates:
    if not p.exists():
        print(f"не найден: {p}")
        continue
    if not _is_allowed(p):
        print(f"отклонён (вне allowlist): {p}")
        continue
    try:
        raw_df = pd.read_csv(p, sep=" ", header=None, skipinitialspace=True)
        used_path = str(p.resolve())
        print(f"загружено из: {used_path}")
        break
    except Exception as e:
        print(f"ошибка {p}: {e}")

if raw_df is None:
    searched = "\n  - ".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"{DATA_NAME} не найден в allowlist. В Colab: Files → загрузите в /content/ "
        f"или /content/data/.\nИскали:\n  - {searched}"
    )

print(f"сырой размер: {raw_df.shape}")
wave_df = raw_df.drop(columns=DROP_COLS, errors="ignore")
X = wave_df.to_numpy(dtype=np.float64)
if X.shape != (N_SAMPLES, 500):
    raise ValueError(f"Expected {(N_SAMPLES, 500)}, got {X.shape}")
if not np.isfinite(X).all():
    raise ValueError("Non-finite values in waveform matrix")
print(f"матрица волн: {X.shape}, min={X.min():.0f}, max={X.max():.0f}")
print(f"пропуски: {np.isnan(X).sum()}, inf: {np.isinf(X).sum()}")
'''

IMPORT_EXTRA = r'''
# --- prep package path (REPO_ROOT only) ---
import sys
_REPO = Path.cwd() if (Path.cwd() / "src").exists() else Path.cwd().parent
_SRC = str((_REPO / "src").resolve())
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
'''

PREP_CONSTANTS = r'''# @title § Preprocessing — константы (EPIC-PREP-0808)
"""Единая ячейка констант prep. Без pip / joblib / pickle."""
from sygnal_clustering.signal_extraction import (
    BASELINE_BINS,
    DECAY_FRAC,
    EPS,
    N_SIGMA,
    PSD_OFFSET,
    PSD_SHORT,
    VALLEY_RATIO_MAX,
    apply_polarity,
    build_qc_flags,
    calibrate_psd_windows,
    extract_prep_features,
    valley_ratio,
)

POLARITY = "negative"  # S0: отрицательный импульс ФЭУ на высоком пьедестале (см. визуал)
PSD_FORMULA = "(long-short)/long"
MAX_GRID_POINTS = 64
CALIB_SUBSAMPLE = 5000
PSD_OFFSET_GRID = [1, 2, 3, 5, 8, 12, 16, 24]
PSD_SHORT_GRID = [10, 15, 20, 30, 40, 50, 60, 80]
if len(PSD_OFFSET_GRID) * len(PSD_SHORT_GRID) > MAX_GRID_POINTS:
    raise ValueError("PSD grid exceeds MAX_GRID_POINTS")

print("prep constants OK", POLARITY, PSD_FORMULA, f"EPS={EPS}")
'''

S0_CELL = r'''# @title § Preprocessing S0 — spike полярности (D0)
"""Сравнение argmax vs argmin; фиксация POLARITY и x_pos."""
row_max = X.max(axis=1)
row_min = X.min(axis=1)
argmax = X.argmax(axis=1)
argmin = X.argmin(axis=1)
baseline_raw = X[:, :BASELINE_BINS].mean(axis=1)

print("row_max: ", float(row_max.mean()), "±", float(row_max.std()))
print("row_min: ", float(row_min.mean()), "±", float(row_min.std()))
print("baseline≈row_max?", float(np.mean(np.abs(row_max - baseline_raw) / (np.abs(baseline_raw) + EPS))))
print("mean |argmax-argmin| bins:", float(np.mean(np.abs(argmax.astype(float) - argmin))))

rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(len(X), size=24, replace=False)

fig, axes = plt.subplots(4, 6, figsize=(14, 8), sharex=True)
axes = axes.ravel()
for ax, i in zip(axes, sample_idx):
    w = X[i]
    ax.plot(w, lw=0.8, color="0.4")
    ax.axvline(argmax[i], color="C3", ls="--", lw=0.8, label="argmax")
    ax.axvline(argmin[i], color="C0", ls="--", lw=0.8, label="argmin")
    ax.set_title(f"i={i}", fontsize=8)
    ax.set_yticks([])
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("S0: raw waves — argmax (red) vs argmin (blue)")
plt.tight_layout()
plt.show()

# Compare orientations around extrema
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for ax, i in zip(axes, sample_idx[:2]):
    ped = float(baseline_raw[i])
    ax.plot(X[i], label="raw", alpha=0.5)
    ax.plot(ped - X[i], label="pedestal - x (negative→pos)", lw=1.2)
    ax.axvline(argmin[i], color="C0", ls="--", label="argmin")
    ax.axvline(argmax[i], color="C3", ls="--", label="argmax")
    ax.legend(fontsize=8)
    ax.set_title(f"wave {i}")
plt.tight_layout()
plt.show()

x_pos = apply_polarity(X, POLARITY)
print(f"POLARITY={POLARITY!r} → x_pos shape {x_pos.shape}, mean max={x_pos.max(axis=1).mean():.2f}")
'''

S0_MD = """## Вывод ML-инженера (S0)

`row_max ≈ baseline`, `row_min` уходит к 0 — типичный **отрицательный** импульс на высоком пьедестале: `argmax` ловит пьедестал, сигнал — провал к `argmin`.

**Решение:** `POLARITY = "negative"`, дальше только `x_pos = pedestal - x`. Без этого ROI/PSD строятся на неверной оси.
"""

S1_CELL = r'''# @title § Preprocessing S1 — baseline, 3σ ROI (D1, D2, D11)
"""Пьедестал → x0 → charge по ROI; ранний isfinite."""
prep = extract_prep_features(
    X,
    polarity=POLARITY,
    baseline_bins=BASELINE_BINS,
    n_sigma=N_SIGMA,
    decay_frac=DECAY_FRAC,
    psd_offset=PSD_OFFSET,
    psd_short=PSD_SHORT,
    eps=EPS,
)

snr_old = X.max(axis=1) / (X[:, :BASELINE_BINS].mean(axis=1) + EPS)
print("SNR old (peak/baseline): mean=", float(snr_old.mean()))
print("SNR new (peak_above/noise): mean=", float(prep.snr.mean()), "std=", float(prep.snr.std()))
print("no_3σ_hit fraction:", float(prep.no_3sigma_hit.mean()))
print("roi_length describe:\n", pd.Series(prep.roi_length).describe())

fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
axes[0].hist(prep.baseline, bins=60, color="C0", alpha=0.85)
axes[0].set_title("baseline")
axes[1].hist(prep.noise_std, bins=60, color="C1", alpha=0.85)
axes[1].set_title("noise_std")
axes[2].hist(prep.roi_length, bins=60, color="C2", alpha=0.85)
axes[2].set_title("roi_length")
plt.tight_layout()
plt.show()

# example ROI on x0
fig, ax = plt.subplots(figsize=(10, 3.5))
for k, i in enumerate(sample_idx[:5]):
    ax.plot(prep.x0[i], alpha=0.8, label=f"i={i}")
    ax.axvline(prep.i_peak[i], color="k", ls=":", lw=0.6)
    ax.axvline(prep.i_end_roi[i], color="C3", ls="--", lw=0.6)
ax.set_title("x0 waves with peak (dotted) and 3σ ROI end (red)")
ax.legend(fontsize=8, ncol=5)
plt.tight_layout()
plt.show()
'''

S1_MD = """## Вывод ML-инженера (S1)

Baseline/noise считаются на `x_pos`, рабочий сигнал — `x0`. Charge и long-gate только по **3σ ROI** (не `sum(500)`). SNR old≈1.0 — диагностика пьедестала; новый SNR = peak_above / noise_std.

`isfinite` на ключевых массивах проверяется внутри `extract_prep_features` (D11).
"""

S2_CELL = r'''# @title § Preprocessing S2 — калибровка PSD + нормировка (D3, D4', D5)
"""Сетка (offset, short) ≤64; valley_ratio gate; decay на x0."""
off_best, short_best, vr_cal, info_cal = calibrate_psd_windows(
    prep.x0,
    prep.i_peak,
    prep.i_end_roi,
    prep.noise_std,
    offsets=PSD_OFFSET_GRID,
    shorts=PSD_SHORT_GRID,
    max_grid_points=MAX_GRID_POINTS,
    subsample=CALIB_SUBSAMPLE,
    random_state=RANDOM_STATE,
    eps=EPS,
)
print(f"calibrated PSD windows: offset={off_best}, short={short_best}, valley_ratio(sub)={vr_cal:.4f}")
print("cal info:", info_cal)

# re-extract with calibrated windows
prep = extract_prep_features(
    X,
    polarity=POLARITY,
    baseline_bins=BASELINE_BINS,
    n_sigma=N_SIGMA,
    decay_frac=DECAY_FRAC,
    psd_offset=off_best,
    psd_short=short_best,
    eps=EPS,
)
PSD_OFFSET, PSD_SHORT = off_best, short_best

vr_psd, info_psd = valley_ratio(prep.psd, eps=EPS)
vr_decay, info_decay = valley_ratio(prep.decay_time, eps=EPS)
print(f"valley_ratio PSD (full)={vr_psd:.4f} | decay={vr_decay:.4f} | threshold={VALLEY_RATIO_MAX}")

GATE_OK = (vr_psd < VALLEY_RATIO_MAX) or (vr_decay < VALLEY_RATIO_MAX)
print("GATE_OK =", GATE_OK)
if not GATE_OK:
    print("STOP: gate FAIL — не открывать EPIC-FE / сравнение новых моделей на «живом» PSD.")

# norm form
x_norm = prep.x0 / (prep.peak_above[:, None] + EPS)
if not np.isfinite(x_norm).all():
    raise ValueError("Non-finite x_norm")

fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
axes[0].hist(prep.psd, bins=80, color="C0", alpha=0.85)
if np.isfinite(vr_psd) and "valley" in info_psd:
    axes[0].axvline(info_psd.get("valley", 0), color="C3", ls="--", label="valley")
    axes[0].legend()
axes[0].set_title(f"PSD (long-short)/long  vr={vr_psd:.3f}")
axes[1].hist(prep.decay_time, bins=80, color="C1", alpha=0.85)
axes[1].set_title(f"decay_time  vr={vr_decay:.3f}")
plt.tight_layout()
plt.show()

# A/B diagnostic short/long (not for X_clust)
psd_short_over_long = prep.short / (prep.charge_roi + EPS)
print("A/B short/long median:", float(np.median(psd_short_over_long)))
'''

S2_MD = """## Вывод ML-инженера (S2)

Default PSD = `(long−short)/long` (Description). Окна калибруются по `valley_ratio` на subsample ≤5000, затем полный прогон.

**Gate:** `valley_ratio < 0.7` на PSD **или** decay. При FAIL — markdown STOP к новым моделям (legacy § ниже не считаются доказательством готовности prep).

Decay: порог `(1−DECAY_FRAC)·peak_above` на `x0`. Нормировка: `x_norm = x0 / (peak_above+EPS)`.
"""

S3_CELL = r'''# @title § Preprocessing S3 — QC, feature contract, scaler (D6–D7, D11–D12)
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
# drop near-zero variance
variances = feat_df.var()
print("feature variances:\n", variances)
live_cols = [c for c in feat_df.columns if variances[c] > 1e-12]
if not live_cols:
    raise ValueError("No live features after variance filter")

feat_live = feat_df[live_cols].to_numpy(dtype=np.float64)
if not np.isfinite(feat_live).all():
    raise ValueError("Non-finite feature matrix (D11)")

print(feat_df.describe().round(4))
display(Markdown("**Что кормим моделям (default):** " + ", ".join(live_cols)))

scaler = RobustScaler()
X_clust = scaler.fit_transform(feat_live)

# keep legacy names for downstream cells that expect features matrix
features = np.column_stack([
    prep.peak_above,
    prep.charge_roi,
    prep.psd,
    prep.decay_time,
    prep.snr,
])
feat_names = ["peak_above", "charge_roi", "psd", "decay_time", "snr"]

# PCA diagnostic on live features
pca = PCA(n_components=min(3, X_clust.shape[1]), random_state=RANDOM_STATE)
pca.fit(X_clust)
print("PCA EVR on live prep features:", np.round(pca.explained_variance_ratio_, 4))
if pca.explained_variance_ratio_[0] > 0.99:
    print("RISK: PC1 dominates — check feature diversity (not a blocker if PSD gate OK)")
'''

S3_MD = """## Вывод ML-инженера (S3)

QC-флаги физические; `candidate_class2` = OR флагов (**не** top-q%). Prep не пишет финальный `cluster`.

Default features: `psd_calibrated`, `decay_time`, `charge_roi`, `tail_ratio` → `RobustScaler` только на них. Бинарные QC не скейлятся.

D12: в § Preprocessing нет pip/joblib/pickle.
"""

CHECKLIST_MD = """## Чеклист DoD EPIC-PREP-0808 (D0–D12)

| # | Критерий | Статус |
|---|---|---|
| D0 | POLARITY + x_pos | ✅ S0 |
| D1 | baseline / noise / x0 + SNR new | ✅ S1 |
| D2 | charge/PSD по 3σ ROI | ✅ S1–S2 |
| D3 | PSD `(long−short)/long` + калибровка окон | ✅ S2 |
| D4' | valley_ratio gate | ✅ S2 (`GATE_OK`) |
| D5 | нормировка; peak не в default X_clust | ✅ S2–S3 |
| D6 | QC без квоты / без финальных меток | ✅ S3 |
| D7 | scaler только на живых признаках | ✅ S3 |
| D8 | § Preprocessing между EDA и FE | ✅ |
| D9 | legacy-баннер перед §5+ | ✅ ниже |
| D10 | allowlist load | ✅ §2 |
| D11 | isfinite | ✅ S1–S3 |
| D12 | без pip/pickle в prep | ✅ |

**Объекты:** `notebooks/avo_sygnal_types_8.ipynb`, `src/sygnal_clustering/signal_extraction.py`.
"""

BANNER_MD = """---

# ⚠️ LEGACY / pre-prep (D9)

Ниже (§5–§9): модели и submission на **исторической** логике сравнения / квоте uncertain ~5%.

- Результат § Preprocessing выше **не отменяет** этот блок и **не является** доказательством, что квота 5% — правильная физика класса 2.
- Актуализация моделей на post-prep признаках — **EPIC-FE / EPIC-Class2**.
- Если `GATE_OK == False`, не интерпретировать silhouette/LB ниже как успех prep.

---
"""

EXTRACT_REPLACEMENT = r'''# @title §4. Feature contract после prep (замена extract_features)
"""Признаки уже посчитаны в § Preprocessing; здесь — сводка и legacy-совместимость."""
print("GATE_OK =", GATE_OK)
print(feat_df.describe().round(4))
print("X_clust shape (live scaled):", X_clust.shape)
# diagnostic: old-style short/long wall should be gone from calibrated PSD
print("psd_calibrated median/std:", float(np.median(prep.psd)), float(np.std(prep.psd)))
'''


def find_cell(nb: dict, predicate) -> int:
    for i, c in enumerate(nb["cells"]):
        if predicate(src_of(c)):
            return i
    raise KeyError("cell not found")


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    clear_outputs(nb)

    # Patch imports cell: add sys.path once
    for c in nb["cells"]:
        s = src_of(c)
        if "from sklearn.cluster import KMeans" in s and "sys.path" not in s:
            bootstrap = (
                "\nimport sys\n"
                "_REPO = Path.cwd() if (Path.cwd() / 'src').exists() else Path.cwd().parent\n"
                "_SRC = str((_REPO / 'src').resolve())\n"
                "if _SRC not in sys.path:\n"
                "    sys.path.insert(0, _SRC)\n"
            )
            c["source"] = (s.rstrip() + bootstrap).splitlines(keepends=True)
            break

    # Replace load cell
    i_load = find_cell(nb, lambda s: "§2. Загрузка" in s or "Загрузка данных" in s and "read_csv" in s)
    nb["cells"][i_load] = code(LOAD_CELL)

    # Replace markdown about upload if present
    for c in nb["cells"]:
        s = src_of(c)
        if "upload" in s.lower() and c["cell_type"] == "markdown":
            c["source"] = md(
                "Данные: только `data/Run200_Wave_0_1.txt` (allowlist D10). "
                "Интерактивный upload запрещён — Run All не зависает.\n"
            )["source"]

    # Find extract_features cell and replace with stub; insert prep BEFORE it
    i_ext = find_cell(nb, lambda s: "def extract_features" in s or "§4. Извлечение признаков" in s)

    prep_cells = [
        md("# § Preprocessing (EPIC-PREP-0808)\n\nФизический prep до FE: polarity → baseline → 3σ ROI → PSD gate → QC.\n"),
        code(PREP_CONSTANTS),
        code(S0_CELL),
        md(S0_MD),
        code(S1_CELL),
        md(S1_MD),
        code(S2_CELL),
        md(S2_MD),
        code(S3_CELL),
        md(S3_MD),
        md(CHECKLIST_MD),
        md(BANNER_MD),
    ]

    nb["cells"][i_ext] = code(EXTRACT_REPLACEMENT)
    for j, cell in enumerate(prep_cells):
        nb["cells"].insert(i_ext + j, cell)

    # After insertion, extract cell index shifted
    # Update scaling cell to not rebuild X_clust from old features[:, :3] if present
    for c in nb["cells"]:
        s = src_of(c)
        if "X_clust = RobustScaler().fit_transform(features[:, :3])" in s:
            new_s = s.replace(
                "X_clust = RobustScaler().fit_transform(features[:, :3])",
                "# X_clust already built in § Preprocessing S3 on live prep features\n"
                "print('X_clust from prep:', X_clust.shape)",
            )
            # also fix comment about peak,charge,psd
            new_s = new_s.replace(
                "# --- масштабирование peak, charge, psd ---",
                "# --- X_clust из prep (psd/decay/charge_roi/tail) ---",
            )
            c["source"] = new_s.splitlines(keepends=True)
        if "Модель 1 — KMeans" in s and "features" in s:
            pass

    # Title tweak
    if nb["cells"] and nb["cells"][0]["cell_type"] == "markdown":
        t = src_of(nb["cells"][0])
        if "avo_sygnal_types_8" not in t:
            nb["cells"][0]["source"] = (
                t.replace(
                    "Кластеризация сигналов сцинтилляционного детектора",
                    "Кластеризация сигналов сцинтилляционного детектора (v8 — prep EPIC-PREP-0808)",
                )
            ).splitlines(keepends=True)

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", NB_PATH, "cells=", len(nb["cells"]))


if __name__ == "__main__":
    main()
