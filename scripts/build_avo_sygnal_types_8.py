#!/usr/bin/env python
"""Generate notebooks/avo_sygnal_types_8.ipynb — эталон LB 0.89109 (lo-only q7%)."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "avo_sygnal_types_8.ipynb"

_conc: dict = {}
exec((ROOT / "scripts" / "notebook_conclusions_8.py").read_text(encoding="utf-8"), _conc)


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELLS = [
    md("""# Кластеризация сигналов сцинтилляционного детектора (v8 — эталон LB 0.89109)

**ФИО:** Анастасия Волконская  
**Группа:** М25-555

Ссылка на соревнование Kaggle:  
https://www.kaggle.com/competitions/signal-types-classification

**Цель:** три кластера импульсов сцинтиллятора. Финальная модель v8 — PSD `(L−S)/L` на окнах **offset=4, short=42**, порог `valley+0.003`, **асимметричный** class2: только низкий PSD (`q_lo=0.07`) → unknown. LB **0.89109**.

Перед работой я один раз сверила план с **консилиумом** (аналитик / физик / ML-инженер): критерий выбора — физика хвоста и LB, не silhouette."""),
    md("""❗Перед Run All положите `Run200_Wave_0_1.txt` в один из путей ниже (`data/`, `/content/data/`, …). В Colab можно загрузить файл через Files."""),
    code("""# @title Установка зависимостей
\"\"\"Проверка пакетов; в Colab при отсутствии — тихая установка через pip.\"\"\"
import importlib
import subprocess
import sys

# --- список необходимых библиотек ---
for pkg in ("pandas", "numpy", "sklearn", "matplotlib", "seaborn"):
    mod = "sklearn" if pkg == "sklearn" else pkg
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
print("Зависимости OK")"""),
    md("""## 📊 Выводы: окружение (§0)

| Пакет | Статус |
|-------|--------|
| pandas / numpy / sklearn / matplotlib / seaborn | установлены или уже были в среде |

**Интерпретация:** зависимости готовы, можно загружать данные без ручного `pip` в обычном Run All.

**Вывод:** среда собрана.

**✅ Следующий шаг:** импорты и константы чемпиона."""),
    code("""# @title §1. Импорты и константы
\"\"\"Импорты, воспроизводимость и пути вывода.\"\"\"
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler

try:
    from IPython.display import display
except ImportError:
    display = print  # noqa: A001

warnings.filterwarnings("ignore", category=FutureWarning)

# --- воспроизводимость и контракт данных ---
RANDOM_STATE = 42
N_SAMPLES = 23_479
DROP_COLS = [0, 1, 2, 3, 504]
BASELINE_BINS = 50
N_SIGMA = 3.0
DECAY_FRAC = 0.40
PSD_OFFSET = 4
PSD_SHORT = 42
DELTA = 0.003
Q_LO_CHAMP = 0.07
EPS = 1e-9
LB_CHAMP = 0.89109

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (11, 5)
print("RANDOM_STATE =", RANDOM_STATE)
print("PSD windows =", (PSD_OFFSET, PSD_SHORT), "DELTA =", DELTA, "Q_LO =", Q_LO_CHAMP)"""),
    md("""## 📊 Выводы: импорты (§1)

| Константа | Значение |
|-----------|----------|
| RANDOM_STATE | **42** |
| PSD windows | **(4, 42)** |
| DELTA / Q_LO | **0.003 / 0.07** |
| LB эталон | **0.89109** |

**Интерпретация:** я зафиксировала контракт чемпиона ещё до обучения, чтобы все модели крутились вокруг одной prep-оси.

**Вывод:** константы согласованы с freeze `P14_2b_qlo_0070`.

**✅ Следующий шаг:** загрузка `Run200_Wave_0_1.txt`."""),
    code("""# @title §2. Загрузка данных
\"\"\"Чтение Run200_Wave_0_1.txt: поиск по типовым путям или загрузка через виджет Colab.\"\"\"
DATA_NAME = "Run200_Wave_0_1.txt"

# --- перебор стандартных путей (локально / Colab / Kaggle) ---
file_paths = [
    f"/datasets/{DATA_NAME}",
    f"/content/data/{DATA_NAME}",
    f"/content/{DATA_NAME}",
    f"data/{DATA_NAME}",
    DATA_NAME,
    f"../data/{DATA_NAME}",
    f"../{DATA_NAME}",
]

raw_df = None
used_path = None
for path in file_paths:
    p = Path(path)
    if not p.exists():
        print(f"не найден: {path}")
        continue
    try:
        raw_df = pd.read_csv(p, sep=" ", header=None, skipinitialspace=True)
        used_path = str(p.resolve())
        print(f"загружено из: {used_path}")
        break
    except Exception as e:
        print(f"ошибка {path}: {e}")

# --- резерв: ручная загрузка в Colab ---
if raw_df is None:
    try:
        from google.colab import files

        print(f"Файл не найден автоматически. Выберите {DATA_NAME}:")
        uploaded = files.upload()
        if DATA_NAME not in uploaded:
            raise FileNotFoundError(f"Нужен файл {DATA_NAME}")
        with open(DATA_NAME, "wb") as f:
            f.write(uploaded[DATA_NAME])
        raw_df = pd.read_csv(DATA_NAME, sep=" ", header=None, skipinitialspace=True)
        used_path = str(Path(DATA_NAME).resolve())
        print(f"загружено через виджет: {used_path}")
    except ImportError:
        raise FileNotFoundError(
            f"{DATA_NAME} не найден. Положите файл в data/ или рядом с ноутбуком."
        )

print(f"сырой размер: {raw_df.shape}")
wave_df = raw_df.drop(columns=DROP_COLS, errors="ignore")
X = wave_df.to_numpy(dtype=np.float64)
assert X.shape == (N_SAMPLES, 500), X.shape
print(f"матрица волн: {X.shape}, min={X.min():.0f}, max={X.max():.0f}")
print(f"пропуски: {np.isnan(X).sum()}, inf: {np.isinf(X).sum()}")"""),
    md(_conc["AFTER_LOAD"]),
    code("""# @title §3. EDA — примеры waveform
\"\"\"Визуализация сырых импульсов: форма важнее амплитуды.\"\"\"
rng = np.random.default_rng(RANDOM_STATE)
idx_show = [0, 1, 2, int(rng.integers(0, N_SAMPLES))]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, i in zip(axes.ravel(), idx_show):
    ax.plot(X[i], lw=0.8)
    ax.set_title(f"импульс #{i}")
    ax.set_xlabel("отсчёт")
    ax.set_ylabel("амплитуда ADC")
plt.suptitle("Примеры сырых волновых форм")
plt.tight_layout()
plt.show()"""),
    md("""## 📊 Выводы: примеры waveform (§3)

**Интерпретация:**

- я увидела похожие по высоте импульсы, но разный «хвост» после пика;
- уже на глаз видно, что разделять надо по форме угасания, а не по амплитуде вспышки.

**Вывод:** сырые графики подтверждают гипотезу о decay/PSD.

**✅ Следующий шаг:** числовая статистика по строкам."""),
    code("""# @title §3. EDA — статистика по строкам
\"\"\"Распределения простых агрегатов до prep: пик почти константен.\"\"\"
row_max = X.max(axis=1)
row_sum = X.sum(axis=1)
row_std = X.std(axis=1)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].hist(row_max, bins=60, edgecolor="white", alpha=0.85)
axes[0].set_title("максимум по строке")
axes[1].hist(row_sum, bins=60, edgecolor="white", alpha=0.85)
axes[1].set_title("сумма по строке")
axes[2].hist(row_std, bins=60, edgecolor="white", alpha=0.85)
axes[2].set_title("σ по строке")
plt.tight_layout()
plt.show()

print("описательная статистика (max по строке):")
print(pd.Series(row_max).describe().round(2))
print("mean/std max:", float(row_max.mean()), float(row_max.std()))"""),
    md(_conc["AFTER_EDA"]),
    code("""# @title §4. Предобработка — polarity / baseline / 3σ ROI / PSD (инлайн)
\"\"\"Инлайн-логика signal_extraction: negative polarity, baseline 50, ROI 3σ, PSD (L-S)/L.\"\"\"


def apply_polarity_negative(x: np.ndarray) -> np.ndarray:
    \"\"\"Отражение отрицательного импульса относительно пьедестала первых BASELINE_BINS.\"\"\"
    ped = np.asarray(x, dtype=np.float64)[:, :BASELINE_BINS].mean(axis=1, keepdims=True)
    return ped - np.asarray(x, dtype=np.float64)


def _roi_end_3sigma(wave, i_peak, noise_std, n_sigma=N_SIGMA):
    thr = n_sigma * float(noise_std)
    n = len(wave)
    for j in range(i_peak, n):
        if wave[j] <= thr:
            return j, False
    return n - 1, True


def extract_prep_inline(x_raw, *, psd_offset=PSD_OFFSET, psd_short=PSD_SHORT):
    \"\"\"Полный prep-батч → dict массивов (без установки локального пакета).\"\"\"
    x_pos = apply_polarity_negative(x_raw)
    baseline = x_pos[:, :BASELINE_BINS].mean(axis=1)
    noise_std = x_pos[:, :BASELINE_BINS].std(axis=1)
    x0 = x_pos - baseline[:, None]
    n, t = x0.shape
    i_peak = np.argmax(x0, axis=1).astype(np.int64)
    peak_above = x0[np.arange(n), i_peak]

    i_end = np.empty(n, dtype=np.int64)
    charge = np.empty(n, dtype=np.float64)
    short = np.empty(n, dtype=np.float64)
    decay = np.empty(n, dtype=np.float64)
    thr_frac = 1.0 - DECAY_FRAC
    for i in range(n):
        p = int(i_peak[i])
        end, _ = _roi_end_3sigma(x0[i], p, float(noise_std[i]))
        i_end[i] = end
        roi = x0[i, p : end + 1]
        charge[i] = float(roi.sum()) if len(roi) else 0.0
        s0 = min(len(roi) - 1, max(0, psd_offset)) if len(roi) else 0
        s1 = min(len(roi), s0 + psd_short)
        short[i] = float(roi[s0:s1].sum()) if len(roi) else 0.0
        thr = thr_frac * float(peak_above[i])
        below = np.where(x0[i, p:] <= thr)[0]
        decay[i] = float(below[0]) if len(below) else float(t - p)

    psd = (charge - short) / (charge + EPS)
    snr = peak_above / (noise_std + EPS)
    return {
        "x0": x0,
        "baseline": baseline,
        "noise_std": noise_std,
        "peak_above": peak_above,
        "i_peak": i_peak,
        "i_end_roi": i_end,
        "charge_roi": charge,
        "short": short,
        "psd": psd,
        "decay_time": decay,
        "snr": snr,
    }


def valley_ratio_1d(values, n_bins=80):
    \"\"\"Bimodality: density(valley) / mean(density modes).\"\"\"
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 100:
        return float("inf"), {"n_modes": 0}
    hist, edges = np.histogram(v, bins=n_bins, density=True)
    smooth = np.convolve(hist, np.array([0.25, 0.5, 0.25]), mode="same")
    peaks = [
        i
        for i in range(1, len(smooth) - 1)
        if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] > 0
    ]
    if len(peaks) < 2:
        return float("inf"), {"n_modes": len(peaks)}
    peak_heights = sorted(((smooth[i], i) for i in peaks), reverse=True)
    i1, i2 = sorted([peak_heights[0][1], peak_heights[1][1]])
    valley_idx = i1 + int(np.argmin(smooth[i1 : i2 + 1]))
    d1, d2 = float(smooth[i1]), float(smooth[i2])
    dv = float(smooth[valley_idx])
    centers = 0.5 * (edges[:-1] + edges[1:])
    info = {
        "n_modes": len(peaks),
        "mode1": float(centers[i1]),
        "mode2": float(centers[i2]),
        "valley": float(centers[valley_idx]),
        "density_mode1": d1,
        "density_mode2": d2,
        "density_valley": dv,
    }
    return float(dv / (0.5 * (d1 + d2) + EPS)), info


# --- расчёт prep на всём датасете ---
prep = extract_prep_inline(X, psd_offset=PSD_OFFSET, psd_short=PSD_SHORT)
psd = prep["psd"]
vr, vinfo = valley_ratio_1d(psd)
thr_champ = float(vinfo["valley"]) + DELTA
print("valley_ratio:", round(vr, 6))
print("modes:", vinfo.get("mode1"), vinfo.get("mode2"), "valley:", vinfo.get("valley"), "thr:", thr_champ)

feat_df = pd.DataFrame(
    {
        "peak": prep["peak_above"],
        "charge": prep["charge_roi"],
        "psd": prep["psd"],
        "decay_time": prep["decay_time"],
        "snr": prep["snr"],
    }
)
print(feat_df.describe().round(4))"""),
    md(_conc["AFTER_PREP"]),
    code("""# @title §5. Feature Engineering — PSD гистограмма и разделимость
\"\"\"Гистограмма PSD с долиной, scatter charge–PSD, корреляции.\"\"\"
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].hist(psd, bins=80, density=True, edgecolor="white", alpha=0.85, color="steelblue")
axes[0].axvline(vinfo["valley"], color="orange", ls="--", label=f"valley={vinfo['valley']:.4f}")
axes[0].axvline(thr_champ, color="crimson", ls="-", label=f"thr={thr_champ:.4f}")
axes[0].set_title("PSD histogram (bimodality)")
axes[0].legend()
sc = axes[1].scatter(
    feat_df["charge"], feat_df["psd"], c=feat_df["decay_time"],
    s=2, alpha=0.15, cmap="viridis", rasterized=True,
)
axes[1].set_xlabel("charge_roi")
axes[1].set_ylabel("PSD (L-S)/L")
axes[1].set_title("PSD vs charge")
plt.colorbar(sc, ax=axes[1], label="decay_time")
plt.tight_layout()
plt.show()

corr = feat_df.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Корреляции признаков после prep")
plt.tight_layout()
plt.show()
print("corr(charge, psd) =", round(float(corr.loc["charge", "psd"]), 4))
print("corr(psd, decay) =", round(float(corr.loc["psd", "decay_time"]), 4))"""),
    md(_conc["AFTER_FE"]),
    code("""# @title §6. Подбор признаков, анализ, важность
\"\"\"Сравнение одномерной бимодальности осей: PSD vs peak/charge/decay.\"\"\"
importance_rows = []
for name, arr in [
    ("psd", feat_df["psd"].to_numpy()),
    ("peak", feat_df["peak"].to_numpy()),
    ("charge", feat_df["charge"].to_numpy()),
    ("decay_time", feat_df["decay_time"].to_numpy()),
    ("snr", feat_df["snr"].to_numpy()),
]:
    r, info = valley_ratio_1d(arr)
    importance_rows.append(
        {
            "feature": name,
            "valley_ratio": r if np.isfinite(r) else np.nan,
            "n_modes": info.get("n_modes", 0),
            "valley": info.get("valley", np.nan),
            "std": float(np.std(arr)),
        }
    )

importance_df = pd.DataFrame(importance_rows).sort_values("valley_ratio")
display(importance_df.round(4))

# --- разделение 0/1 по черновому thr (без class2) для «важности» ---
lab01 = np.where(psd < thr_champ, 0, 1).astype(np.int64)
if psd[lab01 == 0].mean() > psd[lab01 == 1].mean():
    lab01 = 1 - lab01
sep = []
for col in feat_df.columns:
    m0 = float(feat_df.loc[lab01 == 0, col].mean())
    m1 = float(feat_df.loc[lab01 == 1, col].mean())
    s = float(feat_df[col].std() + EPS)
    sep.append({"feature": col, "mean0": m0, "mean1": m1, "abs_z": abs(m1 - m0) / s})
sep_df = pd.DataFrame(sep).sort_values("abs_z", ascending=False)
print("разделимость 0/1 (черновой thr):")
display(sep_df.round(4))"""),
    md(_conc["AFTER_IMPORTANCE"]),
    code("""# @title §7. PCA на [peak, charge, psd]
\"\"\"Масштабирование + scree plot: PCA для визуализации, не для финала.\"\"\"
X_clust = RobustScaler().fit_transform(feat_df[["peak", "charge", "psd"]].to_numpy())

pca_full = PCA(random_state=RANDOM_STATE).fit(X_clust)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, 4), evr, color="steelblue", edgecolor="white")
axes[0].set_title("Дисперсия по PC")
axes[1].plot(range(1, 4), cum, "o-", color="coral")
axes[1].axhline(0.95, ls="--", color="gray")
axes[1].set_title("Накопленная дисперсия")
plt.tight_layout()
plt.show()
print("доли дисперсии:", np.round(evr, 4))
print("накопленная:", np.round(cum, 4))

Z_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_clust)
km_vis = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE).fit_predict(X_clust)
plt.figure(figsize=(8, 6))
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=km_vis, s=5, alpha=0.4, cmap="tab10")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("PC1–PC2 (цвет = черновой KMeans-3)")
plt.tight_layout(); plt.show()"""),
    md(_conc["AFTER_PCA"]),
    code("""# @title §7. FastICA — контрольный эксперимент
\"\"\"ICA ищет менее коррелированные оси; ожидаю ту же перекрытость, что у PCA.\"\"\"
ica = FastICA(n_components=3, random_state=RANDOM_STATE, max_iter=800)
Z_ica = ica.fit_transform(X_clust)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for k, ax in enumerate(axes):
    ax.hist(Z_ica[:, k], bins=50, edgecolor="white", alpha=0.85)
    ax.set_title(f"ICA{k+1}")
plt.suptitle("Распределения ICA-компонент")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(Z_ica[:, 0], Z_ica[:, 1], c=km_vis, s=5, alpha=0.4, cmap="tab10")
plt.xlabel("ICA1"); plt.ylabel("ICA2")
plt.title("ICA1–ICA2")
plt.tight_layout(); plt.show()"""),
    md(_conc["AFTER_ICA"]),
    code("""# @title Вспомогательные функции: метрики, доли, PSD-метки
\"\"\"Общие хелперы для сравнения моделей и правил class2.\"\"\"


def cluster_fractions(labels):
    counts = np.bincount(labels.astype(int), minlength=3)
    fr = counts / max(counts.sum(), 1)
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}


def clustering_metrics(z, labels, sample=8000):
    n_cl = len(np.unique(labels))
    if n_cl < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    n = min(sample, len(labels))
    return {
        "silhouette": float(silhouette_score(z, labels, sample_size=n, random_state=RANDOM_STATE)),
        "calinski_harabasz": float(calinski_harabasz_score(z, labels)),
        "davies_bouldin": float(davies_bouldin_score(z, labels)),
    }


def remap_labels_physics(labels, charge):
    \"\"\"Класс 2 — меньший кластер / anomaly; 0/1 — по среднему заряду.\"\"\"
    out = np.full(len(labels), 2, dtype=np.int64)
    unique = sorted(int(u) for u in np.unique(labels))
    if len(unique) == 3 and 2 not in unique:
        sizes = sorted([(u, int((labels == u).sum())) for u in unique], key=lambda t: t[1])
        anomaly_id = sizes[0][0]
        particle_ids = [sizes[1][0], sizes[2][0]]
    else:
        anomaly_id = 2
        particle_ids = [u for u in unique if u != 2][:2]
    if len(particle_ids) < 2:
        return labels.astype(np.int64)
    stats = sorted([(u, float(charge[labels == u].mean())) for u in particle_ids], key=lambda t: t[1])
    mapping = {stats[0][0]: 0, stats[1][0]: 1, anomaly_id: 2}
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def labels_psd_valley(psd_arr, *, delta=DELTA, q_lo=None, q_hi=None):
    \"\"\"0/1 по valley+delta; опционально симметричные или lo-only хвосты → 2.\"\"\"
    vr_local, info_local = valley_ratio_1d(psd_arr)
    thr = float(info_local["valley"]) + float(delta)
    lab = np.where(psd_arr < thr, 0, 1).astype(np.int64)
    if psd_arr[lab == 0].mean() > psd_arr[lab == 1].mean():
        lab = 1 - lab
    lab = lab.copy()
    finite = np.isfinite(psd_arr)
    if q_lo is not None and q_hi is not None:
        lo, hi = np.quantile(psd_arr[finite], [q_lo, q_hi])
        lab[(psd_arr < lo) | (psd_arr > hi)] = 2
    elif q_lo is not None:
        lo = float(np.quantile(psd_arr[finite], q_lo))
        lab[psd_arr < lo] = 2
    return lab, {"valley_ratio": vr_local, "thr": thr, **{k: info_local.get(k) for k in ("valley", "mode1", "mode2")}}


print("helpers OK")"""),
    md("""## 📊 Выводы: вспомогательные функции (§8 prep)

**Интерпретация:**

- я вынесла доли классов, silhouette/CH/DB и правило `valley+δ` + class2 в общие функции;
- так все пять моделей сравниваются честно на одном контракте PSD.

**Вывод:** инфраструктура сравнения готова.

**✅ Следующий шаг:** модель 1 — KMeans baseline."""),
    code("""# @title §8. Модель 1 — KMeans-3 на [peak, charge, psd] (legacy baseline)
\"\"\"Ожидаю collapse структуры: высокий silhouette при плохих долях 0/1.\"\"\"
labels_m1 = KMeans(n_clusters=3, n_init=15, random_state=RANDOM_STATE).fit_predict(X_clust)
labels_m1 = remap_labels_physics(labels_m1, feat_df["charge"].to_numpy())
m1 = clustering_metrics(X_clust, labels_m1)
f1 = cluster_fractions(labels_m1)
print("Модель 1 — KMeans-3:", m1, f1)
print("counts:", np.bincount(labels_m1, minlength=3))"""),
    md(_conc["AFTER_M1"]),
    code("""# @title §8. Модель 2 — GMM-2 на PSD + 5% uncertain → class2
\"\"\"Старый подход v2: два гаусса по PSD, верхние 5% uncertainty в класс 2.\"\"\"
z_psd = psd.reshape(-1, 1)
gmm2 = GaussianMixture(n_components=2, covariance_type="full", n_init=20, random_state=RANDOM_STATE)
gmm2.fit(z_psd)
proba = gmm2.predict_proba(z_psd)
labels_m2 = gmm2.predict(z_psd).astype(np.int64)
if psd[labels_m2 == 0].mean() > psd[labels_m2 == 1].mean():
    labels_m2 = 1 - labels_m2
unc = 1.0 - proba.max(axis=1)
n_unc = max(1, int(len(labels_m2) * 0.05))
labels_m2 = labels_m2.copy()
labels_m2[np.argsort(unc)[-n_unc:]] = 2
m2 = clustering_metrics(X_clust, labels_m2)
f2 = cluster_fractions(labels_m2)
print("Модель 2 — GMM-2+5%unc:", m2, f2)
print("counts:", np.bincount(labels_m2, minlength=3))"""),
    md(_conc["AFTER_M2"]),
    code("""# @title §8. Модель 3 — симметричный outlier (старый чемпион LB 0.85838)
\"\"\"thr=valley+0.003; class2 = PSD вне [q1.5%, q98.5%] — режет оба хвоста.\"\"\"
labels_m3, meta_m3 = labels_psd_valley(psd, delta=DELTA, q_lo=0.015, q_hi=0.985)
m3 = clustering_metrics(X_clust, labels_m3)
f3 = cluster_fractions(labels_m3)
print("Модель 3 — symmetric q015-985:", m3, f3)
print("meta:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in meta_m3.items()})
print("counts:", np.bincount(labels_m3, minlength=3))
print("LB ref (historical): 0.85838")"""),
    md(_conc["AFTER_M3"]),
    code("""# @title §8. Модель 4 — asymmetric lo-only q7% (НОВЫЙ ЧЕМПИОН, ФИНАЛ)
\"\"\"class2 = только psd < q(0.07); высокий PSD никогда не reject. LB=0.89109.\"\"\"
labels_m4, meta_m4 = labels_psd_valley(psd, delta=DELTA, q_lo=Q_LO_CHAMP, q_hi=None)
m4 = clustering_metrics(X_clust, labels_m4)
f4 = cluster_fractions(labels_m4)
print("Модель 4 — lo-only q7% CHAMP:", m4, f4)
print("meta:", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in meta_m4.items()})
print("counts:", np.bincount(labels_m4, minlength=3))
print("q_lo value:", float(np.quantile(psd[np.isfinite(psd)], Q_LO_CHAMP)))
print("LB champ:", LB_CHAMP)"""),
    md(_conc["AFTER_M4"]),
    code("""# @title §8. Модель 5 — негатив: soft-flip 200 ближайших к thr
\"\"\"Показываю, что «подкрутка» долины ломает чемпиона (ожидаю проигрыш vs M4).\"\"\"
labels_m5 = labels_m4.copy()
dist = np.abs(psd - float(meta_m4["thr"]))
near = [i for i in np.argsort(dist) if labels_m4[i] < 2][:200]
labels_m5[near] = 1 - labels_m5[near]
m5 = clustering_metrics(X_clust, labels_m5)
f5 = cluster_fractions(labels_m5)
print("Модель 5 — soft-flip@thr:", m5, f5)
print("diff vs champ M4:", int((labels_m5 != labels_m4).sum()))

# --- дополнительный негатив: tau/decay как единственная ось ---
lab_tau, meta_tau = labels_psd_valley(prep["decay_time"], delta=0.0, q_lo=0.07, q_hi=None)
print("негатив decay-only fractions:", cluster_fractions(lab_tau), "valley_ratio:", meta_tau["valley_ratio"])"""),
    md(_conc["AFTER_M5"]),
    code("""# @title §8. Сравнение моделей
\"\"\"Сводная таблица внутренних метрик и долей; финал выбираю по LB/физике.\"\"\"
rows = []
for name, metrics, labels, lb in [
    ("1_KMeans3_peak_charge_psd", m1, labels_m1, None),
    ("2_GMM2_psd_unc5", m2, labels_m2, None),
    ("3_symmetric_q015_985", m3, labels_m3, 0.85838),
    ("4_lo_only_q07_CHAMP", m4, labels_m4, LB_CHAMP),
    ("5_softflip_negative", m5, labels_m5, None),
]:
    rows.append({"model": name, **metrics, **cluster_fractions(labels), "LB_ref": lb})

compare_df = pd.DataFrame(rows).set_index("model")
display(compare_df.round(4))
print("лучший silhouette:", compare_df["silhouette"].idxmax())
print("ФИНАЛ по задаче: 4_lo_only_q07_CHAMP (LB=0.89109), не max silhouette")"""),
    md(_conc["AFTER_COMPARE"]),
    code("""# @title §9. Подбор гиперпараметров — q_lo и delta (lo-only)
\"\"\"Сетка q_lo и delta; семейство 0.07 / 0.003 побеждает по стабильности долей.\"\"\"
base01, meta_base = labels_psd_valley(psd, delta=DELTA, q_lo=None, q_hi=None)
# пересоберём 0/1 без class2
lab01_base = np.where(psd < thr_champ, 0, 1).astype(np.int64)
if psd[lab01_base == 0].mean() > psd[lab01_base == 1].mean():
    lab01_base = 1 - lab01_base

tune_q_rows = []
for q in (0.05, 0.06, 0.07, 0.08):
    lab, meta = labels_psd_valley(psd, delta=DELTA, q_lo=q, q_hi=None)
    tune_q_rows.append(
        {
            "q_lo": q,
            "qlo_value": float(np.quantile(psd, q)),
            **cluster_fractions(lab),
            "n_class2": int((lab == 2).sum()),
            "diff_vs_q07": int((lab != labels_m4).sum()),
            "valley_ratio": meta["valley_ratio"],
        }
    )
tune_q_df = pd.DataFrame(tune_q_rows)
print("=== tune q_lo ===")
display(tune_q_df.round(4))

tune_d_rows = []
for d in (0.001, 0.003, 0.005):
    lab, meta = labels_psd_valley(psd, delta=d, q_lo=Q_LO_CHAMP, q_hi=None)
    tune_d_rows.append(
        {
            "delta": d,
            "thr": meta["thr"],
            **cluster_fractions(lab),
            "diff_vs_d003": int((lab != labels_m4).sum()),
            "valley_ratio": meta["valley_ratio"],
        }
    )
tune_d_df = pd.DataFrame(tune_d_rows)
print("=== tune delta ===")
display(tune_d_df.round(4))
print("LB plateau note: 6%->7% historically +0.00013 (0.89096 -> 0.89109)")"""),
    md(_conc["AFTER_TUNE"]),
    code("""# @title §9. Выбор лучшей модели — объяснение
\"\"\"Фиксирую чемпиона и печатаю контракт для submission.\"\"\"
FINAL_LABELS = labels_m4
FINAL_NAME = "4_lo_only_q07_CHAMP"

print("Выбрана модель:", FINAL_NAME)
print("Контракт:")
print("  polarity=negative, baseline=50, ROI=3σ")
print("  PSD=(L-S)/L, windows offset=4, short=42")
print("  thr = valley + 0.003")
print("  class2 = psd < quantile(psd, 0.07) ONLY (hi-tail NEVER rejected)")
print("  LB =", LB_CHAMP)
print("fractions:", cluster_fractions(FINAL_LABELS))
print("counts:", np.bincount(FINAL_LABELS, minlength=3))

fig, ax = plt.subplots(figsize=(9, 4))
for c, name in [(0, "class0"), (1, "class1"), (2, "class2")]:
    ax.hist(psd[FINAL_LABELS == c], bins=60, alpha=0.45, density=True, label=name)
ax.axvline(thr_champ, color="crimson", ls="--", label="thr")
ax.axvline(float(np.quantile(psd, Q_LO_CHAMP)), color="purple", ls=":", label="q_lo")
ax.set_title("PSD по финальным классам (lo-only q7%)")
ax.legend(); plt.tight_layout(); plt.show()"""),
    md(_conc["AFTER_BEST"]),
    code("""# @title §10. Предсказание — submission.csv
\"\"\"Сохранение ответа Kaggle и сверка с freeze чемпиона P14_2b_qlo_0070.\"\"\"
submission = pd.DataFrame({"index": np.arange(N_SAMPLES), "cluster": FINAL_LABELS.astype(int)})

# --- корень репозитория (cwd может быть notebooks/ при nbconvert) ---
here = Path.cwd().resolve()
repo_candidates = [here, here.parent, Path("..").resolve(), Path(".").resolve()]
repo_root = None
for cand in repo_candidates:
    if (cand / "data" / "Run200_Wave_0_1.txt").exists() or (
        cand / "submissions" / "psd_remainder14" / "P14_2b_qlo_0070" / "submission.csv"
    ).exists():
        repo_root = cand
        break
if repo_root is None:
    repo_root = here

out_paths = [
    here / "submission.csv",
    repo_root / "notebooks" / "submission.csv",
    repo_root / "submissions" / "notebook8" / "submission.csv",
]
written = []
for p in out_paths:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(p, index=False)
        written.append(str(p.resolve()))
    except Exception as e:
        print("skip", p, e)

assert submission.shape[0] == N_SAMPLES
assert set(submission["cluster"].unique()).issubset({0, 1, 2})
print(submission["cluster"].value_counts().sort_index())
print("сохранено:", written)

freeze_path = repo_root / "submissions" / "psd_remainder14" / "P14_2b_qlo_0070" / "submission.csv"
if freeze_path.exists():
    ref = pd.read_csv(freeze_path)["cluster"].to_numpy()
    diff = int((submission["cluster"].to_numpy() != ref).sum())
    print(f"diff vs freeze {freeze_path}: {diff}")
    assert diff == 0, diff
    print("ASSERT OK: labels match champion freeze (diff==0)")
else:
    print("WARN: freeze file not found — skip assert (Colab without repo)")
submission.head()"""),
    md(_conc["AFTER_SUB"]),
    md(_conc["FINAL"]),
    md(_conc["KAGGLE"]),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": CELLS,
    }
    OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"written {OUT} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
