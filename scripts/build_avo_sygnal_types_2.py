#!/usr/bin/env python
"""Generate notebooks/avo_sygnal_types_2.ipynb — session submission."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "avo_sygnal_types_2.ipynb"

_conc: dict = {}
exec((ROOT / "scripts" / "notebook_conclusions.py").read_text(encoding="utf-8"), _conc)


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
    md("""# Кластеризация сигналов сцинтилляционного детектора

**ФИО:** Анастасия Волконская  
**Группа:** М25-555"""),
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
    code("""# @title §1. Импорты
\"\"\"Импорты, константы и настройки графиков для всего ноутбука.\"\"\"
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
from sklearn.preprocessing import QuantileTransformer, RobustScaler

try:
    from IPython.display import Markdown, display
except ImportError:
    display = print  # noqa: A001
    Markdown = str

warnings.filterwarnings("ignore", category=FutureWarning)

# --- параметры воспроизводимости и данных ---
RANDOM_STATE = 42
N_SAMPLES = 23_479
DROP_COLS = [0, 1, 2, 3, 504]  # служебные столбцы ФЭУ

# --- оформление графиков ---
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (11, 5)"""),
    md("""❗Запуск следующей ячейки может потребовать выбора файла из каталога вручную, без этого autorun зависнет"""),
    code("""# @title §2. Загрузка данных (3 варианта: платформа / Colab / ручная загрузка)
\"\"\"Чтение Run200_Wave_0_1.txt: поиск по типовым путям или загрузка через виджет Colab.\"\"\"
DATA_NAME = "Run200_Wave_0_1.txt"

# --- перебор стандартных путей ---
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

        print(f"Файл не найден автоматически. Выберите {DATA_NAME} (скачать с Kaggle Data):")
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
            f"{DATA_NAME} не найден. Положите файл рядом с ноутбуком или запустите в Colab."
        )

# --- матрица волновых форм 23479 × 500 ---
print(f"сырой размер: {raw_df.shape}")
wave_df = raw_df.drop(columns=DROP_COLS, errors="ignore")
X = wave_df.to_numpy(dtype=np.float64)
assert X.shape == (N_SAMPLES, 500), X.shape
print(f"матрица волн: {X.shape}, min={X.min():.0f}, max={X.max():.0f}")
print(f"пропуски: {np.isnan(X).sum()}, inf: {np.isinf(X).sum()}")"""),
    md(_conc["AFTER_LOAD"]),
    code("""# @title §3. EDA — примеры waveform
\"\"\"Визуализация нескольких сырых импульсов для оценки формы сигнала.\"\"\"
rng = np.random.default_rng(RANDOM_STATE)
idx_show = [0, 1, 2, int(rng.integers(0, N_SAMPLES))]

# --- сетка графиков ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, i in zip(axes.ravel(), idx_show):
    ax.plot(X[i], lw=0.8)
    ax.set_title(f"импульс #{i}")
    ax.set_xlabel("отсчёт")
    ax.set_ylabel("амплитуда")
plt.suptitle("Примеры сырых волновых форм")
plt.tight_layout()
plt.show()"""),
    code("""# @title §3. EDA — статистика по строкам
\"\"\"Распределения простых агрегатов по каждому импульсу.\"\"\"
row_max = X.max(axis=1)
row_sum = X.sum(axis=1)
row_std = X.std(axis=1)

# --- гистограммы ---
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
print(pd.Series(row_max).describe().round(2))"""),
    md(_conc["AFTER_EDA"]),
    code("""# @title §4. Извлечение признаков (3σ, PSD, decay 40%)
\"\"\"Параметризация импульса: PSD (short/long), decay 40%, базовая линия для SNR.\"\"\"
BASELINE_BINS = 50  # нулевая линия — первые 50 отсчётов (методичка)
DECAY_FRAC = 0.40
PSD_OFFSET = 3   # отступ short gate от пика (в отсчётах)
PSD_SHORT = 30   # длина short gate


def extract_features(matrix):
    \"\"\"peak, charge (long), psd, decay_time, snr для всех волн.

    long gate = сумма от пика до момента спада на 40% (выделенный импульс).
    short gate = 30 отсчётов, начиная с (пик + PSD_OFFSET).
    Базовая линия (первые 50 отсчётов) — для SNR.
    \"\"\"
    x = matrix
    n = x.shape[1]
    peak = x.max(axis=1).astype(np.float64)
    imax = x.argmax(axis=1)

    charge = np.zeros(len(x), dtype=np.float64)
    psd = np.zeros(len(x), dtype=np.float64)
    decay_time = np.zeros(len(x), dtype=np.float64)
    for i in range(len(x)):
        w = x[i]
        p = int(imax[i])
        pk = peak[i]
        # --- граница импульса: первый отсчёт ≤ 60% пика после максимума ---
        thr = pk * (1.0 - DECAY_FRAC)
        i_end = n - 1
        for j, val in enumerate(w[p:]):
            if val <= thr:
                i_end = p + j
                break
        signal = w[p : i_end + 1]
        charge[i] = float(signal.sum())
        s0 = min(len(signal) - 1, PSD_OFFSET)
        s1 = min(len(signal), s0 + PSD_SHORT)
        psd[i] = float(signal[s0:s1].sum()) / (charge[i] + 1e-9)
        decay_time[i] = float(i_end - p)

    baseline = x[:, :BASELINE_BINS].mean(axis=1)
    snr = peak / (baseline + 1e-9)
    return np.column_stack([peak, charge, psd, decay_time, snr])


# --- расчёт признаков по всему датасету ---
features = extract_features(X)
feat_names = ["peak", "charge", "psd", "decay_time", "snr"]
feat_df = pd.DataFrame(features, columns=feat_names)
print(feat_df.describe().round(4))"""),
    md(_conc["AFTER_FEATURES"]),
    code("""# @title §4. PSD vs charge — разделимость признаков
\"\"\"Scatter PSD/заряд и матрица корреляций между признаками.\"\"\"
# --- PSD vs заряд (цвет — время спада); rasterized — видно плотность облака ---
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    feat_df["charge"], feat_df["psd"], c=feat_df["decay_time"],
    s=2, alpha=0.15, cmap="viridis", rasterized=True,
)
ax.set_xlabel("заряд импульса")
ax.set_ylabel("PSD (short/total)")
ax.set_title("PSD vs заряд (цвет — время спада)")
plt.colorbar(sc, label="decay_time")
plt.tight_layout()
plt.show()

# --- корреляции ---
corr = feat_df.corr()
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Корреляции признаков")
plt.tight_layout()
plt.show()"""),
    md(_conc["AFTER_PSD"]),
    code("""# @title §5. Масштабирование и анализ дисперсии признаков
\"\"\"RobustScaler для кластеризации; PCA — оценка, сколько информации несут 3 признака.\"\"\"
# --- масштабирование peak, charge, psd ---
X_clust = RobustScaler().fit_transform(features[:, :3])

# --- полный PCA для оценки вклада компонент ---
pca_full = PCA(random_state=RANDOM_STATE)
pca_full.fit(X_clust)
evr = pca_full.explained_variance_ratio_
cum = np.cumsum(evr)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(range(1, 4), evr, color="steelblue", edgecolor="white")
axes[0].set_xlabel("компонента")
axes[0].set_ylabel("доля дисперсии")
axes[0].set_title("Дисперсия по компонентам")
axes[1].plot(range(1, 4), cum, "o-", color="coral")
axes[1].axhline(0.95, ls="--", color="gray", label="95%")
axes[1].set_xlabel("число компонент")
axes[1].set_ylabel("накопленная дисперсия")
axes[1].legend()
axes[1].set_title("Накопленная дисперсия")
plt.tight_layout()
plt.show()

print("доли дисперсии:", evr.round(3))
print("накопленная:", cum.round(3))"""),
    code("""# @title §5. Визуализация в пространстве главных компонент
\"\"\"Проекция на PC1–PC2; цвет — предварительная разметка KMeans для наглядности.\"\"\"
# --- черновая разметка для визуализации ---
km_vis = KMeans(n_clusters=3, n_init=10, random_state=RANDOM_STATE)
vis_labels = km_vis.fit_predict(X_clust)

# --- проекция на две главные компоненты ---
Z_pca = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_clust)

plt.figure(figsize=(8, 6))
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=vis_labels, s=5, alpha=0.4, cmap="tab10")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PC1 vs PC2 (цвет = KMeans k=3)")
plt.tight_layout()
plt.show()"""),
    md(_conc["AFTER_PCA"]),
    code("""# @title §6. Независимые компоненты признаков (FastICA)
\"\"\"ICA ищет менее коррелированные оси — альтернатива PCA для смешанных признаков.\"\"\"
ica = FastICA(n_components=3, random_state=RANDOM_STATE, max_iter=500)
Z_ica = ica.fit_transform(X_clust)

# --- распределения компонент ---
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for k, ax in enumerate(axes):
    ax.hist(Z_ica[:, k], bins=50, edgecolor="white", alpha=0.85)
    ax.set_title(f"компонента {k + 1}")
plt.suptitle("Распределения ICA-компонент")
plt.tight_layout()
plt.show()

# --- scatter ICA1 vs ICA2 ---
plt.figure(figsize=(8, 6))
plt.scatter(Z_ica[:, 0], Z_ica[:, 1], c=vis_labels, s=5, alpha=0.4, cmap="tab10")
plt.xlabel("ICA1")
plt.ylabel("ICA2")
plt.title("ICA: компоненты 1 и 2")
plt.tight_layout()
plt.show()"""),
    md(_conc["AFTER_ICA"]),
    code("""# @title Вспомогательные функции: remap и метрики кластеризации
\"\"\"Приведение меток к формату ТЗ и расчёт внутренних метрик качества.\"\"\"


def remap_labels_physics(labels, feat):
    \"\"\"Класс 2 — аномалии; 0 и 1 — типы частиц по среднему заряду.\"\"\"
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
    stats = sorted([(u, feat[labels == u, 1].mean()) for u in particle_ids], key=lambda t: t[1])
    mapping = {stats[0][0]: 0, stats[1][0]: 1, anomaly_id: 2}
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def clustering_metrics(z, labels, sample=8000):
    \"\"\"Silhouette, Calinski–Harabasz и Davies–Bouldin (на подвыборке).\"\"\"
    n_cl = len(np.unique(labels))
    if n_cl < 2:
        return {"silhouette": np.nan, "calinski_harabasz": np.nan, "davies_bouldin": np.nan}
    n = min(sample, len(labels))
    return {
        "silhouette": silhouette_score(z, labels, sample_size=n, random_state=RANDOM_STATE),
        "calinski_harabasz": calinski_harabasz_score(z, labels),
        "davies_bouldin": davies_bouldin_score(z, labels),
    }


def cluster_fractions(labels):
    \"\"\"Доли объектов в кластерах 0, 1, 2.\"\"\"
    counts = np.bincount(labels, minlength=3)
    fr = counts / counts.sum()
    return {f"f{i}": round(float(fr[i]), 4) for i in range(3)}"""),
    code("""# @title §7. Модель 1 — KMeans k=3
\"\"\"Baseline: KMeans на масштабированных peak, charge, psd.\"\"\"
z1 = X_clust.copy()
labels_m1 = KMeans(n_clusters=3, n_init=15, random_state=RANDOM_STATE).fit_predict(z1)
labels_m1 = remap_labels_physics(labels_m1, features)
m1 = clustering_metrics(z1, labels_m1)
print("Модель 1 — KMeans:", m1, cluster_fractions(labels_m1))"""),
    code("""# @title §7. Модель 2 — GMM-3 + QuantileTransformer
\"\"\"Смесь из трёх гауссовых распределений после квантильной нормализации.\"\"\"
z2 = QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE).fit_transform(
    features[:, :3]
)
gmm3 = GaussianMixture(n_components=3, covariance_type="full", n_init=20, random_state=RANDOM_STATE)
raw2 = gmm3.fit_predict(z2)
labels_m2 = remap_labels_physics(raw2, features)
m2 = clustering_metrics(z2, labels_m2)
print("Модель 2 — GMM-3:", m2, cluster_fractions(labels_m2))"""),
    code("""# @title §7. Модель 3 — PCA + KMeans k=3
\"\"\"Снижение размерности до 3 PC, затем KMeans — убираем корреляцию признаков.\"\"\"
pca3 = PCA(n_components=3, random_state=RANDOM_STATE)
z3 = pca3.fit_transform(X_clust)
labels_m3 = KMeans(n_clusters=3, n_init=15, random_state=RANDOM_STATE).fit_predict(z3)
labels_m3 = remap_labels_physics(labels_m3, features)
m3 = clustering_metrics(z3, labels_m3)
print("Модель 3 — PCA+KMeans:", m3, cluster_fractions(labels_m3))"""),
    code("""# @title §7. Модель 4 — FastICA + KMeans k=3
\"\"\"Кластеризация в пространстве независимых компонент.\"\"\"
z4 = Z_ica.copy()
labels_m4 = KMeans(n_clusters=3, n_init=15, random_state=RANDOM_STATE).fit_predict(z4)
labels_m4 = remap_labels_physics(labels_m4, features)
m4 = clustering_metrics(z4, labels_m4)
print("Модель 4 — ICA+KMeans:", m4, cluster_fractions(labels_m4))"""),
    code("""# @title §7. Модель 5 — GMM-2 + неуверенные → кластер 2 (финал)
\"\"\"Два основных типа + ~5% импульсов с низкой уверенностью GMM в класс аномалий.\"\"\"
z5 = RobustScaler().fit_transform(features[:, :4])

# --- GMM с двумя компонентами ---
gmm2 = GaussianMixture(n_components=2, covariance_type="full", n_init=20, random_state=RANDOM_STATE)
gmm2.fit(z5)
proba = gmm2.predict_proba(z5)
labels_m5 = gmm2.predict(z5).astype(np.int64)

# --- верхние 5% по неопределённости → класс 2 ---
uncertainty = 1.0 - proba.max(axis=1)
n_unc = max(1, int(len(labels_m5) * 0.05))
labels_m5[np.argsort(uncertainty)[-n_unc:]] = 2
labels_m5 = remap_labels_physics(labels_m5, features)

m5 = clustering_metrics(z5[:, :3], labels_m5)
print("Модель 5 — GMM-2+uncertain:", m5, cluster_fractions(labels_m5))"""),
    code("""# @title §8. Сравнение пяти моделей
\"\"\"Сводная таблица внутренних метрик и долей кластеров.\"\"\"
rows = []
for name, metrics, labels in [
    ("1_KMeans", m1, labels_m1),
    ("2_GMM3_quantile", m2, labels_m2),
    ("3_PCA_KMeans", m3, labels_m3),
    ("4_ICA_KMeans", m4, labels_m4),
    ("5_GMM2_uncertain", m5, labels_m5),
]:
    row = {"model": name, **metrics, **cluster_fractions(labels)}
    rows.append(row)

compare_df = pd.DataFrame(rows).set_index("model")
display(compare_df.round(4))

best_sil = compare_df["silhouette"].idxmax()
print(f"лучший silhouette: {best_sil}")"""),
    md(_conc["AFTER_MODELS"]),
    code("""# @title §8. Подбор uncertain_fraction (модель 5)
\"\"\"Перебор доли «сомнительных» импульсов: 3%, 5%, 7%.\"\"\"
fracs = [0.03, 0.05, 0.07]
tune_rows = []
for uf in fracs:
    gmm = GaussianMixture(n_components=2, covariance_type="full", n_init=15, random_state=RANDOM_STATE)
    gmm.fit(z5)
    pr = gmm.predict_proba(z5)
    lab = gmm.predict(z5).astype(np.int64)
    unc = 1.0 - pr.max(axis=1)
    n_u = max(1, int(len(lab) * uf))
    lab[np.argsort(unc)[-n_u:]] = 2
    lab = remap_labels_physics(lab, features)
    met = clustering_metrics(z5[:, :3], lab)
    tune_rows.append({"uncertain_fraction": uf, **met, **cluster_fractions(lab)})

tune_df = pd.DataFrame(tune_rows)
display(tune_df.round(4))"""),
    md(_conc["AFTER_TUNE"]),
    code("""# @title §9. Финальная модель и submission.csv
\"\"\"Сохранение ответа в формате Kaggle: index, cluster.\"\"\"
FINAL_LABELS = labels_m5
UNCERTAIN_FRAC = 0.05

submission = pd.DataFrame({"index": np.arange(N_SAMPLES), "cluster": FINAL_LABELS.astype(int)})
out_path = Path("submission.csv")
submission.to_csv(out_path, index=False)

assert submission.shape[0] == N_SAMPLES
assert set(submission["cluster"].unique()).issubset({0, 1, 2})
print(submission["cluster"].value_counts().sort_index())
print(f"сохранено: {out_path.resolve()}")
submission.head()"""),
    md(_conc["FINAL"]),
    md(_conc["KAGGLE"]),
    md(_conc["KAGGLE_PLACEHOLDER"]),
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
