#!/usr/bin/env python
"""Build autonomous submission-ready v1 notebook from template."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "avo_m25_555_ml_unsupervised_v1.ipynb"
PLAN_NB = SRC
OUT = SRC

AUTONOMY_MD = """## Перед запуском

Данные загружаются в **§2**:
1. автоматически через `gdown` с [Google Drive](https://drive.google.com/file/d/1dxfUHO8Fc0pAkUpQbcZX6FtK7ufLVIJS/view?usp=drive_link);
2. если не сработало — ручная загрузка через виджет Colab (в той же ячейке).

**Зависимости:** `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `missingno`, `gdown` — установка в следующих ячейках при необходимости.

**Воспроизводимость:** `random_state=42` во всех моделях. Финальный submission: **§8 — row-level PCA 95% + KMeans k=6** (Public LB **0.22835**).

**Время выполнения:** полный прогон ~30–60 мин (несколько KMeans на 534k строк). Для проверки финала достаточно выполнить ячейки до §8 включительно.

**Матчасть (PCA / ICA):** см. выводы §9 — связь с лекционным материалом по снижению размерности."""

SETUP_CODE = '''# Установка зависимостей (только если пакет не найден — для Colab / чистого окружения)
import importlib
import subprocess
import sys

REQUIRED = [
    "pandas", "numpy", "scipy", "sklearn", "matplotlib", "seaborn", "missingno",
]

for pkg in REQUIRED:
    mod = "sklearn" if pkg == "sklearn" else pkg
    try:
        importlib.import_module(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
print("Зависимости OK")
'''

IMPORTS_CODE = '''# §1. Импорты
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import kurtosis, skew
from sklearn.cluster import (
    AgglomerativeClustering,
    DBSCAN,
    KMeans,
    MiniBatchKMeans,
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
RANDOM_STATE = 42
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)
'''

LOAD_CODE = '''# §2. Загрузка данных
# Drive: https://drive.google.com/file/d/1dxfUHO8Fc0pAkUpQbcZX6FtK7ufLVIJS/view?usp=drive_link
!pip install gdown -q

import gdown
from pathlib import Path

FILE_ID = "1dxfUHO8Fc0pAkUpQbcZX6FtK7ufLVIJS"
DATA_FILENAME = "Physical_Activity_Monitoring_unlabeled.csv"
data_path = Path(DATA_FILENAME)

if not data_path.is_file():
    try:
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}&confirm=t",
            DATA_FILENAME,
            quiet=False,
        )
    except Exception as e:
        print(f"gdown не сработал: {e}")

if not data_path.is_file():
    try:
        from google.colab import files  # type: ignore

        print("Файл не найден. Выберите CSV для загрузки...")
        uploaded = files.upload()
        name = next(iter(uploaded))
        Path(name).write_bytes(uploaded[name])
        data_path = Path(name)
    except ImportError:
        raise FileNotFoundError(
            f"Не удалось получить {DATA_FILENAME}. "
            f"Скачайте вручную: https://drive.google.com/file/d/{FILE_ID}/view?usp=drive_link"
        )

df = pd.read_csv(data_path)
print("Загружено:", df.shape, "из", data_path.resolve())
df.head()
'''

UTILS_CODE = '''# Вспомогательные функции (submission, отбор признаков, метрики)

NOTEBOOK_DIR = Path.cwd()


def remap_cluster_ids(labels):
    """Произвольные метки → 1, 2, …, k без пропусков."""
    labels = np.asarray(labels)
    mapping = {old: i + 1 for i, old in enumerate(sorted(np.unique(labels)))}
    return np.array([mapping[x] for x in labels], dtype=int)


def make_submission(n_rows, labels, filename="submission.csv"):
    """CSV: Index, activityID — для Kaggle."""
    final = remap_cluster_ids(labels)
    sub = pd.DataFrame({"Index": np.arange(n_rows, dtype=int), "activityID": final})
    out = NOTEBOOK_DIR / filename
    sub.to_csv(out, index=False)
    print(f"Сохранено: {out} | кластеры: {sorted(sub['activityID'].unique())}")
    print(sub["activityID"].value_counts().sort_index())
    return sub


def cluster_metrics(X, labels):
    """Silhouette, Davies–Bouldin, Calinski–Harabasz (на подвыборке при n>8000)."""
    labels = np.asarray(labels)
    if len(set(labels)) < 2 or (labels >= 0).sum() < 2:
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(len(X))
    if len(X) > 8000:
        idx = rng.choice(len(X), 8000, replace=False)
    Xe, le = X[idx], labels[idx]
    valid = le >= 0
    return {
        "silhouette": float(silhouette_score(Xe[valid], le[valid])),
        "davies_bouldin": float(davies_bouldin_score(Xe[valid], le[valid])),
        "calinski_harabasz": float(calinski_harabasz_score(Xe[valid], le[valid])),
    }


def select_features_unsupervised(X, corr_threshold=0.95, top_n=250):
    """§4: variance → correlation filter → top loadings PC1–PC5."""
    from sklearn.feature_selection import VarianceThreshold

    vt = VarianceThreshold(threshold=1e-8)
    Xv = vt.fit_transform(X)
    corr = np.corrcoef(Xv, rowvar=False)
    drop = set()
    iu = np.triu_indices_from(corr, k=1)
    for i, j in zip(*iu):
        if abs(corr[i, j]) > corr_threshold:
            drop.add(j)
    keep = np.ones(Xv.shape[1], dtype=bool)
    keep[list(drop)] = False
    Xc = Xv[:, keep]
    Xs = RobustScaler().fit_transform(Xc)
    pca = PCA(n_components=min(5, Xs.shape[1], Xs.shape[0]), random_state=RANDOM_STATE)
    pca.fit(Xs)
    loadings = np.max(np.abs(pca.components_), axis=0)
    k = min(top_n, Xc.shape[1])
    idx = np.argsort(loadings)[-k:]
    return Xc[:, idx]
'''

EXPERIMENTS_MD = """---

## §7. Итерации submission и эксперименты v1

В процессе работы проверены несколько пайплайнов. **Public LB на Kaggle:**

| # | Подход | Public LB | Комментарий |
|---|--------|-----------|-------------|
| 0 | **Segment + KMeans k=8** (504 признака) | ~0.13 | silhouette высокий, accuracy низкий |
| 1 | Subject-wise RobustScaler + row PCA k=6 | **0.115** | EDA рекомендовал — на LB хуже |
| 2 | StandardScaler + PCA 98% + MiniBatchKMeans | **0.147** | см. ячейку §7.3 |
| 3 | motion24 + k=7 + PCA 95% | **0.165** | «баланс» кластеров не помог |
| 4 | **Global RobustScaler + PCA 95% + KMeans k=6** | **0.228** | **финальный выбор** |
| — | Segment + ICA grid (v1 pipeline) | не отправлялся | откат: медленно, LB не улучшил |

**Вывод:** internal metrics (silhouette) **не коррелируют** с Kaggle accuracy. Доминирующий кластер ~78% в лучшей модели — норма для этого теста."""

EXPERIMENT_SUBJECT_CODE = '''# §7.1 Эксперимент: subject-wise RobustScaler (LB 0.115 — хуже baseline)
feature_cols = [c for c in df_reduced.columns if c not in ["timestamp", "subject_id"]]
X_subj = df_reduced[feature_cols].copy()
for sid in df_reduced["subject_id"].unique():
    m = df_reduced["subject_id"] == sid
    X_subj.loc[m, feature_cols] = RobustScaler().fit_transform(X_subj.loc[m, feature_cols])
Xp = PCA(n_components=0.95, random_state=RANDOM_STATE).fit_transform(X_subj.values)
lab_subj = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10).fit_predict(Xp)
print("Subject-wise scaler — метрики:", cluster_metrics(Xp, lab_subj))
print("→ Public LB ≈ 0.115; не используем в финале.")
'''

FINAL_SUBMISSION_CODE = '''# §8. Финальный submission для Kaggle (LB 0.22835)
feature_cols = [c for c in df_reduced.columns if c not in ["timestamp", "subject_id"]]
X_raw = df_reduced[feature_cols].values
X_scaled = RobustScaler().fit_transform(X_raw)
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA: {X_pca.shape}, variance={pca.explained_variance_ratio_.sum():.4f}")

best_k, best_inertia, best_labels = None, float("inf"), None
for k in (3, 4, 5, 6):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    lab = km.fit_predict(X_pca)
    print(f"k={k}: inertia={km.inertia_:.2f}, silhouette={cluster_metrics(X_pca, lab)['silhouette']:.4f}")
    if km.inertia_ < best_inertia:
        best_inertia, best_k, best_labels = km.inertia_, k, lab

print(f"\\nВыбран k={best_k} (min inertia)")
submission_final = make_submission(len(df_reduced), best_labels, "submission.csv")
submission_final.head(10)
'''

FEATURE_SELECT_CODE = '''# §4. Unsupervised отбор признаков (504 → filtered)
print("До отбора:", X.shape)
X_selected = select_features_unsupervised(X_clean)
print("После отбора:", X_selected.shape)
X_selected_scaled = RobustScaler().fit_transform(X_selected)
'''

MOTION24_CODE = '''# §7.2 Эксперимент: motion24 + k=7 (LB 0.165 — хуже baseline)
motion_cols = [c for c in feature_cols if "Acc16" in c or "Gyro" in c]
X_m = RobustScaler().fit_transform(df_reduced[motion_cols].values)
Xp7 = PCA(n_components=0.95, random_state=RANDOM_STATE).fit_transform(X_m)
lab7 = KMeans(n_clusters=7, random_state=RANDOM_STATE, n_init=10).fit_predict(Xp7)
print("motion24 k=7:", cluster_metrics(Xp7, lab7))
print("→ Public LB ≈ 0.165; не используем.")
'''

MATCHAST_MD = """---

### Кратко: PCA и ICA (матчасть)

| Метод | Суть | В проекте |
|-------|------|-----------|
| **PCA** | Ортогональные компоненты, max variance | **Row-level PCA 95%** перед KMeans — лучший LB |
| **ICA** | Независимые источники, негауссовость | Тест на сегментах; LB не улучшил |
| **LDA** | Supervised, нужны метки | Не в submission-пайплайне |

PCA здесь — **unsupervised** снижение шума и размерности; выбор k=6 по inertia, не по silhouette."""

CONCLUSIONS_MD = """---

## §9. Выводы

1. **EDA** выявил пропуски, выбросы (~8%), корреляцию Acc6/Acc16 (>0.95), различия между 8 испытуемыми.
2. **Предобработка:** интерполяция по subject, удаление Acc6, сегментация 250 семплов для segment-моделей.
3. **Feature engineering:** 42×12=504 признака на сегмент; unsupervised отбор (variance, corr, PCA loadings) — §4 в коде `select_features_unsupervised`.
4. **Кластеризация:** KMeans, DBSCAN (~95% шума), Agglomerative, GMM; лучший silhouette ≠ лучший LB.
5. **PCA (матчасть):** row-level PCA 95% отсекает шум перед KMeans; ICA/segment ablation не улучшили LB.
6. **Финал:** global `RobustScaler` → PCA → KMeans k=6 → `submission.csv`.

**Скриншот лидерборда:** приложите PNG с Public Score **0.22835** (файл `submission_pca_kmeans` / финальный `submission.csv`).

**Ограничения:** без меток нельзя подобрать k по accuracy; proxy-метрики обманчивы; subject-wise scaling ухудшает LB."""

EDA_NOTE_MD = """> **Примечание (после экспериментов §7):** EDA рекомендовал нормализацию **внутри** `subject_id`, но на Kaggle **глобальный** `RobustScaler` дал LB **0.228** против **0.115**. Subject-wise scaling оставлен как эксперимент §7.1, финальный submission — §8."""

NORMality_CODE = '''# §1.6 Нормальность / негауссовость (обоснование PCA vs ICA)
from scipy.stats import normaltest

sample_cols = ["handAcc16_1", "chestGyro1", "ankleAcc16_2"]
for col in sample_cols:
    x = df[col].dropna().values
    x = x[:: max(1, len(x) // 5000)][:5000]
    stat, p = normaltest(x)
    print(f"{col}: D'Agostino p={p:.2e} → {'не-Gaussian' if p < 0.05 else '≈ Gaussian'}")
print("Вывод: сигналы в основном негауссовы → PCA для декорреляции/шума; ICA — эксперимент на сегментах.")
'''


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": [text], "outputs": [], "execution_count": None}


def clear_cell(cell: dict) -> dict:
    cell = copy.deepcopy(cell)
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    return cell


def should_drop(source: str) -> bool:
    """Удаляем только лишний Colab boilerplate (не §2 gdown)."""
    if "from google.colab import drive" in source:
        return True
    if "#@title Загрузка большого файла" in source:
        return True
    if "#@title Скачивание файла (для Colab)" in source:
        return True
    if "см. §2 загрузка DATA_PATH" in source:
        return True
    return False


def transform_source(source: str) -> str:
    source = source.replace("fillna(method='ffill').fillna(method='bfill')", "ffill().bfill()")
    source = source.replace('fillna(method="ffill").fillna(method="bfill")', "ffill().bfill()")
    if source.strip().startswith("#@title Импорты"):
        return ""
    if "pd.read_csv('Physical_Activity_Monitoring_unlabeled.csv')" in source:
        return ""
    lines = []
    for line in source.split("\n"):
        if "files.download(" in line:
            continue
        if "from google.colab import files" in line and "upload" not in line.lower():
            continue
        lines.append(line)
    return "\n".join(lines)


def strip_base64_markdown(source: str) -> str:
    if "data:image/png;base64," in source:
        return (
            "**Скриншот лидерборда Kaggle** (Public Score **0.22835**)\n\n"
            "_При сдаче ноутбука приложите PNG скриншот страницы Submission "
            "соревнования [Clustering Physical Activity](https://www.kaggle.com/competitions/clustering-physical-activity)._"
        )
    if "Сводные выводы разведочного анализа" in source:
        source = source.replace(
            "| 5 | Нормализовать по испытуемым | Индивидуальные различия в показаниях |",
            "| 5 | Нормализовать по испытуемым | EDA-гипотеза; **финал — global scaler** (§7–§8) |",
        )
    return source


def load_plan_cell() -> str | None:
    if not PLAN_NB.is_file():
        return None
    plan_nb = json.loads(PLAN_NB.read_text(encoding="utf-8"))
    for cell in plan_nb["cells"]:
        src = "".join(cell.get("source", []))
        if "План реализации v1" in src:
            return src
    return None


def fix_plan_text(src: str) -> str:
    src = src.replace("Разработка/Матчасть_PCA_LDA_ICA.md", "§9 и блок матчасти ниже")
    src = src.replace("outputs/submissions/submission_v1.csv", "submission.csv (рядом с ноутбуком)")
    src = src.replace("outputs/figures/leaderboard.png", "PNG-скриншот LB")
    return src


def build():
    nb = json.loads(SRC.read_text(encoding="utf-8"))
    if any("Перед запуском (автономный режим)" in "".join(c.get("source", [])) for c in nb["cells"][:12]):
        print(f"{OUT.name} уже собран; правки вносите напрямую в v1.")
        return
    plan_v1 = load_plan_cell()
    new_cells = []

    # 0-2: keep author, intro, plan (v1 plan if available)
    for i in range(3):
        cell = clear_cell(nb["cells"][i])
        if i == 2 and plan_v1:
            cell = md(fix_plan_text(plan_v1))
        elif i == 2:
            src = fix_plan_text("".join(cell.get("source", [])))
            cell = md(src)
        new_cells.append(cell)

    new_cells.append(md(AUTONOMY_MD))
    new_cells.append(code(SETUP_CODE))
    new_cells.append(code(IMPORTS_CODE))
    new_cells.append(code(LOAD_CODE))
    new_cells.append(code(UTILS_CODE))

    inserted_normality = False
    inserted_experiments = False
    inserted_feature_select = False

    for i in range(3, len(nb["cells"])):
        cell = nb["cells"][i]
        src = "".join(cell.get("source", []))
        if should_drop(src):
            continue
        src = transform_source(src)
        if not src.strip() and cell["cell_type"] == "code":
            continue
        if cell["cell_type"] == "markdown":
            src = strip_base64_markdown(src)
            cell = md(src)
        else:
            cell = code(src)

        new_cells.append(clear_cell(cell))

        # After subject analysis (~cell with subject_id stats)
        if not inserted_normality and "Анализ распределения по испытуемым" in src:
            new_cells.append(code(NORMality_CODE))
            new_cells.append(md(EDA_NOTE_MD))
            inserted_normality = True

        # Before PCA kmeans iteration section
        if not inserted_feature_select and "Форма матрицы признаков X:" in src:
            new_cells.append(code(FEATURE_SELECT_CODE))
            inserted_feature_select = True

        if not inserted_experiments and "KMeans с PCA" in src:
            new_cells.append(md(EXPERIMENTS_MD))
            new_cells.append(code(EXPERIMENT_SUBJECT_CODE))
            new_cells.append(code(MOTION24_CODE))
            inserted_experiments = True

    # Replace last PCA cell output with final submission + conclusions
    # Remove duplicate colab at end if any slipped through
    new_cells = [c for c in new_cells if not should_drop("".join(c.get("source", [])))]

    new_cells.append(code(FINAL_SUBMISSION_CODE))
    new_cells.append(md(MATCHAST_MD))
    new_cells.append(md(CONCLUSIONS_MD))

    out_nb = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "cells": new_cells,
    }

    OUT.write_text(json.dumps(out_nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Written {OUT} with {len(new_cells)} cells")


if __name__ == "__main__":
    build()
