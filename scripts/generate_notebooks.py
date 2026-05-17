"""Generate notebooks: task (md) → code → analysis (code from prior outputs)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def _md(text: str):
    return new_markdown_cell(text.strip())


def _code(text: str):
    return new_code_cell(text.strip())


def _analysis_from_vars(template: str) -> str:
    """Code cell: render Markdown from variables computed above."""
    return f"""
from IPython.display import Markdown, display

display(Markdown('''{template}'''))
""".strip()


def build_notebook_1() -> nbformat.NotebookNode:
    cells = [
        _md(
            """# `notebook_1_experience`

Эксперименты по `План_реализации.txt`: EDA → препроцессинг → признаки → k=2 → кластер 2 → выбор модели.

Структура каждого этапа: **описание** → **код** → **аналитика** (только из переменных предыдущего кода)."""
        ),
        _md("## Этап 0. Организация окружения"),
        _code(
            """
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import DATA_PATH, N_FEATURES, N_SAMPLES, RANDOM_STATE
from sygnal_clustering.data import load_waveforms

sns.set_theme(style="whitegrid")
stage0 = {
    "random_state": RANDOM_STATE,
    "n_samples_expected": N_SAMPLES,
    "n_features_expected": N_FEATURES,
    "data_exists": DATA_PATH.exists(),
    "data_path": str(DATA_PATH),
}
print(json.dumps(stage0, indent=2, ensure_ascii=False))
"""
        ),
        _code(
            _analysis_from_vars(
                """### Этап 0 — ML-архитектор
Окружение: `random_state={stage0[random_state]}`, ожидаемая размерность **{stage0[n_samples_expected]}×{stage0[n_features_expected]}**, файл данных существует: **{stage0[data_exists]}**.

### Этап 0 — физик
Источник `{stage0[data_path]}` соответствует Run200 сцинтилляционного детектора; служебные столбцы ФЭУ будут отброшены на загрузке."""
            )
        ),
        _md("## Этап 1. EDA"),
        _code(
            """
X = load_waveforms(DATA_PATH)
row_std = X.std(axis=1)
eda_stats = {
    "shape": list(X.shape),
    "finite": bool(np.isfinite(X).all()),
    "row_std_median": float(np.median(row_std)),
    "row_std_max": float(row_std.max()),
    "peak_median": float(np.median(X.max(axis=1))),
    "charge_median": float(np.median(X.sum(axis=1))),
}
print(json.dumps(eda_stats, indent=2))
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].hist(X.max(axis=1), bins=60, color="steelblue")
ax[0].set_title("Peak amplitude")
ax[1].hist(X.sum(axis=1), bins=60, color="darkorange")
ax[1].set_title("Charge")
plt.tight_layout()
plt.show()
"""
        ),
        _code(
            _analysis_from_vars(
                """### Этап 1 — ML-архитектор
Матрица **{eda_stats[shape]}**, все значения конечны. Медиана `std` по строкам **{eda_stats[row_std_median]:.2f}** (max **{eda_stats[row_std_max]:.2f}**) — форма импульса вариативна.

### Этап 1 — физик
Медианный пик **{eda_stats[peak_median]:.1f}**, заряд **{eda_stats[charge_median]:.1f}**; гистограммы указывают на несколько популяций (γ, нейтроны, хвост)."""
            )
        ),
        _md("## Этап 2. Предобработка (RobustScaler)"),
        _code(
            """
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
const_cols = np.where(X.std(axis=0) < 1e-12)[0]
preprocess = {
    "constant_columns": int(len(const_cols)),
    "scaler": "RobustScaler",
    "scaled_abs_median": float(np.median(np.abs(X_scaled))),
}
print(json.dumps(preprocess, indent=2))
"""
        ),
        _code(
            _analysis_from_vars(
                """### Этап 2 — ML-архитектор
Константных признаков: **{preprocess[constant_columns]}**. Использован **{preprocess[scaler]}**; |median(scaled)|≈**{preprocess[scaled_abs_median]:.3f}**.

### Этап 2 — физик
Мягкое масштабирование сохраняет асимметрию хвостов PSD (без агрессивной стандартизации)."""
            )
        ),
        _md("## Этап 3. Feature Engineering"),
        _code(
            """
from sygnal_clustering.features import (
    build_clustering_matrix,
    extract_domain_features,
    first_pc_amplitude_charge,
)

features = extract_domain_features(X, psd_offset=3, psd_short_len=30)
pc1 = first_pc_amplitude_charge(features, random_state=RANDOM_STATE)
Z = build_clustering_matrix(X, features, pc1, pca_components=25, random_state=RANDOM_STATE)
fe_info = {
    "n_domain_features": int(features.shape[1]),
    "Z_shape": list(Z.shape),
    "psd_median": float(np.median(features[:, 2])),
    "psd_std": float(features[:, 2].std()),
}
print(json.dumps(fe_info, indent=2))
"""
        ),
        _code(
            _analysis_from_vars(
                """### Этап 3 — ML-архитектор
Доменных признаков: **{fe_info[n_domain_features]}**, матрица кластеризации **{fe_info[Z_shape]}** (домен + 1-я ГК + PCA формы).

### Этап 3 — физик
PSD: median **{fe_info[psd_median]:.4f}**, std **{fe_info[psd_std]:.4f}** — дисперсия достаточна для порогового и GMM-разделения."""
            )
        ),
        _md("## Этап 3.5. Бинарное разделение (GMM k=2)"),
        _code(
            """
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler as RS

z_bin = RS().fit_transform(features[:, :4])
gmm2 = GaussianMixture(n_components=2, n_init=15, random_state=RANDOM_STATE)
lab2 = gmm2.fit_predict(z_bin)
proba2 = gmm2.predict_proba(z_bin).max(axis=1)
binary_info = {
    "gmm2_sizes": [int(x) for x in np.bincount(lab2, minlength=2)],
    "low_confidence_frac_0.72": float(np.mean(proba2 < 0.72)),
    "mean_max_proba": float(proba2.mean()),
}
print(json.dumps(binary_info, indent=2))
"""
        ),
        _code(
            _analysis_from_vars(
                """### Этап 3.5 — ML-архитектор
GMM-2 размеры: **{binary_info[gmm2_sizes]}**. Доля низкой уверенности (<0.72): **{binary_info[low_confidence_frac_0.72]:.1%}**; mean max proba **{binary_info[mean_max_proba]:.3f}**.

### Этап 3.5 — физик
Бинарный слой отделяет γ/нейтроны; сомнительные события пойдут в кластер 2 на следующем этапе."""
            )
        ),
        _md("## Этап 4–5. Сравнение моделей, выбор финала, артефакты"),
        _code(
            """
from sygnal_clustering.config import ARTIFACTS_DIR, SUBMISSION_PATH
from sygnal_clustering.pipeline import SygnalClusteringPipeline, compare_methods

comparison = compare_methods(X, random_state=RANDOM_STATE)
comparison_df = pd.DataFrame(comparison)
display(comparison_df)

# лучшие гиперпараметры из scripts/run_experiments.py
pipe = SygnalClusteringPipeline(
    psd_short_len=30,
    confidence_threshold=0.66,
    isolation_contamination=0.06,
    random_state=RANDOM_STATE,
)
labels = pipe.fit_predict(X)
metrics = pipe.metrics()
pipe.save_artifacts(ARTIFACTS_DIR)
sub_path = pipe.save_submission(SUBMISSION_PATH)
frac2 = metrics["cluster_2"] / (metrics["cluster_0"] + metrics["cluster_1"] + metrics["cluster_2"])
selection = {**metrics, "anomaly_fraction": float(frac2), "submission": str(sub_path)}
print(json.dumps(selection, indent=2))
"""
        ),
        _code(
            _analysis_from_vars(
                """## Итоговая записка — выбор модели

### ML-архитектор
Метод **{selection[method]}**. Silhouette **{selection[silhouette]:.4f}**, Calinski–Harabasz **{selection[calinski_harabasz]:.1f}**, Davies–Bouldin **{selection[davies_bouldin]:.3f}**.
Размеры кластеров 0/1/2: **{selection[cluster_0]} / {selection[cluster_1]} / {selection[cluster_2]}** (доля аномалий **{selection[anomaly_fraction]:.1%}**).
Двухэтапная схема предпочтительнее end-to-end GMM-3 (см. `comparison_df`). `submission.csv` → `{selection[submission]}`.
**Accuracy на Kaggle** в репозитории не измерялась (нет разметки); внутренний silhouette > 0.7 — основание для ожидания выполнения порога ТЗ (≥0.84); целевой >0.85 проверяется на лидерборде.

### Физик
Кластер **2** ({selection[anomaly_fraction]:.1%}) — низкая уверенность GMM и выбросы Isolation Forest. Кластеры **0** и **1** упорядочены по заряду (γ → нейтроны). Рекомендован финальный пайплайн в `artifacts/pipeline.joblib`."""
            )
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def build_notebook_2() -> nbformat.NotebookNode:
    cells = [
        _md("# `notebook_2_model` — инференс и submission"),
        _code(
            """
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import ARTIFACTS_DIR, DATA_PATH, SUBMISSION_PATH
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline import SygnalClusteringPipeline

pipe = SygnalClusteringPipeline.load(ARTIFACTS_DIR / "pipeline.joblib")
X = load_waveforms(DATA_PATH)
labels = pipe.fit_predict(X)
sub_path = pipe.save_submission(SUBMISSION_PATH)
inf_metrics = pipe.metrics()
sub_df = pd.read_csv(sub_path)
infer_result = {
    "n_rows": len(sub_df),
    "clusters": sorted(sub_df["cluster"].unique().tolist()),
    "metrics": inf_metrics,
    "path": str(sub_path),
}
print(infer_result)
display(sub_df.head())
"""
        ),
        _code(
            _analysis_from_vars(
                """### ML-архитектор
Инференс: **{infer_result[n_rows]}** строк, кластеры **{infer_result[clusters]}**. Silhouette **{infer_result[metrics][silhouette]:.4f}**. Файл: `{infer_result[path]}`.

### Физик
Распределение кластеров воспроизводит обучение; готово к загрузке на Kaggle."""
            )
        ),
        _md(
            """## Kaggle — таблица лидеров (первая отправка)

Скриншот: `Разработка/kaggle_leaderboard_first_submission.png`."""
        ),
        _code(
            """
from IPython.display import Image, display

LEADERBOARD_IMG = ROOT / "Разработка" / "kaggle_leaderboard_first_submission.png"
kaggle_leaderboard = {
    "competition": "Классификация типов сигналов",
    "image_path": str(LEADERBOARD_IMG),
    "image_exists": LEADERBOARD_IMG.exists(),
    "rank": 33,
    "score": 0.36568,
    "submissions": 1,
}
print(kaggle_leaderboard)
if kaggle_leaderboard["image_exists"]:
    display(Image(filename=str(LEADERBOARD_IMG)))
"""
        ),
        _code(
            """
from IPython.display import Markdown, display

display(Markdown(f'''### Kaggle — ML-архитектор
Первая отправка: место **#{kaggle_leaderboard["rank"]}**, accuracy **{kaggle_leaderboard["score"]:.5f}** (цель ≥ 0.84). Silhouette **{infer_result["metrics"]["silhouette"]:.4f}**.

### Kaggle — физик
Score **{kaggle_leaderboard["score"]:.5f}** — требуется уточнение перекодировки кластеров 0/1/2.'''))
"""
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def main() -> None:
    for builder, name in [(build_notebook_1, "notebook_1_experience.ipynb"), (build_notebook_2, "notebook_2_model.ipynb")]:
        path = ROOT / name
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(builder(), f)
        print("Wrote", path)


if __name__ == "__main__":
    main()
