"""Generate notebook_3_expeerience and notebook_4_model2."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def _md(text: str):
    return new_markdown_cell(text.strip())


def _code(text: str):
    return new_code_cell(text.strip())


def _analysis(template: str) -> str:
    return f"""
from IPython.display import Markdown, display
display(Markdown('''{template}'''))
""".strip()


def build_notebook_3() -> nbformat.NotebookNode:
    cells = [
        _md(
            """# `notebook_3_expeerience`

Альтернативный пайплайн **v2** (баланс кластеров, без Isolation Forest v1).

Рекомендации: `Разработка/Рекомендации_улучшения_Kaggle.md`.  
Выход: `submission2.csv`, артефакты: `artifacts/v2/`."""
        ),
        _md("## Этап 0. Окружение"),
        _code(
            """
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import DATA_PATH, RANDOM_STATE
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline_v2 import (
    ARTIFACTS_V2_DIR,
    SUBMISSION2_PATH,
    SygnalClusteringPipelineV2,
    compare_v2_methods,
)

stage0_v2 = {"data_path": str(DATA_PATH), "random_state": RANDOM_STATE}
print(json.dumps(stage0_v2, indent=2))
"""
        ),
        _code(
            _analysis(
                """### Этап 0 — ML-архитектор / физик
Запуск v2: критерий отбора — баланс кластеров, не только silhouette v1."""
            )
        ),
        _md("## Этап 1. Сравнение вариантов v2"),
        _code(
            """
X = load_waveforms(DATA_PATH)
comparison_v2 = compare_v2_methods(X, random_state=RANDOM_STATE)
comparison_v2_df = pd.DataFrame(comparison_v2)
display(comparison_v2_df)
"""
        ),
        _code(
            _analysis(
                """### Этап 1 — ML-архитектор
Сравнены `balanced_gmm_quantile_psd` и `pc1_psd_tails`. См. `max_cluster_fraction` и доли кластеров в таблице — v2 избегает ~90% в одном классе как v1.

### Этап 1 — физик
GMM на квантиль-нормированном PSD-блоке ближе к двум популяциям + отдельный хвост аномалий."""
            )
        ),
        _md("## Этап 2. Финальная модель v2 и submission2.csv"),
        _code(
            """
pipe_v2 = SygnalClusteringPipelineV2(
    psd_short_len=30,
    anomaly_quantile=0.10,
    use_gmm_primary=True,
    random_state=RANDOM_STATE,
)
labels_v2 = pipe_v2.fit_predict(X)
metrics_v2 = pipe_v2.metrics()
pipe_v2.save_artifacts(ARTIFACTS_V2_DIR)
sub2_path = pipe_v2.save_submission(SUBMISSION2_PATH)
selection_v2 = {**metrics_v2, "submission2": str(sub2_path)}
print(json.dumps(selection_v2, indent=2, ensure_ascii=False))
"""
        ),
        _code(
            _analysis(
                """## Итог v2 — выбор модели

### ML-архитектор
Метод **{selection_v2[method]}**. Silhouette **{selection_v2[silhouette]:.4f}**, max доля класса **{selection_v2[max_cluster_fraction]:.3f}**.
Кластеры: **{selection_v2[cluster_0]} / {selection_v2[cluster_1]} / {selection_v2[cluster_2]}**.
Файл: `{selection_v2[submission2]}`. Сравнить accuracy на Kaggle с v1 (`submission.csv`).

### Физик
Более равномерное разбиение γ/нейтронов; класс 2 — хвосты PSD / малый GMM-компонент. Ожидается рост accuracy относительно 0.36568."""
            )
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def build_notebook_4() -> nbformat.NotebookNode:
    cells = [
        _md("# `notebook_4_model2` — инференс v2 → `submission2.csv`"),
        _code(
            """
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import DATA_PATH
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline_v2 import ARTIFACTS_V2_DIR, SUBMISSION2_PATH, SygnalClusteringPipelineV2

pipe_v2 = SygnalClusteringPipelineV2.load(ARTIFACTS_V2_DIR / "pipeline_v2.joblib")
X = load_waveforms(DATA_PATH)
labels_v2 = pipe_v2.fit_predict(X)
sub2_path = pipe_v2.save_submission(SUBMISSION2_PATH)
inf_v2 = pipe_v2.metrics()
sub2_df = pd.read_csv(sub2_path)
infer_v2 = {"path": str(sub2_path), "metrics": inf_v2, "head": sub2_df.head(3).to_dict()}
print(infer_v2)

LEADERBOARD_IMG = ROOT / "Разработка" / "kaggle_leaderboard_first_submission.png"
if LEADERBOARD_IMG.exists():
    display(Image(filename=str(LEADERBOARD_IMG)))
"""
        ),
        _code(
            _analysis(
                """### ML-архитектор
`submission2.csv`: silhouette **{infer_v2[metrics][silhouette]:.4f}**, max_frac **{infer_v2[metrics][max_cluster_fraction]:.3f}**. Отправить на Kaggle рядом с v1.

### Физик
Скриншот v1 — baseline 0.36568; v2 — гипотеза более физичного баланса классов."""
            )
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def main() -> None:
    for builder, name in [
        (build_notebook_3, "notebook_3_expeerience.ipynb"),
        (build_notebook_4, "notebook_4_model2.ipynb"),
    ]:
        path = ROOT / name
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(builder(), f)
        print("Wrote", path)


if __name__ == "__main__":
    main()
