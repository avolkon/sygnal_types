"""Generate production notebooks (5 and 6)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def _md(t: str):
    return new_markdown_cell(t.strip())


def _code(t: str):
    return new_code_cell(t.strip())


def _ana(t: str):
    return _code(
        f"""
from IPython.display import Markdown, display
display(Markdown('''{t}'''))
""".strip()
    )


def build_nb5():
    cells = [
        _md(
            """# `notebook_5_v3_experience`

Три контрольных submission для Kaggle (план: `Разработка/Ревью/План_v3_Kaggle.md`).

| Вариант | Файл | Метод |
|---------|------|--------|
| A | submission3a.csv | третили `col2` |
| B | submission3b.csv | Description + GMM-3 |
| C | submission3c.csv | GMM-2 + 5% uncertain → 2 |
| основной | submission3.csv | копия **C** |"""
        ),
        _md("## Запуск трёх методов"),
        _code(
            """
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline import run_all_v3

X = load_waveforms()
report_v3 = run_all_v3(X)
print(json.dumps(report_v3, indent=2, ensure_ascii=False))
"""
        ),
        _ana(
            """### ML-архитектор / физик (после run_all_v3)
Рекомендован **C** (Kaggle 0.44571). Отправить все три файла на Kaggle для сравнения."""
        ),
        _md("## Просмотр распределений"),
        _code(
            """
for key, path in report_v3["paths"].items():
    df = pd.read_csv(path)
    print(key, path)
    print(df["cluster"].value_counts().sort_index(), "\\n")
"""
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def build_nb6():
    cells = [
        _md(
            """# `notebook_6_v3_model` — основной инференс (метод C, score 0.44571)

Финальный пайплайн: Description-признаки + GMM-2 + 5% uncertain → кластер 2.
Файл для Kaggle: **`submission3c.csv`** (копии: `submission3.csv`, `submission.csv`)."""
        ),
        _code(
            """
import sys
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import (
    EXPERIMENTS_DIR,
    KAGGLE_LEADERBOARD_BEST,
    SUBMISSION3C_PATH,
    SUBMISSION3_PATH,
    SUBMISSION_PATH,
)
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline import (
    labels_to_submission,
    method_c_gmm2_low_confidence,
)

X = load_waveforms()
labels_c, _ = method_c_gmm2_low_confidence(X)
labels_to_submission(labels_c, SUBMISSION3C_PATH)
labels_to_submission(labels_c, SUBMISSION3_PATH)
labels_to_submission(labels_c, SUBMISSION_PATH)
sub = pd.read_csv(SUBMISSION3C_PATH)
print(sub["cluster"].value_counts().sort_index())

if KAGGLE_LEADERBOARD_BEST.exists():
    display(Image(filename=str(KAGGLE_LEADERBOARD_BEST)))
else:
    print("Скриншот не найден:", KAGGLE_LEADERBOARD_BEST)
"""
        ),
        _ana(
            """### Итог
**method_c_gmm2_low_confidence** — production pipeline. Kaggle score **0.44571**."""
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def main():
    out_dir = ROOT / "notebooks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for fn, builder in [
        ("notebook_5_v3_experience.ipynb", build_nb5),
        ("notebook_6_v3_model.ipynb", build_nb6),
    ]:
        p = out_dir / fn
        with open(p, "w", encoding="utf-8") as f:
            nbformat.write(builder(), f)
        print("Wrote", p)


if __name__ == "__main__":
    main()
