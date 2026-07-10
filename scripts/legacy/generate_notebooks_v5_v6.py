"""Generate notebook_5_v3_experience and notebook_6_v3_model."""

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

Три контрольных submission для Kaggle (план: `Разработка/План_v3_Kaggle.md`).

| Вариант | Файл | Метод |
|---------|------|--------|
| A | submission3a.csv | третили `col2` |
| B | submission3b.csv | Description + GMM-3 |
| C | submission3c.csv | GMM-2 + 5% uncertain → 2 |
| основной | submission3.csv | копия B |"""
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

from sygnal_clustering.config import DATA_PATH
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline_v3 import run_all_v3

X = load_waveforms(DATA_PATH)
report_v3 = run_all_v3(X)
print(json.dumps(report_v3, indent=2, ensure_ascii=False))
"""
        ),
        _ana(
            """### ML-архитектор / физик (после run_all_v3)
См. `report_v3`: размеры кластеров и silhouette по A/B/C. Отправить **все три** файла на Kaggle и записать score."""
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
        _ana(
            """### Сравнение v1/v2
- v1: ~88% в классе 0 (score 0.36568)
- v2: ~48/22/30 (score 0.34)
- v3A/B/C: см. вывод выше — выбрать лучший по лидерборду."""
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def build_nb6():
    cells = [
        _md("# `notebook_6_v3_model` — воспроизведение submission3"),
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
from sygnal_clustering.pipeline_v3 import (
    SUBMISSION3_PATH,
    method_b_description_gmm3,
    labels_to_submission,
)

X = load_waveforms(DATA_PATH)
labels_b, _ = method_b_description_gmm3(X)
labels_to_submission(labels_b, SUBMISSION3_PATH)
sub3 = pd.read_csv(SUBMISSION3_PATH)
print(sub3["cluster"].value_counts().sort_index())
display(sub3.head())

img = ROOT / "Разработка" / "kaggle_leaderboard_first_submission.png"
if img.exists():
    display(Image(filename=str(img)))
"""
        ),
        _ana(
            """### Итог
`submission3.csv` обновлён (вариант B). Сравнить новый score с 0.36568 и 0.34."""
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
