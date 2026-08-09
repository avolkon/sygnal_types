# sygnal_types

Кластеризация сигналов сцинтилляционного детектора (γ / нейтроны / аномалии) — соревнование [Signal types classification](https://www.kaggle.com/competitions/signal-types-classification).

**Лучший результат Kaggle (архив v3):** accuracy **0.44571** (место **#29**), файл `submission3c.csv`.  
**Сессионный проект:** accuracy **0.45968** (место **#160**), ноутбук `avo_sygnal_types_2.ipynb`.

## Быстрый старт

| Что | Где |
|-----|-----|
| **Сдача (основной ноутбук)** | [`notebooks/avo_sygnal_types_2.ipynb`](notebooks/avo_sygnal_types_2.ipynb) |
| Submission из ноутбука | [`notebooks/submission.csv`](notebooks/submission.csv) |
| Архив: EDA + сравнение A/B/C | [`notebooks/notebook_5_v3_experience.ipynb`](notebooks/notebook_5_v3_experience.ipynb) |
| Архив: инференс метод C | [`notebooks/notebook_6_v3_model.ipynb`](notebooks/notebook_6_v3_model.ipynb) |
| Датасет | [`data/Run200_Wave_0_1.txt`](data/Run200_Wave_0_1.txt) |

```bash
pip install -r requirements.txt
# Windows: set PYTHONPATH=src
# Linux/macOS: export PYTHONPATH=src
pytest tests -q
python scripts/run_experiments.py
jupyter notebook notebooks/avo_sygnal_types_2.ipynb
```

## Структура проекта

```
sygnal_types/
├── data/                  # Run200_Wave_0_1.txt (23 479 × 500)
├── notebooks/             # все .ipynb, submission.csv, скриншоты Kaggle
├── artifacts/             # артефакты пайплайнов
│   ├── v1/                # legacy pipeline v1
│   ├── v2/                # pipeline v2
│   └── v3/                # production v3 (method C)
├── src/sygnal_clustering/ # код: pipeline, признаки, модели
├── scripts/               # генераторы ноутбуков, эксперименты
├── tests/
└── Разработка/            # ТЗ, отчёты, архив экспериментов, примеры
```

## Финальная модель (production)

- **Метод:** `method_c_gmm2_low_confidence` (`pipeline.py` + `signal_extraction`)
- **Схема:** признаки по Description (PSD, decay 40%) → GMM k=2 → 5% uncertain → кластер 2
- **Артефакты:** `artifacts/v3/`

## Google Colab

1. Клонировать репозиторий или загрузить `data/Run200_Wave_0_1.txt` в `/content/data/`
2. `pip install -r requirements.txt`, `PYTHONPATH=src`
3. Открыть `notebooks/avo_sygnal_types_2.ipynb` → Run All

## Лицензия и данные

Данные `Run200_Wave_0_1.txt` — в рамках учебного проекта МИФИ / Kaggle.
