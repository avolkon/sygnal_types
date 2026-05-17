# sygnal_types

Кластеризация сигналов сцинтилляционного детектора (γ / нейтроны / аномалии) — соревнование [Signal types classification](https://www.kaggle.com/competitions/signal-types-classification).

**Лучший результат Kaggle:** accuracy **0.44571** (место **#29**), файл **`submission3c.csv`**.

## Быстрый старт (сдача)

| Что | Где |
|-----|-----|
| Эксперименты + EDA | [`notebook_5_v3_experience.ipynb`](notebook_5_v3_experience.ipynb) |
| Финальная модель + скриншот Kaggle | [`notebook_6_v3_model.ipynb`](notebook_6_v3_model.ipynb) |
| Ответ для Kaggle | [`submission3c.csv`](submission3c.csv) или [`submission.csv`](submission.csv) |
| Итоговый отчёт | [`Разработка/Ревью/Итоговый_отчёт.md`](Разработка/Ревью/Итоговый_отчёт.md) |
| Чеклист сдачи | [`Разработка/Ревью/Чеклист_сдачи.md`](Разработка/Ревью/Чеклист_сдачи.md) |

```bash
pip install -r requirements.txt
# Windows: set PYTHONPATH=src
# Linux/macOS: export PYTHONPATH=src
pytest tests -q
python scripts/run_experiments_v3.py
jupyter notebook notebook_5_v3_experience.ipynb
```

## Структура репозитория

**Корень:**

| Путь | Назначение |
|------|------------|
| `Run200_Wave_0_1.txt` | Датасет (23 479 × 500) |
| `notebook_5_v3_experience.ipynb` | EDA, сравнение моделей A/B/C |
| `notebook_6_v3_model.ipynb` | **Финальный инференс** (метод C) |
| `submission3c.csv` | Лучший submission |
| `submission.csv` | Копия 3c (формат ТЗ) |
| `src/sygnal_clustering/` | Код пайплайна |
| `requirements.txt` | Зависимости |

**`Разработка/`:**

| Путь | Назначение |
|------|------------|
| `Подготовка/` | ТЗ, Description, план |
| `Ревью/` | Итоговый отчёт, чеклист сдачи |
| `Эксперименты/` | Архив ноутбуков v1–v2, скриншоты, промежуточные submission |

## Финальная модель

- **Метод:** `method_c_gmm2_low_confidence` (`pipeline_v3` + `signal_extraction`)
- **Схема:** признаки по Description (3σ, PSD, decay 40%) → GMM k=2 → 5% uncertain → кластер 2
- **Кластеры:** ~52% / ~43% / ~5%

## Google Colab

1. Клонировать репозиторий в `/content/sygnal_types` или положить `Run200_Wave_0_1.txt` в `/content/`
2. Установить зависимости, задать `PYTHONPATH=src`
3. Выполнить ноутбуки 5 → 6
4. Закрепить скриншот лидерборда (уже в notebook_6)

## Лицензия и данные

Данные `Run200_Wave_0_1.txt` — в рамках учебного проекта МФТИ / Kaggle.
