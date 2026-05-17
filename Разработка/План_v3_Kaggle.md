# План v3 — калибровка под Kaggle

**Результаты v1 / v2:** accuracy **0.36568** / **0.34** — около случайного угадывания для 3 классов (~0.33).

## Цель v3

Три **контрольных** submission для проверки гипотез на лидерборде (не оптимизация silhouette).

| ID | Файл | Гипотеза |
|----|------|----------|
| **A** | `submission3a.csv` | Метка коррелирует с метаданными ФЭУ (`col2`) — третили |
| **B** | `submission3b.csv` | Параметризация по **Description** (3σ, PSD, 40% decay) + GMM-3 |
| **C** | `submission3c.csv` | GMM-2 на Description-признаках + **5%** низкой уверенности → кластер 2 |

**Рекомендуемый основной файл:** `submission3.csv` — копия варианта **C** (лучший score на Kaggle).

## Фактические результаты Kaggle (2026-05-17)

| Файл | Score | Комментарий |
|------|-------|-------------|
| submission.csv (v1) | **0.36568** | двухэтапный GMM + Isolation Forest |
| submission2.csv (v2) | **0.34447** | баланс кластеров |
| submission3a.csv (A) | **0.29426** | третили col2 — гипотеза не подтвердилась |
| submission3b.csv (B) | **0.36666** | Description + GMM-3 |
| **submission3c.csv (C)** | **0.44571** | **лучший** — GMM-2 + 5% uncertain → кластер 2 |
| submission3.csv | 0.36666 | была копия B; теперь копия C |

Скриншоты: `Разработка/Эксперименты/kaggle_leaderboard_best_0.44571.png`, `Разработка/Эксперименты/kaggle_submissions_v3_scores.png`.

**Следующий шаг (v4):** сетка `uncertain_fraction` для метода C → `submission4.csv`.

## Отличия от v1 / v2

| Версия | Проблема |
|--------|----------|
| v1 | ~90% в одном кластере; Isolation Forest |
| v2 | Искусственный баланс без физической разметки |
| **v3** | Три независимые гипотезы; выделение импульса по методичке |

## Действия после отправки

1. Загрузить `submission3a.csv`, `submission3b.csv`, `submission3c.csv` на Kaggle.
2. Зафиксировать score в `notebook_5_v3_experience`.
3. Развивать лучший вариант (сетка порогов PSD / short gate / доля кластера 2).

## Файлы репозитория

- `src/sygnal_clustering/signal_extraction.py`
- `src/sygnal_clustering/pipeline_v3.py`
- `scripts/run_experiments_v3.py`
- `notebook_5_v3_experience.ipynb`, `notebook_6_v3_model.ipynb`
- `artifacts_v3/`
