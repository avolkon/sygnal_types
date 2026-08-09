# FE-0 baseline (EPIC-FE-0808)

| Поле | Значение |
|---|---|
| **Дата** | 09.08.2026 |
| **Submission** | `submissions/fe0_baseline/submission.csv` |
| **Модель** | legacy notebook §9: GMM-2 + `uncertain_fraction=0.05` на prep `features[:, :4]` |
| **Цель этапа** | число «до файнтюна» + freeze prep |

## Freeze prep

| Константа | Значение |
|---|---|
| DATA | `Run200_Wave_0_1.txt` |
| SHA256 | `87626b55d2b8659d7dc4296649fe49e7d6da36ca72396659de1ef1d1b54d33d9` |
| POLARITY | `negative` |
| PSD windows | offset=5, short=50 |
| valley_ratio PSD / decay | 0.0603 / 0.0000 |
| GATE_OK | True |
| X_clust | psd, decay_time, charge_roi, tail_ratio |
| PCA EVR (live) | [0.6103, 0.1992, 0.1051] |
| candidate_class2 (QC OR) | 0.0291 |

## Baseline labels

| Класс | Count | Fraction |
|---|---:|---:|
| 0 | 13311 | 0.5669 |
| 1 | 8995 | 0.3831 |
| 2 | 1173 | 0.05 |

## LB результат

| Версия | Score | Комментарий |
|---|---:|---|
| Исторический (ноутбук, pre-prep) | **0.45968** | GMM-2 + 5% |
| **FE-0** (`submission08bl.csv`) | **0.44938** | тот же рецепт на prep-признаках |

**Вывод:** FE-0 ниже истории на ≈0.010. Legacy-рецепт на новых признаках не улучшил LB. Дальше — кандидаты на live `X_clust` / PSD.

## Следующий шаг

Залить **B:** `submissions/fe_candidates/B_gmm2_xclust_overlap_qc/submission.csv`
