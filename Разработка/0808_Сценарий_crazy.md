# Сценарий CRAZY: головоломка остатка ~11% (0808)

| Поле | Значение |
|---|---|
| **Ветка** | `crazy` (fork от `main`) |
| **База** | asymmetric lo-only PSD, LB **0.89109** |
| **Роль** | puzzle / non-linear residual (не матчасть PSD) |
| **Код** | `src/sygnal_clustering/crazy_hypotheses.py` |
| **Runner** | `scripts/run_crazy_pipelines.py` |

## Мандат

Традиционная логика вывела на плато ~0.89–0.90.  
Остаток — головоломка Description; каждая гипотеза — отдельный пайплайн.

## Пайплайны

| ID | Идея | Папка |
|---|---|---|
| CRAZY_0 | freeze base | `CRAZY_0_BASE` |
| A0 | PSD vs decay disagree → 2 | `A0_disagree_decay` |
| A0b | PSD vs PCA disagree → 2 | `A0b_disagree_pca` |
| A1 | isthmus flip 0↔1 | `A1_isthmus_flip` |
| A2 | multi-peak → 2 | `A2_multipeak` |
| A2b | non-proportional amp–charge → 2 | `A2b_nonproportional` |
| A3 | index blocks own valley | `A3_index_blocks` |
| A3b | rolling pedestal | `A3b_rolling_baseline` |
| A4b | decay extremes → 2 | `A4b_decay_extremes` |
| A5b | ritual f2≈10% | `A5b_balance_ritual` |
| A6b | seed 27052019 punch 10%→2 | `A6b_seed_text` |
| A6c | mirror in peak quartile 0..3 | `A6c_q*_mirror` |

## Правила

- Не смешивать две гипотезы в одном upload.  
- База 0/1+lo class2 не откатывать без LB-роста.  
- Стоп семейства: 2× ≤ 0.89109.

## Reproduce

```bash
git checkout crazy
python scripts/run_crazy_pipelines.py
```
