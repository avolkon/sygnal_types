# FE formula grid: поиск «их» gate (ветка `exp/alt-formula-gate`)

Чемпион main: **0.85838** — `(L−S)/L`, окна (4,42), valley+0.003, q015–985.

## Стратегия
- Крупная локальная сетка формул × окон (без массовых upload).
- Локальный proxy: `valley_ratio`, разделение мод, умеренный `diff_vs_champ`.
- На Kaggle — только top-8 из `submissions/fe_formula_grid/export/`.

## Первый upload (приоритет)

`submissions\fe_formula_grid\export\FG_W4_42_diff_over_sum\submission.csv`

| Поле | Значение |
|---|---|
| formula | `diff_over_sum` |
| windows | (4, 42) |
| diff vs champ | 16 |
| valley_ratio | 0.1077 |
| proxy_score | 1.1192 |

Ориентир: **> 0.85838**. Если ≤ — следующий из `export_order` в `grid_summary.json`.

## LB

| Вариант | Score |
|---|---:|
| champ (main) | **0.85838** |
| ALT_f_diff_over_sum (старый alt) | *(ожидает upload)* |
| `FG_W4_42_diff_over_sum` | *(ожидает upload)* |
