# FE angel: карта перешейка + probe (ветка `angel`)

Чемпион снаружи не трогаем: **0.85838**, окна (4,42), valley+0.003, q015–985.

Перешеек: `|PSD − thr| < 0.02` (thr=0.6398).

## Карта

| Регион | n | snr_med | peak_med | charge_med | roi_len_med |
|---|---:|---:|---:|---:|---:|
| soft & label0 (быстрые у края) | 429 | 112.1 | 300 | 1509 | 18 |
| soft & label1 (медленные у края) | 950 | 301.1 | 801 | 3947 | 24 |
| soft weak SNR≤q25 | 345 | 57.8 | — | — | — |

Полный JSON: `submissions\angel_isthmus\isthmus_map.json`.

**Вывод карты:** у края быстрые — тусклые; медленные — ярче. `decay` в перешейке бесполезен. Гипотеза probe: тусклые soft — плохая видимость формы → не насиловать 0/1.

## Первый upload

`submissions\angel_isthmus\ISTH_weak_to2\submission.csv`

| Поле | Значение |
|---|---|
| правило | soft & SNR≤q25 → **класс 2**; иначе champ |
| diff vs champ | **345** |
| mutated in soft | 345 |
| fractions | {'f0': 0.4845, 'f1': 0.4708, 'f2': 0.0448} |

Ориентир: **> 0.85838**. Если ≤ — `ISTH_narrow_weak_to2`, затем stop isthmus.
