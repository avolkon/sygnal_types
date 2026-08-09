# FE ось B: альтернативная физика PSD / ROI

Чемпион-референс: **0.85838** (3σ ROI, `(long−short)/long`, окна 4/42, q015–985).

## Гипотезы
1. Другой ROI (`n_sigma≠3`, спад до доли пика, фиксированная длина)
2. Другая формула PSD (`(L−S)/(L+S)`, `short/long`, …)
3. PSD на `x_norm`, другие baseline bins

Без гарантии 0.88.

## Важно

`short/long` ≈ инверсия `(long−short)/long` → почти swap 0↔1 (раньше swap **вредил**). Не первый тест.

## Upload order

| # | Файл | Идея |
|---|---|---|
| **1** | `ALT_nsig_40/` | ROI до **4σ** |
| 2 | `ALT_nsig_25/` | ROI 2.5σ |
| 3 | `ALT_fixlen_80/` | фиксированная длина ROI=80 |
| 4 | `ALT_roifrac_20/` | ROI до 0.2·peak |
| 5 | `ALT_f_diff_over_sum/` | (L−S)/(L+S) |
| 6 | `ALT_f_short_over_long/` | риск swap |

## LB

| Вариант | Score |
|---|---:|
| champ 3σ | **0.85838** |
| ALT_nsig_40 (4σ) | 0.85633 |

**Сейчас лей:** `submissions/fe_alt/ALT_nsig_25/submission.csv` (ROI 2.5σ)  
Ориентир: **> 0.85838**.
