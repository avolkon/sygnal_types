# FE radio: энергозависимый PSD-cut (ветка `radio`)

Чемпион: **0.85838** — глобальный valley+0.003, окна (4,42), class2 q015–985.

Гипотеза H-radio: порог 0/1 = f(энергия), не одна вертикаль на PSD.
Заморожено: формула `(L−S)/L`, окна, class2. Меняется только per-bin thr.

## Первый upload

`submissions\radio_energy\RAD_peak_q4_valley\submission.csv`

| Поле | Значение |
|---|---|
| энергия | `peak_above`, 4 квартиля |
| thr | valley_bin + 0.003 (fallback = global) |
| diff vs champ | **134** |
| fractions | {'f0': 0.4908, 'f1': 0.4791, 'f2': 0.0301} |

Ориентир: **> 0.85838**. Если ≤ — `RAD_charge_q4_valley`, затем stop H-radio.

## LB

| Вариант | Score |
|---|---:|
| champ | **0.85838** |
| `RAD_peak_q4_valley` | *(ожидает upload)* |
