# FE crazy: residual puzzle pipelines

Branch: **`crazy`**. Base LB **0.89109** (`P14_2b_qlo_0070` contract).

Каждая гипотеза = отдельный пайплайн / submission. Одна заливка — одна гипотеза.

## Сводка

| name | fractions | diff vs base | rule |
|---|---|---:|---|
| `A0_disagree_decay` | {'f0': 0.1243, 'f1': 0.4565, 'f2': 0.4192} | 8199 | psd_vs_decay_disagree->2 + lo_q07 |
| `A0b_disagree_pca` | {'f0': 0.2236, 'f1': 0.3425, 'f2': 0.4338} | 8542 | psd_vs_pca_disagree->2 + lo_q07 |
| `A1_isthmus_flip` | {'f0': 0.4475, 'f1': 0.4825, 'f2': 0.07} | 821 | flip 0<->1 if |psd-thr|<0.015 |
| `A2_multipeak` | {'f0': 0.4303, 'f1': 0.4926, 'f2': 0.0771} | 166 | multi_peak->2 on base |
| `A2b_nonproportional` | {'f0': 0.4367, 'f1': 0.4903, 'f2': 0.073} | 69 | |logCharge - (a+b logPeak)| > 2.5 sigma -> 2 |
| `A3_index_blocks` | {'f0': 0.4338, 'f1': 0.496, 'f2': 0.0702} | 112 | per-index 8 blocks valley+lo7% |
| `A4b_decay_extremes` | {'f0': 0.4276, 'f1': 0.4917, 'f2': 0.0807} | 251 | decay hang|<=1|q99 -> 2 on base |
| `A5b_balance_ritual` | {'f0': 0.4068, 'f1': 0.4932, 'f2': 0.1} | 704 | ritual f2~10% lo-only |
| `A6b_seed_text` | {'f0': 0.3897, 'f1': 0.4403, 'f2': 0.17} | 2348 | rng(seed=27052019) punch 10% of run -> 2 |
| `A6c_q0_mirror` | {'f0': 0.4412, 'f1': 0.4888, 'f2': 0.07} | 5392 | mirror 0<->1 in peak quartile 0 |
| `A6c_q1_mirror` | {'f0': 0.4613, 'f1': 0.4687, 'f2': 0.07} | 4944 | mirror 0<->1 in peak quartile 1 |
| `A6c_q2_mirror` | {'f0': 0.4133, 'f1': 0.5167, 'f2': 0.07} | 5629 | mirror 0<->1 in peak quartile 2 |
| `A6c_q3_mirror` | {'f0': 0.4877, 'f1': 0.4423, 'f2': 0.07} | 5870 | mirror 0<->1 in peak quartile 3 |
| `A3b_rolling_baseline` | {'f0': 0.4368, 'f1': 0.4932, 'f2': 0.07} | 0 | rolling pedestal window=500 then lo-only PSD |

## Upload order (абсурд → риск)

1. `A0_disagree_decay` — PSD∧decay несогласие → 2  
2. `A2b_nonproportional` — нарушение amp∝charge → 2  
3. `A4b_decay_extremes` — дискретный decay как ключ 2  
4. `A1_isthmus_flip` — flip только в перешейке  
5. далее по `summary.json`

Ориентир: **> 0.89109**.  
Сценарий: `0808_Сценарий_crazy.md`.
