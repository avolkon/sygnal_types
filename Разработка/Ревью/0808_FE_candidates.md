# FE-1…FE-3 candidates (ожидают LB после FE-0)

| Поле | Значение |
|---|---|
| **Дата** | 09.08.2026 |
| **Prep freeze** | PSD windows (5, 50), GATE_OK, см. `0808_FE0_baseline.md` |
| **Вход моделей** | live `X_clust` = RobustScaler(psd, decay, charge_roi, tail_ratio) |
| **Remap 0↔1** | по среднему **PSD** (не charge) |

## Кандидаты

| ID | Путь submission | f0 / f1 / f2 | Идея |
|---|---|---|---|
| **FE-0** | `submissions/fe0_baseline/submission.csv` | 0.57 / 0.38 / **0.05** | legacy GMM-2 + top-5% (baseline) |
| A | `submissions/fe_candidates/A_gmm2_xclust_qc/` | 0.43 / 0.54 / 0.029 | GMM-2 на X_clust; class2=QC |
| B | `…/B_gmm2_xclust_overlap_qc/` | 0.43 / 0.51 / 0.061 | + overlap долины PSD |
| C | `…/C_gmm2_psd1d_qc/` | 0.46 / 0.51 / 0.029 | GMM-2 только по PSD |
| D | `…/D_gmm2_xclust_abs_unc_qc/` | 0.42 / 0.53 / 0.058 | абсолютный τ(unc) + QC |
| E | `…/E_gmm2_bestbic_qc/` | = A (full) | BIC-выбор covariance |

## LB loop

| ID | Score | Файл | Статус |
|---|---:|---|---|
| hist (pre-prep) | 0.45968 | — | старый ориентир |
| FE-0 | 0.44938 | submission08bl | хуже hist |
| B | 0.45811 | submission08gmm | ниже hist |
| B_swap01 | **0.39767** | submission08swap01 | ориентация B верная; swap вредит |
| A | 0.47063 | submission08A | била hist |
| C | 0.84564 | submission08C | GMM-PSD + QC |
| C_no_qc | 0.84539 | submissionCno_qc | без QC чуть хуже |
| **C_valley_p003** | **0.85182** | submissionC_valley_p003 | **чемпион** |
| C_valley_p002 | 0.85173 | submissionC_valley_p002 | −0.00009 |
| C_valley_p005 | 0.85165 | submissionC_valley_p005 | |
| C_valley_qc | 0.85161 | submissionC_valley_qc | |
| C_valley_p01 | 0.85148 | submissionC_valley_p01 | |
| C_valley_m01 | 0.84986 | submissionC_valley_m01 | |
| C_overlap_qc | 0.83461 | submissionC_overlap_qc | |
| **C_valley_p004** | ? | — | **следующий** (valley+0.004) |
| C_valley_p001 / p0025 | ? | — | очередь; затем стоп/апрув на фиксацию |

## Развилка

Пик у **+0.003**. Сосед слева хуже → проверяем справа:  
`submissions/fe_candidates/C_valley_p004/submission.csv`.  
Если ≤ p003 — фиксируем чемпиона (нужен апрув: стоп тюнинга порога / ещё 1–2 пробы).
