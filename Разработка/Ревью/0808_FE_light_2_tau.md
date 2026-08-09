# FE LIGHT-2: tau_eff + honesty

- Champ ref: **0.85838**
- Axis: honest single-exp `tau_eff` after 40% amplitude drop (not discrete decay=3)
- Split: GMM-2 equal-responsibility on `log(tau)` (means ≈ 5.19 / 15.37, thr=8.9060)
- Why not raw valley: mode2≈35.64036289077534 is a heavy-tail bump; valley_ratio_raw=0.046 is a false gate pass
- Clip diagnostic valley (p01–p95): 9.337340201390646 (vr=0.555)
- Class2: honesty (`qc_bad_decay_fit`) ∪ thin tau tails q[0.015,0.985] — not SNR bulk
- Polarity: large tau → 0 (LB-aligned; tau anti-corr PSD=-0.389)
- Fractions: `{'f0': 0.4398, 'f1': 0.5099, 'f2': 0.0503}`; diff vs M0: **7556**; agree on 0/1: **0.7164**
- honesty→2: 490; tail→2: 690

Upload: `submissions/light/LIGHT_2_tau_eff/submission.csv`

```json
{
  "LIGHT_2": {
    "fractions": {
      "f0": 0.4398,
      "f1": 0.5099,
      "f2": 0.0503
    },
    "diff_vs_M0": 7556,
    "agree_M0_on_01": 0.7163509927715253,
    "split": "gmm2_log",
    "thr": 8.90603176890801,
    "means_log": [
      1.6466591960685626,
      2.732441467904242
    ],
    "means_lin": [
      5.189613353646042,
      15.370367506530537
    ],
    "weights": [
      0.5306026908919714,
      0.46939730910802846
    ],
    "n_fit": 22989,
    "qlo": 2.7342019048284443,
    "qhi": 35.31042536830993,
    "n_honesty_class2": 490,
    "n_tail_class2": 690,
    "polarity": "large_tau_is_0",
    "tau_mean_0": 16.619982944355506,
    "tau_mean_1": 5.413860662010456
  }
}
```
