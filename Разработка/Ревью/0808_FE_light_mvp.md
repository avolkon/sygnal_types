# FE LIGHT MVP

- Champ ref: **0.85838**
- LIGHT-0 M0 fractions: `{'f0': 0.4918, 'f1': 0.4782, 'f2': 0.0301}` valley_ratio=0.1083 thr=0.639819
- LIGHT-1 psd_front: `{'f0': 0.3059, 'f1': 0.664, 'f2': 0.0301}` valley_ratio=0.9213 thr=0.431591
- LIGHT-2 tau_eff: `{'f0': 0.4398, 'f1': 0.5099, 'f2': 0.0503}` thr=8.9060 agree_M0=0.7164
- L1 diff vs M0: **5842**; L2 diff vs M0: **7556**

```json
{
  "title": "LIGHT MVP (0/1/2)",
  "champion_score_ref": 0.85838,
  "LIGHT_0": {
    "fractions": {
      "f0": 0.4918,
      "f1": 0.4782,
      "f2": 0.0301
    },
    "valley_ratio": 0.10831234255542994,
    "valley": 0.6368187270804372,
    "thr": 0.6398187270804372,
    "qlo": 0.47838671991316434,
    "qhi": 0.7683496648036692,
    "mode1": 0.53581160970174,
    "mode2": 0.6784098930599007
  },
  "LIGHT_1": {
    "fractions": {
      "f0": 0.3059,
      "f1": 0.664,
      "f2": 0.0301
    },
    "diff_vs_M0": 5842,
    "corr_psd_front_argmax": 0.8085153858897927,
    "n_qc_no_front": 0,
    "valley_ratio": 0.9213319759601858,
    "valley": 0.42859103391204745,
    "thr": 0.43159103391204745,
    "qlo": 0.32947025083073644,
    "qhi": 0.6105699681524317,
    "mode1": 0.4209530484387648,
    "mode2": 0.45914297580517827
  },
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
  },
  "upload_hint": "Upload LIGHT_2_tau_eff/submission.csv as upload #2 (tau hypothesis).",
  "next": "LIGHT-3 rho two-exp if tau axis is not worse diagnostically"
}
```
