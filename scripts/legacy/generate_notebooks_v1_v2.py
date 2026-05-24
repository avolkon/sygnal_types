"""Generate notebooks: task (md) тЖТ code тЖТ analysis (code from prior outputs)."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]


def _md(text: str):
    return new_markdown_cell(text.strip())


def _code(text: str):
    return new_code_cell(text.strip())


def _analysis_from_vars(template: str) -> str:
    """Code cell: render Markdown from variables computed above."""
    return f"""
from IPython.display import Markdown, display

display(Markdown('''{template}'''))
""".strip()


def build_notebook_1() -> nbformat.NotebookNode:
    cells = [
        _md(
            """# `notebook_1_experience`

╨н╨║╤Б╨┐╨╡╤А╨╕╨╝╨╡╨╜╤В╤Л ╨┐╨╛ `╨Я╨╗╨░╨╜_╤А╨╡╨░╨╗╨╕╨╖╨░╤Ж╨╕╨╕.txt`: EDA тЖТ ╨┐╤А╨╡╨┐╤А╨╛╤Ж╨╡╤Б╤Б╨╕╨╜╨│ тЖТ ╨┐╤А╨╕╨╖╨╜╨░╨║╨╕ тЖТ k=2 тЖТ ╨║╨╗╨░╤Б╤В╨╡╤А 2 тЖТ ╨▓╤Л╨▒╨╛╤А ╨╝╨╛╨┤╨╡╨╗╨╕.

╨б╤В╤А╤Г╨║╤В╤Г╤А╨░ ╨║╨░╨╢╨┤╨╛╨│╨╛ ╤Н╤В╨░╨┐╨░: **╨╛╨┐╨╕╤Б╨░╨╜╨╕╨╡** тЖТ **╨║╨╛╨┤** тЖТ **╨░╨╜╨░╨╗╨╕╤В╨╕╨║╨░** (╤В╨╛╨╗╤М╨║╨╛ ╨╕╨╖ ╨┐╨╡╤А╨╡╨╝╨╡╨╜╨╜╤Л╤Е ╨┐╤А╨╡╨┤╤Л╨┤╤Г╤Й╨╡╨│╨╛ ╨║╨╛╨┤╨░)."""
        ),
        _md("## ╨н╤В╨░╨┐ 0. ╨Ю╤А╨│╨░╨╜╨╕╨╖╨░╤Ж╨╕╤П ╨╛╨║╤А╤Г╨╢╨╡╨╜╨╕╤П"),
        _code(
            """
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import DATA_PATH, N_FEATURES, N_SAMPLES, RANDOM_STATE
from sygnal_clustering.data import load_waveforms

sns.set_theme(style="whitegrid")
stage0 = {
    "random_state": RANDOM_STATE,
    "n_samples_expected": N_SAMPLES,
    "n_features_expected": N_FEATURES,
    "data_exists": DATA_PATH.exists(),
    "data_path": str(DATA_PATH),
}
print(json.dumps(stage0, indent=2, ensure_ascii=False))
"""
        ),
        _code(
            _analysis_from_vars(
                """### ╨н╤В╨░╨┐ 0 тАФ ML-╨░╤А╤Е╨╕╤В╨╡╨║╤В╨╛╤А
╨Ю╨║╤А╤Г╨╢╨╡╨╜╨╕╨╡: `random_state={stage0[random_state]}`, ╨╛╨╢╨╕╨┤╨░╨╡╨╝╨░╤П ╤А╨░╨╖╨╝╨╡╤А╨╜╨╛╤Б╤В╤М **{stage0[n_samples_expected]}├Ч{stage0[n_features_expected]}**, ╤Д╨░╨╣╨╗ ╨┤╨░╨╜╨╜╤Л╤Е ╤Б╤Г╤Й╨╡╤Б╤В╨▓╤Г╨╡╤В: **{stage0[data_exists]}**.

### ╨н╤В╨░╨┐ 0 тАФ ╤Д╨╕╨╖╨╕╨║
╨Ш╤Б╤В╨╛╤З╨╜╨╕╨║ `{stage0[data_path]}` ╤Б╨╛╨╛╤В╨▓╨╡╤В╤Б╤В╨▓╤Г╨╡╤В Run200 ╤Б╤Ж╨╕╨╜╤В╨╕╨╗╨╗╤П╤Ж╨╕╨╛╨╜╨╜╨╛╨│╨╛ ╨┤╨╡╤В╨╡╨║╤В╨╛╤А╨░; ╤Б╨╗╤Г╨╢╨╡╨▒╨╜╤Л╨╡ ╤Б╤В╨╛╨╗╨▒╤Ж╤Л ╨д╨н╨г ╨▒╤Г╨┤╤Г╤В ╨╛╤В╨▒╤А╨╛╤И╨╡╨╜╤Л ╨╜╨░ ╨╖╨░╨│╤А╤Г╨╖╨║╨╡."""
            )
        ),
        _md("## ╨н╤В╨░╨┐ 1. EDA"),
        _code(
            """
X = load_waveforms(DATA_PATH)
row_std = X.std(axis=1)
eda_stats = {
    "shape": list(X.shape),
    "finite": bool(np.isfinite(X).all()),
    "row_std_median": float(np.median(row_std)),
    "row_std_max": float(row_std.max()),
    "peak_median": float(np.median(X.max(axis=1))),
    "charge_median": float(np.median(X.sum(axis=1))),
}
print(json.dumps(eda_stats, indent=2))
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].hist(X.max(axis=1), bins=60, color="steelblue")
ax[0].set_title("Peak amplitude")
ax[1].hist(X.sum(axis=1), bins=60, color="darkorange")
ax[1].set_title("Charge")
plt.tight_layout()
plt.show()
"""
        ),
        _code(
            _analysis_from_vars(
                """### ╨н╤В╨░╨┐ 1 тАФ ML-╨░╤А╤Е╨╕╤В╨╡╨║╤В╨╛╤А
╨Ь╨░╤В╤А╨╕╤Ж╨░ **{eda_stats[shape]}**, ╨▓╤Б╨╡ ╨╖╨╜╨░╤З╨╡╨╜╨╕╤П ╨║╨╛╨╜╨╡╤З╨╜╤Л. ╨Ь╨╡╨┤╨╕╨░╨╜╨░ `std` ╨┐╨╛ ╤Б╤В╤А╨╛╨║╨░╨╝ **{eda_stats[row_std_median]:.2f}** (max **{eda_stats[row_std_max]:.2f}**) тАФ ╤Д╨╛╤А╨╝╨░ ╨╕╨╝╨┐╤Г╨╗╤М╤Б╨░ ╨▓╨░╤А╨╕╨░╤В╨╕╨▓╨╜╨░.

### ╨н╤В╨░╨┐ 1 тАФ ╤Д╨╕╨╖╨╕╨║
╨Ь╨╡╨┤╨╕╨░╨╜╨╜╤Л╨╣ ╨┐╨╕╨║ **{eda_stats[peak_median]:.1f}**, ╨╖╨░╤А╤П╨┤ **{eda_stats[charge_median]:.1f}**; ╨│╨╕╤Б╤В╨╛╨│╤А╨░╨╝╨╝╤Л ╤Г╨║╨░╨╖╤Л╨▓╨░╤О╤В ╨╜╨░ ╨╜╨╡╤Б╨║╨╛╨╗╤М╨║╨╛ ╨┐╨╛╨┐╤Г╨╗╤П╤Ж╨╕╨╣ (╬│, ╨╜╨╡╨╣╤В╤А╨╛╨╜╤Л, ╤Е╨▓╨╛╤Б╤В)."""
            )
        ),
        _md("## ╨н╤В╨░╨┐ 2. ╨Я╤А╨╡╨┤╨╛╨▒╤А╨░╨▒╨╛╤В╨║╨░ (RobustScaler)"),
        _code(
            """
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
const_cols = np.where(X.std(axis=0) < 1e-12)[0]
preprocess = {
    "constant_columns": int(len(const_cols)),
    "scaler": "RobustScaler",
    "scaled_abs_median": float(np.median(np.abs(X_scaled))),
}
print(json.dumps(preprocess, indent=2))
"""
        ),
        _code(
            _analysis_from_vars(
                """### ╨н╤В╨░╨┐ 2 тАФ ML-╨░╤А╤Е╨╕╤В╨╡╨║╤В╨╛╤А
╨Ъ╨╛╨╜╤Б╤В╨░╨╜╤В╨╜╤Л╤Е ╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓: **{preprocess[constant_columns]}**. ╨Ш╤Б╨┐╨╛╨╗╤М╨╖╨╛╨▓╨░╨╜ **{preprocess[scaler]}**; |median(scaled)|тЙИ**{preprocess[scaled_abs_median]:.3f}**.

### ╨н╤В╨░╨┐ 2 тАФ ╤Д╨╕╨╖╨╕╨║
╨Ь╤П╨│╨║╨╛╨╡ ╨╝╨░╤Б╤И╤В╨░╨▒╨╕╤А╨╛╨▓╨░╨╜╨╕╨╡ ╤Б╨╛╤Е╤А╨░╨╜╤П╨╡╤В ╨░╤Б╨╕╨╝╨╝╨╡╤В╤А╨╕╤О ╤Е╨▓╨╛╤Б╤В╨╛╨▓ PSD (╨▒╨╡╨╖ ╨░╨│╤А╨╡╤Б╤Б╨╕╨▓╨╜╨╛╨╣ ╤Б╤В╨░╨╜╨┤╨░╤А╤В╨╕╨╖╨░╤Ж╨╕╨╕)."""
            )
        ),
        _md("## ╨н╤В╨░╨┐ 3. Feature Engineering"),
        _code(
            """
from sygnal_clustering.features import (
    build_clustering_matrix,
    extract_domain_features,
    first_pc_amplitude_charge,
)

features = extract_domain_features(X, psd_offset=3, psd_short_len=30)
pc1 = first_pc_amplitude_charge(features, random_state=RANDOM_STATE)
Z = build_clustering_matrix(X, features, pc1, pca_components=25, random_state=RANDOM_STATE)
fe_info = {
    "n_domain_features": int(features.shape[1]),
    "Z_shape": list(Z.shape),
    "psd_median": float(np.median(features[:, 2])),
    "psd_std": float(features[:, 2].std()),
}
print(json.dumps(fe_info, indent=2))
"""
        ),
        _code(
            _analysis_from_vars(
                """### ╨н╤В╨░╨┐ 3 тАФ ML-╨░╤А╤Е╨╕╤В╨╡╨║╤В╨╛╤А
╨Ф╨╛╨╝╨╡╨╜╨╜╤Л╤Е ╨┐╤А╨╕╨╖╨╜╨░╨║╨╛╨▓: **{fe_info[n_domain_features]}**, ╨╝╨░╤В╤А╨╕╤Ж╨░ ╨║╨╗╨░╤Б╤В╨╡╤А╨╕╨╖╨░╤Ж╨╕╨╕ **{fe_info[Z_shape]}** (╨┤╨╛╨╝╨╡╨╜ + 1-╤П ╨У╨Ъ + PCA ╤Д╨╛╤А╨╝╤Л).

### ╨н╤В╨░╨┐ 3 тАФ ╤Д╨╕╨╖╨╕╨║
PSD: median **{fe_info[psd_median]:.4f}**, std **{fe_info[psd_std]:.4f}** тАФ ╨┤╨╕╤Б╨┐╨╡╤А╤Б╨╕╤П ╨┤╨╛╤Б╤В╨░╤В╨╛╤З╨╜╨░ ╨┤╨╗╤П ╨┐╨╛╤А╨╛╨│╨╛╨▓╨╛╨│╨╛ ╨╕ GMM-╤А╨░╨╖╨┤╨╡╨╗╨╡╨╜╨╕╤П."""
            )
        ),
        _md("## ╨н╤В╨░╨┐ 3.5. ╨С╨╕╨╜╨░╤А╨╜╨╛╨╡ ╤А╨░╨╖╨┤╨╡╨╗╨╡╨╜╨╕╨╡ (GMM k=2)"),
        _code(
            """
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler as RS

z_bin = RS().fit_transform(features[:, :4])
gmm2 = GaussianMixture(n_components=2, n_init=15, random_state=RANDOM_STATE)
lab2 = gmm2.fit_predict(z_bin)
proba2 = gmm2.predict_proba(z_bin).max(axis=1)
binary_info = {
    "gmm2_sizes": [int(x) for x in np.bincount(lab2, minlength=2)],
    "low_confidence_frac_0.72": float(np.mean(proba2 < 0.72)),
    "mean_max_proba": float(proba2.mean()),
}
print(json.dumps(binary_info, indent=2))
"""
        ),
        _code(
            _analysis_from_vars(
                """### ╨н╤В╨░╨┐ 3.5 тАФ ML-╨░╤А╤Е╨╕╤В╨╡╨║╤В╨╛╤А
GMM-2 ╤А╨░╨╖╨╝╨╡╤А╤Л: **{binary_info[gmm2_sizes]}**. ╨Ф╨╛╨╗╤П ╨╜╨╕╨╖╨║╨╛╨╣ ╤Г╨▓╨╡╤А╨╡╨╜╨╜╨╛╤Б╤В╨╕ (<0.72): **{binary_info[low_confidence_frac_0.72]:.1%}**; mean max proba **{binary_info[mean_max_proba]:.3f}**.

### ╨н╤В╨░╨┐ 3.5 тАФ ╤Д╨╕╨╖╨╕╨║
╨С╨╕╨╜╨░╤А╨╜╤Л╨╣ ╤Б╨╗╨╛╨╣ ╨╛╤В╨┤╨╡╨╗╤П╨╡╤В ╬│/╨╜╨╡╨╣╤В╤А╨╛╨╜╤Л; ╤Б╨╛╨╝╨╜╨╕╤В╨╡╨╗╤М╨╜╤Л╨╡ ╤Б╨╛╨▒╤Л╤В╨╕╤П ╨┐╨╛╨╣╨┤╤Г╤В ╨▓ ╨║╨╗╨░╤Б╤В╨╡╤А 2 ╨╜╨░ ╤Б╨╗╨╡╨┤╤Г╤О╤Й╨╡╨╝ ╤Н╤В╨░╨┐╨╡."""
            )
        ),
        _md("## ╨н╤В╨░╨┐ 4тАУ5. ╨б╤А╨░╨▓╨╜╨╡╨╜╨╕╨╡ ╨╝╨╛╨┤╨╡╨╗╨╡╨╣, ╨▓╤Л╨▒╨╛╤А ╤Д╨╕╨╜╨░╨╗╨░, ╨░╤А╤В╨╡╤Д╨░╨║╤В╤Л"),
        _code(
            """
from sygnal_clustering.config import ARTIFACTS_DIR, SUBMISSION_PATH
from sygnal_clustering.pipeline import SygnalClusteringPipeline, compare_methods

comparison = compare_methods(X, random_state=RANDOM_STATE)
comparison_df = pd.DataFrame(comparison)
display(comparison_df)

# ╨╗╤Г╤З╤И╨╕╨╡ ╨│╨╕╨┐╨╡╤А╨┐╨░╤А╨░╨╝╨╡╤В╤А╤Л ╨╕╨╖ scripts/run_experiments.py
pipe = SygnalClusteringPipeline(
    psd_short_len=30,
    confidence_threshold=0.66,
    isolation_contamination=0.06,
    random_state=RANDOM_STATE,
)
labels = pipe.fit_predict(X)
metrics = pipe.metrics()
pipe.save_artifacts(ARTIFACTS_DIR)
sub_path = pipe.save_submission(SUBMISSION_PATH)
frac2 = metrics["cluster_2"] / (metrics["cluster_0"] + metrics["cluster_1"] + metrics["cluster_2"])
selection = {**metrics, "anomaly_fraction": float(frac2), "submission": str(sub_path)}
print(json.dumps(selection, indent=2))
"""
        ),
        _code(
            _analysis_from_vars(
                """## ╨Ш╤В╨╛╨│╨╛╨▓╨░╤П ╨╖╨░╨┐╨╕╤Б╨║╨░ тАФ ╨▓╤Л╨▒╨╛╤А ╨╝╨╛╨┤╨╡╨╗╨╕

### ML-╨░╤А╤Е╨╕╤В╨╡╨║╤В╨╛╤А
╨Ь╨╡╤В╨╛╨┤ **{selection[method]}**. Silhouette **{selection[silhouette]:.4f}**, CalinskiтАУHarabasz **{selection[calinski_harabasz]:.1f}**, DaviesтАУBouldin **{selection[davies_bouldin]:.3f}**.
╨а╨░╨╖╨╝╨╡╤А╤Л ╨║╨╗╨░╤Б╤В╨╡╤А╨╛╨▓ 0/1/2: **{selection[cluster_0]} / {selection[cluster_1]} / {selection[cluster_2]}** (╨┤╨╛╨╗╤П ╨░╨╜╨╛╨╝╨░╨╗╨╕╨╣ **{selection[anomaly_fraction]:.1%}**).
╨Ф╨▓╤Г╤Е╤Н╤В╨░╨┐╨╜╨░╤П ╤Б╤Е╨╡╨╝╨░ ╨┐╤А╨╡╨┤╨┐╨╛╤З╤В╨╕╤В╨╡╨╗╤М╨╜╨╡╨╡ end-to-end GMM-3 (╤Б╨╝. `comparison_df`). `submission.csv` тЖТ `{selection[submission]}`.
**Accuracy ╨╜╨░ Kaggle** ╨▓ ╤А╨╡╨┐╨╛╨╖╨╕╤В╨╛╤А╨╕╨╕ ╨╜╨╡ ╨╕╨╖╨╝╨╡╤А╤П╨╗╨░╤Б╤М (╨╜╨╡╤В ╤А╨░╨╖╨╝╨╡╤В╨║╨╕); ╨▓╨╜╤Г╤В╤А╨╡╨╜╨╜╨╕╨╣ silhouette > 0.7 тАФ ╨╛╤Б╨╜╨╛╨▓╨░╨╜╨╕╨╡ ╨┤╨╗╤П ╨╛╨╢╨╕╨┤╨░╨╜╨╕╤П ╨▓╤Л╨┐╨╛╨╗╨╜╨╡╨╜╨╕╤П ╨┐╨╛╤А╨╛╨│╨░ ╨в╨Ч (тЙе0.84); ╤Ж╨╡╨╗╨╡╨▓╨╛╨╣ >0.85 ╨┐╤А╨╛╨▓╨╡╤А╤П╨╡╤В╤Б╤П ╨╜╨░ ╨╗╨╕╨┤╨╡╤А╨▒╨╛╤А╨┤╨╡.

### ╨д╨╕╨╖╨╕╨║
╨Ъ╨╗╨░╤Б╤В╨╡╤А **2** ({selection[anomaly_fraction]:.1%}) тАФ ╨╜╨╕╨╖╨║╨░╤П ╤Г╨▓╨╡╤А╨╡╨╜╨╜╨╛╤Б╤В╤М GMM ╨╕ ╨▓╤Л╨▒╤А╨╛╤Б╤Л Isolation Forest. ╨Ъ╨╗╨░╤Б╤В╨╡╤А╤Л **0** ╨╕ **1** ╤Г╨┐╨╛╤А╤П╨┤╨╛╤З╨╡╨╜╤Л ╨┐╨╛ ╨╖╨░╤А╤П╨┤╤Г (╬│ тЖТ ╨╜╨╡╨╣╤В╤А╨╛╨╜╤Л). ╨а╨╡╨║╨╛╨╝╨╡╨╜╨┤╨╛╨▓╨░╨╜ ╤Д╨╕╨╜╨░╨╗╤М╨╜╤Л╨╣ ╨┐╨░╨╣╨┐╨╗╨░╨╣╨╜ ╨▓ `artifacts/pipeline.joblib`."""
            )
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def build_notebook_2() -> nbformat.NotebookNode:
    cells = [
        _md("# `notebook_2_model` тАФ ╨╕╨╜╤Д╨╡╤А╨╡╨╜╤Б ╨╕ submission"),
        _code(
            """
import sys
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import ARTIFACTS_DIR, DATA_PATH, SUBMISSION_PATH
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline import SygnalClusteringPipeline

pipe = SygnalClusteringPipeline.load(ARTIFACTS_DIR / "pipeline.joblib")
X = load_waveforms(DATA_PATH)
labels = pipe.fit_predict(X)
sub_path = pipe.save_submission(SUBMISSION_PATH)
inf_metrics = pipe.metrics()
sub_df = pd.read_csv(sub_path)
infer_result = {
    "n_rows": len(sub_df),
    "clusters": sorted(sub_df["cluster"].unique().tolist()),
    "metrics": inf_metrics,
    "path": str(sub_path),
}
print(infer_result)
display(sub_df.head())
"""
        ),
        _code(
            _analysis_from_vars(
                """### ML-╨░╤А╤Е╨╕╤В╨╡╨║╤В╨╛╤А
╨Ш╨╜╤Д╨╡╤А╨╡╨╜╤Б: **{infer_result[n_rows]}** ╤Б╤В╤А╨╛╨║, ╨║╨╗╨░╤Б╤В╨╡╤А╤Л **{infer_result[clusters]}**. Silhouette **{infer_result[metrics][silhouette]:.4f}**. ╨д╨░╨╣╨╗: `{infer_result[path]}`.

### ╨д╨╕╨╖╨╕╨║
╨а╨░╤Б╨┐╤А╨╡╨┤╨╡╨╗╨╡╨╜╨╕╨╡ ╨║╨╗╨░╤Б╤В╨╡╤А╨╛╨▓ ╨▓╨╛╤Б╨┐╤А╨╛╨╕╨╖╨▓╨╛╨┤╨╕╤В ╨╛╨▒╤Г╤З╨╡╨╜╨╕╨╡; ╨│╨╛╤В╨╛╨▓╨╛ ╨║ ╨╖╨░╨│╤А╤Г╨╖╨║╨╡ ╨╜╨░ Kaggle."""
            )
        ),
        _md(
            """## Kaggle тАФ ╤В╨░╨▒╨╗╨╕╤Ж╨░ ╨╗╨╕╨┤╨╡╤А╨╛╨▓ (╨┐╨╡╤А╨▓╨░╤П ╨╛╤В╨┐╤А╨░╨▓╨║╨░)

╨б╨║╤А╨╕╨╜╤И╨╛╤В: `╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░/kaggle_leaderboard_first_submission.png`."""
        ),
        _code(
            """
from IPython.display import Image, display

LEADERBOARD_IMG = ROOT / "╨а╨░╨╖╤А╨░╨▒╨╛╤В╨║╨░" / "kaggle_leaderboard_first_submission.png"
kaggle_leaderboard = {
    "competition": "╨Ъ╨╗╨░╤Б╤Б╨╕╤Д╨╕╨║╨░╤Ж╨╕╤П ╤В╨╕╨┐╨╛╨▓ ╤Б╨╕╨│╨╜╨░╨╗╨╛╨▓",
    "image_path": str(LEADERBOARD_IMG),
    "image_exists": LEADERBOARD_IMG.exists(),
    "rank": 33,
    "score": 0.36568,
    "submissions": 1,
}
print(kaggle_leaderboard)
if kaggle_leaderboard["image_exists"]:
    display(Image(filename=str(LEADERBOARD_IMG)))
"""
        ),
        _code(
            """
from IPython.display import Markdown, display

display(Markdown(f'''### Kaggle тАФ ML-╨░╤А╤Е╨╕╤В╨╡╨║╤В╨╛╤А
╨Я╨╡╤А╨▓╨░╤П ╨╛╤В╨┐╤А╨░╨▓╨║╨░: ╨╝╨╡╤Б╤В╨╛ **#{kaggle_leaderboard["rank"]}**, accuracy **{kaggle_leaderboard["score"]:.5f}** (╤Ж╨╡╨╗╤М тЙе 0.84). Silhouette **{infer_result["metrics"]["silhouette"]:.4f}**.

### Kaggle тАФ ╤Д╨╕╨╖╨╕╨║
Score **{kaggle_leaderboard["score"]:.5f}** тАФ ╤В╤А╨╡╨▒╤Г╨╡╤В╤Б╤П ╤Г╤В╨╛╤З╨╜╨╡╨╜╨╕╨╡ ╨┐╨╡╤А╨╡╨║╨╛╨┤╨╕╤А╨╛╨▓╨║╨╕ ╨║╨╗╨░╤Б╤В╨╡╤А╨╛╨▓ 0/1/2.'''))
"""
        ),
    ]
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


def main() -> None:
    for builder, name in [(build_notebook_1, "notebook_1_experience.ipynb"), (build_notebook_2, "notebook_2_model.ipynb")]:
        path = ROOT / name
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(builder(), f)
        print("Wrote", path)


if __name__ == "__main__":
    main()
