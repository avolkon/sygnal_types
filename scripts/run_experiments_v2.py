"""Train v2 pipeline and write submission2.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import DATA_PATH, RANDOM_STATE  # noqa: E402
from sygnal_clustering.data import load_waveforms  # noqa: E402
from sygnal_clustering.pipeline_v2 import (  # noqa: E402
    ARTIFACTS_V2_DIR,
    SUBMISSION2_PATH,
    SygnalClusteringPipelineV2,
    compare_v2_methods,
)


def balance_score(metrics: dict) -> float:
    """Higher is better: prefer moderate silhouette and max cluster fraction < 0.65."""
    sil = metrics.get("silhouette", 0.0)
    max_frac = metrics.get("max_cluster_fraction", 1.0)
    penalty = max(0.0, max_frac - 0.65) * 2.0
    frac2 = metrics.get("fraction_2", 0.0)
    if frac2 == 0.0:
        c2 = metrics.get("cluster_2", 0)
        n = metrics.get("cluster_0", 0) + metrics.get("cluster_1", 0) + c2
        frac2 = c2 / n if n else 0.0
    penalty2 = 0.0
    if frac2 < 0.03 or frac2 > 0.35:
        penalty2 = 0.1
    return sil - penalty - penalty2


def main() -> None:
    x = load_waveforms(DATA_PATH)
    comparison = compare_v2_methods(x, random_state=RANDOM_STATE)

    best_score = -1.0
    best_pipe: SygnalClusteringPipelineV2 | None = None
    best_metrics: dict | None = None

    for psd_len in (25, 30, 35):
        for aq in (0.08, 0.10, 0.12):
            for use_gmm in (True, False):
                pipe = SygnalClusteringPipelineV2(
                    psd_short_len=psd_len,
                    anomaly_quantile=aq,
                    use_gmm_primary=use_gmm,
                    random_state=RANDOM_STATE,
                )
                pipe.fit_predict(x)
                m = pipe.metrics()
                score = balance_score(m)
                if score > best_score:
                    best_score = score
                    best_pipe = pipe
                    best_metrics = m

    assert best_pipe is not None and best_metrics is not None
    best_pipe.save_artifacts(ARTIFACTS_V2_DIR)
    path = best_pipe.save_submission(SUBMISSION2_PATH)

    report = {
        "best_balance_score": best_score,
        "best_metrics": best_metrics,
        "comparison": comparison,
        "submission2": str(path),
    }
    with open(ARTIFACTS_V2_DIR / "experiment_report_v2.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
