"""Grid search hyperparameters and save best pipeline artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import ARTIFACTS_DIR, DATA_PATH, RANDOM_STATE  # noqa: E402
from sygnal_clustering.data import load_waveforms  # noqa: E402
from sygnal_clustering.pipeline import SygnalClusteringPipeline, compare_methods  # noqa: E402


def selection_score(metrics: dict) -> float:
    """Composite score: silhouette + penalty if anomaly cluster size is unrealistic."""
    sil = metrics.get("silhouette", 0.0)
    c2 = metrics.get("cluster_2", 0)
    n = c2 + metrics.get("cluster_0", 0) + metrics.get("cluster_1", 0)
    frac = c2 / n if n else 0.0
    penalty = 0.0
    if frac < 0.02 or frac > 0.25:
        penalty = 0.15
    return sil - penalty


def main() -> None:
    x = load_waveforms(DATA_PATH)
    comparison = compare_methods(x, random_state=RANDOM_STATE)

    best_score = -1.0
    best_pipe: SygnalClusteringPipeline | None = None
    best_metrics: dict | None = None

    for psd_short in (25, 30, 35, 40):
        for conf in (0.66, 0.70, 0.72, 0.74, 0.78):
            for contam in (0.06, 0.08, 0.10):
                pipe = SygnalClusteringPipeline(
                    psd_short_len=psd_short,
                    confidence_threshold=conf,
                    isolation_contamination=contam,
                    random_state=RANDOM_STATE,
                )
                pipe.fit_predict(x)
                m = pipe.metrics()
                score = selection_score(m)
                if score > best_score:
                    best_score = score
                    best_pipe = pipe
                    best_metrics = m

    assert best_pipe is not None and best_metrics is not None
    best_pipe.save_artifacts(ARTIFACTS_DIR)
    best_pipe.save_submission()

    report = {
        "best_selection_score": best_score,
        "best_metrics": best_metrics,
        "comparison": comparison,
    }
    with open(ARTIFACTS_DIR / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Best metrics:", json.dumps(best_metrics, indent=2))
    print("Saved artifacts to", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
