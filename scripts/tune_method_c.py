"""Tune method C uncertain_fraction and write submission4.csv."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sygnal_clustering.config import ARTIFACTS_V3_DIR, RANDOM_STATE, SUBMISSION4_PATH  # noqa: E402
from sygnal_clustering.data import load_waveforms  # noqa: E402
from sygnal_clustering.io import write_submission  # noqa: E402
from sygnal_clustering.pipeline import (  # noqa: E402
    method_c_gmm2_low_confidence,
    metrics_for_labels,
)

TARGET_CLUSTER2_FRAC = 0.05


def main() -> None:
    x = load_waveforms()
    results = []
    best_frac = 0.05
    best_dist = float("inf")
    best_labels = None

    for frac in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
        lab, fe = method_c_gmm2_low_confidence(x, uncertain_fraction=frac, random_state=RANDOM_STATE)
        m = metrics_for_labels(lab, fe, f"gmm2_unc_{frac:.2f}")
        m["uncertain_fraction"] = frac
        results.append(m)
        dist = abs(m["fraction_2"] - TARGET_CLUSTER2_FRAC)
        if dist < best_dist:
            best_dist = dist
            best_frac = frac
            best_labels = lab

    assert best_labels is not None
    write_submission(best_labels, SUBMISSION4_PATH)
    report_path = ARTIFACTS_V3_DIR / "tune_method_c.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "best_uncertain_fraction": best_frac,
        "target_cluster2_fraction": TARGET_CLUSTER2_FRAC,
        "grid": results,
        "note": "Kaggle best was 3c at uncertain_fraction=0.05, score 0.44571",
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
