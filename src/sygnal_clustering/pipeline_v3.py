"""Backward compatibility: v3 API lives in sygnal_clustering.pipeline."""

from sygnal_clustering.pipeline import (  # noqa: F401
    SUBMISSION3_PATH,
    SUBMISSION3A_PATH,
    SUBMISSION3B_PATH,
    SUBMISSION3C_PATH,
    labels_to_submission,
    method_a_meta_col2_tercile,
    method_b_description_gmm3,
    method_c_gmm2_low_confidence,
    metrics_for_labels,
    run_all_v3,
)

__all__ = [
    "SUBMISSION3A_PATH",
    "SUBMISSION3B_PATH",
    "SUBMISSION3C_PATH",
    "SUBMISSION3_PATH",
    "labels_to_submission",
    "method_a_meta_col2_tercile",
    "method_b_description_gmm3",
    "method_c_gmm2_low_confidence",
    "metrics_for_labels",
    "run_all_v3",
]
