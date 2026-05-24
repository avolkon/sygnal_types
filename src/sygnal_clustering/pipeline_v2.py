"""Backward compatibility: v2 API lives in sygnal_clustering.legacy.pipeline_v2."""

from sygnal_clustering.config import ARTIFACTS_V2_DIR, SUBMISSION2_PATH
from sygnal_clustering.legacy.pipeline_v2 import (
    SygnalClusteringPipelineV2,
    compare_v2_methods,
)

__all__ = [
    "ARTIFACTS_V2_DIR",
    "SUBMISSION2_PATH",
    "SygnalClusteringPipelineV2",
    "compare_v2_methods",
]
