"""Legacy v1/v2 pipelines (archived experiments)."""

from sygnal_clustering.legacy.pipeline_v1 import SygnalClusteringPipeline, compare_methods
from sygnal_clustering.legacy.pipeline_v2 import SygnalClusteringPipelineV2, compare_v2_methods

__all__ = [
    "SygnalClusteringPipeline",
    "SygnalClusteringPipelineV2",
    "compare_methods",
    "compare_v2_methods",
]
