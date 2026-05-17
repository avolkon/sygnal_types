import numpy as np
import pytest

from sygnal_clustering.config import DATA_PATH, N_SAMPLES
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline_v2 import SUBMISSION2_PATH, SygnalClusteringPipelineV2


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_pipeline_v2_labels_and_balance():
    x = load_waveforms(DATA_PATH)
    pipe = SygnalClusteringPipelineV2()
    labels = pipe.fit_predict(x)
    assert labels.shape == (N_SAMPLES,)
    assert set(np.unique(labels)) <= {0, 1, 2}
    m = pipe.metrics()
    assert m["max_cluster_fraction"] < 0.75


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_submission2_format(tmp_path):
    x = load_waveforms(DATA_PATH)
    pipe = SygnalClusteringPipelineV2()
    pipe.fit_predict(x)
    out = tmp_path / "submission2.csv"
    pipe.save_submission(out)
    import pandas as pd

    df = pd.read_csv(out)
    assert list(df.columns) == ["index", "cluster"]
    assert len(df) == N_SAMPLES


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_submission2_default_path():
    x = load_waveforms(DATA_PATH)
    pipe = SygnalClusteringPipelineV2()
    pipe.fit_predict(x)
    path = pipe.save_submission(SUBMISSION2_PATH)
    assert path.exists()
