import numpy as np
import pytest

from sygnal_clustering.config import DATA_PATH, N_SAMPLES
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline import SygnalClusteringPipeline


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_pipeline_labels_valid():
    x = load_waveforms(DATA_PATH)
    pipe = SygnalClusteringPipeline()
    labels = pipe.fit_predict(x)
    assert labels.shape == (N_SAMPLES,)
    assert set(np.unique(labels)) <= {0, 1, 2}


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_submission_format(tmp_path):
    x = load_waveforms(DATA_PATH)
    pipe = SygnalClusteringPipeline()
    pipe.fit_predict(x)
    out = tmp_path / "submission.csv"
    pipe.save_submission(out)
    import pandas as pd

    df = pd.read_csv(out)
    assert list(df.columns) == ["index", "cluster"]
    assert len(df) == N_SAMPLES
    assert df["cluster"].isin([0, 1, 2]).all()
