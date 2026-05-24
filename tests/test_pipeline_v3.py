import numpy as np
import pandas as pd
import pytest

from sygnal_clustering.config import DATA_PATH, N_SAMPLES
from sygnal_clustering.data import load_waveforms
from sygnal_clustering.pipeline import (
    SUBMISSION3_PATH,
    SUBMISSION3A_PATH,
    SUBMISSION3B_PATH,
    SUBMISSION3C_PATH,
    method_a_meta_col2_tercile,
    method_b_description_gmm3,
    method_c_gmm2_low_confidence,
    run_all_v3,
)


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_v3_methods_labels():
    x = load_waveforms()
    for labels in [
        method_a_meta_col2_tercile(),
        method_b_description_gmm3(x)[0],
        method_c_gmm2_low_confidence(x)[0],
    ]:
        assert labels.shape == (N_SAMPLES,)
        u = set(np.unique(labels))
        assert u <= {0, 1, 2}
        assert len(u) == 3


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_run_all_v3_writes_files():
    run_all_v3()
    for p in (SUBMISSION3A_PATH, SUBMISSION3B_PATH, SUBMISSION3C_PATH, SUBMISSION3_PATH):
        assert p.exists()
        df = pd.read_csv(p)
        assert list(df.columns) == ["index", "cluster"]
        assert len(df) == N_SAMPLES
