from pathlib import Path

import numpy as np
import pytest

from sygnal_clustering.config import DATA_PATH, N_FEATURES, N_SAMPLES
from sygnal_clustering.data import load_waveforms


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_load_waveforms_shape():
    x = load_waveforms(DATA_PATH)
    assert x.shape == (N_SAMPLES, N_FEATURES)
    assert x.dtype == np.float64


@pytest.mark.skipif(not DATA_PATH.exists(), reason="dataset not present")
def test_load_waveforms_finite():
    x = load_waveforms(DATA_PATH)
    assert np.isfinite(x).all()


def test_load_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_waveforms(tmp_path / "missing.txt")
