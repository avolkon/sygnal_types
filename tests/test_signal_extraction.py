import numpy as np

from sygnal_clustering.models import remap_labels_physics
from sygnal_clustering.signal_extraction import extract_description_features


def test_extract_single_waveform_shape():
    x = np.random.default_rng(42).random((3, 500))
    features = extract_description_features(x)
    assert features.shape == (3, 7)
    assert np.isfinite(features).all()


def test_remap_labels_physics_charge_order():
    features = np.array(
        [
            [1.0, 10.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [1.0, 20.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [1.0, 15.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    labels = np.array([0, 1, 2])
    out = remap_labels_physics(labels, features)
    assert out[0] == 0  # lower charge
    assert out[1] == 1  # higher charge
    assert out[2] == 2  # anomaly preserved
