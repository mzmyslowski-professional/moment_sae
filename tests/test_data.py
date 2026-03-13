# tests/test_data.py
import numpy as np
import pytest
from generate_activations import make_series, generate_dataset
from config import CFG


def test_make_series_shape():
    s = make_series(freq=0.05, trend_slope=0, noise_sigma=0.1, series_len=512)
    assert s.shape == (512,)
    assert s.dtype == np.float32


def test_make_series_trend_visible():
    """Positive trend series should have mean of second half > first half."""
    s = make_series(freq=0.0, trend_slope=2.0, noise_sigma=0.0, series_len=512)
    assert s[256:].mean() > s[:256].mean()


def test_make_series_deterministic_with_rng():
    rng = np.random.default_rng(0)
    s1 = make_series(0.1, 0, 0.5, rng=rng)
    rng2 = np.random.default_rng(0)
    s2 = make_series(0.1, 0, 0.5, rng=rng2)
    assert np.allclose(s1, s2)


def test_generate_dataset_shapes():
    series, labels = generate_dataset(n_series=270, series_len=512)
    assert series.shape == (270, 512)
    assert labels.shape == (270, 3)
    assert labels.dtype == np.int8


def test_generate_dataset_balanced():
    """Each of the 27 (freq, trend, noise) cells should have equal count."""
    series, labels = generate_dataset(n_series=270, series_len=512)
    for fi in range(3):
        for ti in range(3):
            for ni in range(3):
                mask = (
                    (labels[:, 0] == fi) &
                    (labels[:, 1] == ti) &
                    (labels[:, 2] == ni)
                )
                assert mask.sum() == 10, \
                    f"Cell ({fi},{ti},{ni}) has {mask.sum()} samples, expected 10"


def test_generate_dataset_label_range():
    _, labels = generate_dataset(n_series=270, series_len=512)
    assert labels[:, 0].max() == 2  # 3 freq bins: 0,1,2
    assert labels[:, 1].max() == 2  # 3 trend bins: 0,1,2
    assert labels[:, 2].max() == 2  # 3 noise bins: 0,1,2
    assert labels.min() == 0
