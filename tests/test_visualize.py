# tests/test_visualize.py
import numpy as np
import pytest
from visualize import compute_selectivity


def test_selectivity_monosemantic_freq_feature():
    """A feature that only activates for freq_bin=2 should have freq_sel >> others."""
    n_series = 270
    feature_acts = np.zeros((n_series, 4))
    labels = np.zeros((n_series, 3), dtype=np.int8)

    for i in range(n_series):
        labels[i] = [i % 3, (i // 3) % 3, (i // 9) % 3]

    for i in range(n_series):
        if labels[i, 0] == 2:
            feature_acts[i, 0] = 10.0
        else:
            feature_acts[i, 0] = 0.1

    table = compute_selectivity(feature_acts, labels)

    row = table[table["feature_id"] == 0].iloc[0]
    assert row["freq_sel"] > 2.0
    assert row["trend_sel"] < 1.5
    assert row["noise_sel"] < 1.5
    assert row["dominant_axis"] == "freq"


def test_selectivity_excludes_dead_features():
    """Features with global_mean == 0 should not appear in output table."""
    n_series = 27
    feature_acts = np.zeros((n_series, 4))
    labels = np.zeros((n_series, 3), dtype=np.int8)
    for i in range(n_series):
        labels[i] = [i % 3, (i // 3) % 3, (i // 9) % 3]
    feature_acts[:, 1] = 1.0  # only feature 1 is alive

    table = compute_selectivity(feature_acts, labels)
    assert 0 not in table["feature_id"].values
    assert 1 in table["feature_id"].values


def test_selectivity_polysemantic_no_dominant():
    """A uniformly-activating feature should not be flagged as monosemantic."""
    n_series = 270
    feature_acts = np.ones((n_series, 1))
    labels = np.zeros((n_series, 3), dtype=np.int8)
    for i in range(n_series):
        labels[i] = [i % 3, (i // 3) % 3, (i // 9) % 3]

    table = compute_selectivity(feature_acts, labels)
    row = table.iloc[0]
    assert row["dominant_axis"] == "none"
