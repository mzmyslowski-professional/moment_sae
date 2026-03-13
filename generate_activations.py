# generate_activations.py
"""
Generate synthetic time series, run through MOMENT layer 18,
save activations and labels to data/.
"""
import os
import numpy as np
import torch
from config import CFG


# Ground-truth axes (indices into CFG.freqs, CFG.trends, CFG.noises)
FREQ_VALUES = list(CFG.freqs)
TREND_VALUES = list(CFG.trends)
NOISE_VALUES = list(CFG.noises)


def make_series(
    freq: float,
    trend_slope: float,
    noise_sigma: float,
    series_len: int = 512,
    rng=None,
) -> np.ndarray:
    """
    Build a single synthetic time series:
      signal = sin(2π·freq·t) + trend_slope·linspace(0,1) + N(0, noise_sigma)

    Returns float32 array of shape [series_len].
    """
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(series_len, dtype=np.float32)
    sinusoid = np.sin(2 * np.pi * freq * t).astype(np.float32)
    trend = (trend_slope * np.linspace(0, 1, series_len)).astype(np.float32)
    noise = rng.normal(0, noise_sigma, series_len).astype(np.float32)
    return sinusoid + trend + noise


def generate_dataset(
    n_series: int = CFG.n_series,
    series_len: int = CFG.series_len,
    seed: int = 42,
):
    """
    Generate a balanced dataset of synthetic time series with known labels.

    Returns:
        series: float32 [n_series, series_len]
        labels: int8    [n_series, 3]  columns = (freq_bin, trend_bin, noise_bin)

    Note: n_series is rounded down to the nearest multiple of 27 (n_cells).
    """
    rng = np.random.default_rng(seed)
    n_cells = len(FREQ_VALUES) * len(TREND_VALUES) * len(NOISE_VALUES)  # 27
    per_cell = n_series // n_cells

    series_list = []
    labels_list = []

    for fi, freq in enumerate(FREQ_VALUES):
        for ti, trend in enumerate(TREND_VALUES):
            for ni, noise in enumerate(NOISE_VALUES):
                for _ in range(per_cell):
                    s = make_series(freq, trend, noise, series_len, rng)
                    series_list.append(s)
                    labels_list.append([fi, ti, ni])

    series = np.stack(series_list)                    # [N, series_len]
    labels = np.array(labels_list, dtype=np.int8)     # [N, 3]
    return series, labels
