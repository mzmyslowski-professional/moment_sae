# tests/test_train.py
import pytest
from train_sae import get_lr_scale, get_l1_coeff
from config import CFG


def test_lr_scale_at_step_0():
    assert get_lr_scale(0, warmup_steps=1000) == pytest.approx(0.0)


def test_lr_scale_at_warmup_end():
    assert get_lr_scale(1000, warmup_steps=1000) == pytest.approx(1.0)


def test_lr_scale_flat_after_warmup():
    assert get_lr_scale(5000, warmup_steps=1000) == pytest.approx(1.0)
    assert get_lr_scale(50000, warmup_steps=1000) == pytest.approx(1.0)


def test_lr_scale_linear_during_warmup():
    assert get_lr_scale(500, warmup_steps=1000) == pytest.approx(0.5)


def test_l1_coeff_at_step_0():
    assert get_l1_coeff(0) == pytest.approx(CFG.l1_start)


def test_l1_coeff_at_warmup_end():
    assert get_l1_coeff(CFG.l1_warmup_steps) == pytest.approx(CFG.l1_end)


def test_l1_coeff_flat_after_warmup():
    assert get_l1_coeff(CFG.l1_warmup_steps + 1000) == pytest.approx(CFG.l1_end)


def test_l1_coeff_monotone_during_warmup():
    values = [get_l1_coeff(s) for s in range(0, CFG.l1_warmup_steps + 1, 100)]
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


import torch
from train_sae import resample_dead_features
from sae import SparseAutoencoder


def test_resample_fires_on_correct_features():
    """Features that never fired in buffer should get new W_enc rows."""
    sae = SparseAutoencoder(d_model=8, n_features=16)
    # Mark only features 0,1,2 as having fired
    fired_buffer = torch.zeros(16, dtype=torch.bool)
    fired_buffer[[0, 1, 2]] = True

    original_enc = sae.W_enc.data.clone()
    residuals = torch.randn(100, 8)

    n_resampled = resample_dead_features(sae, fired_buffer, residuals)

    assert n_resampled == 13  # 16 - 3 fired = 13 dead
    # Fired features (0,1,2) should be unchanged
    assert torch.allclose(sae.W_enc.data[[0, 1, 2]], original_enc[[0, 1, 2]])
    # Dead features should have changed
    assert not torch.allclose(sae.W_enc.data[3:], original_enc[3:])


def test_resample_resets_dead_biases():
    sae = SparseAutoencoder(d_model=8, n_features=16)
    with torch.no_grad():
        sae.b_enc.data.fill_(9.99)  # corrupt biases

    fired_buffer = torch.zeros(16, dtype=torch.bool)
    fired_buffer[0] = True  # only feature 0 fired

    resample_dead_features(sae, fired_buffer, torch.randn(50, 8))

    # Dead features (1-15) should have b_enc reset to 0
    assert torch.allclose(sae.b_enc.data[1:], torch.zeros(15), atol=1e-6)
    # Feature 0 was alive, bias should be unchanged
    assert sae.b_enc.data[0].item() == pytest.approx(9.99)


def test_resample_returns_zero_when_all_alive():
    sae = SparseAutoencoder(d_model=8, n_features=16)
    fired_buffer = torch.ones(16, dtype=torch.bool)  # all fired
    n = resample_dead_features(sae, fired_buffer, torch.randn(50, 8))
    assert n == 0
