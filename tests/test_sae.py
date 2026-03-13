# tests/test_sae.py
import torch
import torch.nn.functional as F
import pytest
from sae import SparseAutoencoder


@pytest.fixture
def sae():
    return SparseAutoencoder(d_model=16, n_features=64)


def test_forward_output_shapes(sae):
    x = torch.randn(32, 16)
    z, x_rec = sae(x)
    assert z.shape == (32, 64), f"z shape {z.shape} != (32, 64)"
    assert x_rec.shape == (32, 16), f"x_rec shape {x_rec.shape} != (32, 16)"


def test_features_are_nonnegative(sae):
    x = torch.randn(32, 16)
    z, _ = sae(x)
    assert (z >= 0).all(), "SAE features must be non-negative (ReLU)"


def test_b_pre_is_subtracted_then_added(sae):
    """If W_enc and W_dec are zero, reconstruction should equal b_pre."""
    with torch.no_grad():
        sae.W_enc.zero_()
        sae.W_dec.zero_()
        sae.b_enc.zero_()
        x = torch.randn(4, 16)
        sae.b_pre.data = x[0].clone()  # set b_pre to first sample
        _, x_rec = sae(x[0:1])
        assert torch.allclose(x_rec, x[0:1], atol=1e-6)


def test_decoder_columns_unit_norm_after_normalize(sae):
    with torch.no_grad():
        sae.W_dec.data *= 5.0
    sae.normalize_decoder()
    norms = sae.W_dec.norm(dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"W_dec column norms not unit: min={norms.min():.4f} max={norms.max():.4f}"


def test_loss_components_positive(sae):
    x = torch.randn(32, 16)
    z, x_rec = sae(x)
    total, mse, l1 = sae.compute_loss(x, z, x_rec, l1_coeff=1e-4)
    assert mse.item() >= 0
    assert l1.item() >= 0
    assert torch.isclose(total, mse + 1e-4 * l1, atol=1e-5)


def test_loss_uses_correct_reduction():
    """MSE uses mean, L1 uses sum-per-sample then mean over batch."""
    sae = SparseAutoencoder(d_model=4, n_features=8)
    x = torch.zeros(10, 4)
    x_rec = torch.ones(10, 4)   # residual = 1.0 everywhere
    z = torch.ones(10, 8)        # all features active at 1.0
    total, mse, l1 = sae.compute_loss(x, z, x_rec, l1_coeff=0.0)
    # MSE = mean((x - x_rec)^2) = mean(1.0) = 1.0
    assert torch.isclose(mse, torch.tensor(1.0), atol=1e-5), f"mse={mse}"
    # L1 = mean(sum(|z|, dim=-1)) = mean(8.0) = 8.0
    total_with_l1, _, l1_val = sae.compute_loss(x, z, x_rec, l1_coeff=1.0)
    assert torch.isclose(l1_val, torch.tensor(8.0), atol=1e-5), f"l1={l1_val}"
    # When l1_coeff=0, total should equal mse
    total_no_l1, mse_no_l1, _ = sae.compute_loss(x, z, x_rec, l1_coeff=0.0)
    assert torch.isclose(total_no_l1, mse_no_l1, atol=1e-5)
