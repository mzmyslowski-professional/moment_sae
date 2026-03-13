# train_sae.py
"""
SAE training loop.
Usage: python train_sae.py
Resumes from checkpoint if checkpoints/sae_state.pt exists.
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from config import CFG
from sae import SparseAutoencoder


def get_lr_scale(step: int, warmup_steps: int = CFG.l1_warmup_steps) -> float:
    """Linear warmup, then flat. Returns multiplier in [0, 1]."""
    if step >= warmup_steps:
        return 1.0
    return step / warmup_steps


def get_l1_coeff(step: int) -> float:
    """Linear warmup from l1_start to l1_end over l1_warmup_steps, then flat."""
    if step >= CFG.l1_warmup_steps:
        return CFG.l1_end
    t = step / CFG.l1_warmup_steps
    return CFG.l1_start + t * (CFG.l1_end - CFG.l1_start)


@torch.no_grad()
def resample_dead_features(
    sae: SparseAutoencoder,
    fired_buffer: torch.Tensor,
    recent_residuals: torch.Tensor,
) -> int:
    """
    Reinitialize W_enc rows for features that never fired in the current window.

    Args:
        sae: the SAE model
        fired_buffer: bool tensor [n_features], True if feature fired this window
        recent_residuals: float tensor [B, d_model], reconstruction residuals

    Returns:
        number of features resampled
    """
    dead_mask = ~fired_buffer
    n_dead = dead_mask.sum().item()
    if n_dead == 0:
        return 0

    dead_indices = dead_mask.nonzero(as_tuple=True)[0]

    # Pick random residual vectors as new directions
    rand_idx = torch.randint(0, recent_residuals.shape[0], (n_dead,))
    new_directions = recent_residuals[rand_idx].to(sae.W_enc.device)

    # Normalize to unit norm, scale by 0.2 * average encoder norm
    new_directions = F.normalize(new_directions, dim=-1)
    avg_enc_norm = sae.W_enc.norm(dim=-1).mean()
    new_directions = new_directions * avg_enc_norm * 0.2

    sae.W_enc.data[dead_indices] = new_directions
    sae.b_enc.data[dead_indices] = 0.0

    return n_dead
