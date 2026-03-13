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
