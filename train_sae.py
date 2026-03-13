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


def train(device: str = None):
    """
    Full SAE training loop.
    Loads data/activations.npy, trains SAE, saves checkpoints.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on {device}")

    os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    # --- Load activations ---
    acts_path = os.path.join(CFG.data_dir, "activations.npy")
    print(f"Loading activations from {acts_path}...")
    activations = torch.tensor(np.load(acts_path), dtype=torch.float32)
    print(f"  Shape: {activations.shape}  ({activations.element_size() * activations.nelement() / 1e9:.2f} GB)")

    dataset = TensorDataset(activations)
    loader = DataLoader(dataset, batch_size=CFG.batch_size, shuffle=True, drop_last=True)

    # --- SAE + optimizer ---
    sae = SparseAutoencoder(CFG.d_model, CFG.n_features).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=CFG.lr, betas=(0.9, 0.999))

    # --- Rolling fired buffer for dead feature tracking ---
    fired_buffer = torch.zeros(CFG.n_features, dtype=torch.bool, device=device)

    # --- Resume from checkpoint if available ---
    start_step = 0
    ckpt_path = os.path.join(CFG.checkpoint_dir, "sae_state.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        sae.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        if "fired_buffer" in ckpt:
            fired_buffer = ckpt["fired_buffer"].to(device)
        print(f"Resumed from step {start_step}")

    step = start_step
    data_iter = iter(loader)

    print(f"Starting training from step {start_step} to {CFG.total_steps}...")
    while step < CFG.total_steps:
        try:
            (x,) = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            (x,) = next(data_iter)

        x = x.to(device)

        # LR warmup
        lr_scale = get_lr_scale(step)
        for pg in optimizer.param_groups:
            pg["lr"] = CFG.lr * lr_scale

        # Forward
        z, x_rec = sae(x)
        l1_coeff = get_l1_coeff(step)
        total_loss, mse_loss, l1_loss = sae.compute_loss(x, z, x_rec, l1_coeff)

        # Backward
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
        optimizer.step()
        sae.normalize_decoder()

        # Update fired buffer
        with torch.no_grad():
            fired_buffer |= (z > 0).any(dim=0)

        step += 1

        # Dead feature resampling
        if step % CFG.resample_every == 0:
            residuals = (x - x_rec).detach()
            n_resampled = resample_dead_features(sae, fired_buffer, residuals)
            pct_dead = (~fired_buffer).float().mean().item() * 100
            print(f"[step {step}] Resampled {n_resampled} dead features "
                  f"({pct_dead:.1f}% dead in window)")
            fired_buffer.zero_()

        # Logging
        if step % CFG.log_every == 0:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
                pct_dead = (~fired_buffer).float().mean().item() * 100
            print(
                f"step {step:6d} | "
                f"loss {total_loss.item():.4f} | "
                f"mse {mse_loss.item():.4f} | "
                f"l1 {l1_loss.item():.4f} | "
                f"L0 {l0:.1f} | "
                f"dead {pct_dead:.1f}% | "
                f"λ {l1_coeff:.2e}"
            )

        # Checkpointing
        if step % CFG.checkpoint_every == 0:
            torch.save(
                {
                    "model": sae.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "fired_buffer": fired_buffer.cpu(),
                },
                ckpt_path,
            )
            print(f"Checkpoint saved at step {step}")

    # Final checkpoint
    torch.save(
        {
            "model": sae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "fired_buffer": fired_buffer.cpu(),
        },
        ckpt_path,
    )
    print(f"Training complete. Final checkpoint saved.")
    return sae


def main():
    train()


if __name__ == "__main__":
    main()
