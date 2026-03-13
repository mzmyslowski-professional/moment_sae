# sae.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, d_model: int, n_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_features = n_features

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(n_features, d_model))
        self.b_enc = nn.Parameter(torch.zeros(n_features))
        self.W_dec = nn.Parameter(torch.empty(d_model, n_features))

        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, d_model] → z: [B, n_features]"""
        x_hat = x - self.b_pre
        return F.relu(x_hat @ self.W_enc.T + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, n_features] → x_rec: [B, d_model]"""
        return z @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_rec = self.decode(z)
        return z, x_rec

    @torch.no_grad()
    def normalize_decoder(self):
        """Renormalize W_dec columns to unit norm. Call after each optimizer step."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def compute_loss(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        x_rec: torch.Tensor,
        l1_coeff: float,
    ):
        """Returns (total_loss, mse_loss, l1_loss)."""
        mse = F.mse_loss(x_rec, x, reduction="mean")
        l1 = z.abs().sum(dim=-1).mean()
        total = mse + l1_coeff * l1
        return total, mse, l1
