# visualize.py
"""
Visualization and feature evaluation for trained SAE.
Usage: python visualize.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab scripts
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import CFG
from sae import SparseAutoencoder

AXIS_NAMES = ["freq", "trend", "noise"]
AXIS_LABELS = {
    "freq":  [f"{v} c/s" for v in CFG.freqs],
    "trend": [str(v) for v in CFG.trends],
    "noise": [f"σ={v}" for v in CFG.noises],
}


def compute_selectivity(
    feature_acts: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Compute per-feature selectivity scores across the three ground-truth axes.

    Args:
        feature_acts: [n_series, n_features]  max activation per series
        labels:       [n_series, 3]           (freq_bin, trend_bin, noise_bin)

    Returns:
        DataFrame with columns [feature_id, freq_sel, trend_sel, noise_sel, dominant_axis]
        Dead features (global_mean == 0) are excluded.
    """
    rows = []
    n_features = feature_acts.shape[1]

    for f in range(n_features):
        acts = feature_acts[:, f]
        global_mean = acts.mean()

        if global_mean == 0:
            continue

        sels = {}
        for axis_idx, axis_name in enumerate(AXIS_NAMES):
            group_means = np.array([
                acts[labels[:, axis_idx] == g].mean()
                for g in range(3)
            ])
            sels[axis_name] = group_means.max() / global_mean

        dominant = "none"
        for axis_name in AXIS_NAMES:
            other_axes = [a for a in AXIS_NAMES if a != axis_name]
            if (
                sels[axis_name] > 2.0
                and all(sels[a] < 1.2 for a in other_axes)
            ):
                dominant = axis_name
                break

        rows.append({
            "feature_id":  f,
            "freq_sel":    sels["freq"],
            "trend_sel":   sels["trend"],
            "noise_sel":   sels["noise"],
            "dominant_axis": dominant,
        })

    return pd.DataFrame(rows)


def get_series_level_acts(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    n_patches: int,
    device: str,
    batch_size: int = 4096,
) -> np.ndarray:
    """
    Run all activations through SAE, return max activation per series per feature.

    Returns: float32 [n_series, n_features]
    """
    sae.eval()
    all_z = []
    loader = DataLoader(TensorDataset(activations), batch_size=batch_size)

    with torch.no_grad():
        for (x,) in loader:
            x = x.to(device)
            z, _ = sae(x)
            all_z.append(z.cpu().float().numpy())

    z_all = np.concatenate(all_z, axis=0)

    n_series = z_all.shape[0] // n_patches
    z_reshaped = z_all[: n_series * n_patches].reshape(n_series, n_patches, -1)
    return z_reshaped.max(axis=1)   # [n_series, n_features]


def plot_feature_top20(
    feature_id: int,
    feature_acts: np.ndarray,
    series: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
):
    """
    Plot the 20 time series that most strongly activate a given feature.
    Saves to output_dir/feature_<N>_top20.png incrementally.
    """
    acts = feature_acts[:, feature_id]
    top_idx = np.argsort(acts)[::-1][:20]

    fig, axes = plt.subplots(4, 5, figsize=(20, 12))
    fig.suptitle(f"Feature {feature_id} — top-20 activating series", fontsize=14)

    for plot_i, series_i in enumerate(top_idx):
        ax = axes[plot_i // 5][plot_i % 5]
        ax.plot(series[series_i], linewidth=0.8)
        fi, ti, ni = labels[series_i]
        ax.set_title(
            f"f={CFG.freqs[fi]} | t={CFG.trends[ti]:+d} | σ={CFG.noises[ni]}",
            fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"feature_{feature_id:04d}_top20.png")
    plt.savefig(out_path, dpi=80)
    plt.close(fig)


def plot_summary_dashboard(
    feature_acts: np.ndarray,
    labels: np.ndarray,
    top_feature_ids: list,
    output_dir: str,
):
    """
    For each of the top features, plot mean activation by group on 3 axes.
    Saves to output_dir/feature_summary_dashboard.png.
    """
    n = len(top_feature_ids)
    fig, axes = plt.subplots(n, 3, figsize=(12, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, fid in enumerate(top_feature_ids):
        acts = feature_acts[:, fid]
        for col, (axis_idx, axis_name) in enumerate(
            [(0, "freq"), (1, "trend"), (2, "noise")]
        ):
            ax = axes[row][col]
            group_means = [
                acts[labels[:, axis_idx] == g].mean() for g in range(3)
            ]
            ax.bar(AXIS_LABELS[axis_name], group_means)
            ax.set_title(f"Feature {fid} — by {axis_name}", fontsize=8)
            ax.set_ylabel("mean activation", fontsize=7)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "feature_summary_dashboard.png")
    plt.savefig(out_path, dpi=80)
    plt.close(fig)
    print(f"Dashboard saved → {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CFG.output_dir, exist_ok=True)

    # Load SAE
    ckpt_path = os.path.join(CFG.checkpoint_dir, "sae_state.pt")
    sae = SparseAutoencoder(CFG.d_model, CFG.n_features).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    sae.load_state_dict(ckpt["model"])
    sae.eval()
    print(f"Loaded SAE from {ckpt_path} (step {ckpt['step']})")

    # Load data
    activations = torch.tensor(
        np.load(os.path.join(CFG.data_dir, "activations.npy")), dtype=torch.float32
    )
    patch_labels = np.load(os.path.join(CFG.data_dir, "labels.npy"))
    # Series-level labels: take every n_patches-th row (first patch of each series)
    series_labels = patch_labels[:: CFG.n_patches]   # [n_series, 3]

    # Reconstruct original series for plotting (same seed as generation)
    from generate_activations import generate_dataset
    series, _ = generate_dataset()

    # Get series-level max activations
    print("Computing feature activations across all series...")
    feature_acts = get_series_level_acts(sae, activations, CFG.n_patches, device)

    # Compute selectivity table
    print("Computing selectivity scores...")
    table = compute_selectivity(feature_acts, series_labels)
    csv_path = os.path.join(CFG.output_dir, "selectivity_table.csv")
    table.to_csv(csv_path, index=False)
    print(f"Selectivity table saved → {csv_path}")
    print(f"Monosemantic candidates: {(table['dominant_axis'] != 'none').sum()}")
    print(table[table["dominant_axis"] != "none"].head(20).to_string())

    # Top-50 features by peak activation
    peak_acts = feature_acts.max(axis=0)   # [n_features]
    top50_ids = np.argsort(peak_acts)[::-1][:50].tolist()

    # Per-feature top-20 plots (incremental saves)
    print(f"Plotting top-20 series for {len(top50_ids)} features...")
    for fid in top50_ids:
        plot_feature_top20(fid, feature_acts, series, series_labels, CFG.output_dir)
    print("Per-feature plots done.")

    # Summary dashboard for top-20 features
    plot_summary_dashboard(feature_acts, series_labels, top50_ids[:20], CFG.output_dir)


if __name__ == "__main__":
    main()
