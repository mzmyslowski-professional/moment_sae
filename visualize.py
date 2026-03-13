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
    series_batch: int = 64,
) -> np.ndarray:
    """
    Run all activations through SAE, return max activation per series per feature.

    Processes `series_batch` series at a time to avoid materialising the full
    [N*n_patches, n_features] intermediate array (~5 GB at default settings).

    Returns: float32 [n_series, n_features]
    """
    sae.eval()
    n_total = activations.shape[0]   # N * n_patches
    n_series = n_total // n_patches
    series_max = []

    with torch.no_grad():
        for s_start in range(0, n_series, series_batch):
            s_end = min(s_start + series_batch, n_series)
            x = activations[s_start * n_patches : s_end * n_patches].to(device)
            z, _ = sae(x)                                     # [(b*n_patches), n_features]
            z = z.cpu().float()
            b = s_end - s_start
            z_max = z.reshape(b, n_patches, -1).max(dim=1).values  # [b, n_features]
            series_max.append(z_max.numpy())

    return np.concatenate(series_max, axis=0)   # [n_series, n_features]


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


def plot_cell_heatmap(
    feature_id: int,
    feature_acts: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
):
    """
    Plot mean activation across all 27 cells as three 2D heatmaps.

    Layout: one heatmap per noise level, axes = freq (x) vs trend (y).
    This reveals whether a feature is axis-aligned, conjunction, or polysemantic.

    Saves to output_dir/feature_<N>_heatmap.png.
    """
    acts = feature_acts[:, feature_id]
    n_freq = len(CFG.freqs)
    n_trend = len(CFG.trends)
    n_noise = len(CFG.noises)

    # Build [n_freq, n_trend, n_noise] mean-activation grid
    grid = np.zeros((n_freq, n_trend, n_noise))
    for fi in range(n_freq):
        for ti in range(n_trend):
            for ni in range(n_noise):
                mask = (
                    (labels[:, 0] == fi) &
                    (labels[:, 1] == ti) &
                    (labels[:, 2] == ni)
                )
                grid[fi, ti, ni] = acts[mask].mean() if mask.any() else 0.0

    vmax = grid.max()

    fig, axes = plt.subplots(1, n_noise, figsize=(5 * n_noise, 4))
    fig.suptitle(f"Feature {feature_id} — mean activation per cell", fontsize=13)

    for ni, ax in enumerate(axes):
        im = ax.imshow(
            grid[:, :, ni].T,   # [trend, freq] for natural orientation
            aspect="auto",
            vmin=0, vmax=vmax,
            cmap="viridis",
        )
        ax.set_title(f"noise σ={CFG.noises[ni]}", fontsize=9)
        ax.set_xlabel("freq", fontsize=8)
        ax.set_ylabel("trend", fontsize=8)
        ax.set_xticks(range(n_freq))
        ax.set_xticklabels([str(v) for v in CFG.freqs], fontsize=7)
        ax.set_yticks(range(n_trend))
        ax.set_yticklabels([str(v) for v in CFG.trends], fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"feature_{feature_id:04d}_heatmap.png")
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
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

    # Per-feature top-20 plots and cell heatmaps (incremental saves)
    print(f"Plotting top-20 series and cell heatmaps for {len(top50_ids)} features...")
    for fid in top50_ids:
        plot_feature_top20(fid, feature_acts, series, series_labels, CFG.output_dir)
        plot_cell_heatmap(fid, feature_acts, series_labels, CFG.output_dir)
    print("Per-feature plots done.")

    # Summary dashboard for top-20 features
    plot_summary_dashboard(feature_acts, series_labels, top50_ids[:20], CFG.output_dir)


if __name__ == "__main__":
    main()
