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


def extract_activations(
    series: np.ndarray,
    device: str = "cuda",
    extraction_batch: int = 32,
):
    """
    Run series through frozen MOMENT-1-large, capture layer 18 activations.

    Args:
        series: float32 [N, series_len]
        device: torch device string
        extraction_batch: number of series per MOMENT forward pass

    Returns:
        activations: float32 numpy [N * n_patches, d_model]
    """
    from momentfm import MOMENTPipeline

    print("Loading MOMENT-1-large...")
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large",
        model_kwargs={"task_name": "reconstruction"},
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.init()
    model.to(device)
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)

    # Register hook on encoder block 18
    # Path: model.encoder.block[18]  (MOMENTPipeline exposes encoder directly)
    hook_output = {}

    def _hook_fn(module, input, output):
        # T5Block returns a tuple; first element is hidden states [B, n_patches, d_model]
        hook_output["acts"] = output[0].detach().cpu().float()  # cast fp16→fp32

    # Validate hook path before registering
    assert hasattr(model, 'encoder') and hasattr(model.encoder, 'block'), \
           f"Hook path 'model.encoder.block' not found. " \
           f"Available model attrs: {list(model._modules.keys())}"
    assert len(model.encoder.block) > CFG.layer_idx, \
           f"layer_idx={CFG.layer_idx} out of range, encoder has {len(model.encoder.block)} blocks"
    handle = model.encoder.block[CFG.layer_idx].register_forward_hook(_hook_fn)

    all_acts = []
    n = len(series)

    print(f"Extracting activations for {n} series (batch={extraction_batch})...")
    for start in range(0, n, extraction_batch):
        end = min(start + extraction_batch, n)
        batch_np = series[start:end]           # [b, series_len]
        b = len(batch_np)

        # MOMENT expects [B, n_channels, seq_len]
        x_enc = torch.tensor(batch_np).unsqueeze(1).to(device)  # [b, 1, series_len]
        mask = torch.ones(b, CFG.series_len, device=device)

        model(x_enc=x_enc, input_mask=mask)

        if "acts" not in hook_output:
            raise RuntimeError(
                "Hook did not fire. Check that model.model.encoder.block[18] is the correct path."
            )
        acts = hook_output["acts"]             # [b, n_patches, d_model]
        acts_flat = acts.reshape(-1, acts.shape[-1])  # [b*n_patches, d_model]
        all_acts.append(acts_flat.numpy())

        if (start // extraction_batch) % 10 == 0:
            print(f"  {end}/{n} series processed")

    handle.remove()
    return np.concatenate(all_acts, axis=0)   # [N*n_patches, d_model]


def main():
    """Generate dataset, extract activations, save to data/."""
    os.makedirs(CFG.data_dir, exist_ok=True)
    acts_path = os.path.join(CFG.data_dir, "activations.npy")
    labels_path = os.path.join(CFG.data_dir, "labels.npy")

    print("Generating synthetic dataset...")
    series, labels = generate_dataset()
    print(f"  Series shape: {series.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    activations = extract_activations(series, device=device)
    print(f"  Activations shape: {activations.shape}")

    # Replicate labels 64x (one per patch per series)
    labels_patch = np.repeat(labels, CFG.n_patches, axis=0)   # [N*n_patches, 3]
    print(f"  Labels (patch-level) shape: {labels_patch.shape}")

    np.save(acts_path, activations)
    np.save(labels_path, labels_patch)
    print(f"Saved activations → {acts_path}  ({activations.nbytes / 1e9:.2f} GB)")
    print(f"Saved labels      → {labels_path}")


if __name__ == "__main__":
    main()
