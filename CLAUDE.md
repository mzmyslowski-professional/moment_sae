# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Trains a Sparse Autoencoder (SAE) on frozen activations from layer 18 of MOMENT-1-large (a T5-style time series foundation model) to decompose polysemantic neurons into interpretable features for frequency, trend, and noise level.

## Commands

**Install (Colab only — do not install torch, it's pre-installed):**
```bash
pip install momentfm --no-deps
pip install transformers einops psutil
```

**Run tests:**
```bash
pytest tests/ -v
pytest tests/test_sae.py::test_forward_shapes -v   # single test
```

**Run pipeline steps:**
```bash
python generate_activations.py   # ~13 min on T4, saves data/activations.npy (~1.3 GB)
python train_sae.py              # ~25 min on T4, saves checkpoints/sae_state.pt
python visualize.py              # saves outputs/
```

## Architecture

All hyperparameters live in `config.py` (`CFG` singleton). Every other file imports from it — change things there only.

**Data flow:**
1. `generate_activations.py` — generates 27-cell synthetic dataset (3 freq × 3 trend × 3 noise), runs it through frozen MOMENT-1-large, captures `model.encoder.block[18]` activations via `register_forward_hook`. Output: `[N*64, 1024]` float32 array (64 patches per 512-length series).
2. `train_sae.py` — trains `SparseAutoencoder` on those activations. Includes λ warmup (l1_start→l1_end over 1000 steps), dead feature resampling every 2500 steps, and checkpoint resumption.
3. `visualize.py` — loads checkpoint + activations, runs SAE forward pass, computes per-feature selectivity scores, saves plots and CSV.

**SAE architecture** (`sae.py`): Anthropic-style ReLU SAE. Subtracts `b_pre` before encoding, adds it back after decoding. `W_dec` columns are unit-norm (enforced after every optimizer step via `normalize_decoder()`). Loss = MSE + λ·mean(L1 over features).

**Key MOMENT detail:** `MOMENTPipeline.from_pretrained` exposes the encoder directly as `model.encoder.block[N]` — not `model.model.encoder.block[N]`. The model is loaded in fp16 on CUDA to avoid OOM during the CPU→GPU weight transfer.

**Selectivity scoring:** a feature is monosemantic if `max_group_mean / global_mean > 2.0` on one axis and `< 1.2` on both others. Series-level activations are computed as the max over 64 patches — this is intentional for freq/noise but causes trend features to be invisible (trend is a global signal; switch to mean pooling to surface them).

## Known issues / tuning history

- `n_features=1024` (current). 4096 caused persistent ~25% dead features and plateau because the 27-cell dataset lacks diversity to fill a 4x dictionary.
- `l1_start=1e-3, l1_end=4e-3`. Original 1e-4/4e-4 produced L0≈640; 10x increase brings it to ~120.
- Dead % shown as 100% immediately after resampling is cosmetic — `fired_buffer` is zeroed before the log line.
- Colab: `data/`, `checkpoints/`, `outputs/` are symlinked to Google Drive. Delete checkpoint before restarting with changed hyperparameters.
