# config.py
from dataclasses import dataclass, field


@dataclass
class Config:
    # Model architecture
    d_model: int = 1024        # MOMENT-1-large hidden dim
    n_features: int = 4096     # SAE width (4x expansion)
    layer_idx: int = 18        # MOMENT encoder block to hook

    # MOMENT input
    patch_len: int = 8         # MOMENT default patch length
    series_len: int = 512      # Input time series length

    # Synthetic dataset
    n_series: int = 5000       # Total synthetic series
    freqs: tuple = (0.05, 0.20, 0.40)   # cycles/sample
    trends: tuple = (-1, 0, 1)          # normalized slopes
    noises: tuple = (0.1, 0.5, 1.5)    # Gaussian sigma

    # SAE training
    batch_size: int = 2048
    lr: float = 3e-4
    l1_start: float = 1e-3
    l1_end: float = 4e-3
    l1_warmup_steps: int = 1000
    total_steps: int = 50000
    resample_every: int = 2500
    checkpoint_every: int = 10000
    log_every: int = 500

    # Paths
    data_dir: str = "data"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"

    @property
    def n_patches(self) -> int:
        return self.series_len // self.patch_len

    @property
    def n_activations(self) -> int:
        return self.n_series * self.n_patches


CFG = Config()
