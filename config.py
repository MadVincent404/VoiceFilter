"""
config.py  — Hyperparameters & YAML loader
"""

import os
import yaml
from dataclasses import dataclass, field


@dataclass
class Config:
    # Audio
    sample_rate: int      = 16000
    segment_length: float = 3.0        # seconds
    n_fft: int            = 512
    hop_length: int       = 128
    win_length: int       = 512

    # Model
    d_vec_size: int   = 192
    lstm_hidden: int  = 400
    lstm_layers: int  = 7
    fc_hidden: int    = 512
    dropout: float    = 0.5

    # Training
    epochs: int        = 100
    batch_size: int    = 64
    lr: float          = 1e-4
    grad_clip: float   = 5.0
    num_workers: int   = 8
    log_interval: int  = 50
    num_mixes: int     = 2
    snr_min: float = -5.0
    snr_max: float = 5.0
    weight_decay: float = 0.0

    # Paths
    librispeech_root: str  = "/workspace/data/LibriSpeech"
    d_vec_path: str        = "/workspace/data/d_vectors.pt"
    checkpoint_dir: str    = "checkpoints"
    log_dir: str           = "logs"
    tensorboard_dir: str   = "runs"


def get_config(path: str = None) -> Config:
    cfg = Config()
    if path and os.path.exists(path):
        with open(path) as f:
            overrides = yaml.safe_load(f) or {}
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg
