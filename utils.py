"""
utils.py  — Logger, checkpoint I/O, SDR loss
"""

import os
import logging
import torch
import torch.nn as nn

# Logger
def get_logger(log_dir: str, name: str = "voicefilter") -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

# Checkpoints
def save_checkpoint(path: str, model, optimizer, epoch: int, step: int, logger):
    torch.save({
        "epoch":      epoch,
        "step":       step,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(path: str, model, optimizer, logger):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    epoch = ckpt.get("epoch", 0)
    step  = ckpt.get("step",  0)
    logger.info(f"Resumed from {path} (epoch {epoch}, step {step})")
    return epoch, step

# SDR Loss  (Signal-to-Distortion Ratio, higher = better → minimize negative)
class SDRLoss(nn.Module):
    """
    Negative Scale-Invariant SDR (SI-SDR).

    SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)

    where:
      s_target = <s_hat, s> / ||s||^2 * s
      e_noise  = s_hat - s_target
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, estimated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            estimated : (B, T, F)  predicted magnitude
            target    : (B, T, F)  clean magnitude
        Returns:
            loss      : scalar (mean over batch)
        """
        # Flatten time-frequency dims
        est = estimated.reshape(estimated.shape[0], -1)   # (B, T*F)
        tgt = target.reshape(target.shape[0], -1)         # (B, T*F)

        dot = (est * tgt).sum(dim=-1, keepdim=True)       # (B, 1)
        s_target_energy = (tgt * tgt).sum(dim=-1, keepdim=True).clamp(self.eps)

        s_target = dot / s_target_energy * tgt
        e_noise  = est - s_target

        si_sdr = 10 * torch.log10(
            (s_target * s_target).sum(-1).clamp(self.eps) /
            (e_noise  * e_noise).sum(-1).clamp(self.eps)
        )   # (B,)

        return -si_sdr.mean()

# MSE Loss on magnitude (alternative / simpler)
class MagMSELoss(nn.Module):
    def forward(self, estimated, target):
        return nn.functional.mse_loss(estimated, target)
