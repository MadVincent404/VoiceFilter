"""
VoiceFilter Training Script - LibriSpeech multi-split
Paper: https://arxiv.org/abs/1810.04826
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import VoiceFilter
from dataset import LibriSpeechVoiceFilterDataset
from utils import get_logger, load_checkpoint, save_checkpoint, SDRLoss
from config import get_config


def train(args, config):
    logger = get_logger(config.log_dir)
    writer = SummaryWriter(config.tensorboard_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ----- Splits -----
    train_splits = getattr(config, "train_splits",
                           ["train-clean-360", "train-clean-100"])
    snr_range = (getattr(config, "snr_min", -3.0),
                 getattr(config, "snr_max",  3.0))
    logger.info(f"Train splits : {train_splits}")
    logger.info(f"SNR range    : {snr_range}")

    # ----- Datasets -----
    train_dataset = LibriSpeechVoiceFilterDataset(
        librispeech_root=config.librispeech_root,
        split=train_splits,
        sample_rate=config.sample_rate,
        segment_length=config.segment_length,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        d_vec_path=config.d_vec_path,
        num_mixes=config.num_mixes,
        snr_range=snr_range,
    )
    val_dataset = LibriSpeechVoiceFilterDataset(
        librispeech_root=config.librispeech_root,
        split=["dev-clean"],
        sample_rate=config.sample_rate,
        segment_length=config.segment_length,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        d_vec_path=config.d_vec_path,
        num_mixes=1,
        snr_range=(0.0, 0.0),
    )

    logger.info(f"Train items : {len(train_dataset):,}  |  "
                f"Speakers : {len(train_dataset.speakers)}")
    logger.info(f"Val   items : {len(val_dataset):,}  |  "
                f"Speakers : {len(val_dataset.speakers)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=config.num_workers > 0,
    )

    # ----- Model -----
    model = VoiceFilter(config).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=getattr(config, "weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, 
    )
    criterion = SDRLoss()

    start_epoch = 0
    global_step = 0
    best_val    = float("inf")

    if args.resume:
        start_epoch, global_step = load_checkpoint(
            args.resume, model, optimizer, logger
        )

    # ----- Training loop -----
    for epoch in range(start_epoch, config.epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in pbar:
            mixed_mag  = batch["mixed_mag"].to(device)
            target_mag = batch["target_mag"].to(device)
            d_vec      = batch["d_vec"].to(device)

            mask          = model(mixed_mag, d_vec)
            estimated_mag = mixed_mag * mask
            loss          = criterion(estimated_mag, target_mag)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            train_loss  += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % config.log_interval == 0:
                writer.add_scalar("Loss/train_step", loss.item(), global_step)

        avg_train_loss = train_loss / len(train_loader)
        writer.add_scalar("Loss/train_epoch", avg_train_loss, epoch)
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # ----- Validation -----
        val_loss = validate(model, val_loader, criterion, device)
        writer.add_scalar("Loss/val", val_loss, epoch)
        logger.info(f"Epoch {epoch+1} | Val Loss:   {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                os.path.join(config.checkpoint_dir, "best.pt"),
                model, optimizer, epoch + 1, global_step, logger,
            )
            logger.info(f"  -> New best val loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        # Checkpoint tous les 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                os.path.join(config.checkpoint_dir, f"epoch_{epoch+1:03d}.pt"),
                model, optimizer, epoch + 1, global_step, logger,
            )

    writer.close()
    logger.info("Training complete.")


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            mixed_mag  = batch["mixed_mag"].to(device)
            target_mag = batch["target_mag"].to(device)
            d_vec      = batch["d_vec"].to(device)
            mask          = model(mixed_mag, d_vec)
            estimated_mag = mixed_mag * mask
            total_loss   += criterion(estimated_mag, target_mag).item()
    return total_loss / len(loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = get_config(args.config)
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.tensorboard_dir, exist_ok=True)

    train(args, config)