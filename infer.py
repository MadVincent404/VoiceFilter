"""
infer.py — Run VoiceFilter on a mixed audio file given a reference utterance.

Usage:
  python infer.py ^
      --checkpoint checkpoints/epoch_050.pt ^
      --mixed     mixed.wav ^
      --reference reference.wav ^
      --output    output_filtered.wav
"""

import argparse
import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np

from model import VoiceFilter
from config import get_config

try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ImportError:
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        EncoderClassifier = None


def load_wav(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Charge un fichier audio avec soundfile (compatible Windows)."""
    wav, sr = sf.read(path)
    wav = torch.from_numpy(wav).float()

    if wav.dim() == 2:
        wav = wav.mean(1)   # stereo -> mono: (samples, channels) -> (samples,)
    # wav.dim() == 1 : déjà mono

    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return wav  # (samples,)


def save_wav(path: str, wav: torch.Tensor, sr: int):
    """Sauvegarde avec soundfile (compatible Windows, pas besoin de FFmpeg)."""
    data = wav.squeeze().cpu().numpy()
    sf.write(path, data, sr)


def get_d_vec(reference_path: str, device: torch.device) -> torch.Tensor:
    assert EncoderClassifier is not None, "Install speechbrain: pip install speechbrain"
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa",
        run_opts={"device": "cpu"},
    )
    encoder.eval()
    
    # Charger plusieurs segments et moyenner pour un embedding stable
    wav = load_wav(reference_path)
    
    # Découper en segments de 3s et moyenner
    seg_len = 16000 * 3
    segments = []
    if wav.shape[0] >= seg_len:
        for start in range(0, wav.shape[0] - seg_len + 1, seg_len):
            segments.append(wav[start:start + seg_len].unsqueeze(0))
    else:
        segments = [wav.unsqueeze(0)]
    
    embeddings = []
    for seg in segments:
        with torch.no_grad():
            emb = encoder.encode_batch(seg).squeeze()
            embeddings.append(emb)
    
    emb = torch.stack(embeddings).mean(0)
    emb = emb / emb.norm().clamp(min=1e-8)
    return emb.to(device)


@torch.no_grad()
def infer(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Modèle ---
    model = VoiceFilter(config).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # --- Audio ---
    mixed_wav = load_wav(args.mixed, config.sample_rate).to(device)  # (samples,)

    # --- STFT ---
    stft = T.Spectrogram(
        n_fft=config.n_fft, hop_length=config.hop_length,
        win_length=config.win_length, power=None,
    ).to(device)
    istft = T.InverseSpectrogram(
        n_fft=config.n_fft, hop_length=config.hop_length,
        win_length=config.win_length,
    ).to(device)

    mixed_spec  = stft(mixed_wav)        # (F, T) complex
    mixed_mag   = mixed_spec.abs()       # (F, T)
    mixed_phase = mixed_spec.angle()     # (F, T)
    mixed_mag_in = mixed_mag.T.unsqueeze(0)   # (1, T, F)

    # --- D-vector ---
    d_vec = get_d_vec(args.reference, device).unsqueeze(0)   # (1, D)

    # --- Inférence ---
    mask          = model(mixed_mag_in, d_vec)               # (1, T, F)
    estimated_mag = (mixed_mag_in * mask).squeeze(0).T       # (F, T)

    # --- Reconstruction ---
    real = estimated_mag * torch.cos(mixed_phase)
    imag = estimated_mag * torch.sin(mixed_phase)
    estimated_spec = torch.complex(real, imag)               # (F, T)

    output_wav = istft(
        estimated_spec.unsqueeze(0), length=mixed_wav.shape[0]
    ).cpu()   # (1, samples)

    # --- Normalisation RMS ---
    eps        = 1e-8
    rms_mixed  = mixed_wav.cpu().pow(2).mean().sqrt().clamp(min=eps)
    rms_output = output_wav.pow(2).mean().sqrt().clamp(min=eps)
    output_wav = output_wav * (rms_mixed / rms_output)

    # Évite le clipping
    peak = output_wav.abs().max()
    if peak > 0.99:
        output_wav = output_wav * (0.99 / peak)

    print(f"Mask mean : {mask.mean().item():.4f}  "
          f"min : {mask.min().item():.4f}  "
          f"max : {mask.max().item():.4f}")
    print(f"RMS mixed : {rms_mixed.item():.4f}  "
          f"RMS output (avant norm) : {rms_output.item():.4f}")

    # --- Sauvegarde via soundfile ---
    save_wav(args.output, output_wav, config.sample_rate)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mixed",      required=True)
    parser.add_argument("--reference",  required=True)
    parser.add_argument("--output",     required=True)
    parser.add_argument("--config",     default="config.yaml")
    args = parser.parse_args()

    config = get_config(args.config)
    infer(args, config)
    