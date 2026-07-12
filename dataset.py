"""
LibriSpeech Dataset for VoiceFilter training.

For each item:
  1. Sample a target utterance from a random speaker.
  2. Sample an interfering utterance from a DIFFERENT speaker.
  3. Mix them at a random SNR in [-5, 5] dB.
  4. Load (or compute) the d-vector for the target speaker.
  5. Return magnitude spectrograms.

LibriSpeech directory structure expected:
  <root>/train-clean-360/<speaker>/<chapter>/<file>.flac
  <root>/dev-clean/<speaker>/<chapter>/<file>.flac
"""

import os
import random
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset


class LibriSpeechVoiceFilterDataset(Dataset):
    def __init__(
        self,
        librispeech_root,
        split,
        sample_rate=16000,
        segment_length=3.0,
        n_fft=512,
        hop_length=128,
        win_length=512,
        d_vec_path=None,
        num_mixes=1,
        snr_range=(-5.0, 5.0),
    ):
        super().__init__()
        self.sample_rate     = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.n_fft           = n_fft
        self.hop_length      = hop_length
        self.win_length      = win_length
        self.num_mixes       = num_mixes
        self.snr_min, self.snr_max = snr_range

        self.stft = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, power=1, normalized=False,
        )

        splits = [split] if isinstance(split, str) else list(split)
        root   = Path(librispeech_root)

        self.speaker_files = defaultdict(list)
        for sp in splits:
            split_root = root / sp
            if not split_root.exists():
                print(f"[WARN] Split not found: {split_root}")
                continue
            for flac in split_root.rglob("*.flac"):
                try:
                    speaker_id = flac.relative_to(split_root).parts[0]
                except (ValueError, IndexError):
                    speaker_id = flac.parts[-3]
                self.speaker_files[speaker_id].append(flac)

        self.speakers = sorted(self.speaker_files.keys())
        assert len(self.speakers) >= 2, \
            f"Need at least 2 speakers, found {len(self.speakers)}. Splits: {splits}"

        self.items = [
            (spk, fp)
            for spk, files in self.speaker_files.items()
            for fp in files
        ] * num_mixes

        if d_vec_path and os.path.exists(d_vec_path):
            self.d_vecs = torch.load(d_vec_path, map_location="cpu")
        else:
            self.d_vecs = None
            print(f"[WARN] d_vec_path not found: {d_vec_path}")


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        target_spk, target_path = self.items[idx]
        other_spk   = random.choice([s for s in self.speakers if s != target_spk])
        interf_path = random.choice(self.speaker_files[other_spk])

        target_wav = self._load_audio(target_path)
        interf_wav = self._load_audio(interf_path)

        snr_db    = random.uniform(self.snr_min, self.snr_max)
        mixed_wav = self._mix(target_wav, interf_wav, snr_db)

        target_mag = self.stft(target_wav).T
        mixed_mag  = self.stft(mixed_wav).T
        d_vec      = self._get_d_vec(target_spk)

        return {"mixed_mag": mixed_mag, "target_mag": target_mag,
                "d_vec": d_vec, "target_spk": target_spk}

    # Helpers
    def _load_audio(self, path):
        wav_np, sr = sf.read(str(path), dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np).T
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = T.Resample(sr, self.sample_rate)(wav)
        if wav.shape[0] < self.segment_samples:
            wav = torch.nn.functional.pad(wav, (0, self.segment_samples - wav.shape[0]))
        else:
            start = random.randint(0, wav.shape[0] - self.segment_samples)
            wav   = wav[start: start + self.segment_samples]
        return wav

    def _mix(self, target, interf, snr_db):
        eps   = 1e-8
        scale = (target.pow(2).mean().clamp(eps) /
                 (10 ** (snr_db / 10) * interf.pow(2).mean().clamp(eps))).sqrt()
        return target + scale * interf

    def _get_d_vec(self, speaker_id):
        if self.d_vecs is not None and speaker_id in self.d_vecs:
            return self.d_vecs[speaker_id].float()
        return torch.zeros(192)
