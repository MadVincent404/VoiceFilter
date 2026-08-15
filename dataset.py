"""
LibriSpeech Dataset for VoiceFilter training.
Phase 3: ajout de bruit ambiant et musique depuis MUSAN.

Pour chaque item :
  1. Utterance cible (locuteur A)
  2. Utterance interférente (locuteur B différent)
  3. [Phase 3] Bruit ambiant ou musique depuis MUSAN (optionnel)
  4. Mélange à SNR aléatoire
  5. D-vector du locuteur cible
"""

import os
import random
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset


class LibriSpeechVoiceFilterDataset(Dataset):
    def __init__(
        self,
        librispeech_root: str,
        split,                          # str ou list[str]
        sample_rate: int   = 16000,
        segment_length: float = 3.0,
        n_fft: int         = 512,
        hop_length: int    = 128,
        win_length: int    = 512,
        d_vec_path: str    = None,
        num_mixes: int     = 1,
        snr_range: tuple   = (-5.0, 5.0),
        # Phase 3 — bruit/musique
        noise_dir: str     = None,      # ex: data/musan/noise
        music_dir: str     = None,      # ex: data/musan/music
        noise_prob: float  = 0.5,       # proba d'ajouter du bruit par item
        music_prob: float  = 0.3,       # proba d'ajouter de la musique
        noise_snr_range: tuple = (5.0, 20.0),  # SNR bruit/musique vs cible
    ):
        super().__init__()
        self.sample_rate     = sample_rate
        self.segment_samples = int(segment_length * sample_rate)
        self.n_fft           = n_fft
        self.hop_length      = hop_length
        self.win_length      = win_length
        self.num_mixes       = num_mixes
        self.snr_min, self.snr_max = snr_range

        # Phase 3
        self.noise_prob      = noise_prob
        self.music_prob      = music_prob
        self.noise_snr_min, self.noise_snr_max = noise_snr_range

        self.stft = T.Spectrogram(
            n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, power=1, normalized=False,
        )

        # ------------------------------------------------------------------
        # Index locuteurs
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Index fichiers de bruit / musique (MUSAN)
        # ------------------------------------------------------------------
        self.noise_files = self._index_audio(noise_dir)
        self.music_files = self._index_audio(music_dir)

        if self.noise_files:
            print(f"[INFO] Noise files  : {len(self.noise_files)}")
        if self.music_files:
            print(f"[INFO] Music files  : {len(self.music_files)}")

        # ------------------------------------------------------------------
        # D-vectors
        # ------------------------------------------------------------------
        if d_vec_path and os.path.exists(d_vec_path):
            self.d_vecs = torch.load(d_vec_path, map_location="cpu")
        else:
            self.d_vecs = None
            print(f"[WARN] d_vec_path not found: {d_vec_path}")

    def _index_audio(self, directory):
        if not directory:
            return []
        p = Path(directory)
        if not p.exists():
            print(f"[WARN] Audio dir not found: {directory}")
            return []
        files = list(p.rglob("*.wav")) + list(p.rglob("*.flac"))
        return files

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        target_spk, target_path = self.items[idx]
        other_spk   = random.choice([s for s in self.speakers if s != target_spk])
        interf_path = random.choice(self.speaker_files[other_spk])

        target_wav = self._load_audio(target_path)
        interf_wav = self._load_audio(interf_path)

        # Mélange speech + interférant
        snr_db    = random.uniform(self.snr_min, self.snr_max)
        mixed_wav = self._mix(target_wav, interf_wav, snr_db)

        # Phase 3 — ajout bruit ambiant
        if self.noise_files and random.random() < self.noise_prob:
            noise = self._load_audio(random.choice(self.noise_files))
            noise_snr = random.uniform(self.noise_snr_min, self.noise_snr_max)
            mixed_wav = self._mix(mixed_wav, noise, noise_snr)

        # Phase 3 — ajout musique
        if self.music_files and random.random() < self.music_prob:
            music = self._load_audio(random.choice(self.music_files))
            music_snr = random.uniform(self.noise_snr_min, self.noise_snr_max)
            mixed_wav = self._mix(mixed_wav, music, music_snr)

        target_mag = self.stft(target_wav).T
        mixed_mag  = self.stft(mixed_wav).T
        d_vec      = self._get_d_vec(target_spk)

        return {
            "mixed_mag":  mixed_mag,
            "target_mag": target_mag,
            "d_vec":      d_vec,
            "target_spk": target_spk,
        }

    def _load_audio(self, path):
        wav, sr = torchaudio.load(str(path))
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
