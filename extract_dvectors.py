"""
extract_dvectors.py
Pre-compute speaker d-vectors (embeddings) for all speakers in LibriSpeech.
Uses SpeechBrain's ECAPA-TDNN speaker encoder.

Usage:
  python extract_dvectors.py \
      --librispeech_root /data/LibriSpeech \
      --splits train-clean-360 dev-clean \
      --output /data/d_vectors.pt
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import soundfile as sf

import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm


try:
    from speechbrain.inference.classifiers import EncoderClassifier
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("[WARN] SpeechBrain not found. Install with: pip install speechbrain")


def get_speaker_files(root: Path, splits: list[str]) -> dict[str, list[Path]]:
    speaker_files = defaultdict(list)
    for split in splits:
        for flac in (root / split).rglob("*.flac"):
            speaker_id = flac.parts[-3]
            speaker_files[speaker_id].append(flac)
    return dict(speaker_files)


def compute_d_vector(encoder, wav: torch.Tensor, sr: int) -> torch.Tensor:
    """Compute a single d-vector for a waveform."""
    if sr != 16000:
        wav = T.Resample(sr, 16000)(wav)
    wav = wav.mean(0, keepdim=True)  # mono, shape (1, samples)
    with torch.no_grad():
        emb = encoder.encode_batch(wav)   # (1, 1, 192) for ECAPA
    return emb.squeeze()  # (192,) or (256,) depending on model


def main(args):
    assert SPEECHBRAIN_AVAILABLE, "Please install speechbrain: pip install speechbrain"

    root = Path(args.librispeech_root)
    speaker_files = get_speaker_files(root, args.splits)
    print(f"Found {len(speaker_files)} speakers across {args.splits}")

    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/spkrec-ecapa",
        run_opts={"device": "cuda:0" if torch.cuda.is_available() else "cpu"},
    )
    encoder.eval()

    d_vectors = {}
    for speaker_id, files in tqdm(speaker_files.items(), desc="Speakers"):
        embeddings = []
        # Use up to 10 utterances per speaker for a robust embedding
        for fpath in files[:10]:
            try:
                wav_np, sr = sf.read(str(fpath), dtype="float32", always_2d=True)
                wav = torch.from_numpy(wav_np).T
                emb = compute_d_vector(encoder, wav, sr)
                embeddings.append(emb)
            except Exception as e:
                print(f"  [WARN] skip {fpath}: {e}")
                continue

        if embeddings:
            dvec = torch.stack(embeddings).mean(0)
            d_vectors[speaker_id] = dvec / dvec.norm().clamp(min=1e-8)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(d_vectors, args.output)
    print(f"Saved {len(d_vectors)} d-vectors → {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--librispeech_root", type=str, required=True)
    parser.add_argument("--splits", nargs="+",
                        default=["train-clean-360", "dev-clean"])
    parser.add_argument("--output", type=str, default="/data/d_vectors.pt")
    main(parser.parse_args())
