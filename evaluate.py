"""
evaluate.py — Évaluation complète VoiceFilter sur LibriSpeech test-clean
Métriques : SI-SDR, SI-SNR, PESQ (wideband), WER

Usage :
  python evaluate.py \
      --checkpoint checkpoints_v5/best.pt \
      --librispeech_root data/LibriSpeech/test-clean \
      --d_vec_path checkpoints_v5/d_vectors.pt \
      --config config.yaml \
      --num_pairs 500 \
      --output_dir eval_results

Prérequis :
  pip install pesq torchmetrics jiwer speechbrain soundfile
"""

import os
import argparse
import random
import csv
import shutil
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio.transforms as T
import soundfile as sf
import numpy as np
from tqdm import tqdm

# --- Métriques ---
try:
    from pesq import pesq as pesq_fn
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    print("[WARN] pesq non installé : pip install pesq")

try:
    from torchmetrics.audio import (
        ScaleInvariantSignalNoiseRatio,
        ScaleInvariantSignalDistortionRatio,
        SignalDistortionRatio,
    )
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("[WARN] torchmetrics non installé : pip install torchmetrics")

try:
    from jiwer import wer as compute_wer_fn
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("[WARN] jiwer non installé : pip install jiwer")

try:
    from speechbrain.inference.ASR import EncoderDecoderASR
    ASR_AVAILABLE = True
except ImportError:
    ASR_AVAILABLE = False
    print("[WARN] speechbrain ASR non disponible")

try:
    from speechbrain.inference.classifiers import EncoderClassifier
    SB_AVAILABLE = True
except ImportError:
    SB_AVAILABLE = False

from model import VoiceFilter
from config import get_config


# Audio helpers

def load_wav(path, target_sr=16000):
    wav, sr = sf.read(str(path))
    wav = torch.from_numpy(wav).float()
    if wav.dim() == 2:
        wav = wav.mean(1)
    if sr != target_sr:
        wav = T.Resample(sr, target_sr)(wav)
    return wav


def save_wav(path, wav: torch.Tensor, sr=16000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = wav.squeeze().cpu().numpy()
    sf.write(path, data, sr)


def mix_signals(target, interferer, snr_db=0.0):
    eps = 1e-8
    tp = target.pow(2).mean().clamp(eps)
    ip = interferer.pow(2).mean().clamp(eps)
    scale = (tp / (10 ** (snr_db / 10) * ip)).sqrt()
    return target + scale * interferer


def pad_or_trim(wav, length):
    if wav.shape[0] < length:
        return torch.nn.functional.pad(wav, (0, length - wav.shape[0]))
    return wav[:length]


def normalize_rms(wav, reference, eps=1e-8):
    rms_ref = reference.pow(2).mean().sqrt().clamp(eps)
    rms_wav = wav.pow(2).mean().sqrt().clamp(eps)
    wav = wav * (rms_ref / rms_wav)
    peak = wav.abs().max()
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    return wav


# Métriques

def compute_pesq(ref_np, deg_np, sr=16000):
    if not PESQ_AVAILABLE:
        return float("nan")
    try:
        return float(pesq_fn(sr, ref_np, deg_np, "wb"))
    except Exception:
        return float("nan")


def compute_si_sdr(est: torch.Tensor, ref: torch.Tensor):
    if TORCHMETRICS_AVAILABLE:
        m = ScaleInvariantSignalDistortionRatio()
        return m(est.unsqueeze(0), ref.unsqueeze(0)).item()
    eps = 1e-8
    ref_zm = ref - ref.mean()
    est_zm = est - est.mean()
    dot = (est_zm * ref_zm).sum()
    proj = dot / ref_zm.pow(2).sum().clamp(eps) * ref_zm
    noise = est_zm - proj
    return (10 * torch.log10(proj.pow(2).sum().clamp(eps) /
                             noise.pow(2).sum().clamp(eps))).item()


def compute_si_snr(est: torch.Tensor, ref: torch.Tensor):
    if TORCHMETRICS_AVAILABLE:
        m = ScaleInvariantSignalNoiseRatio()
        return m(est.unsqueeze(0), ref.unsqueeze(0)).item()
    return compute_si_sdr(est, ref)


def compute_sdr(est: torch.Tensor, ref: torch.Tensor):
    if TORCHMETRICS_AVAILABLE:
        m = SignalDistortionRatio()
        return m(est.unsqueeze(0), ref.unsqueeze(0)).item()
    eps = 1e-8
    dot = (est * ref).sum()
    proj = dot / ref.pow(2).sum().clamp(eps) * ref
    noise = est - proj
    return (10 * torch.log10(proj.pow(2).sum().clamp(eps) /
                             noise.pow(2).sum().clamp(eps))).item()


def compute_wer(asr_model, wav: torch.Tensor, reference_text: str, sr=16000):
    if not JIWER_AVAILABLE or asr_model is None:
        return float("nan")
    try:
        wav_np = wav.squeeze().cpu().numpy()
        # SpeechBrain ASR attend un tensor (1, samples)
        wav_t = torch.from_numpy(wav_np).unsqueeze(0)
        with torch.no_grad():
            transcription = asr_model.transcribe_batch(wav_t, torch.tensor([1.0]))[0][0]
        return compute_wer_fn(reference_text.lower(), transcription.lower())
    except Exception as e:
        return float("nan")


# D-vector

def load_dvecs(d_vec_path):
    if d_vec_path and os.path.exists(d_vec_path):
        dv = torch.load(d_vec_path, map_location="cpu")
        # Vérifier la normalisation
        norms = [v.norm().item() for v in list(dv.values())[:10]]
        mean_norm = sum(norms) / len(norms)
        if mean_norm > 10:
            print(f"[WARN] D-vectors non normalisés (norm={mean_norm:.1f}), normalisation appliquée")
            dv = {k: v / v.norm().clamp(min=1e-8) for k, v in dv.items()}
        return dv
    return None


def get_d_vec_from_dict(d_vecs, speaker_id, device):
    if d_vecs and speaker_id in d_vecs:
        return d_vecs[speaker_id].float().to(device)
    return torch.zeros(192, device=device)


# Chargement des transcriptions LibriSpeech

def load_transcripts(librispeech_root):
    """Charge toutes les transcriptions .trans.txt de LibriSpeech."""
    transcripts = {}
    root = Path(librispeech_root)
    for trans_file in root.rglob("*.trans.txt"):
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    transcripts[parts[0]] = parts[1]
    return transcripts


# Sauvegarde des meilleurs/pires échantillons

def save_samples(results, output_dir, sr=16000, n=5):
    """Sauvegarde les n meilleurs et n pires selon PESQ."""
    valid = [r for r in results if not np.isnan(r["est_pesq"])]
    if not valid:
        print("[WARN] Aucun résultat PESQ valide pour les échantillons")
        return

    sorted_by_pesq = sorted(valid, key=lambda x: x["est_pesq"])
    worst_n = sorted_by_pesq[:n]
    best_n  = sorted_by_pesq[-n:][::-1]

    for label, samples in [("best", best_n), ("worst", worst_n)]:
        for i, r in enumerate(samples, 1):
            base = os.path.join(output_dir, f"{label}_data")

            # Dossiers
            for sub in ["mixed", "interfered", "reference", "voice_filtered"]:
                os.makedirs(os.path.join(base, sub), exist_ok=True)

            prefix = f"{i:02d}_spk{r['target_speaker']}"

            # Sauvegarder les fichiers audio
            save_wav(os.path.join(base, "mixed",         f"{prefix}_mixed.wav"),
                     r["mixed_wav"], sr)
            save_wav(os.path.join(base, "interfered",    f"{prefix}_interferer.wav"),
                     r["interferer_wav"], sr)
            save_wav(os.path.join(base, "reference",     f"{prefix}_reference.wav"),
                     r["target_wav"], sr)
            save_wav(os.path.join(base, "voice_filtered",f"{prefix}_filtered.wav"),
                     r["est_wav"], sr)

            # Fiche métriques
            info_path = os.path.join(base, f"{prefix}_metrics.txt")
            with open(info_path, "w") as f:
                f.write(f"Rank #{i} ({label})\n")
                f.write(f"Target speaker  : {r['target_speaker']}\n")
                f.write(f"Target file     : {r['target_file']}\n")
                f.write(f"Interferer spk  : {r['interf_speaker']}\n")
                f.write(f"\n--- Mélange vs Référence ---\n")
                f.write(f"PESQ   : {r['mixed_pesq']:.3f}\n")
                f.write(f"SI-SDR : {r['mixed_si_sdr']:.3f} dB\n")
                f.write(f"SI-SNR : {r['mixed_si_snr']:.3f} dB\n")
                f.write(f"SDR    : {r['mixed_sdr']:.3f} dB\n")
                f.write(f"\n--- Estimé vs Référence ---\n")
                f.write(f"PESQ   : {r['est_pesq']:.3f}\n")
                f.write(f"SI-SDR : {r['est_si_sdr']:.3f} dB\n")
                f.write(f"SI-SNR : {r['est_si_snr']:.3f} dB\n")
                f.write(f"SDR    : {r['est_sdr']:.3f} dB\n")
                f.write(f"\n--- Delta ---\n")
                f.write(f"PESQ   : {r['delta_pesq']:+.3f}\n")
                f.write(f"SI-SDR : {r['delta_si_sdr']:+.3f} dB\n")
                f.write(f"SI-SNR : {r['delta_si_snr']:+.3f} dB\n")
                f.write(f"SDR    : {r['delta_sdr']:+.3f} dB\n")
                if not np.isnan(r.get("wer_mixed", float("nan"))):
                    f.write(f"\n--- WER ---\n")
                    f.write(f"WER mixed   : {r['wer_mixed']:.3f}\n")
                    f.write(f"WER filtered: {r['wer_filtered']:.3f}\n")
                    f.write(f"Delta WER   : {r['delta_wer']:+.3f}\n")

    print(f"Echantillons best/worst sauvegardes dans {output_dir}/best_data et worst_data")


# Evaluation principale

@torch.no_grad()
def evaluate(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Modèle ---
    model = VoiceFilter(config).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # --- STFT ---
    stft = T.Spectrogram(
        n_fft=config.n_fft, hop_length=config.hop_length,
        win_length=config.win_length, power=None,
    ).to(device)
    istft = T.InverseSpectrogram(
        n_fft=config.n_fft, hop_length=config.hop_length,
        win_length=config.win_length,
    ).to(device)

    seg_len = int(config.segment_length * config.sample_rate)

    # --- Index locuteurs ---
    root = Path(args.librispeech_root)
    speaker_files = defaultdict(list)
    for flac in root.rglob("*.flac"):
        try:
            speaker_id = flac.relative_to(root).parts[0]
        except (ValueError, IndexError):
            speaker_id = flac.parts[-3]
        speaker_files[speaker_id].append(flac)

    speakers = sorted(speaker_files.keys())
    print(f"Speakers test-clean: {len(speakers)}")
    assert len(speakers) >= 2

    # --- Transcriptions pour WER ---
    transcripts = load_transcripts(args.librispeech_root)
    print(f"Transcriptions chargees: {len(transcripts)}")

    # --- ASR pour WER ---
    asr_model = None
    if ASR_AVAILABLE and JIWER_AVAILABLE and args.compute_wer:
        print("Chargement modèle ASR...")
        try:
            asr_model = EncoderDecoderASR.from_hparams(
                source="speechbrain/asr-crdnn-rnnlm-librispeech",
                savedir="pretrained_models/asr-crdnn",
                run_opts={"device": str(device)},
            )
            asr_model.eval()
            print("ASR charge.")
        except Exception as e:
            print(f"[WARN] ASR non charge: {e}")

    # --- D-vectors ---
    d_vecs = load_dvecs(args.d_vec_path)
    if d_vecs:
        print(f"D-vectors: {len(d_vecs)} locuteurs")
    else:
        print("[WARN] D-vectors non trouves, utilisation de zeros")

    # --- Paires de test ---
    random.seed(42)
    pairs = []
    for _ in range(args.num_pairs):
        spk1, spk2 = random.sample(speakers, 2)
        f1 = random.choice(speaker_files[spk1])
        f2 = random.choice(speaker_files[spk2])
        pairs.append((spk1, f1, spk2, f2))

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")

    results      = []
    metrics_mix  = defaultdict(list)
    metrics_est  = defaultdict(list)
    all_wer_mix  = []
    all_wer_filt = []

    for spk1, f1, spk2, f2 in tqdm(pairs, desc="Evaluation"):
        # Charger et aligner
        target_wav = pad_or_trim(load_wav(f1, config.sample_rate), seg_len)
        interf_wav = pad_or_trim(load_wav(f2, config.sample_rate), seg_len)
        mixed_wav  = mix_signals(target_wav, interf_wav, snr_db=0.0)

        # STFT
        mixed_d     = mixed_wav.to(device)
        mixed_spec  = stft(mixed_d)
        mixed_mag   = mixed_spec.abs()
        mixed_phase = mixed_spec.angle()
        mixed_mag_in = mixed_mag.T.unsqueeze(0)   # (1, T, F)

        # D-vector
        dvec = get_d_vec_from_dict(d_vecs, spk1, device).unsqueeze(0)

        # Inférence
        mask    = model(mixed_mag_in, dvec)
        est_mag = (mixed_mag_in * mask).squeeze(0).T

        real     = est_mag * torch.cos(mixed_phase)
        imag     = est_mag * torch.sin(mixed_phase)
        est_spec = torch.complex(real, imag)
        est_wav  = istft(est_spec.unsqueeze(0), length=seg_len).squeeze(0).cpu()

        # Normalisation RMS
        est_wav = normalize_rms(est_wav, mixed_wav)

        # Numpy pour PESQ
        ref_np  = target_wav.numpy()
        mix_np  = mixed_wav.numpy()
        est_np  = est_wav.numpy()

        # --- Métriques mélange ---
        m_pesq   = compute_pesq(ref_np, mix_np)
        m_si_sdr = compute_si_sdr(mixed_wav, target_wav)
        m_si_snr = compute_si_snr(mixed_wav, target_wav)
        m_sdr    = compute_sdr(mixed_wav,    target_wav)

        # --- Métriques estimé ---
        e_pesq   = compute_pesq(ref_np, est_np)
        e_si_sdr = compute_si_sdr(est_wav, target_wav)
        e_si_snr = compute_si_snr(est_wav, target_wav)
        e_sdr    = compute_sdr(est_wav,    target_wav)

        # --- WER ---
        file_id = f1.stem
        ref_text = transcripts.get(file_id, "")
        wer_mix  = compute_wer(asr_model, mixed_wav, ref_text) if ref_text else float("nan")
        wer_filt = compute_wer(asr_model, est_wav,   ref_text) if ref_text else float("nan")

        metrics_mix["pesq"].append(m_pesq)
        metrics_mix["si_sdr"].append(m_si_sdr)
        metrics_mix["si_snr"].append(m_si_snr)
        metrics_mix["sdr"].append(m_sdr)
        metrics_est["pesq"].append(e_pesq)
        metrics_est["si_sdr"].append(e_si_sdr)
        metrics_est["si_snr"].append(e_si_snr)
        metrics_est["sdr"].append(e_sdr)
        if not np.isnan(wer_mix):
            all_wer_mix.append(wer_mix)
            all_wer_filt.append(wer_filt)

        results.append({
            # Infos
            "target_speaker":  spk1,
            "target_file":     str(f1),
            "interf_speaker":  spk2,
            # Mélange
            "mixed_pesq":      round(m_pesq,   3),
            "mixed_si_sdr":    round(m_si_sdr, 3),
            "mixed_si_snr":    round(m_si_snr, 3),
            "mixed_sdr":       round(m_sdr,    3),
            # Estimé
            "est_pesq":        round(e_pesq,   3),
            "est_si_sdr":      round(e_si_sdr, 3),
            "est_si_snr":      round(e_si_snr, 3),
            "est_sdr":         round(e_sdr,    3),
            # Delta
            "delta_pesq":      round(e_pesq   - m_pesq,   3),
            "delta_si_sdr":    round(e_si_sdr - m_si_sdr, 3),
            "delta_si_snr":    round(e_si_snr - m_si_snr, 3),
            "delta_sdr":       round(e_sdr    - m_sdr,    3),
            # WER
            "wer_mixed":       round(wer_mix,  3) if not np.isnan(wer_mix)  else "nan",
            "wer_filtered":    round(wer_filt, 3) if not np.isnan(wer_filt) else "nan",
            "delta_wer":       round(wer_filt - wer_mix, 3) if not np.isnan(wer_mix) else "nan",
            # Audio (pour save_samples)
            "mixed_wav":       mixed_wav,
            "target_wav":      target_wav,
            "interferer_wav":  interf_wav,
            "est_wav":         est_wav,
        })

    # --- CSV (sans les tensors) ---
    csv_keys = [k for k in results[0].keys()
                if k not in ("mixed_wav", "target_wav", "interferer_wav", "est_wav")]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in csv_keys})

    # --- Résumé ---
    def nanmean(lst):
        arr = [x for x in lst if not np.isnan(x)]
        return sum(arr) / len(arr) if arr else float("nan")

    print("\n" + "="*65)
    print(f"{'Metric':<12} {'Mixed':>10} {'Filtered':>10} {'Delta':>10}")
    print("-"*65)
    for key in ["pesq", "si_sdr", "si_snr", "sdr"]:
        m = nanmean(metrics_mix[key])
        e = nanmean(metrics_est[key])
        print(f"{key.upper():<12} {m:>10.3f} {e:>10.3f} {e-m:>+10.3f}")

    if all_wer_mix:
        wm = nanmean(all_wer_mix)
        wf = nanmean(all_wer_filt)
        print(f"{'WER':<12} {wm:>10.3f} {wf:>10.3f} {wf-wm:>+10.3f}")
    print("="*65)
    print(f"\nResultats CSV -> {csv_path}")

    # --- Meilleurs / pires échantillons ---
    save_samples(results, args.output_dir, sr=config.sample_rate, n=5)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True)
    parser.add_argument("--librispeech_root", required=True)
    parser.add_argument("--d_vec_path",       required=True)
    parser.add_argument("--config",           default="config.yaml")
    parser.add_argument("--num_pairs",        type=int, default=500)
    parser.add_argument("--output_dir",       default="eval_results")
    parser.add_argument("--compute_wer",      action="store_true",
                        help="Activer le calcul WER (lent, necessite speechbrain ASR)")
    args = parser.parse_args()

    config = get_config(args.config)
    evaluate(args, config)