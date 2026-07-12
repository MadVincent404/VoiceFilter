# VoiceFilter — Speaker-Conditioned Voice Isolation

> Reproduction of **"VoiceFilter: Targeted Voice Separation by Speaker-Conditioned Spectrogram Masking"**  
> Wang et al., Google AI, 2018 — [arXiv:1810.04826](https://arxiv.org/abs/1810.04826)

---

## Introduction

In real-world audio scenarios, speech signals are rarely isolated: background speakers, ambient noise, and overlapping conversations degrade both intelligibility and downstream tasks like automatic speech recognition (ASR). VoiceFilter addresses this by conditioning a neural network on a short reference utterance from the target speaker (encoded as a d-vector), and predicting a soft time-frequency mask that suppresses all other speakers in the mixture.

The key insight of the paper is the decoupling of speaker representation (trained separately as a speaker encoder) from the masking network (CNN + LSTM), allowing the system to generalize to unseen speakers at inference time without retraining.

---

## The Approach

### Architecture

The VoiceFilter network takes two inputs at inference time:

1. **Magnitude spectrogram** of the noisy mixture (STFT with n_fft=512, hop=128)
2. **D-vector** of the target speaker extracted by a pre-trained ECAPA-TDNN speaker encoder

The network predicts a soft mask ∈ [0, 1] applied element-wise to the input spectrogram. The enhanced waveform is reconstructed via inverse STFT using the phase of the original mixture.

```
Reference audio ──► ECAPA-TDNN ──► d-vector (192-dim, L2-normalized)
                                          │
Noisy audio ──► STFT ──► magnitude ──► CNN (8 layers) ──► concat ──► LSTM ──► FC1 ──► FC2 ──► sigmoid mask
                  │                                                                                    │
                  └──────────────────────────────── phase ◄──────── masked magnitude ◄───────────────┘
                                                        │
                                                   iSTFT ──► enhanced audio
```

**CNN Block** (from Table 1 of the paper):

| Layer  | Width (t×f) | Dilation (t×f) | Filters |
|--------|-------------|----------------|---------|
| CNN 1  | 1×7         | 1×1            | 64      |
| CNN 2  | 7×1         | 1×1            | 64      |
| CNN 3  | 5×5         | 1×1            | 64      |
| CNN 4  | 5×5         | 2×1            | 64      |
| CNN 5  | 5×5         | 4×1            | 64      |
| CNN 6  | 5×5         | 8×1            | 64      |
| CNN 7  | 5×5         | 16×1           | 64      |
| CNN 8  | 1×1         | 1×1            | 8       |
| LSTM   | —           | —              | 400     |
| FC 1   | —           | —              | 600     |
| FC 2   | —           | —              | 600     |

The d-vector is concatenated to the CNN output at every time frame before being fed to the LSTM, following the paper's design decision to inject speaker information between the convolutional and recurrent layers.

### Training Objective

Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) loss, minimized as its negative:

```
SI-SDR = 10 · log₁₀(‖s_target‖² / ‖e_noise‖²)
```

where `s_target = (<ŝ, s> / ‖s‖²) · s` and `e_noise = ŝ − s_target`.

---

## Experimentation Setup

### Requirements

```bash
pip install torch torchaudio speechbrain pesq torchmetrics jiwer soundfile tqdm pyyaml tensorboard streamlit pandas
```

### Environment

| Component     | Version              |
|---------------|----------------------|
| Python        | 3.11                 |
| PyTorch       | 2.12.0+cu126         |
| torchaudio    | 2.12.0               |
| CUDA          | 13.2                 |
| GPU           | NVIDIA RTX 4090 (24 GB) |
| Platform      | VastAI / Linux |

### Data Preparation

**Phase 1 training data:**
- `train-clean-360` (~360h, 921 speakers)
- `train-clean-100` (~100h, 251 speakers)
- Total: ~460h, 1172 speakers

**Validation:** `dev-clean` (40 speakers)  
**Test:** `test-clean` (40 speakers)

```bash
# Download LibriSpeech
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/test-clean.tar.gz

# Extract d-vectors (L2-normalized ECAPA-TDNN embeddings)
python extract_dvectors.py \
    --librispeech_root data/LibriSpeech \
    --splits train-clean-360 train-clean-100 dev-clean test-clean \
    --output data/d_vectors.pt
```

**Important:** D-vectors must be L2-normalized before training. The `extract_dvectors.py` script handles this automatically.

### Training

```bash
# Phase 1 — clean speech
python train.py --config config.yaml

# Resume from checkpoint
python train.py --config config.yaml --resume checkpoints/best.pt
```

Key hyperparameters:

| Parameter       | Value              |
|-----------------|--------------------|
| Batch size      | 48                 |
| Learning rate   | 1e-4 (Adam)        |
| Weight decay    | 1e-4               |
| Grad clip       | 5.0                |
| SNR range       | ±3 dB              |
| Segment length  | 3s                 |
| num_mixes       | 2                  |
| Epochs          | 60                 |

---

## Results

Evaluated on 500 random speaker pairs from `test-clean` at 0 dB SNR.

| Metric | Mixed (baseline) | VoiceFilter | Delta |
|--------|-----------------|-------------|-------|
| PESQ   | 1.132           | 1.813       | +0.681 |
| SI-SDR | 0.003 dB        | 7.828 dB    | +7.83 dB |
| SI-SNR | 0.003 dB        | 7.834 dB    | +7.83 dB |
| SDR    | 0.115 dB        | 8.516 dB    | +8.40 dB |
| WER    | 124.2%          | 76.2%       | −48.1 pp |

The WER reduction of 48 percentage points demonstrates that the model significantly improves speech intelligibility for downstream ASR tasks, which is the primary motivation of the original paper.

Training converged to val SI-SDR ≈ −11.77 dB at epoch 30, with a train/val gap of only ~1.8 dB indicating minimal overfitting.

**Convergence was dramatically faster with the full CNN architecture** (−8.44 dB at epoch 6) compared to our initial LSTM-only implementation (~50 epochs to reach the same level).

---

## Conclusion

This reproduction successfully validates the VoiceFilter approach on LibriSpeech. The CNN layers are critical: they extract local time-frequency features before the LSTM processes temporal dynamics, and their inclusion reduced convergence time by approximately 8×. L2-normalization of d-vectors is essential — without it, the model fails to learn meaningful speaker discrimination regardless of training duration.

Limitations of this phase-1 model:
- Trained only on clean read speech (LibriSpeech clean sets)
- Evaluated at 0 dB SNR only (equal energy mixture)
- Phase reconstruction uses noisy phase (no learned phase estimation)

---

## Future Work

### Phase 2 — Fine-tuning on Diverse Speech

Fine-tune the phase-1 checkpoint on `train-other-500` (~500h of accented and noisy speech) with a reduced learning rate (3e-5) to improve generalization to real-world conditions.

### Music and Noise Robustness

Mix speech with music tracks and environmental noise (MUSAN, FreeSound) during training to build a model robust to non-speech interference.

### Reverberation Simulation

Use [pyroomacoustics](https://github.com/LCAV/pyroomacoustics) to simulate room impulse responses and train on reverberant mixtures:

```python
import pyroomacoustics as pra
room = pra.ShoeBox([6, 4, 3], fs=16000, materials=pra.Material(0.35))
room.add_source([2, 3, 1.5], signal=clean_speech)
room.simulate()
reverberant = room.mic_array.signals[0]
```

### Ambient Noise Augmentation

Add street noise, cafe ambience, and babble noise at varying SNRs (−5 to +20 dB) to simulate cocktail-party conditions.

### Phase Estimation

Replace the noisy-phase reconstruction with a learned phase estimator or use a complex-valued spectrogram mask to improve high-frequency reconstruction quality.

### Real-time Inference

Optimize the model for streaming inference using causal convolutions and a unidirectional LSTM, targeting <50ms latency for real-time applications.
