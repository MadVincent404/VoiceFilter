"""
VoiceFilter — Streamlit Demo App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Config
GITHUB_URL   = "https://github.com/MadVincent404/VoiceFilter.git"
RESULTS_CSV  = "results.csv"
TRAIN_LOG    = "miscellaneous/logs/voicefilter.log"
BEST_DIR     = "audios/best_data"
WORST_DIR    = "audios/worst_data"
SR           = 16000

st.set_page_config(
    page_title="VoiceFilter",
    layout="wide",
)

# Style minimal
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-val  { font-size: 2rem; font-weight: 700; color: #1f77b4; }
    .metric-delta{ font-size: 1rem; color: #2ca02c; }
    .metric-label{ font-size: 0.85rem; color: #555; margin-top: 4px; }
    h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.title(" VoiceFilter — Speaker-Conditioned Voice Isolation")
st.markdown(
    f"Reproduction of **[VoiceFilter (Wang et al., 2018)](https://arxiv.org/abs/1810.04826)** "
    f"trained on LibriSpeech 360h + 100h clean. "
    f"&nbsp;|&nbsp; [GitHub]({GITHUB_URL})"
)
st.divider()

# Métriques globales
st.subheader("Evaluation Results — LibriSpeech test-clean (500 pairs, SNR=0 dB)")

METRICS = [
    ("PESQ",   1.132, 1.813, "+0.681"),
    ("SI-SDR", 0.003, 7.828, "+7.83 dB"),
    ("SI-SNR", 0.003, 7.834, "+7.83 dB"),
    ("SDR",    0.115, 8.516, "+8.40 dB"),
    ("WER",    1.242, 0.762, "−48.1 pp"),
]

cols = st.columns(len(METRICS))
for col, (name, mixed, filtered, delta) in zip(cols, METRICS):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{filtered}</div>
            <div class="metric-delta">{delta} vs mixed</div>
            <div class="metric-label">{name}</div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Audio samples
def list_audio_files(folder):
    p = Path(folder)
    if not p.exists():
        return []
    return sorted(p.glob("*.wav"))


def render_audio_table(data_dir, label):
    st.subheader(f"{'Best !' if label == 'best' else 'Need improvements !'} {label.capitalize()} 5 Samples (ranked by PESQ)")

    mixed_files    = list_audio_files(os.path.join(data_dir, "mixed"))
    ref_files      = list_audio_files(os.path.join(data_dir, "reference"))
    interf_files   = list_audio_files(os.path.join(data_dir, "interfered"))
    filtered_files = list_audio_files(os.path.join(data_dir, "voice_filtered"))

    n = max(len(mixed_files), 1)

    # En-tête du tableau
    hcols = st.columns([1, 2, 2, 2, 2])
    for col, title in zip(hcols, ["#", "Mixed", "Reference (target)", "Interferer", "VoiceFilter output"]):
        col.markdown(f"**{title}**")
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for i in range(min(5, n)):
        row = st.columns([1, 2, 2, 2, 2])
        row[0].markdown(f"**{i+1}**")

        for col_idx, files in enumerate([mixed_files, ref_files, interf_files, filtered_files], 1):
            if i < len(files):
                row[col_idx].audio(str(files[i]), format="audio/wav")
            else:
                row[col_idx].markdown("—")


tab_best, tab_worst = st.tabs(["Best samples", "Worst samples"])

with tab_best:
    if Path(BEST_DIR).exists():
        render_audio_table(BEST_DIR, "best")
    else:
        st.info(f"Dossier non trouvé : {BEST_DIR}")

with tab_worst:
    if Path(WORST_DIR).exists():
        render_audio_table(WORST_DIR, "worst")
    else:
        st.info(f"Dossier non trouvé : {WORST_DIR}")


st.divider()

# Training logs
st.subheader("Training Log")

log_path = Path(TRAIN_LOG)
if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    epoch_lines = [l for l in lines if "Train Loss" in l or "Val Loss" in l]

    # Parser pour graphique
    train_losses, val_losses, epochs = [], [], []
    epoch_num = 0
    for l in epoch_lines:
        if "Train Loss" in l:
            epoch_num += 1
            try:
                train_losses.append(float(l.split("Train Loss:")[-1].strip()))
                epochs.append(epoch_num)
            except Exception:
                pass
        elif "Val Loss" in l:
            try:
                val_losses.append(float(l.split("Val Loss:")[-1].strip()))
            except Exception:
                pass

    if train_losses and val_losses:
        n = min(len(train_losses), len(val_losses), len(epochs))
        chart_df = pd.DataFrame({
            "Epoch":      epochs[:n],
            "Train Loss": train_losses[:n],
            "Val Loss":   val_losses[:n],
        }).set_index("Epoch")
        st.line_chart(chart_df, color=["#1f77b4", "#ff7f0e"])

    with st.expander("Raw log (last 50 lines)"):
        st.code("\n".join(lines[-50:]), language="text")
else:
    st.info(f"Log non trouvé : {TRAIN_LOG}")

st.divider()

# Footer
st.markdown(
    f"Built with [Streamlit](https://streamlit.io) · "
    f"[VoiceFilter paper](https://arxiv.org/abs/1810.04826) · "
    f"[GitHub]({GITHUB_URL})",
    unsafe_allow_html=False,
)