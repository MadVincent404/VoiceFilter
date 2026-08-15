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
GITHUB_URL = "https://github.com/MadVincent404/VoiceFilter.git"
SR         = 16000

PHASES = {
    "Phase 1 — Clean Speech": {
        "desc":      "Trained on LibriSpeech train-clean-360 + train-clean-100 (~460h)",
        "log":       "miscellaneous/logs/voicefilter.log",
        "best_dir":  "audios/eval_results_phase1/best_data",
        "worst_dir": "audios/eval_results_phase1/worst_data",
        "csv":       "audios/eval_results_phase1/results.csv",
        "metrics": [
            ("PESQ",   1.132, 1.813, "+0.681"),
            ("SI-SDR", 0.003, 7.828, "+7.83 dB"),
            ("SI-SNR", 0.003, 7.834, "+7.83 dB"),
            ("SDR",    0.115, 8.516, "+8.40 dB"),
            ("WER",    1.242, 0.762, "-48.1 pp"),
        ],
        "noisy": False,
    },
    "Phase 2 — Diverse Speech": {
        "desc":      "Fine-tuned on train-other-500 + train-clean-100 (~600h, accents & noisy recordings)",
        "log":       "miscellaneous/logs_phase2/voicefilter.log",
        "best_dir":  "audios/eval_results_phase2/best_data",
        "worst_dir": "audios/eval_results_phase2/worst_data",
        "csv":       "audios/eval_results_phase2/results.csv",
        "metrics": [
            ("PESQ",   1.138, 1.826, "+0.688"),
            ("SI-SDR",-0.007, 7.119, "+7.126 dB"),
            ("SI-SNR",-0.007, 7.123, "+7.130 dB"),
            ("SDR",    0.116, 7.923, "+7.807 dB"),
            ("WER",    "—",   "—",   "—"),
        ],
        "noisy": False,
    },
    "Phase 3 — Noise & Music": {
        "desc":       "Fine-tuned with MUSAN ambient noise & music augmentation (50% noise, 30% music)",
        "log":        "miscellaneous/logs_phase3/voicefilter.log",
        "best_dir":   "audios/eval_results_phase3/best_data",
        "worst_dir":  "audios/eval_results_phase3/worst_data",
        "csv_speech": "audios/eval_results_phase3/results_speech_only.csv",
        "csv_noise":  "audios/eval_results_phase3/results_with_noise.csv",
        "csv_music":  "audios/eval_results_phase3/results_with_music.csv",
        "metrics": [
            ("PESQ",   "+0.698", "+0.321", "+0.264"),
            ("SI-SDR", "+7.619", "+6.081", "+5.544"),
            ("SI-SNR", "+7.622", "+6.083", "+5.545"),
            ("SDR",    "+8.197", "+6.553", "+5.966"),
            ("WER",    "—",      "—",      "—"),
        ],
        "noisy": True,
    },
}

# Page setup
st.set_page_config(page_title="VoiceFilter", layout="wide")

st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa; border-radius: 8px; padding: 16px;
        text-align: center; border: 1px solid #e9ecef; height: 110px;
    }
    .metric-val   { font-size: 1.8rem; font-weight: 700; color: #1f77b4; }
    .metric-delta { font-size: 0.95rem; color: #2ca02c; }
    .metric-label { font-size: 0.82rem; color: #555; margin-top: 4px; }
    .phase-desc   { background:#eef4fb; border-left:4px solid #1f77b4;
                    padding:10px 14px; border-radius:4px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("VoiceFilter — Speaker-Conditioned Voice Isolation")
st.markdown(
    f"Reproduction of **[VoiceFilter (Wang et al., 2018)](https://arxiv.org/abs/1810.04826)** · "
    f"[GitHub]({GITHUB_URL})"
)
st.divider()

# Helpers
def list_wav(folder):
    p = Path(folder)
    return sorted(p.glob("*.wav")) if p.exists() else []


def render_metrics(metrics):
    cols = st.columns(len(metrics))
    for col, (name, mixed, filtered, delta) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{filtered}</div>
                <div class="metric-delta">{delta} vs mixed</div>
                <div class="metric-label">{name}</div>
            </div>""", unsafe_allow_html=True)


def render_audio_table(data_dir, label, extra_cols=None):
    """
    Affiche un tableau d'échantillons audio.
    extra_cols : liste de (sous-dossier, titre) supplémentaires (ex: noise, music)
    """
    icon  = "🏆" if label == "best" else "⚠️"
    title = "Best 5 Samples" if label == "best" else "Worst 5 Samples"
    st.markdown(f"**{icon} {title}** (ranked by PESQ)")

    base_cols = [
        ("mixed",         "Mixed"),
        ("reference",     "Reference (target)"),
        ("interfered",    "Interferer"),
        ("voice_filtered","VoiceFilter output"),
    ]
    if extra_cols:
        base_cols += extra_cols

    files_by_col = {
        sub: list_wav(os.path.join(data_dir, sub))
        for sub, _ in base_cols
    }
    n = max((len(v) for v in files_by_col.values()), default=0)
    if n == 0:
        st.info(f"Aucun fichier audio dans {data_dir}")
        return

    # Header
    widths = [1] + [2] * len(base_cols)
    hrow   = st.columns(widths)
    hrow[0].markdown("**#**")
    for i, (_, title_) in enumerate(base_cols, 1):
        hrow[i].markdown(f"**{title_}**")
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)

    for i in range(min(5, n)):
        row = st.columns(widths)
        row[0].markdown(f"**{i+1}**")
        for j, (sub, _) in enumerate(base_cols, 1):
            fs = files_by_col[sub]
            if i < len(fs):
                row[j].audio(str(fs[i]), format="audio/wav")
            else:
                row[j].markdown("—")


def render_csv(csv_path, label=""):
    if not Path(csv_path).exists():
        st.info(f"CSV non trouvé : {csv_path}")
        return
    df = pd.read_csv(csv_path)
    skip = {"mixed_wav","target_wav","interferer_wav","est_wav",
            "noise_wav","music_wav"}
    cols = [c for c in df.columns if c not in skip]

    delta_cols = [c for c in cols if c.startswith("delta_")]

    def color_delta(val):
        try:
            v = float(val)
            color = "green" if v > 0 else ("red" if v < 0 else "gray")
            # WER : amélioration = delta négatif
            return f"color: {color}"
        except Exception:
            return ""

    if label:
        st.caption(f"Condition : **{label}**")
    st.dataframe(
        df[cols].style.map(color_delta, subset=delta_cols),
        use_container_width=True,
        height=250,
    )


def render_log(log_path):
    p = Path(log_path)
    if not p.exists():
        st.info(f"Log non trouvé : {log_path}")
        return

    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    train_losses, val_losses, epochs = [], [], []
    ep = 0
    for l in lines:
        if "Train Loss" in l:
            ep += 1
            try:
                train_losses.append(float(l.split("Train Loss:")[-1].strip()))
                epochs.append(ep)
            except Exception:
                pass
        elif "Val Loss" in l:
            try:
                val_losses.append(float(l.split("Val Loss:")[-1].strip()))
            except Exception:
                pass

    if train_losses and val_losses:
        n = min(len(train_losses), len(val_losses), len(epochs))
        df = pd.DataFrame({
            "Epoch":      epochs[:n],
            "Train Loss": train_losses[:n],
            "Val Loss":   val_losses[:n],
        }).set_index("Epoch")
        st.line_chart(df, color=["#1f77b4", "#ff7f0e"])

    with st.expander("Raw log (last 50 lines)"):
        st.code("\n".join(lines[-50:]), language="text")


# Phase tabs
tab1, tab2, tab3 = st.tabs(list(PHASES.keys()))

for tab, (phase_name, cfg) in zip([tab1, tab2, tab3], PHASES.items()):
    with tab:
        # Description
        st.markdown(f'<div class="phase-desc">{cfg["desc"]}</div>',
                    unsafe_allow_html=True)

        # Métriques
        st.subheader("Evaluation Metrics — test-clean (500 pairs, SNR=0 dB)")
        render_metrics(cfg["metrics"])
        st.caption("Mixed = baseline (no processing) · Filtered = VoiceFilter output")
        st.divider()

        # Samples audio
        st.subheader("Audio Samples")

        if cfg["noisy"]:
            # Phase 3 — 3 sous-onglets par condition
            sub_speech, sub_noise, sub_music = st.tabs(
                ["Speech only", "Speech + Noise", "Speech + Music"]
            )
            for sub_tab, cond_label, extra in [
                (sub_speech, "speech_only",  []),
                (sub_noise,  "with_noise",   [("noise", "Ambient noise")]),
                (sub_music,  "with_music",   [("music", "Music")]),
            ]:
                with sub_tab:
                    c1, c2 = st.columns(2)
                    with c1:
                        render_audio_table(cfg["best_dir"],  "best",  extra)
                    with c2:
                        render_audio_table(cfg["worst_dir"], "worst", extra)
        else:
            c1, c2 = st.columns(2)
            with c1:
                render_audio_table(cfg["best_dir"],  "best")
            with c2:
                render_audio_table(cfg["worst_dir"], "worst")

        st.divider()

        # CSV résultats
        st.subheader("Per-pair Results")
        if cfg["noisy"]:
            csv_tab1, csv_tab2, csv_tab3 = st.tabs(
                ["Speech only", "With noise", "With music"]
            )
            with csv_tab1:
                render_csv(cfg.get("csv_speech", ""), "Speech only")
            with csv_tab2:
                render_csv(cfg.get("csv_noise",  ""), "Speech + Noise")
            with csv_tab3:
                render_csv(cfg.get("csv_music",  ""), "Speech + Music")
        else:
            render_csv(cfg.get("csv", ""))

        st.divider()

        # Training log
        st.subheader("Training Log")
        render_log(cfg["log"])

# Footer
st.divider()
st.markdown(
    f"Built with [Streamlit](https://streamlit.io) · "
    f"[VoiceFilter paper](https://arxiv.org/abs/1810.04826) · "
    f"[GitHub]({GITHUB_URL})"
)