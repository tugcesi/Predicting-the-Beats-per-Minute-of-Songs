"""
app.py – BPM Predictor (Streamlit)
Notebook pipeline ile birebir uyumlu.
Çalıştır: streamlit run app.py
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Sayfa Ayarları ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 BPM Predictor",
    page_icon="🎵",
    layout="wide"
)

# ── Artifact Yükleme ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    files = ['model.joblib', 'feature_columns.joblib', 'bpm_stats.joblib']
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        return None, None, None, f"Eksik dosyalar: {missing}"
    model    = joblib.load('model.joblib')
    features = joblib.load('feature_columns.joblib')
    stats    = joblib.load('bpm_stats.joblib')
    return model, features, stats, None

model, feature_columns, bpm_stats, err = load_artifacts()

if err:
    st.error(f"⚠️ {err}")
    st.info("Önce şunu çalıştırın: `python save_model.py`")
    st.stop()

# ── Pipeline Fonksiyonu (save_model.py ile birebir) ──────────────────────────
def build_input_df(inputs: dict) -> pd.DataFrame:
    vocal_content              = inputs['VocalContent']
    live_performance_likelihood = inputs['LivePerformanceLikelihood']
    audio_loudness             = inputs['AudioLoudness']
    mood_score                 = inputs['MoodScore']
    energy                     = inputs['Energy']
    rhythm_score               = inputs['RhythmScore']
    track_duration_ms          = inputs['TrackDurationMs']

    row = {
        'MoodScore'                  : mood_score,
        'TrackDurationMs'            : track_duration_ms,
        'vocal_live'                 : vocal_content * live_performance_likelihood,
        'loudness_mood'              : abs(audio_loudness) * mood_score,
        'RhythmScore'                : rhythm_score,
        'VocalContent'               : vocal_content,
        'Energy'                     : energy,
        'LivePerformanceLikelihood'  : live_performance_likelihood,
        'AudioLoudness'              : audio_loudness,
        'InstrumentalScore'          : inputs['InstrumentalScore'],
        'AcousticQuality'            : inputs['AcousticQuality'],
    }
    return pd.DataFrame([row])[feature_columns]


# ── Gauge Chart ───────────────────────────────────────────────────────────────
def make_gauge(bpm: float, bpm_min: float, bpm_max: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=bpm,
        title={'text': "Tahmini BPM"},
        number={'suffix': " BPM", 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [bpm_min * 0.9, bpm_max * 1.05]},
            'bar' : {'color': '#7C3AED'},
            'steps': [
                {'range': [bpm_min * 0.9,  80],  'color': '#DBEAFE'},
                {'range': [80,             120],  'color': '#D1FAE5'},
                {'range': [120,            160],  'color': '#FEF9C3'},
                {'range': [160, bpm_max * 1.05],  'color': '#FFE4E6'},
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 3},
                'thickness': 0.8,
                'value': 120
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
    return fig


# ── BPM Tempo Kategorisi ──────────────────────────────────────────────────────
def bpm_category(bpm: float) -> tuple[str, str]:
    if bpm < 60:
        return "🐢 Larghissimo / Çok Yavaş", "#93C5FD"
    elif bpm < 80:
        return "🚶 Largo / Yavaş", "#6EE7B7"
    elif bpm < 100:
        return "🎻 Andante / Orta Yavaş", "#A7F3D0"
    elif bpm < 120:
        return "🎸 Moderato / Orta", "#FDE68A"
    elif bpm < 140:
        return "🎤 Allegro / Hızlı", "#FCA5A5"
    elif bpm < 160:
        return "🔥 Vivace / Çok Hızlı", "#F87171"
    else:
        return "⚡ Presto / Aşırı Hızlı", "#EF4444"


# ── Başlık ────────────────────────────────────────────────────────────────────
st.title("🎵 Beats Per Minute (BPM) Predictor")
st.caption("Kaggle Playground Series S5E9 | GradientBoostingRegressor")
st.divider()

# ── Sidebar: Kullanıcı Girdileri ─────────────────────────────────────────────
with st.sidebar:
    st.header("🎛️ Şarkı Özellikleri")

    st.subheader("🎵 Ritim & Enerji")
    rhythm_score = st.slider(
        "RhythmScore", 0.0, 1.0, 0.60,
        help="Şarkının ritim yoğunluğu (0=düşük, 1=yüksek)"
    )
    energy = st.slider(
        "Energy", 0.0, 1.0, 0.50,
        help="Şarkının enerji seviyesi"
    )
    audio_loudness = st.slider(
        "AudioLoudness (dB)", -30.0, -1.0, -8.4,
        help="Ses yüksekliği (negatif dB)"
    )

    st.subheader("🎤 Vokal & Performans")
    vocal_content = st.slider(
        "VocalContent", 0.0, 0.26, 0.07,
        help="Vokallerin müzikteki oranı"
    )
    live_perf = st.slider(
        "LivePerformanceLikelihood", 0.024, 0.60, 0.18,
        help="Canlı performans olasılığı"
    )

    st.subheader("🎹 Akustik & Enstrümantal")
    acoustic_quality = st.slider(
        "AcousticQuality", 0.0, 1.0, 0.26,
        help="Akustik kalite skoru"
    )
    instrumental_score = st.slider(
        "InstrumentalScore", 0.0, 1.0, 0.12,
        help="Enstrümantal içerik skoru"
    )

    st.subheader("😊 Duygu & Süre")
    mood_score = st.slider(
        "MoodScore", 0.0, 1.0, 0.56,
        help="Şarkının ruh hali skoru (0=negatif, 1=pozitif)"
    )
    track_duration_ms = st.number_input(
        "TrackDurationMs (milisaniye)", 
        min_value=60000, max_value=600000,
        value=241866, step=1000,
        help="Şarkı süresi (ms). Örn: 3 dk = 180,000 ms"
    )

    predict_btn = st.button(
        "🎵 BPM Tahmin Et", use_container_width=True, type="primary"
    )

# ── Tahmin ────────────────────────────────────────────────────────────────────
if predict_btn:
    inputs = {
        'RhythmScore'               : rhythm_score,
        'AudioLoudness'             : audio_loudness,
        'VocalContent'              : vocal_content,
        'AcousticQuality'           : acoustic_quality,
        'InstrumentalScore'         : instrumental_score,
        'LivePerformanceLikelihood' : live_perf,
        'MoodScore'                 : mood_score,
        'TrackDurationMs'           : track_duration_ms,
        'Energy'                    : energy,
    }

    try:
        X_input  = build_input_df(inputs)
        pred_bpm = float(model.predict(X_input)[0])

        tempo_label, tempo_color = bpm_category(pred_bpm)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.success("### ✅ Tahmin Tamamlandı!")
            st.metric("Tahmini BPM", f"{pred_bpm:.1f} BPM")
            
            low  = max(0.0, pred_bpm * 0.95)
            high = pred_bpm * 1.05
            st.caption(f"📉 Tahmini aralık: **{low:.1f}** – **{high:.1f}** BPM (±5%)")

            st.markdown(
                f"<div style='background:{tempo_color};padding:12px;border-radius:8px;"
                f"text-align:center;font-size:18px;font-weight:bold'>{tempo_label}</div>",
                unsafe_allow_html=True
            )
            st.plotly_chart(
                make_gauge(pred_bpm, bpm_stats['min'], bpm_stats['max']),
                use_container_width=True
            )

        with col2:
            st.info("### 📋 Girilen Şarkı Özellikleri")

            # Computed features göster
            vocal_live_val    = vocal_content * live_perf
            loudness_mood_val = abs(audio_loudness) * mood_score
            track_sec         = track_duration_ms / 1000

            display = pd.DataFrame({
                'Özellik': [
                    'RhythmScore', 'AudioLoudness', 'VocalContent',
                    'AcousticQuality', 'InstrumentalScore', 'LivePerformanceLikelihood',
                    'MoodScore', 'TrackDurationMs', 'Energy',
                    '── Türetilen ──',
                    'vocal_live', 'loudness_mood', 'TrackDurationSec'
                ],
                'Değer': [
                    f"{rhythm_score:.4f}", f"{audio_loudness:.4f}", f"{vocal_content:.4f}",
                    f"{acoustic_quality:.4f}", f"{instrumental_score:.4f}", f"{live_perf:.4f}",
                    f"{mood_score:.4f}", f"{track_duration_ms:,}", f"{energy:.4f}",
                    '──────────────',
                    f"{vocal_live_val:.4f}", f"{loudness_mood_val:.4f}", f"{track_sec:.1f} sn"
                ]
            })
            st.dataframe(display, hide_index=True, use_container_width=True)

        # ── Feature Importance ─────────────────────────────────────────────
        st.divider()
        st.subheader("📈 Feature Importance")

        fi = pd.Series(
            model.feature_importances_, index=feature_columns
        ).sort_values(ascending=True)

        fig_fi = go.Figure(go.Bar(
            x=fi.values, y=fi.index,
            orientation='h',
            marker_color='#7C3AED'
        ))
        fig_fi.update_layout(
            xaxis_title="Importance", height=380,
            margin=dict(l=10, r=10, t=20, b=10)
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        # ── BPM Referans Tablosu ───────────────────────────────────────────
        st.divider()
        st.subheader("🎼 BPM Tempo Referansı")
        ref_df = pd.DataFrame({
            'Tempo': ['Larghissimo', 'Largo', 'Andante', 'Moderato', 'Allegro', 'Vivace', 'Presto'],
            'BPM Aralığı': ['< 60', '60–80', '80–100', '100–120', '120–140', '140–160', '> 160'],
            'Örnek Tür': [
                'Hüzünlü baladlar', 'Sakin akustik', 'Slow pop',
                'Pop / R&B', 'Dans / EDM hafif', 'Uptempo pop',
                'Drum & Bass / Hardcore'
            ]
        })
        st.dataframe(ref_df, hide_index=True, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Hata: {e}")

else:
    # ── Karşılama Ekranı ──────────────────────────────────────────────────
    st.info("👈 Sol panelden şarkı özelliklerini ayarlayın ve **BPM Tahmin Et** butonuna tıklayın.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model",       "Gradient Boosting")
    c2.metric("Yarışma",     "Kaggle S5E9")
    c3.metric("Train Rows",  "524,164")
    c4.metric("Hedef",       "BeatsPerMinute")

    st.subheader("🔬 Feature Engineering Özeti")
    fe_df = pd.DataFrame({
        'Türetilen Özellik': [
            'vocal_live', 'loudness_mood', 'mood_energy',
            'energy_rhythm', 'energy_loudness', 'TrackDurationSec'
        ],
        'Formül': [
            'VocalContent × LivePerformanceLikelihood',
            '|AudioLoudness| × MoodScore',
            'MoodScore × Energy',
            'Energy × RhythmScore',
            'Energy × |AudioLoudness|',
            'TrackDurationMs / 1000'
        ],
        'Amacı': [
            'Vokal × canlı performans etkileşimi',
            'Ses şiddeti × ruh hali etkileşimi',
            'Enerji × duygu yoğunluğu',
            'Enerji × ritim sinerji',
            'Enerji × ses şiddeti sinerji',
            'Süre saniye cinsinden'
        ]
    })
    st.dataframe(fe_df, hide_index=True, use_container_width=True)

    st.subheader("📊 Model Karşılaştırması (Notebook'tan)")
    model_df = pd.DataFrame({
        'Model': ['Gradient Boosting', 'Ridge', 'LightGBM', 'XGBoost', 'KNeighbors', 'Decision Tree'],
        'R²':    [0.0005, 0.0001, -0.0001, -0.0090, -0.1992, -1.1142],
        'RMSE':  [26.4388, 26.4439, 26.4470, 26.5642, 28.9600, 38.4519],
        'MAE':   [21.1799, 21.1841, 21.1883, 21.2762, 23.1588, 30.7964],
    })
    st.dataframe(
        model_df.style.highlight_min(subset=['RMSE', 'MAE'], color='#D1FAE5')
                      .highlight_max(subset=['R²'],           color='#D1FAE5'),
        hide_index=True, use_container_width=True
    )
    st.caption("✅ Gradient Boosting en yüksek R² ve en düşük RMSE/MAE ile seçildi.")