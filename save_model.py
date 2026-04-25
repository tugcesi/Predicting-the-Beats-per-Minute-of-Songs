"""
save_model.py – BPM Predictor
Pipeline (notebook ile birebir uyumlu):
  1. train.csv yükle
  2. Feature engineering
  3. GradientBoostingRegressor eğit
  4. model.joblib + feature_columns.joblib kaydet

Çalıştır: python save_model.py
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ── 1. Veri Yükleme ───────────────────────────────────────────────────────────
print("📂 Veri yükleniyor...")
train = pd.read_csv("train.csv")

print(f"Train shape: {train.shape}")

# ── 2. Feature Engineering (notebook ile birebir) ────────────────────────────
print("⚙️  Feature engineering...")

df = train.copy()

# Interaction features
df['vocal_live']      = df['VocalContent'] * df['LivePerformanceLikelihood']
df['loudness_mood']   = df['AudioLoudness'].abs() * df['MoodScore']
df['mood_energy']     = df['MoodScore'] * df['Energy']
df['energy_rhythm']   = df['Energy'] * df['RhythmScore']
df['energy_loudness'] = df['Energy'] * df['AudioLoudness'].abs()
df['TrackDurationSec'] = df['TrackDurationMs'] / 1000

# ── 3. Feature Listesi (notebook'taki sıra ile) ───────────────────────────────
features = [
    'MoodScore',
    'TrackDurationMs',
    'vocal_live',
    'loudness_mood',
    'RhythmScore',
    'VocalContent',
    'Energy',
    'LivePerformanceLikelihood',
    'AudioLoudness',
    'InstrumentalScore',
    'AcousticQuality',
]

x = df[features]
y = df['BeatsPerMinute']

# ── 4. Train/Val Split ────────────────────────────────────────────────────────
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# ── 5. Model Eğitimi ──────���───────────────────────────────────────────────────
print("🚀 Model eğitiliyor (GradientBoostingRegressor)...")
model = GradientBoostingRegressor(random_state=42)
model.fit(x_train, y_train)

# ── 6. Değerlendirme ──────────────────────────────────────────────────────────
preds = model.predict(x_val)
print(f"\n📊 Validation Sonuçları:")
print(f"   R²   : {r2_score(y_val, preds):.4f}")
print(f"   RMSE : {mean_squared_error(y_val, preds)**0.5:.4f}")
print(f"   MAE  : {mean_absolute_error(y_val, preds):.4f}")

# ── 7. Artifact Kaydet ────────────────────────────────────────────────────────
joblib.dump(model,    'model.joblib')
joblib.dump(features, 'feature_columns.joblib')

# BPM istatistikleri (app için)
bpm_stats = {
    'min'    : float(y.min()),
    'max'    : float(y.max()),
    'mean'   : float(y.mean()),
    'median' : float(y.median()),
}
joblib.dump(bpm_stats, 'bpm_stats.joblib')

print("\n✅ Kaydedilen dosyalar:")
print("   model.joblib")
print("   feature_columns.joblib")
print("   bpm_stats.joblib")