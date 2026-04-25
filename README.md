# 🎵 Predicting the Beats Per Minute of Songs

Kaggle **Playground Series S5E9** yarışması için geliştirilmiş bir makine öğrenmesi projesi.  
Şarkıların akustik özelliklerinden yola çıkarak **BeatsPerMinute (BPM)** değerini tahmin eder.

---

## 📁 Proje Yapısı

```
├── predicting-the-beats-per-minute-of-songs.ipynb  # Keşifsel analiz & model denemeleri
├── save_model.py                                    # Model eğitimi & artifact kaydetme
├── app.py                                           # Streamlit web uygulaması
├── requirements.txt                                 # Bağımlılıklar
├── train.csv                                        # Eğitim verisi (Kaggle'dan indirilir)
├── model.joblib                                     # Eğitilmiş model (save_model.py ile oluşur)
├── feature_columns.joblib                           # Özellik listesi (save_model.py ile oluşur)
└── bpm_stats.joblib                                 # BPM istatistikleri (save_model.py ile oluşur)
```

---

## 📊 Veri Seti

**Kaynak:** [Kaggle Playground Series S5E9](https://www.kaggle.com/competitions/playground-series-s5e9)

| Özellik | Açıklama |
|---|---|
| `RhythmScore` | Ritim yoğunluğu (0–1) |
| `AudioLoudness` | Ses yüksekliği (negatif dB) |
| `VocalContent` | Vokallerin oranı |
| `AcousticQuality` | Akustik kalite skoru |
| `InstrumentalScore` | Enstrümantal içerik |
| `LivePerformanceLikelihood` | Canlı performans olasılığı |
| `MoodScore` | Ruh hali skoru (0=negatif, 1=pozitif) |
| `TrackDurationMs` | Şarkı süresi (milisaniye) |
| `Energy` | Enerji seviyesi (0–1) |
| `BeatsPerMinute` | 🎯 **Hedef değişken** |

- **Train:** 524,164 satır  
- **Test:** 174,722 satır  
- **Eksik değer:** Yok

---

## ⚙️ Feature Engineering

Notebook'taki analize dayanarak türetilen etkileşim özellikleri:

| Türetilen Özellik | Formül | Açıklama |
|---|---|---|
| `vocal_live` | `VocalContent × LivePerformanceLikelihood` | Vokal–canlı performans etkileşimi |
| `loudness_mood` | `\|AudioLoudness\| × MoodScore` | Ses şiddeti–ruh hali etkileşimi |
| `mood_energy` | `MoodScore × Energy` | Duygu–enerji yoğunluğu |
| `energy_rhythm` | `Energy × RhythmScore` | Enerji–ritim sinerjisi |
| `energy_loudness` | `Energy × \|AudioLoudness\|` | Enerji–ses sinerjisi |
| `TrackDurationSec` | `TrackDurationMs / 1000` | Süre (saniye) |

---

## 🤖 Model Karşılaştırması

| Model | R² | RMSE | MAE |
|---|---|---|---|
| ✅ **Gradient Boosting** | **0.0005** | **26.44** | **21.18** |
| Ridge | 0.0001 | 26.44 | 21.18 |
| LightGBM | -0.0001 | 26.45 | 21.19 |
| XGBoost | -0.0090 | 26.56 | 21.28 |
| KNeighbors | -0.1992 | 28.96 | 23.16 |
| Decision Tree | -1.1142 | 38.45 | 30.80 |

> 📌 BPM tahmini zor bir regresyon problemidir; veri setinde tüm özellikler ile hedef arasında düşük korelasyon bulunmaktadır (max ~0.007). **Gradient Boosting** en iyi sonucu vermiştir.

---

## 🚀 Kurulum ve Çalıştırma

### 1. Bağımlılıkları Kur

```bash
pip install -r requirements.txt
```

### 2. Veriyi İndir

[Kaggle](https://www.kaggle.com/competitions/playground-series-s5e9/data) sayfasından `train.csv` dosyasını indirip proje klasörüne koy.

### 3. Modeli Eğit

```bash
python save_model.py
```

Bu komut şu dosyaları oluşturur:
- `model.joblib` — eğitilmiş GradientBoostingRegressor
- `feature_columns.joblib` — özellik sırası
- `bpm_stats.joblib` — BPM istatistikleri (gauge chart için)

### 4. Uygulamayı Başlat

```bash
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` adresini aç.

---

## 🖥️ Uygulama Özellikleri

- 🎛️ **Sidebar** — tüm şarkı özelliklerini sezgisel slider'larla ayarla
- 📊 **Gauge Chart** — BPM değerini görsel olarak gösterir
- 🎼 **Tempo Kategorisi** — Larghissimo'dan Presto'ya otomatik sınıflandırma
- 📈 **Feature Importance** — modelin hangi özelliklere ağırlık verdiğini gösterir
- 📋 **Türetilen Özellikler** — hesaplanan interaction feature'ları anlık görüntüle
- 🎼 **BPM Referans Tablosu** — müzik tempo terminolojisi

---

## 🎼 BPM Tempo Referansı

| Tempo | BPM Aralığı | Örnek Tür |
|---|---|---|
| Larghissimo | < 60 | Hüzünlü baladlar |
| Largo | 60–80 | Sakin akustik |
| Andante | 80–100 | Slow pop |
| Moderato | 100–120 | Pop / R&B |
| Allegro | 120–140 | Dans / Hafif EDM |
| Vivace | 140–160 | Uptempo pop |
| Presto | > 160 | Drum & Bass / Hardcore |

---

## 🔗 Bağlantılar

- 📓 [Kaggle Notebook](https://www.kaggle.com/competitions/playground-series-s5e9)
- 🏆 [Yarışma Sayfası](https://www.kaggle.com/competitions/playground-series-s5e9)

---

## 📄 Lisans

MIT License
