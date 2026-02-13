# 🚶 Yaya Tespit Sistemi

Python ve OpenCV kullanarak **HOG (Histogram of Oriented Gradients) + SVM** tabanlı yaya tespiti yapan modüler bir görüntü işleme sistemi.

## ✨ Özellikler

- **Stock Video Desteği** — Video dosyaları üzerinden kare kare yaya tespiti
- **Real-Time Destek** — Kamera ile canlı yaya tespiti
- **Non-Maximum Suppression (NMS)** — Çakışan tespitleri elemine eder
- **FPS Göstergesi** — Anlık performans takibi (kayan ortalama)
- **Güven Skoru** — Her tespit için güvenilirlik değeri
- **Video Çıktısı Kaydetme** — İşlenmiş videoyu dosyaya yazma
- **CLI Arayüzü** — Argümanlarla esnek kullanım

## 🏗️ Mimari

### Design Pattern'ler

| Pattern | Modül | Amaç |
|---|---|---|
| **Strategy** | `core/source/` | Dosya ve kamera kaynakları arasında geçiş |
| **Factory** | `source_factory.py` | Kaynak tipine göre nesne üretimi |
| **Template Method** | `base_detector.py` | Tespit algoritması iskeleti |
| **Pipeline** | `detection_pipeline.py` | İşleme adımlarının orkestrasyonu |

### İşleme Akışı

```
Video Kaynağı → Ön-İşleme → HOG+SVM Tespit → NMS Filtreleme → Görselleştirme
     │              │              │                │                │
  file/camera    resize       detectMultiScale    overlap          bbox +
                grayscale                         elimination     info panel
```

### Proje Yapısı

```
YayaTespit/
├── main.py                              # CLI giriş noktası
├── requirements.txt                     # Bağımlılıklar
├── .gitignore
│
├── config/
│   └── settings.py                      # Dataclass konfigürasyonlar
│
├── core/
│   ├── source/                          # Video kaynağı (Strategy)
│   │   ├── base_source.py               # Abstract VideoSource
│   │   ├── file_source.py               # Dosya tabanlı kaynak
│   │   ├── camera_source.py             # Kamera kaynağı
│   │   └── source_factory.py            # Factory
│   │
│   ├── preprocessing/
│   │   └── preprocessor.py              # Resize, renk dönüşümü
│   │
│   ├── detection/
│   │   ├── base_detector.py             # Abstract Detector + Detection
│   │   └── hog_detector.py              # HOG + SVM implementasyonu
│   │
│   ├── postprocessing/
│   │   └── postprocessor.py             # NMS filtreleme
│   │
│   └── visualization/
│       └── visualizer.py                # Bounding box + bilgi paneli
│
├── pipeline/
│   └── detection_pipeline.py            # Orkestrasyon
│
├── utils/
│   ├── logger.py                        # Merkezi logging
│   └── fps_counter.py                   # Kayan ortalama FPS
│
└── input/                               # Test videoları
```

## 🔧 Kurulum

### Gereksinimler

- Python 3.10+
- Web kamerası (real-time tespit için)

### Adımlar

```bash
# 1. Sanal ortamı oluştur
python -m venv venv

# 2. Sanal ortamı aktifle
.\venv\Scripts\activate        # Windows
source venv/bin/activate       # Linux/Mac

# 3. Bağımlılıkları kur
pip install -r requirements.txt
```

## 🚀 Kullanım

### Stock Video ile Tespit

```bash
python main.py --source file --input input/video.mp4
```

### Kamera ile Real-Time Tespit

```bash
python main.py --source camera
```

### Çıktıyı Kaydetme

```bash
python main.py --source file --input input/video.mp4 --save-output --output-path output/sonuc.avi
```

### Tüm Parametreler

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--source` | `file` | Kaynak tipi: `file` veya `camera` |
| `--input` | — | Video dosya yolu (file modu için zorunlu) |
| `--camera-index` | `0` | Kamera cihaz indeksi |
| `--target-width` | `640` | Ön-işleme hedef genişlik (piksel) |
| `--save-output` | `False` | Çıktı videosunu kaydet |
| `--output-path` | `output/result.avi` | Çıktı dosya yolu |

### Kontroller

- **`q`** — Programı durdur ve pencereyi kapat

## ⚙️ Konfigürasyon

Tüm ayarlar `config/settings.py` içindeki dataclass'lar ile yönetilir:

```python
# HOG + SVM Tespit Parametreleri
DetectionConfig(
    win_stride=(8, 8),       # Kayma penceresi adımı
    padding=(8, 8),          # ROI dolgusu
    scale=1.05,              # Piramit ölçek faktörü
    confidence_threshold=0.3, # Güven eşiği
    nms_threshold=0.4,       # NMS örtüşme eşiği
    min_detection_size=(40, 80),  # Minimum tespit boyutu
)
```

## 🧩 SOLID Prensipleri

| Prensip | Uygulama |
|---|---|
| **Single Responsibility** | Her modül tek bir sorumluluğa sahip |
| **Open/Closed** | Yeni tespit algoritması eklemek mevcut kodu değiştirmez |
| **Liskov Substitution** | `CameraSource` ↔ `FileVideoSource` birbirinin yerine kullanılabilir |
| **Interface Segregation** | Küçük, odaklı arayüzler (`VideoSource`, `BaseDetector`) |
| **Dependency Inversion** | Pipeline soyutlamalara bağımlı, somut sınıflara değil |

## 📊 Teknik Detaylar

### HOG + SVM

- **HOG**: Görüntüdeki kenar yönelimlerinin histogramını çıkarır
- **SVM**: OpenCV'nin önceden eğitilmiş `DefaultPeopleDetector` modeli
- **Multi-Scale**: `detectMultiScale` ile farklı boyutlardaki yayaları tespit eder
- **NMS**: `cv2.dnn.NMSBoxes` ile çakışan kutuları elemine eder

### Performans Optimizasyonları

- Frame küçültme ile işlem hızlandırma (`--target-width`)
- Koordinat ölçekleme ile orijinal boyutta doğru konumlama
- `deque` tabanlı kayan ortalama FPS (sabit bellek)
- Context manager ile güvenli kaynak yönetimi

## 📝 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.
