"""
Merkezi Konfigürasyon Modülü
============================
Tüm sistem ayarları burada dataclass olarak tanımlanır.
Immutable (frozen) dataclass'lar kullanılarak ayarların çalışma zamanında
değiştirilmesi engellenir — bu "yapılandırma kararlılığı" sağlar.

Mimari notu:
    Tüm bileşenler (dedektör, preprocessor, visualizer vb.) kendi
    config sınıfını constructor'da alır. Bu sayede bileşenler birbirinden
    bağımsız test edilebilir (Dependency Injection prensibi).

Sınıf hiyerarşisi:
    PipelineConfig (üst düzey)
    ├── DetectionConfig      — HOG+SVM tespit parametreleri
    ├── PreprocessConfig     — Ön-işleme (resize, CLAHE, sharpening)
    ├── VisualizationConfig  — Görselleştirme (renk, çizgi, panel)
    └── ReportingConfig      — Raporlama ve frame örnekleme
"""

from dataclasses import dataclass, field
from typing import Tuple


# =============================================================================
# TESPİT KONFİGÜRASYONU
# =============================================================================
@dataclass(frozen=True)
class DetectionConfig:
    """
    HOG + SVM tespit parametreleri.

    HOG (Histogram of Oriented Gradients) dedektörü, görüntüdeki kenar
    yönelimlerini analiz ederek yaya silüetini tanır. Bu parametreler
    dedektörün hassasiyetini ve doğruluğunu doğrudan etkiler.
    """

    # --- Kayan Pencere Parametreleri ---
    # win_stride: HOG kayan penceresinin X ve Y yönünde kayma adımı (piksel).
    #   Küçük değer → daha yoğun tarama, daha fazla tespit ama yavaş.
    #   Büyük değer → daha hızlı ama bazı yayaları kaçırabilir.
    win_stride: Tuple[int, int] = (8, 8)

    # padding: Görüntü kenarlarına eklenen dolgu (piksel).
    #   Kenar yakınındaki yayaların tespitini iyileştirir.
    padding: Tuple[int, int] = (8, 8)

    # scale: Görüntü piramidi ölçek faktörü (1.0'dan büyük olmalı).
    #   1.03 gibi küçük değer → daha ince piramit, daha doğru ama yavaş.
    #   1.10 gibi büyük değer → daha hızlı ama uzaktaki küçük yayaları kaçırır.
    scale: float = 1.03

    # --- Eşik Değerleri ---
    # hit_threshold: HOG dedektörünün SVM karar sınırı eşiği.
    #   Yüksek değer → daha az tespit, daha az false positive.
    #   Bu parametre HOG seviyesinde filtreleme yapar (EN HIZLI filtre).
    hit_threshold: float = 0.5

    # confidence_threshold: Post-processing'de uygulanan güven eşiği.
    #   Bu değerin altındaki tespitler elenir.
    #   0.5 → orta düzey filtreleme (tabela/bina false positive'lerini azaltır).
    confidence_threshold: float = 0.5

    # nms_threshold: Non-Maximum Suppression (NMS) çakışma eşiği.
    #   İki kutunun IoU değeri bu eşiği aşarsa, düşük güvenli olan elenir.
    #   0.4 → %40'tan fazla çakışan kutular birleştirilir.
    nms_threshold: float = 0.4

    # --- Boyut Filtreleri ---
    # min_detection_size: Minimum bounding box boyutu (genişlik, yükseklik) piksel.
    #   Çok küçük tespitler genellikle gürültüdür.
    min_detection_size: Tuple[int, int] = (40, 80)

    # max_detection_size: Maksimum bounding box boyutu (genişlik, yükseklik) piksel.
    #   Çok büyük kutular genellikle birden fazla kişiyi veya arka planı kapsar.
    #   200x400 → tek bir yaya için makul üst sınır.
    max_detection_size: Tuple[int, int] = (200, 400)

    # high_confidence_threshold: Yüksek güvenli tespit eşiği.
    #   Bu değerin üstündeki tespitler görselleştirmede YEŞİL gösterilir.
    #   Altındakiler TURUNCU (şüpheli) olarak işaretlenir.
    high_confidence_threshold: float = 1.0

    # --- En-Boy Oranı Filtresi ---
    # Yayalar fiziksel olarak dikey yapıdadır. En-boy oranı (h/w) belirli
    # bir aralıkta olmalıdır. Bu filtre yatay nesneleri (araç, tabela) eler.
    # min_aspect_ratio: Minimum yükseklik/genişlik oranı (1.3 → biraz dikey).
    min_aspect_ratio: float = 1.3
    # max_aspect_ratio: Maksimum oran (3.5 → çok uzun ince nesneleri eler).
    max_aspect_ratio: float = 3.5

    # --- Multi-Pass Tespit ---
    # Kalabalık ortamlarda tek geçişin kaçırdığı yayaları farklı
    # parametrelerle ikinci bir tarama yaparak yakalamaya çalışır.
    # NOT: Performans maliyeti yüksektir, varsayılan kapalı.
    enable_multi_pass: bool = False

    # İkinci geçiş parametreleri (daha yoğun tarama):
    second_pass_win_stride: Tuple[int, int] = (4, 4)   # Daha ince adım
    second_pass_scale: float = 1.02                      # Daha ince piramit
    second_pass_padding: Tuple[int, int] = (16, 16)     # Daha geniş dolgu
    second_pass_hit_threshold: float = 0.3               # Daha düşük eşik


# =============================================================================
# ÖN-İŞLEME KONFİGÜRASYONU
# =============================================================================
@dataclass(frozen=True)
class PreprocessConfig:
    """
    Görüntü ön-işleme parametreleri.

    Ön-işleme, dedektöre girmeden önce görüntü kalitesini iyileştirir.
    İşlem sırası: resize → denoise → CLAHE → sharpen → (gri dönüşüm)

    Her adım bağımsız olarak açılıp kapatılabilir.
    """

    # target_width: Yeniden boyutlandırma hedef genişliği (piksel).
    #   Daha küçük → daha hızlı ama uzak yayaları kaçırır.
    #   640 → hız/doğruluk dengesi için optimal.
    target_width: int = 640

    # convert_to_gray: True ise son adımda gri tonlamaya dönüştürür.
    #   HOG renkli veya gri görüntüde çalışabilir.
    convert_to_gray: bool = False

    # --- CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
    # Düşük kontrastlı veya gölgeli alanlardaki detayları ortaya çıkarır.
    # LAB renk uzayında L kanalına uygulanır — renkleri bozmaz.
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.5           # Kontrast sınırlama değeri
    clahe_grid_size: Tuple[int, int] = (8, 8)  # Bölge ızgarası boyutu

    # --- Keskinleştirme (Unsharp Mask) ---
    # Bulanık video karelerindeki kenar bilgisini güçlendirir.
    # HOG kenar tabanlı çalıştığı için keskinlik doğrudan tespit kalitesini etkiler.
    enable_sharpening: bool = True
    sharpen_strength: float = 0.5  # 0.0 = etkisiz, 1.0 = maksimum keskinleştirme

    # --- Gürültü Azaltma (Gaussian Blur) ---
    # Hafif bulanıklaştırma ile sensör gürültüsünü azaltır.
    # Kernel boyutu küçük tutulur (3-5) — kenar bilgisini korumak için.
    enable_denoising: bool = True
    denoise_strength: int = 3  # GaussianBlur kernel boyutu (tek sayı olmalı)


# =============================================================================
# GÖRSELLEŞTİRME KONFİGÜRASYONU
# =============================================================================
@dataclass(frozen=True)
class VisualizationConfig:
    """
    Görselleştirme (çizim ve panel) parametreleri.

    NOT: box_color artık doğrudan kullanılmıyor — Visualizer sınıfı
    güven seviyesine göre dinamik renk atar. Bu alan yedek olarak korunur.
    """

    box_color: Tuple[int, int, int] = (0, 255, 0)    # Varsayılan kutu rengi (BGR: yeşil)
    box_thickness: int = 2                             # Kutu çizgi kalınlığı (piksel)
    font_scale: float = 0.5                            # Güven etiketi yazı boyutu
    font_color: Tuple[int, int, int] = (0, 255, 0)    # Bilgi paneli yazı rengi
    info_panel_color: Tuple[int, int, int] = (0, 0, 0) # Bilgi paneli arka plan (siyah)
    info_panel_alpha: float = 0.6                       # Panel saydamlık (0=saydam, 1=opak)
    show_confidence: bool = True                        # Güven skoru etiketini göster
    show_info_panel: bool = True                        # FPS/tespit sayısı panelini göster
    save_output: bool = False                           # Çıktı videosunu kaydet
    output_path: str = "output/result.avi"              # Çıktı dosya yolu


# =============================================================================
# RAPORLAMA KONFİGÜRASYONU
# =============================================================================
@dataclass(frozen=True)
class ReportingConfig:
    """
    Loglama, raporlama ve frame örnekleme parametreleri.

    Pipeline çalışması sırasında kare başı istatistik toplar ve
    işlem sonunda detaylı JSON raporu üretir. Ayrıca belirli aralıklarla
    tespit içeren kareleri diske kaydeder (analiz amaçlı).
    """

    enable_reporting: bool = True            # Rapor oluşturmayı etkinleştir
    enable_frame_sampling: bool = True       # Frame örnekleme etkinleştir
    report_output_dir: str = "output"        # Rapor dosyası dizini
    sample_output_dir: str = "output/samples"  # Örnek kare dizini
    sample_interval: int = 10                # Her N tespitli karede bir kaydet
    save_raw_frames: bool = False            # Çizimsiz orijinal kareyi de kaydet
    high_confidence_threshold: float = 1.5   # Bu güvenin üstündeki tespitlerde
                                             # interval'e bakmadan hemen kaydet


# =============================================================================
# ANA PİPELINE KONFİGÜRASYONU
# =============================================================================
@dataclass
class PipelineConfig:
    """
    Pipeline genel konfigürasyonu.

    Tüm alt konfigürasyonları bir araya getirir.
    NOT: Bu sınıf frozen DEĞİL — çünkü çalışma zamanında
    bileşenler tarafından okunması gerekir.
    """

    # Alt konfigürasyonlar — her biri kendi varsayılanlarıyla başlar
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    # Pencere adı — cv2.imshow tarafından kullanılır
    display_window_name: str = "Yaya Tespit Sistemi"

    # Çıkış tuşu — bu tuşa basıldığında pipeline durur
    quit_key: str = "q"
