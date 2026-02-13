"""
Görselleştirme Modülü
======================
Frame üzerine tespit sonuçlarını görsel olarak çizer:
    - Bounding box (sınırlayıcı kutu) çizimi
    - Güven skoru etiketleri
    - FPS ve tespit sayısı bilgi paneli
    - Güven seviyesine göre renk kodlaması

Renk Kodlaması:
    - YEŞİL (0, 255, 0): Yüksek güvenli tespit (≥ high_confidence_threshold)
    - TURUNCU (0, 180, 255): Düşük güvenli tespit (şüpheli, gözle doğrulanmalı)

Bu renk sistemi sayesinde kullanıcı hangi tespitlerin güvenilir,
hangilerinin şüpheli olduğunu bir bakışta anlayabilir.
"""

from typing import List, Optional

import cv2
import numpy as np

from config.settings import VisualizationConfig, DetectionConfig
from core.detection.base_detector import Detection
from utils.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# GÜVEN TABANLI RENK SABİTLERİ (BGR formatında)
# =============================================================================
# NOT: OpenCV BGR renk sırası kullanır (RGB değil!)
_COLOR_HIGH = (0, 255, 0)          # Yeşil — yüksek güvenli tespit kutu rengi
_COLOR_LOW = (0, 180, 255)         # Turuncu — düşük güvenli tespit kutu rengi
_COLOR_TEXT_BG_HIGH = (0, 180, 0)  # Koyu yeşil — yüksek güven etiket arka planı
_COLOR_TEXT_BG_LOW = (0, 140, 200) # Koyu turuncu — düşük güven etiket arka planı


class Visualizer:
    """
    Frame üzerine tespit sonuçlarını ve bilgi panelini çizer.

    İki ana çizim fonksiyonu:
        1. Bounding box + güven etiketi (her tespit için)
        2. Bilgi paneli (FPS + toplam tespit sayısı)

    Opsiyonel olarak çıktıyı video dosyasına da kaydeder.
    """

    # OpenCV yazı tipi — tüm metin çizimlerinde kullanılır
    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        config: VisualizationConfig,
        detection_config: DetectionConfig = None,
    ) -> None:
        """
        Visualizer'ı başlatır.

        Args:
            config: Görselleştirme konfigürasyonu (renkler, kalınlık, panel ayarları).
            detection_config: Tespit konfigürasyonu (yüksek güven eşiği için).
                              None ise varsayılan eşik (1.0) kullanılır.
        """
        self._config = config
        self._det_config = detection_config

        # Yüksek güven eşiği — bu değerin üstü yeşil, altı turuncu gösterilir
        self._high_conf = 1.0
        if detection_config is not None:
            self._high_conf = detection_config.high_confidence_threshold

        # Video yazıcı — save_output etkinse setup_writer() ile başlatılır
        self._writer: Optional[cv2.VideoWriter] = None

        logger.info("Visualizer başlatıldı | Çıktı kaydetme: %s", config.save_output)

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Frame üzerine tüm tespit sonuçlarını çizer.

        İşlem sırası:
            1. Frame'in kopyasını al (orijinal korunur)
            2. Her tespit için bounding box + etiket çiz
            3. Bilgi paneli çiz (FPS, tespit sayısı)
            4. Video yazıcıya yaz (etkinse)

        Args:
            frame: Orijinal BGR frame (değiştirilmez).
            detections: Post-processing sonrası tespit listesi.
            fps: Mevcut işlem FPS değeri (bilgi paneli için).

        Returns:
            Çizim yapılmış frame kopyası (orijinal bozulmaz).
        """
        # Frame'in kopyası üzerinde çalış — orijinali koru
        # (örnekleme modülü orijinali kaydetmek isteyebilir)
        output = frame.copy()

        # Her tespit için bounding box ve güven etiketi çiz
        for detection in detections:
            self._draw_bounding_box(output, detection)

        # Sol üst köşeye FPS + tespit sayısı paneli çiz
        if self._config.show_info_panel:
            self._draw_info_panel(output, len(detections), fps)

        # Video dosyasına yaz (yazıcı aktifse)
        if self._writer is not None:
            self._writer.write(output)

        return output

    def _get_box_color(self, confidence: float) -> tuple:
        """
        Güven seviyesine göre kutu ve etiket arka plan rengini belirler.

        Renk kodlaması mantığı:
            güven ≥ high_confidence_threshold → YEŞİL (güvenilir tespit)
            güven < high_confidence_threshold → TURUNCU (şüpheli tespit)

        Args:
            confidence: Tespite ait güven skoru.

        Returns:
            (kutu_rengi, etiket_arka_plan_rengi) tuple'ı.
        """
        if confidence >= self._high_conf:
            return _COLOR_HIGH, _COLOR_TEXT_BG_HIGH
        return _COLOR_LOW, _COLOR_TEXT_BG_LOW

    def _draw_bounding_box(self, frame: np.ndarray, det: Detection) -> None:
        """
        Tek bir tespit kutusunu güven rengine göre çizer.

        Çizim elementleri:
            1. Dikdörtgen kutu (bounding box)
            2. Güven skoru etiketi (kutunun üstünde)
            3. Etiket arka planı (okunabilirlik için)

        Args:
            frame: Üzerine çizim yapılacak frame (in-place değiştirilir).
            det: Çizilecek tespit nesnesi.
        """
        # Kutunun köşe koordinatlarını hesapla
        top_left = (det.x, det.y)
        bottom_right = (det.x + det.w, det.y + det.h)

        # Güven seviyesine göre renk belirle
        box_color, text_bg_color = self._get_box_color(det.confidence)

        # --- Bounding Box Çizimi ---
        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            box_color,                       # Güven tabanlı renk
            self._config.box_thickness,      # Çizgi kalınlığı (piksel)
        )

        # --- Güven Skoru Etiketi ---
        if self._config.show_confidence:
            # Etiket metni: "Yaya 1.44"
            label = f"Yaya {det.confidence:.2f}"

            # Metin boyutunu hesapla (etiket arka planı için gerekli)
            label_size, _ = cv2.getTextSize(
                label, self._FONT, self._config.font_scale, 1
            )

            # Etiket arka planı — metin okunabilirliğini artırır
            # Kutunun hemen üstüne yarı saydam dikdörtgen çizer
            cv2.rectangle(
                frame,
                (det.x, det.y - label_size[1] - 8),   # Sol üst (metin üstü)
                (det.x + label_size[0] + 4, det.y),     # Sağ alt (kutu üstü)
                text_bg_color,                            # Güven tabanlı arka plan rengi
                cv2.FILLED,                               # İçi dolu dikdörtgen
            )

            # Metin çizimi (beyaz yazı, anti-aliased)
            cv2.putText(
                frame,
                label,
                (det.x + 2, det.y - 4),       # Metin konumu (sol alt referans)
                self._FONT,
                self._config.font_scale,
                (255, 255, 255),               # Beyaz yazı rengi
                1,                              # Yazı kalınlığı
                cv2.LINE_AA,                    # Anti-aliasing (pürüzsüz kenarlar)
            )

    def _draw_info_panel(
        self, frame: np.ndarray, count: int, fps: float
    ) -> None:
        """
        Sol üst köşeye yarı saydam bilgi paneli çizer.

        Panel içeriği:
            Satır 1: "FPS: 23.5" — Gerçek zamanlı işlem hızı
            Satır 2: "Tespit: 4" — Mevcut karedeki toplam tespit sayısı

        Yarı saydamlık tekniği:
            1. Frame kopyası oluştur
            2. Kopya üzerine opak dikdörtgen çiz
            3. Orijinal ile kopyayı alpha-blending ile birleştir
            Bu yöntem altındaki görüntünün görünmesini sağlar.

        Args:
            frame: Üzerine çizim yapılacak frame.
            count: Tespit sayısı.
            fps: Anlık FPS değeri.
        """
        panel_height = 60   # Panel yüksekliği (piksel)
        panel_width = 200    # Panel genişliği (piksel)

        # Alpha blending için frame kopyası oluştur
        overlay = frame.copy()

        # Kopya üzerine siyah/opak dikdörtgen çiz
        cv2.rectangle(
            overlay,
            (0, 0),                              # Sol üst köşe
            (panel_width, panel_height),          # Sağ alt köşe
            self._config.info_panel_color,        # Panel arka plan rengi
            cv2.FILLED,                           # İçi dolu
        )

        # Alpha blending: overlay * alpha + frame * (1 - alpha)
        # Bu işlem yarı saydam efekt yaratır
        cv2.addWeighted(
            overlay,
            self._config.info_panel_alpha,        # Panel saydamlığı (0.6)
            frame,
            1 - self._config.info_panel_alpha,    # Arka plan ağırlığı (0.4)
            0,                                     # Ek parlaklık (gamma)
            frame,                                 # Sonuç → orijinal frame'e yaz
        )

        # --- FPS Metni ---
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 22),                     # Metin konumu
            self._FONT,
            0.55,                         # Yazı boyutu
            self._config.font_color,      # Yazı rengi (yeşil)
            1,                            # Yazı kalınlığı
            cv2.LINE_AA,
        )

        # --- Tespit Sayısı Metni ---
        cv2.putText(
            frame,
            f"Tespit: {count}",
            (10, 48),                     # İkinci satır konumu
            self._FONT,
            0.55,
            self._config.font_color,
            1,
            cv2.LINE_AA,
        )

    def setup_writer(
        self, output_path: str, fps: float, frame_size: tuple[int, int]
    ) -> None:
        """
        Video yazıcıyı (VideoWriter) başlatır.

        XVID codec'i ile AVI formatında çıktı üretir.
        draw() metodu her çağrıldığında otomatik olarak yazılır.

        Args:
            output_path: Çıktı dosya yolu (örn: "output/result.avi").
            fps: Video FPS değeri.
            frame_size: (genişlik, yükseklik) tuple'ı.
        """
        # XVID codec — AVI container ile uyumlu, yaygın desteklenen codec
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        logger.info("Video yazıcı başlatıldı: %s", output_path)

    def release_writer(self) -> None:
        """
        Video yazıcıyı kapatır ve kaynakları serbest bırakır.
        Pipeline sonunda çağrılmalıdır — aksi halde video bozulabilir.
        """
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Video yazıcı kapatıldı.")
