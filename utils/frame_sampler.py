"""
Frame Örnekleme Modülü
========================
Tespit içeren kareleri belirli aralıklarla diske kaydeder.
Bu kayıtlar, tespit kalitesini görsel olarak analiz etmek
ve doğrulamak için kullanılır.

Kullanım Senaryoları:
    - False positive analizi (yanlış tespitleri görsel olarak bulmak)
    - Parametric tuning (ayar değişikliklerinin etkisini görmek)
    - Raporlama (müşteriye sonuç göstermek)

Kaydetme Koşulları:
    1. Karede en az 1 tespit varsa VE
    2. Aşağıdaki koşullardan biri sağlanıyorsa:
       a. Yüksek güvenli tespit var (eşik üstü) → hemen kaydet
       b. Belirli aralıkta (her N tespitli karede bir) → periyodik kaydet

Dosya Adı Formatı:
    frame_000020_2det.jpg
    ^      ^       ^
    |      |       └── Tespit sayısı
    |      └── Kare numarası (6 haneli, sıfır dolgulu)
    └── Sabit önek
"""

import os
from typing import List

import cv2
import numpy as np

from core.detection.base_detector import Detection
from utils.logger import get_logger

logger = get_logger(__name__)


class FrameSampler:
    """
    Tespit içeren kareleri diske kaydeden örnekleyici.

    İki kaydetme modu:
        1. Güven tabanlı: Yüksek güvenli tespit varsa hemen kaydet
        2. Periyodik: Her N tespitli karede bir kaydet

    Opsiyonel olarak orijinal (çizimsiz) kare de kaydedilebilir
    (eğitim verisi oluşturmak veya yeniden analiz için).
    """

    def __init__(
        self,
        output_dir: str = "output/samples",
        sample_interval: int = 10,
        min_confidence_to_save: float = 0.0,
        save_raw: bool = False,
    ) -> None:
        """
        FrameSampler'ı başlatır ve çıktı dizinlerini oluşturur.

        Args:
            output_dir: Kaydedilen karelerin dizini.
            sample_interval: Kaç tespitli karede bir kaydedeceği.
                Örn: 10 → her 10. tespitli kare kaydedilir.
            min_confidence_to_save: Bu değerin üstünde güven skoru varsa
                interval'e bakmadan hemen kaydeder.
                0.0 → bu özellik devre dışı.
            save_raw: True ise çizimsiz orijinal frame de kaydedilir
                (output_dir/raw/ altına).
        """
        self._output_dir = output_dir
        # sample_interval en az 1 olmalı (sıfıra bölme koruması)
        self._sample_interval = max(1, sample_interval)
        self._min_confidence = min_confidence_to_save
        self._save_raw = save_raw

        # İstatistik sayaçları
        self._frames_with_detections = 0  # Tespit içeren toplam kare sayısı
        self._total_saved = 0             # Diske kaydedilen toplam kare sayısı

        # Çıktı dizinlerini oluştur (yoksa)
        os.makedirs(output_dir, exist_ok=True)

        # Raw (çizimsiz) kare dizini
        if save_raw:
            os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)

        logger.info(
            "FrameSampler başlatıldı | Dizin: %s | Aralık: %d | "
            "Raw kayıt: %s",
            output_dir,
            sample_interval,
            save_raw,
        )

    def process(
        self,
        frame_number: int,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        detections: List[Detection],
    ) -> None:
        """
        Kareyi değerlendirip gerekirse kaydeder.

        Karar mantığı:
            1. Tespit yoksa → hiçbir şey yapma
            2. Yüksek güvenli tespit varsa → hemen kaydet
            3. Periyodik interval'e ulaşıldıysa → kaydet
            4. Hiçbiri değilse → atla

        Args:
            frame_number: Kare numarası (1'den başlar).
            raw_frame: Orijinal (çizimsiz) frame.
            annotated_frame: Bounding box çizilmiş frame.
            detections: Bu karedeki tespit listesi.
        """
        # Tespit yoksa kaydetme — veri israfını önle
        if not detections:
            return

        # Tespitli kare sayacını artır
        self._frames_with_detections += 1

        # --- Yüksek güven kontrolü ---
        # Karadeki en yüksek güven skorunu bul
        max_conf = max(d.confidence for d in detections)
        # min_confidence > 0 kontrolü → 0.0 ise bu özellik devre dışı
        high_confidence = max_conf >= self._min_confidence > 0

        # --- Kaydetme kararı ---
        # Ya yüksek güvenli tespit var → hemen kaydet
        # Ya da periyodik interval'e ulaşıldı → periyodik kaydet
        should_save = (
            high_confidence
            or self._frames_with_detections % self._sample_interval == 0
        )

        if should_save:
            self._save_frame(frame_number, raw_frame, annotated_frame, detections)

    def _save_frame(
        self,
        frame_number: int,
        raw_frame: np.ndarray,
        annotated_frame: np.ndarray,
        detections: List[Detection],
    ) -> None:
        """
        Kareyi JPEG olarak diske yazar.

        Dosya adı formatı: frame_000020_2det.jpg
        JPEG kalitesi: 90 (dosya boyutu / kalite dengesi)

        Args:
            frame_number: Kare numarası.
            raw_frame: Orijinal frame.
            annotated_frame: Çizimli frame.
            detections: Tespit listesi (dosya adı için tespit sayısı).
        """
        # Dosya adı oluştur — kare no ve tespit sayısı bilgilendirici
        filename = f"frame_{frame_number:06d}_{len(detections)}det.jpg"

        # --- Çizimli kareyi kaydet ---
        annotated_path = os.path.join(self._output_dir, filename)
        cv2.imwrite(annotated_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # --- Orijinal (çizimsiz) kareyi kaydet (opsiyonel) ---
        if self._save_raw:
            raw_path = os.path.join(self._output_dir, "raw", filename)
            cv2.imwrite(raw_path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        self._total_saved += 1

    @property
    def total_saved(self) -> int:
        """Toplam kaydedilen kare sayısını döndürür."""
        return self._total_saved

    @property
    def frames_with_detections(self) -> int:
        """Tespit içeren toplam kare sayısını döndürür."""
        return self._frames_with_detections
