"""
Frame örnekleme modülü.
Tespit içeren kareleri belirli aralıklarla diske kaydeder.
Analiz ve doğrulama amacıyla kullanılır.
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

    Kaydetme koşulları:
    - Karede en az 1 tespit varsa
    - Belirlenen kare aralığı (her N karede bir)
    - Veya güven skoru eşiği üstünde tespit varsa
    """

    def __init__(
        self,
        output_dir: str = "output/samples",
        sample_interval: int = 10,
        min_confidence_to_save: float = 0.0,
        save_raw: bool = False,
    ) -> None:
        """
        Args:
            output_dir: Kayıt dizini.
            sample_interval: Kaç karede bir kaydedeceği (tespit varsa).
            min_confidence_to_save: Bu değerin üstünde güven skoru varsa
                                    interval'e bakmadan kaydeder.
            save_raw: True ise orijinal (çizimsiz) frame de kaydedilir.
        """
        self._output_dir = output_dir
        self._sample_interval = max(1, sample_interval)
        self._min_confidence = min_confidence_to_save
        self._save_raw = save_raw

        self._frames_with_detections = 0
        self._total_saved = 0

        os.makedirs(output_dir, exist_ok=True)

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

        Args:
            frame_number: Kare numarası.
            raw_frame: Orijinal (çizimsiz) frame.
            annotated_frame: Görselleştirilmiş frame.
            detections: Tespit listesi.
        """
        if not detections:
            return

        self._frames_with_detections += 1

        # Yüksek güvenli tespit varsa hemen kaydet
        max_conf = max(d.confidence for d in detections)
        high_confidence = max_conf >= self._min_confidence > 0

        # Interval kontrolü veya yüksek güven
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
        """Kareyi diske yazar."""
        filename = f"frame_{frame_number:06d}_{len(detections)}det.jpg"

        # Çizimli kareyi kaydet
        annotated_path = os.path.join(self._output_dir, filename)
        cv2.imwrite(annotated_path, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Orijinal kareyi kaydet
        if self._save_raw:
            raw_path = os.path.join(self._output_dir, "raw", filename)
            cv2.imwrite(raw_path, raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

        self._total_saved += 1

    @property
    def total_saved(self) -> int:
        """Toplam kaydedilen kare sayısı."""
        return self._total_saved

    @property
    def frames_with_detections(self) -> int:
        """Tespit içeren toplam kare sayısı."""
        return self._frames_with_detections
