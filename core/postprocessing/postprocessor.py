"""
Son-işleme modülü.
Non-Maximum Suppression (NMS) ve güven eşiği filtreleme uygular.
"""

from typing import List

import cv2
import numpy as np

from config.settings import DetectionConfig
from core.detection.base_detector import Detection
from utils.logger import get_logger

logger = get_logger(__name__)


class Postprocessor:
    """Tespit sonuçlarını filtreler ve düzenler."""

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        logger.info(
            "Postprocessor başlatıldı | NMS eşiği: %.2f | Güven eşiği: %.2f",
            config.nms_threshold,
            config.confidence_threshold,
        )

    def process(self, detections: List[Detection]) -> List[Detection]:
        """
        Tespit sonuçlarına filtreleme ve NMS uygular.
        Sıra: güven eşiği → en-boy oranı → NMS

        Args:
            detections: Ham tespit listesi.

        Returns:
            Filtrelenmiş tespit listesi.
        """
        if not detections:
            return []

        # 1. Güven eşiği filtreleme
        filtered = [
            d for d in detections
            if d.confidence >= self._config.confidence_threshold
        ]

        # 2. Boyut ve en-boy oranı filtresi
        filtered = [
            d for d in filtered
            if self._is_valid_detection(d)
        ]

        if not filtered:
            return []

        # 3. NMS uygula
        return self._apply_nms(filtered)

    def _is_valid_detection(self, det: Detection) -> bool:
        """
        Tespitteki bounding box'ın yaya boyutuna ve oranına
        uygun olup olmadığını kontrol eder.
        """
        # Maksimum boyut kontrolü (dev kutular sahte tespit)
        max_w, max_h = self._config.max_detection_size
        if det.w > max_w or det.h > max_h:
            return False

        # En-boy oranı kontrolü (yayalar dikeydir)
        if det.w == 0:
            return False
        ratio = det.h / det.w
        return self._config.min_aspect_ratio <= ratio <= self._config.max_aspect_ratio

    def _is_valid_aspect_ratio(self, det: Detection) -> bool:
        """
        Tespitteki bounding box'ın yaya oranına uygun olup olmadığını kontrol eder.
        Yayalar genellikle dikey yapıdadır (yükseklik/genişlik > 1.2).
        """
        if det.w == 0:
            return False
        ratio = det.h / det.w
        return self._config.min_aspect_ratio <= ratio <= self._config.max_aspect_ratio

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """
        Non-Maximum Suppression uygular.
        Çakışan kutuları eleme ile tekrar tespitleri önler.
        """
        boxes = np.array(
            [[d.x, d.y, d.w, d.h] for d in detections], dtype=np.int32
        )
        confidences = np.array(
            [d.confidence for d in detections], dtype=np.float32
        )

        # OpenCV NMS (x, y, w, h formatında kutular bekler)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=confidences.tolist(),
            score_threshold=self._config.confidence_threshold,
            nms_threshold=self._config.nms_threshold,
        )

        if len(indices) == 0:
            return []

        # indices farklı OpenCV sürümlerinde farklı boyutta olabilir
        indices = np.array(indices).flatten()

        return [detections[i] for i in indices]
