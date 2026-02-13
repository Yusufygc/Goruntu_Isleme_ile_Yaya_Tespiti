"""
HOG + SVM tabanlı yaya tespit motoru.
OpenCV'nin yerleşik HOGDescriptor ve DefaultPeopleDetector kullanır.
"""

from typing import List

import cv2
import numpy as np

from config.settings import DetectionConfig
from core.detection.base_detector import BaseDetector, Detection
from utils.logger import get_logger

logger = get_logger(__name__)


class HOGDetector(BaseDetector):
    """HOG (Histogram of Oriented Gradients) + SVM yaya dedektörü."""

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._hog: cv2.HOGDescriptor = None

    def initialize(self) -> None:
        """HOG descriptor ve SVM dedektörünü yükler."""
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        logger.info(
            "HOG Dedektör başlatıldı | winStride: %s | scale: %.2f",
            self._config.win_stride,
            self._config.scale,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        HOG + SVM ile yaya tespiti yapar.

        Args:
            frame: BGR veya gri tonlama girdi frame.

        Returns:
            Tespit edilen yayaların listesi.
        """
        if self._hog is None:
            raise RuntimeError("Dedektör başlatılmadı. Önce initialize() çağırın.")

        regions, weights = self._hog.detectMultiScale(
            frame,
            winStride=self._config.win_stride,
            padding=self._config.padding,
            scale=self._config.scale,
            hitThreshold=self._config.hit_threshold,
        )

        detections: List[Detection] = []

        for (x, y, w, h), weight in zip(regions, weights):
            confidence = float(weight)

            # Minimum boyut filtresi
            if w < self._config.min_detection_size[0]:
                continue
            if h < self._config.min_detection_size[1]:
                continue

            detections.append(Detection(
                x=int(x), y=int(y), w=int(w), h=int(h),
                confidence=confidence,
            ))

        return detections
