"""
HOG + SVM tabanlı yaya tespit motoru.
OpenCV'nin yerleşik HOGDescriptor ve DefaultPeopleDetector kullanır.

Multi-pass tespit: Farklı parametrelerle birden fazla geçiş
yaparak kalabalık ortamda daha fazla yaya yakalar.
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
            "HOG Dedektör başlatıldı | winStride: %s | scale: %.2f | "
            "multi-pass: %s",
            self._config.win_stride,
            self._config.scale,
            self._config.enable_multi_pass,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        HOG + SVM ile yaya tespiti yapar.
        Multi-pass etkinse iki farklı parametre seti ile tarar.

        Args:
            frame: BGR veya gri tonlama girdi frame.

        Returns:
            Tespit edilen yayaların listesi.
        """
        if self._hog is None:
            raise RuntimeError("Dedektör başlatılmadı. Önce initialize() çağırın.")

        # Geçiş 1: Standart parametreler
        detections = self._single_pass(
            frame,
            win_stride=self._config.win_stride,
            padding=self._config.padding,
            scale=self._config.scale,
        )

        # Geçiş 2: Yoğun tarama (kalabalık ortam için)
        if self._config.enable_multi_pass:
            second_pass = self._single_pass(
                frame,
                win_stride=self._config.second_pass_win_stride,
                padding=self._config.second_pass_padding,
                scale=self._config.second_pass_scale,
                hit_threshold=self._config.second_pass_hit_threshold,
            )
            detections.extend(second_pass)

        return detections

    def _single_pass(
        self,
        frame: np.ndarray,
        win_stride: tuple,
        padding: tuple,
        scale: float,
        hit_threshold: float = 0.0,
    ) -> List[Detection]:
        """Tek geçişlik HOG tespiti yapar."""
        regions, weights = self._hog.detectMultiScale(
            frame,
            winStride=win_stride,
            padding=padding,
            scale=scale,
            hitThreshold=hit_threshold,
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
