"""
Görüntü ön-işleme modülü.
Tespit öncesi frame hazırlığını yapar.
"""

import cv2
import numpy as np

from config.settings import PreprocessConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Frame ön-işleme işlemlerini yönetir."""

    def __init__(self, config: PreprocessConfig) -> None:
        self._config = config
        self._scale_factor: float = 1.0
        logger.info(
            "Preprocessor başlatıldı | Hedef genişlik: %d | Gri tonlama: %s",
            config.target_width,
            config.convert_to_gray,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame üzerinde ön-işleme uygular.

        Args:
            frame: BGR formatında girdi frame.

        Returns:
            İşlenmiş frame.
        """
        processed = self._resize(frame)

        if self._config.convert_to_gray:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        return processed

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """
        En-boy oranını koruyarak yeniden boyutlandırır.

        Orijinal boyut hedef genişlikten küçükse işlemi atlar.
        """
        height, width = frame.shape[:2]

        if width <= self._config.target_width:
            self._scale_factor = 1.0
            return frame

        self._scale_factor = self._config.target_width / width
        new_height = int(height * self._scale_factor)

        return cv2.resize(
            frame,
            (self._config.target_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    @property
    def scale_factor(self) -> float:
        """Son işlemdeki ölçekleme faktörünü döndürür."""
        return self._scale_factor
