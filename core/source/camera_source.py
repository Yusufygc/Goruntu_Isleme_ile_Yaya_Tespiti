"""
Kamera tabanlı video kaynağı.
Real-time tespit için hazır iskelet.
"""

from typing import Optional

import cv2
import numpy as np

from core.source.base_source import VideoSource
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraSource(VideoSource):
    """Kameradan canlı kare okuyan kaynak."""

    def __init__(self, camera_index: int = 0) -> None:
        """
        Args:
            camera_index: Kamera cihaz indeksi. Varsayılan 0 (birincil kamera).
        """
        self._camera_index = camera_index
        self._capture: Optional[cv2.VideoCapture] = None
        self._fps_value: float = 30.0
        self._width: int = 0
        self._height: int = 0

    def open(self) -> None:
        """Kamerayı açar."""
        self._capture = cv2.VideoCapture(self._camera_index)

        if not self._capture.isOpened():
            raise IOError(
                f"Kamera açılamadı (index: {self._camera_index}). "
                "Kameranın bağlı ve kullanılabilir olduğundan emin olun."
            )

        self._fps_value = self._capture.get(cv2.CAP_PROP_FPS) or 30.0
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            "Kamera açıldı (index: %d) | %dx%d | %.1f FPS",
            self._camera_index,
            self._width,
            self._height,
            self._fps_value,
        )

    def read_frame(self) -> Optional[np.ndarray]:
        """Kameradan bir kare okur."""
        if self._capture is None or not self._capture.isOpened():
            return None

        ret, frame = self._capture.read()
        return frame if ret else None

    def release(self) -> None:
        """Kamerayı serbest bırakır."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Kamera kapatıldı (index: %d)", self._camera_index)

    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def fps(self) -> float:
        return self._fps_value

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)
