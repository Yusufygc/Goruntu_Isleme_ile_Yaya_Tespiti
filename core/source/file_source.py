"""
Dosya tabanlı video kaynağı.
Stock video dosyalarından kare kare okuma yapar.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.source.base_source import VideoSource
from utils.logger import get_logger

logger = get_logger(__name__)


class FileVideoSource(VideoSource):
    """Video dosyasından kare okuyan kaynak."""

    def __init__(self, file_path: str) -> None:
        """
        Args:
            file_path: Video dosyasının yolu.

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
        """
        self._file_path = Path(file_path)
        if not self._file_path.exists():
            raise FileNotFoundError(f"Video dosyası bulunamadı: {file_path}")

        self._capture: Optional[cv2.VideoCapture] = None
        self._fps_value: float = 0.0
        self._width: int = 0
        self._height: int = 0
        self._total_frames: int = 0

    def open(self) -> None:
        """Video dosyasını açar ve meta verileri okur."""
        self._capture = cv2.VideoCapture(str(self._file_path))

        if not self._capture.isOpened():
            raise IOError(f"Video dosyası açılamadı: {self._file_path}")

        self._fps_value = self._capture.get(cv2.CAP_PROP_FPS)
        self._width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Video açıldı: %s | %dx%d | %.1f FPS | %d kare",
            self._file_path.name,
            self._width,
            self._height,
            self._fps_value,
            self._total_frames,
        )

    def read_frame(self) -> Optional[np.ndarray]:
        """Bir sonraki kareyi okur."""
        if self._capture is None or not self._capture.isOpened():
            return None

        ret, frame = self._capture.read()
        return frame if ret else None

    def release(self) -> None:
        """VideoCapture nesnesini serbest bırakır."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Video kaynağı kapatıldı: %s", self._file_path.name)

    def is_opened(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    @property
    def fps(self) -> float:
        return self._fps_value

    @property
    def frame_size(self) -> tuple[int, int]:
        return (self._width, self._height)

    @property
    def total_frames(self) -> int:
        """Toplam kare sayısını döndürür."""
        return self._total_frames
