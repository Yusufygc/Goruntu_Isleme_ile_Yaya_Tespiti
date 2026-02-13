"""
Kamera Tabanlı Video Kaynağı
==============================
Web kamerası veya USB kameradan canlı (real-time) görüntü akışı sağlar.
OpenCV'nin VideoCapture sınıfını kamera modunda kullanır.

Kullanım:
    source = CameraSource(camera_index=0)  # Varsayılan kamera
    with source:
        frame = source.read_frame()
        # frame ile tespit yap...
"""

from typing import Optional

import cv2
import numpy as np

from core.source.base_source import VideoSource
from utils.logger import get_logger

logger = get_logger(__name__)


class CameraSource(VideoSource):
    """
    Kameradan canlı kare okuyan kaynak.

    NOT: Kamera FPS değeri her zaman doğru raporlanmaz —
    varsayılan 30.0 kullanılır ve gerçek FPS, FPSCounter ile ölçülür.
    """

    def __init__(self, camera_index: int = 0) -> None:
        """
        Args:
            camera_index: Kamera cihaz indeksi.
                0 = birincil kamera (dahili webcam),
                1 = ikincil kamera (harici USB kamera), vb.
        """
        self._camera_index = camera_index
        self._capture: Optional[cv2.VideoCapture] = None

        # Kamera FPS değeri — donanımdan okunamazsa 30.0 varsayılır
        self._fps_value: float = 30.0
        self._width: int = 0
        self._height: int = 0

    def open(self) -> None:
        """
        Kamerayı açar ve çözünürlük bilgilerini okur.

        Raises:
            IOError: Kamera açılamazsa (bağlı değil, başka uygulama kullanıyor vb.).
        """
        # camera_index ile kamera cihazını aç
        self._capture = cv2.VideoCapture(self._camera_index)

        if not self._capture.isOpened():
            raise IOError(
                f"Kamera açılamadı (index: {self._camera_index}). "
                "Kameranın bağlı ve kullanılabilir olduğundan emin olun."
            )

        # FPS değerini oku — bazı kameralar 0 döndürür, bu durumda 30.0 kullan
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
        """
        Kameradan bir kare okur.

        Returns:
            BGR numpy array veya okuma başarısızsa None.
        """
        if self._capture is None or not self._capture.isOpened():
            return None

        ret, frame = self._capture.read()
        return frame if ret else None

    def release(self) -> None:
        """Kamerayı serbest bırakır ve kaynakları temizler."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Kamera kapatıldı (index: %d)", self._camera_index)

    def is_opened(self) -> bool:
        """Kameranın açık olup olmadığını döndürür."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def fps(self) -> float:
        """Kamera FPS değerini döndürür (donanım raporu veya 30.0 varsayılan)."""
        return self._fps_value

    @property
    def frame_size(self) -> tuple[int, int]:
        """(genişlik, yükseklik) tuple'ı olarak kamera çözünürlüğünü döndürür."""
        return (self._width, self._height)
