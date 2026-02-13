"""
Dosya Tabanlı Video Kaynağı
=============================
Disk üzerindeki video dosyalarından kare kare okuma yapar.
OpenCV'nin VideoCapture sınıfını kullanır.

Desteklenen formatlar:
    MP4, AVI, MKV, MOV ve OpenCV'nin desteklediği tüm formatlar.

Örnek Kullanım:
    source = FileVideoSource("input/video.mp4")
    with source:
        while True:
            frame = source.read_frame()
            if frame is None:
                break  # Video sonu
            # frame ile işlem yap...
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.source.base_source import VideoSource
from utils.logger import get_logger

logger = get_logger(__name__)


class FileVideoSource(VideoSource):
    """
    Video dosyasından kare okuyan kaynak.

    Dosya yolu constructor'da doğrulanır — dosya yoksa
    hemen FileNotFoundError fırlatılır (fail-fast prensibi).
    """

    def __init__(self, file_path: str) -> None:
        """
        Video kaynağını dosya yolu ile oluşturur.

        Args:
            file_path: Video dosyasının yolu (göreceli veya mutlak).

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
        """
        # Path nesnesi — platform bağımsız dosya yolu yönetimi
        self._file_path = Path(file_path)

        # Dosya varlık kontrolü — erken başarısızlık (fail-fast)
        if not self._file_path.exists():
            raise FileNotFoundError(f"Video dosyası bulunamadı: {file_path}")

        # OpenCV VideoCapture nesnesi — open() ile başlatılır
        self._capture: Optional[cv2.VideoCapture] = None

        # Video meta verileri — open() sonrası doldurulur
        self._fps_value: float = 0.0       # Saniyedeki kare sayısı
        self._width: int = 0                # Çözünürlük genişliği (piksel)
        self._height: int = 0               # Çözünürlük yüksekliği (piksel)
        self._total_frames: int = 0         # Toplam kare sayısı

    def open(self) -> None:
        """
        Video dosyasını açar ve meta verileri okur.

        OpenCV'nin CAP_PROP_* sabitleri ile video bilgilerini alır:
            - FPS: Oynatma hızı
            - FRAME_WIDTH/HEIGHT: Çözünürlük
            - FRAME_COUNT: Toplam kare sayısı (ilerleme hesabı için)

        Raises:
            IOError: Dosya açılamazsa (bozuk dosya, desteklenmeyen codec vb.).
        """
        # VideoCapture ile dosyayı aç
        self._capture = cv2.VideoCapture(str(self._file_path))

        # Açılma kontrolü — codec hatası, izin sorunu vb. olabilir
        if not self._capture.isOpened():
            raise IOError(f"Video dosyası açılamadı: {self._file_path}")

        # Meta verileri oku
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
        """
        Bir sonraki kareyi okur.

        Returns:
            BGR numpy array veya video bittiyse/hata olduysa None.
        """
        # Kaynak açık değilse okuma yapma
        if self._capture is None or not self._capture.isOpened():
            return None

        # ret: Okuma başarılı mı? frame: BGR numpy array
        ret, frame = self._capture.read()
        return frame if ret else None  # Video sonunda ret=False olur

    def release(self) -> None:
        """
        VideoCapture nesnesini serbest bırakır.
        Dosya tanıtıcısı (file handle) serbest bırakılır.
        """
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Video kaynağı kapatıldı: %s", self._file_path.name)

    def is_opened(self) -> bool:
        """VideoCapture'ın açık olup olmadığını döndürür."""
        return self._capture is not None and self._capture.isOpened()

    @property
    def fps(self) -> float:
        """Video dosyasının FPS değerini döndürür."""
        return self._fps_value

    @property
    def frame_size(self) -> tuple[int, int]:
        """(genişlik, yükseklik) tuple'ı olarak çözünürlüğü döndürür."""
        return (self._width, self._height)

    @property
    def total_frames(self) -> int:
        """
        Toplam kare sayısını döndürür.
        İlerleme çubuğu veya rapor oluşturma için kullanılır.
        """
        return self._total_frames
