"""
Soyut Video Kaynağı Arayüzü
=============================
Tüm video kaynakları bu sınıftan türetilir.
Strategy Pattern kullanılır — kaynak tipi çalışma zamanında seçilir.

Desteklenen türetilmiş sınıflar:
    - FileVideoSource  → Video dosyasından okuma
    - CameraSource      → Canlı kamera akışından okuma

Context Manager desteği sayesinde 'with' bloğu ile kullanılabilir:
    with FileVideoSource("video.mp4") as source:
        source.open()
        frame = source.read_frame()
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VideoSource(ABC):
    """
    Video kaynağı için soyut temel sınıf.

    Alt sınıflar şu metotları implement etmelidir:
        - open(): Kaynağı açar
        - read_frame(): Sonraki kareyi okur
        - release(): Kaynağı serbest bırakır
        - is_opened(): Durum kontrolü
        - fps (property): Saniyedeki kare sayısı
        - frame_size (property): Çözünürlük
    """

    @abstractmethod
    def open(self) -> None:
        """Kaynağı açar ve meta verileri yükler."""

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Bir sonraki kareyi okur.

        Returns:
            BGR numpy array veya kaynak bittiyse None.
        """

    @abstractmethod
    def release(self) -> None:
        """Kaynağı serbest bırakır ve kaynakları temizler."""

    @abstractmethod
    def is_opened(self) -> bool:
        """Kaynağın açık ve okumaya hazır olup olmadığını döndürür."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Kaynak FPS (Frames Per Second) değerini döndürür."""

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(genişlik, yükseklik) piksel olarak çözünürlük döndürür."""

    # --- Context Manager Desteği ---
    # 'with' bloğu ile otomatik açma/kapama sağlar
    def __enter__(self) -> "VideoSource":
        """Context manager girişi — kaynağı açar."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager çıkışı — kaynağı kapatır (hata olsa bile)."""
        self.release()
