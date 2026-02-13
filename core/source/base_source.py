"""
Soyut video kaynağı arayüzü.
Tüm video kaynakları bu sınıftan türetilir (Strategy Pattern).
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class VideoSource(ABC):
    """Video kaynağı için soyut temel sınıf."""

    @abstractmethod
    def open(self) -> None:
        """Kaynağı açar."""

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Bir sonraki kareyi okur.

        Returns:
            Frame (BGR numpy array) veya kaynak bittiyse None.
        """

    @abstractmethod
    def release(self) -> None:
        """Kaynağı serbest bırakır."""

    @abstractmethod
    def is_opened(self) -> bool:
        """Kaynağın açık olup olmadığını döndürür."""

    @property
    @abstractmethod
    def fps(self) -> float:
        """Kaynak FPS değerini döndürür."""

    @property
    @abstractmethod
    def frame_size(self) -> tuple[int, int]:
        """(genişlik, yükseklik) tuple döndürür."""

    def __enter__(self) -> "VideoSource":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
