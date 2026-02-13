"""
FPS (Frames Per Second) ölçüm sınıfı.
Kayan ortalama ile stabil FPS hesabı yapar.
"""

import time
from collections import deque


class FPSCounter:
    """Kayan ortalama tabanlı FPS sayacı."""

    def __init__(self, window_size: int = 30) -> None:
        """
        Args:
            window_size: Kayan ortalama pencere boyutu.
        """
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._fps: float = 0.0

    def tick(self) -> None:
        """Yeni bir kare işlendiğinde çağrılır."""
        now = time.perf_counter()
        self._timestamps.append(now)

        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._fps = (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        """Mevcut FPS değerini döndürür."""
        return self._fps

    def reset(self) -> None:
        """Sayacı sıfırlar."""
        self._timestamps.clear()
        self._fps = 0.0
