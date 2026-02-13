"""
FPS (Frames Per Second) Ölçüm Sınıfı
=======================================
Kayan ortalama (sliding window) yöntemiyle stabil FPS hesabı yapar.

Neden kayan ortalama?
    - Anlık FPS değeri her kare arasında çok değişir
    - 30 karelik pencere ile yumuşatılmış, tutarlı FPS gösterilir
    - Kullanıcı deneyimi: Sabit bir sayı görmek daha anlaşılır

Algoritma:
    1. Her tick() çağrısında zaman damgası kaydedilir
    2. Penceredeki ilk ve son zaman damgası arasındaki fark hesaplanır
    3. FPS = (kare_sayısı - 1) / geçen_süre

Kullanım:
    counter = FPSCounter(window_size=30)
    for frame in video:
        counter.tick()
        print(f"FPS: {counter.fps:.1f}")
"""

import time
from collections import deque


class FPSCounter:
    """
    Kayan ortalama tabanlı FPS sayacı.

    deque (double-ended queue) kullanılır — maxlen aşıldığında
    eski zaman damgaları otomatik silinir (bellek verimliliği).
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        Args:
            window_size: Kayan ortalama pencere boyutu (kare sayısı).
                Büyük değer → daha stabil ama gecikmeli FPS.
                Küçük değer → daha hızlı tepki ama dalgalı FPS.
                30 → ~1 saniyelik pencere (30 FPS video için).
        """
        # Zaman damgaları kuyruğu — maxlen ile otomatik boyut sınırlama
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._fps: float = 0.0

    def tick(self) -> None:
        """
        Yeni bir kare işlendiğinde çağrılır.

        time.perf_counter() kullanılır — en yüksek çözünürlüklü
        monoton zamanlayıcı (nanosaniye hassasiyeti).
        """
        now = time.perf_counter()
        self._timestamps.append(now)

        # En az 2 zaman damgası varsa FPS hesapla
        if len(self._timestamps) >= 2:
            # İlk ve son zaman damgası arasındaki fark
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                # FPS = (kare_sayısı - 1) / geçen_süre
                # -1 çünkü N zaman damgası arasında N-1 aralık var
                self._fps = (len(self._timestamps) - 1) / elapsed

    @property
    def fps(self) -> float:
        """Mevcut kayan ortalama FPS değerini döndürür."""
        return self._fps

    def reset(self) -> None:
        """
        Sayacı sıfırlar.
        Sahne değişikliği veya duraklatma sonrası kullanılabilir.
        """
        self._timestamps.clear()
        self._fps = 0.0
