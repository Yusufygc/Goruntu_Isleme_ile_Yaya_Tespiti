"""
Görüntü ön-işleme modülü.
Tespit öncesi frame hazırlığını yapar.

İşlem sırası: resize → denoise → CLAHE → sharpen
Her adım konfigürasyondan bağımsız olarak etkinleştirilebilir.
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

        # CLAHE nesnesini bir kez oluştur (bellek verimliliği)
        self._clahe: cv2.CLAHE = None
        if config.enable_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,
                tileGridSize=config.clahe_grid_size,
            )

        # Keskinleştirme çekirdeğini önceden hesapla
        self._sharpen_kernel: np.ndarray = None
        if config.enable_sharpening:
            self._sharpen_kernel = self._build_sharpen_kernel(
                config.sharpen_strength
            )

        logger.info(
            "Preprocessor başlatıldı | Genişlik: %d | CLAHE: %s | "
            "Keskinleştirme: %s | Gürültü azaltma: %s",
            config.target_width,
            config.enable_clahe,
            config.enable_sharpening,
            config.enable_denoising,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame üzerinde ön-işleme uygular.
        Sıra: resize → denoise → CLAHE → sharpen

        Args:
            frame: BGR formatında girdi frame.

        Returns:
            İşlenmiş frame.
        """
        processed = self._resize(frame)

        if self._config.enable_denoising:
            processed = self._denoise(processed)

        if self._config.enable_clahe:
            processed = self._apply_clahe(processed)

        if self._config.enable_sharpening:
            processed = self._sharpen(processed)

        if self._config.convert_to_gray:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        return processed

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """En-boy oranını koruyarak yeniden boyutlandırır."""
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

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Hafif Gaussian blur ile gürültü azaltma.
        Kernel boyutu küçük tutulur — kenar bilgisini korumak için.
        """
        k = self._config.denoise_strength
        # Kernel boyutu tek sayı olmalı
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(frame, (k, k), 0)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization).
        LAB renk uzayında L kanalına uygular — renkleri bozmaz.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        l_channel = self._clahe.apply(l_channel)

        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        """
        Unsharp Mask tabanlı keskinleştirme.
        Bulanık karelerde kenar bilgisini güçlendirir (HOG kenar tabanlı).
        """
        return cv2.filter2D(frame, -1, self._sharpen_kernel)

    @staticmethod
    def _build_sharpen_kernel(strength: float) -> np.ndarray:
        """
        Ayarlanabilir keskinleştirme çekirdeği oluşturur.

        strength=0 → orijinal, strength=1 → maksimum keskinleştirme.
        """
        base = np.array(
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]], dtype=np.float32
        )
        identity = np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]], dtype=np.float32
        )
        # Orijinal ile keskin arası interpolasyon
        return identity * (1 - strength) + base * strength

    @property
    def scale_factor(self) -> float:
        """Son işlemdeki ölçekleme faktörünü döndürür."""
        return self._scale_factor
