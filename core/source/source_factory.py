"""
Video kaynağı fabrikası.
Factory Pattern ile kaynak tipine göre doğru nesneyi üretir.
"""

from enum import Enum, auto
from typing import Optional

from core.source.base_source import VideoSource
from core.source.file_source import FileVideoSource
from core.source.camera_source import CameraSource


class SourceType(Enum):
    """Desteklenen kaynak tipleri."""

    FILE = auto()
    CAMERA = auto()


class SourceFactory:
    """Kaynak tipine göre VideoSource nesnesi üreten fabrika."""

    @staticmethod
    def create(
        source_type: SourceType,
        path: Optional[str] = None,
        camera_index: int = 0,
    ) -> VideoSource:
        """
        Kaynak tipine göre uygun VideoSource nesnesi üretir.

        Args:
            source_type: Kaynak tipi (FILE veya CAMERA).
            path: Video dosya yolu (FILE tipi için zorunlu).
            camera_index: Kamera indeksi (CAMERA tipi için).

        Returns:
            VideoSource implementasyonu.

        Raises:
            ValueError: Geçersiz kaynak tipi veya eksik parametre.
        """
        if source_type == SourceType.FILE:
            if path is None:
                raise ValueError("FILE kaynağı için 'path' parametresi zorunludur.")
            return FileVideoSource(path)

        if source_type == SourceType.CAMERA:
            return CameraSource(camera_index)

        raise ValueError(f"Desteklenmeyen kaynak tipi: {source_type}")
