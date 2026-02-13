"""
Video Kaynağı Fabrikası (Factory Pattern)
==========================================
Kaynak tipine göre doğru VideoSource nesnesini üretir.

Factory Pattern burada neden kullanılır?
    - Kullanıcı "file" veya "camera" seçimi yapar
    - Bu seçime göre farklı sınıf örneklenir
    - Fabrika bu kararı merkezileştirir — yeni kaynak tipleri
      (örn: RTSP stream, IP kamera) eklendiğinde sadece fabrika güncellenir

Kullanım:
    source = SourceFactory.create(SourceType.FILE, path="video.mp4")
    source = SourceFactory.create(SourceType.CAMERA, camera_index=0)
"""

from enum import Enum, auto
from typing import Optional

from core.source.base_source import VideoSource
from core.source.file_source import FileVideoSource
from core.source.camera_source import CameraSource


class SourceType(Enum):
    """
    Desteklenen kaynak tipleri.

    auto() ile otomatik değer atanır — sayısal değer önemsizdir,
    enum karşılaştırması kullanılır.
    """

    FILE = auto()      # Video dosyası (MP4, AVI vb.)
    CAMERA = auto()    # Canlı kamera akışı


class SourceFactory:
    """
    Kaynak tipine göre VideoSource nesnesi üreten fabrika.

    @staticmethod ile instance oluşturmadan çağrılabilir:
        SourceFactory.create(SourceType.FILE, path="video.mp4")
    """

    @staticmethod
    def create(
        source_type: SourceType,
        path: Optional[str] = None,
        camera_index: int = 0,
    ) -> VideoSource:
        """
        Kaynak tipine göre uygun VideoSource nesnesi üretir.

        Args:
            source_type: Kaynak tipi enum'u (FILE veya CAMERA).
            path: Video dosya yolu (FILE tipi için zorunlu).
            camera_index: Kamera cihaz indeksi (CAMERA tipi için, varsayılan: 0).

        Returns:
            VideoSource alt sınıfı (FileVideoSource veya CameraSource).

        Raises:
            ValueError: Geçersiz kaynak tipi veya FILE seçilip path verilmezse.
        """
        # --- Dosya Kaynağı ---
        if source_type == SourceType.FILE:
            if path is None:
                raise ValueError("FILE kaynağı için 'path' parametresi zorunludur.")
            return FileVideoSource(path)

        # --- Kamera Kaynağı ---
        if source_type == SourceType.CAMERA:
            return CameraSource(camera_index)

        # Bilinmeyen tip — gelecekte yeni tipler eklendiğinde buraya düşer
        raise ValueError(f"Desteklenmeyen kaynak tipi: {source_type}")
