"""
Video kaynağı modülleri.
Strategy Pattern ile farklı kaynak tipleri desteklenir.
"""

from core.source.base_source import VideoSource
from core.source.file_source import FileVideoSource
from core.source.camera_source import CameraSource
from core.source.source_factory import SourceFactory

__all__ = ["VideoSource", "FileVideoSource", "CameraSource", "SourceFactory"]
