"""
Merkezi konfigürasyon modülü.
Tüm ayarlar dataclass olarak tanımlanır.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class DetectionConfig:
    """HOG + SVM tespit parametreleri."""

    win_stride: Tuple[int, int] = (8, 8)
    padding: Tuple[int, int] = (8, 8)
    scale: float = 1.05
    hit_threshold: float = 0.0
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.4
    min_detection_size: Tuple[int, int] = (40, 80)


@dataclass(frozen=True)
class PreprocessConfig:
    """Ön-işleme parametreleri."""

    target_width: int = 640
    convert_to_gray: bool = False


@dataclass(frozen=True)
class VisualizationConfig:
    """Görselleştirme parametreleri."""

    box_color: Tuple[int, int, int] = (0, 255, 0)
    box_thickness: int = 2
    font_scale: float = 0.5
    font_color: Tuple[int, int, int] = (0, 255, 0)
    info_panel_color: Tuple[int, int, int] = (0, 0, 0)
    info_panel_alpha: float = 0.6
    show_confidence: bool = True
    show_info_panel: bool = True
    save_output: bool = False
    output_path: str = "output/result.avi"


@dataclass
class PipelineConfig:
    """Pipeline genel konfigürasyonu."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    display_window_name: str = "Yaya Tespit Sistemi"
    quit_key: str = "q"
