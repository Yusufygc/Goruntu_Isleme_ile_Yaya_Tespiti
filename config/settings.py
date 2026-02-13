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
    scale: float = 1.03
    hit_threshold: float = 0.3
    confidence_threshold: float = 0.3
    nms_threshold: float = 0.4
    min_detection_size: Tuple[int, int] = (40, 80)
    max_detection_size: Tuple[int, int] = (300, 500)

    # En-boy oranı filtresi (yayalar dikeydir)
    min_aspect_ratio: float = 1.3
    max_aspect_ratio: float = 3.5

    # Multi-pass tespit (kalabalık ortam desteği)
    enable_multi_pass: bool = False
    second_pass_win_stride: Tuple[int, int] = (4, 4)
    second_pass_scale: float = 1.02
    second_pass_padding: Tuple[int, int] = (16, 16)
    second_pass_hit_threshold: float = 0.3


@dataclass(frozen=True)
class PreprocessConfig:
    """Ön-işleme parametreleri."""

    target_width: int = 640
    convert_to_gray: bool = False

    # CLAHE (uyarlamalı histogram eşitleme)
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_grid_size: Tuple[int, int] = (8, 8)

    # Keskinleştirme (bulanık videolar için)
    enable_sharpening: bool = True
    sharpen_strength: float = 0.5

    # Gürültü azaltma
    enable_denoising: bool = True
    denoise_strength: int = 3


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
