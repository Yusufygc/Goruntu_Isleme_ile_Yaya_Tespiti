"""
Görselleştirme modülü.
Bounding box, güven skoru, FPS ve bilgi paneli çizimini yönetir.
"""

from typing import List, Optional

import cv2
import numpy as np

from config.settings import VisualizationConfig
from core.detection.base_detector import Detection
from utils.logger import get_logger

logger = get_logger(__name__)


class Visualizer:
    """Frame üzerine tespit sonuçlarını ve bilgi panelini çizer."""

    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, config: VisualizationConfig) -> None:
        self._config = config
        self._writer: Optional[cv2.VideoWriter] = None
        logger.info("Visualizer başlatıldı | Çıktı kaydetme: %s", config.save_output)

    def draw(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Frame üzerine tespit sonuçlarını çizer.

        Args:
            frame: Orijinal BGR frame.
            detections: Tespit listesi.
            fps: Mevcut FPS değeri.

        Returns:
            Çizim yapılmış frame (kopyası).
        """
        output = frame.copy()

        for detection in detections:
            self._draw_bounding_box(output, detection)

        if self._config.show_info_panel:
            self._draw_info_panel(output, len(detections), fps)

        if self._writer is not None:
            self._writer.write(output)

        return output

    def _draw_bounding_box(self, frame: np.ndarray, det: Detection) -> None:
        """Tek bir tespit kutusunu çizer."""
        top_left = (det.x, det.y)
        bottom_right = (det.x + det.w, det.y + det.h)

        cv2.rectangle(
            frame,
            top_left,
            bottom_right,
            self._config.box_color,
            self._config.box_thickness,
        )

        if self._config.show_confidence:
            label = f"Yaya {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(
                label, self._FONT, self._config.font_scale, 1
            )

            # Etiket arka planı
            cv2.rectangle(
                frame,
                (det.x, det.y - label_size[1] - 8),
                (det.x + label_size[0] + 4, det.y),
                self._config.box_color,
                cv2.FILLED,
            )

            cv2.putText(
                frame,
                label,
                (det.x + 2, det.y - 4),
                self._FONT,
                self._config.font_scale,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_info_panel(
        self, frame: np.ndarray, count: int, fps: float
    ) -> None:
        """Sol üst köşeye bilgi paneli çizer."""
        panel_height = 60
        panel_width = 200
        overlay = frame.copy()

        cv2.rectangle(
            overlay,
            (0, 0),
            (panel_width, panel_height),
            self._config.info_panel_color,
            cv2.FILLED,
        )

        cv2.addWeighted(
            overlay,
            self._config.info_panel_alpha,
            frame,
            1 - self._config.info_panel_alpha,
            0,
            frame,
        )

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 22),
            self._FONT,
            0.55,
            self._config.font_color,
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Tespit: {count}",
            (10, 48),
            self._FONT,
            0.55,
            self._config.font_color,
            1,
            cv2.LINE_AA,
        )

    def setup_writer(
        self, output_path: str, fps: float, frame_size: tuple[int, int]
    ) -> None:
        """Video yazıcıyı başlatır."""
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        logger.info("Video yazıcı başlatıldı: %s", output_path)

    def release_writer(self) -> None:
        """Video yazıcıyı kapatır."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Video yazıcı kapatıldı.")
