"""
Tespit raporu oluşturma modülü.
Video işleme sonunda detaylı JSON raporu üretir.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FrameStats:
    """Tek bir karenin tespit istatistikleri."""

    frame_number: int
    detection_count: int
    confidences: List[float]
    fps: float


@dataclass
class DetectionReport:
    """Video işleme sonuç raporu."""

    # Video bilgileri
    video_source: str = ""
    video_resolution: str = ""
    video_fps: float = 0.0
    total_video_frames: int = 0

    # İşleme istatistikleri
    total_processed_frames: int = 0
    frames_with_detections: int = 0
    frames_without_detections: int = 0
    total_detections: int = 0

    # Güven istatistikleri
    avg_confidence: float = 0.0
    min_confidence: float = 0.0
    max_confidence: float = 0.0

    # Performans
    avg_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0
    total_processing_time_sec: float = 0.0

    # Kare detayları (en yoğun kareler)
    top_detection_frames: List[Dict] = field(default_factory=list)

    # Konfigürasyon özeti
    config_summary: Dict = field(default_factory=dict)


class ReportGenerator:
    """
    Pipeline çalışması sırasında istatistik toplar,
    sonunda detaylı JSON raporu üretir.
    """

    def __init__(self, output_dir: str = "output") -> None:
        self._output_dir = output_dir
        self._frame_stats: List[FrameStats] = []
        self._all_confidences: List[float] = []
        self._all_fps: List[float] = []
        self._start_time: float = 0.0

        os.makedirs(output_dir, exist_ok=True)
        logger.info("ReportGenerator başlatıldı | Dizin: %s", output_dir)

    def start(self) -> None:
        """İşleme başlangıç zamanını kaydeder."""
        self._start_time = time.time()

    def record_frame(
        self,
        frame_number: int,
        detection_count: int,
        confidences: List[float],
        fps: float,
    ) -> None:
        """
        Bir karenin istatistiklerini kaydeder.

        Args:
            frame_number: Kare numarası.
            detection_count: Tespit sayısı.
            confidences: Güven skorları listesi.
            fps: Anlık FPS değeri.
        """
        stats = FrameStats(
            frame_number=frame_number,
            detection_count=detection_count,
            confidences=confidences,
            fps=fps,
        )
        self._frame_stats.append(stats)
        self._all_confidences.extend(confidences)

        if fps > 0:
            self._all_fps.append(fps)

    def generate(
        self,
        video_source: str = "",
        video_resolution: str = "",
        video_fps: float = 0.0,
        total_video_frames: int = 0,
        config_summary: Dict = None,
    ) -> str:
        """
        Raporu oluşturur ve JSON dosyasına yazar.

        Returns:
            Rapor dosyasının yolu.
        """
        elapsed = time.time() - self._start_time if self._start_time else 0

        frames_with = sum(1 for s in self._frame_stats if s.detection_count > 0)
        total_det = sum(s.detection_count for s in self._frame_stats)

        report = DetectionReport(
            video_source=video_source,
            video_resolution=video_resolution,
            video_fps=video_fps,
            total_video_frames=total_video_frames,
            total_processed_frames=len(self._frame_stats),
            frames_with_detections=frames_with,
            frames_without_detections=len(self._frame_stats) - frames_with,
            total_detections=total_det,
            avg_confidence=self._safe_avg(self._all_confidences),
            min_confidence=min(self._all_confidences) if self._all_confidences else 0,
            max_confidence=max(self._all_confidences) if self._all_confidences else 0,
            avg_fps=self._safe_avg(self._all_fps),
            min_fps=min(self._all_fps) if self._all_fps else 0,
            max_fps=max(self._all_fps) if self._all_fps else 0,
            total_processing_time_sec=round(elapsed, 2),
            top_detection_frames=self._get_top_frames(5),
            config_summary=config_summary or {},
        )

        output_path = os.path.join(self._output_dir, "detection_report.json")
        report_dict = asdict(report)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info("Rapor oluşturuldu: %s", output_path)
        self._log_summary(report)

        return output_path

    def _get_top_frames(self, n: int) -> List[Dict]:
        """En çok tespit içeren N kareyi döndürür."""
        sorted_stats = sorted(
            self._frame_stats,
            key=lambda s: s.detection_count,
            reverse=True,
        )
        return [asdict(s) for s in sorted_stats[:n]]

    def _log_summary(self, report: DetectionReport) -> None:
        """Rapor özetini log'a yazar."""
        logger.info("=" * 50)
        logger.info("TESPIT RAPORU ÖZETİ")
        logger.info("=" * 50)
        logger.info("Toplam kare: %d", report.total_processed_frames)
        logger.info("Tespitli kare: %d (%%%.1f)",
                     report.frames_with_detections,
                     (report.frames_with_detections / max(1, report.total_processed_frames)) * 100)
        logger.info("Toplam tespit: %d", report.total_detections)
        logger.info("Ortalama güven: %.2f", report.avg_confidence)
        logger.info("Ortalama FPS: %.1f", report.avg_fps)
        logger.info("Toplam süre: %.1f sn", report.total_processing_time_sec)
        logger.info("=" * 50)

    @staticmethod
    def _safe_avg(values: list) -> float:
        """Güvenli ortalama hesaplama."""
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)
