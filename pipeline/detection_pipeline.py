"""
Tespit pipeline'ı.
Tüm işleme adımlarını orkestre eder:
  source → preprocess → detect → postprocess → visualize → sample → report
"""

import os
from dataclasses import asdict

import cv2

from config.settings import PipelineConfig
from core.source.base_source import VideoSource
from core.preprocessing.preprocessor import Preprocessor
from core.detection.hog_detector import HOGDetector
from core.postprocessing.postprocessor import Postprocessor
from core.visualization.visualizer import Visualizer
from utils.fps_counter import FPSCounter
from utils.frame_sampler import FrameSampler
from utils.report_generator import ReportGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPipeline:
    """
    Yaya tespit pipeline'ı.

    Tek sorumluluk: Tüm bileşenleri sıralı çalıştırmak
    ve kullanıcı etkileşimini yönetmek.
    """

    def __init__(self, source: VideoSource, config: PipelineConfig) -> None:
        """
        Args:
            source: Video kaynağı (dosya veya kamera).
            config: Pipeline konfigürasyonu.
        """
        self._source = source
        self._config = config

        self._preprocessor = Preprocessor(config.preprocess)
        self._detector = HOGDetector(config.detection)
        self._postprocessor = Postprocessor(config.detection)
        self._visualizer = Visualizer(config.visualization)
        self._fps_counter = FPSCounter()

        # Raporlama ve örnekleme
        self._reporter: ReportGenerator = None
        self._sampler: FrameSampler = None

        if config.reporting.enable_reporting:
            self._reporter = ReportGenerator(
                output_dir=config.reporting.report_output_dir
            )

        if config.reporting.enable_frame_sampling:
            self._sampler = FrameSampler(
                output_dir=config.reporting.sample_output_dir,
                sample_interval=config.reporting.sample_interval,
                min_confidence_to_save=config.reporting.high_confidence_threshold,
                save_raw=config.reporting.save_raw_frames,
            )

    def run(self) -> None:
        """
        Pipeline'ı çalıştırır.
        'q' tuşu ile durdurulur.
        """
        logger.info("Pipeline başlatılıyor...")

        self._detector.initialize()

        if self._reporter:
            self._reporter.start()

        with self._source:
            self._setup_output_writer()
            self._process_frames()
            self._visualizer.release_writer()

        cv2.destroyAllWindows()

        # Rapor oluştur
        self._generate_report()

        logger.info("Pipeline tamamlandı.")

    def _process_frames(self) -> None:
        """Kare kare işleme döngüsü."""
        frame_count = 0

        while True:
            frame = self._source.read_frame()
            if frame is None:
                logger.info("Video sonu (toplam %d kare işlendi).", frame_count)
                break

            self._fps_counter.tick()
            frame_count += 1

            # 1. Ön-işleme
            processed = self._preprocessor.process(frame)

            # 2. Tespit
            detections = self._detector.detect(processed)

            # 3. Koordinat ölçekleme (eğer resize yapıldıysa)
            scale = self._preprocessor.scale_factor
            if scale != 1.0:
                detections = [d.scale(scale) for d in detections]

            # 4. Son-işleme (NMS)
            detections = self._postprocessor.process(detections)

            # 5. Görselleştirme (orijinal frame üzerine)
            output = self._visualizer.draw(
                frame, detections, self._fps_counter.fps
            )

            # 6. Frame örnekleme
            if self._sampler:
                self._sampler.process(
                    frame_number=frame_count,
                    raw_frame=frame,
                    annotated_frame=output,
                    detections=detections,
                )

            # 7. Kare istatistiğini kaydet
            if self._reporter:
                confidences = [d.confidence for d in detections]
                self._reporter.record_frame(
                    frame_number=frame_count,
                    detection_count=len(detections),
                    confidences=confidences,
                    fps=self._fps_counter.fps,
                )

            # Göster
            cv2.imshow(self._config.display_window_name, output)

            # Çıkış kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord(self._config.quit_key):
                logger.info("Kullanıcı tarafından durduruldu.")
                break

    def _generate_report(self) -> None:
        """Pipeline sonunda rapor oluşturur."""
        if self._reporter is None:
            return

        w, h = self._source.frame_size

        config_summary = {
            "detection": asdict(self._config.detection),
            "preprocess": asdict(self._config.preprocess),
        }

        self._reporter.generate(
            video_source=str(getattr(self._source, '_file_path', 'camera')),
            video_resolution=f"{w}x{h}",
            video_fps=self._source.fps,
            total_video_frames=getattr(self._source, 'total_frames', 0),
            config_summary=config_summary,
        )

        if self._sampler:
            logger.info(
                "Frame örnekleme: %d kare kaydedildi (%d tespitli kareden)",
                self._sampler.total_saved,
                self._sampler.frames_with_detections,
            )

    def _setup_output_writer(self) -> None:
        """Çıktı video yazıcıyı ayarlar (konfigürasyonda etkinse)."""
        if not self._config.visualization.save_output:
            return

        output_path = self._config.visualization.output_path
        output_dir = os.path.dirname(output_path)

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        self._visualizer.setup_writer(
            output_path=output_path,
            fps=self._source.fps,
            frame_size=self._source.frame_size,
        )
