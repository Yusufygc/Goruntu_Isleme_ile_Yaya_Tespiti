"""
Tespit pipeline'ı.
Tüm işleme adımlarını orkestre eder:
  source → preprocess → detect → postprocess → visualize
"""

import os

import cv2

from config.settings import PipelineConfig
from core.source.base_source import VideoSource
from core.preprocessing.preprocessor import Preprocessor
from core.detection.hog_detector import HOGDetector
from core.postprocessing.postprocessor import Postprocessor
from core.visualization.visualizer import Visualizer
from utils.fps_counter import FPSCounter
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

    def run(self) -> None:
        """
        Pipeline'ı çalıştırır.
        'q' tuşu ile durdurulur.
        """
        logger.info("Pipeline başlatılıyor...")

        self._detector.initialize()

        with self._source:
            self._setup_output_writer()
            self._process_frames()
            self._visualizer.release_writer()

        cv2.destroyAllWindows()
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

            # Göster
            cv2.imshow(self._config.display_window_name, output)

            # Çıkış kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord(self._config.quit_key):
                logger.info("Kullanıcı tarafından durduruldu.")
                break

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
