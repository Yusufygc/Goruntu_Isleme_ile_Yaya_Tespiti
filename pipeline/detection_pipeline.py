"""
Tespit Pipeline'ı (Ana Orkestratör)
====================================
Tüm işleme adımlarını sıralı çalıştıran merkezi modül.

Pipeline Akış Şeması:
    ┌─────────┐   ┌──────────┐   ┌────────┐   ┌────────────┐   ┌───────────┐
    │ Source   │ → │ Preproc. │ → │ Detect │ → │ Postproc.  │ → │ Visualize │
    │ (video)  │   │ (resize, │   │ (HOG+  │   │ (NMS,      │   │ (draw,    │
    │          │   │  CLAHE)  │   │  SVM)  │   │  filter)   │   │  panel)   │
    └─────────┘   └──────────┘   └────────┘   └────────────┘   └───────────┘
                                                                       │
                                                              ┌────────┴────────┐
                                                              │                 │
                                                         ┌────▼──┐       ┌──────▼──┐
                                                         │Sample │       │ Report  │
                                                         │(save) │       │ (JSON)  │
                                                         └───────┘       └─────────┘

Tasarım Prensipleri:
    - Single Responsibility: Pipeline yalnızca sıralama ve koordinasyondan sorumlu
    - Dependency Injection: Tüm bileşenler config üzerinden yapılandırılır
    - Context Manager: Video kaynağı 'with' bloğu ile güvenli kapatılır
"""

import os
from dataclasses import asdict

import cv2

# Konfigürasyon
from config.settings import PipelineConfig

# Kaynak
from core.source.base_source import VideoSource

# İşleme bileşenleri (pipeline sırasına göre)
from core.preprocessing.preprocessor import Preprocessor
from core.detection.hog_detector import HOGDetector
from core.postprocessing.postprocessor import Postprocessor
from core.visualization.visualizer import Visualizer

# Yardımcı modüller
from utils.fps_counter import FPSCounter
from utils.frame_sampler import FrameSampler
from utils.report_generator import ReportGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


class DetectionPipeline:
    """
    Yaya tespit pipeline'ı — ana orkestratör sınıf.

    Tek sorumluluk: Tüm bileşenleri doğru sırada çalıştırmak
    ve kullanıcı etkileşimini (pencere, klavye) yönetmek.

    Bileşenler:
        - Preprocessor: Görüntü ön-işleme (resize, CLAHE, sharpen)
        - HOGDetector: Yaya tespiti (HOG + SVM)
        - Postprocessor: Filtreleme ve NMS
        - Visualizer: Sonuçları frame üzerine çizme
        - FrameSampler: Kareleri diske kaydetme (opsiyonel)
        - ReportGenerator: İstatistik toplama ve rapor üretme (opsiyonel)
        - FPSCounter: İşlem hızı ölçümü
    """

    def __init__(self, source: VideoSource, config: PipelineConfig) -> None:
        """
        Pipeline'ı tüm bileşenleriyle başlatır.

        Args:
            source: Video kaynağı (dosya veya kamera).
            config: Pipeline konfigürasyonu (tüm alt ayarları içerir).
        """
        self._source = source
        self._config = config

        # --- Ana İşleme Bileşenleri ---
        # Her bileşen kendi config sınıfını alır (Dependency Injection)
        self._preprocessor = Preprocessor(config.preprocess)
        self._detector = HOGDetector(config.detection)
        self._postprocessor = Postprocessor(config.detection)
        self._visualizer = Visualizer(config.visualization, config.detection)
        self._fps_counter = FPSCounter()

        # --- Raporlama ve Örnekleme (Opsiyonel) ---
        # Bu bileşenler konfigürasyonda devre dışı bırakılabilir
        self._reporter: ReportGenerator = None
        self._sampler: FrameSampler = None

        # Rapor oluşturucuyu başlat (etkinse)
        if config.reporting.enable_reporting:
            self._reporter = ReportGenerator(
                output_dir=config.reporting.report_output_dir
            )

        # Frame örnekleyiciyi başlat (etkinse)
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

        Akış:
            1. Dedektörü başlat (SVM ağırlıklarını yükle)
            2. Raporlayıcıyı başlat (zamanlayıcıyı başlat)
            3. Video kaynağını aç (context manager)
            4. Çıktı yazıcıyı hazırla (etkinse)
            5. Kare kare işleme döngüsü
            6. Yazıcıyı kapat
            7. Rapor oluştur

        Durma koşulları:
            - Video sonu (read_frame() → None)
            - 'q' tuşuna basılması
            - Ctrl+C (KeyboardInterrupt — main.py'de yakalanır)
        """
        logger.info("Pipeline başlatılıyor...")

        # Dedektörü bir kez başlat — SVM ağırlıklarını yükler
        self._detector.initialize()

        # Raporlayıcıyı başlat — işlem süresini ölçmek için zamanlayıcı başlatır
        if self._reporter:
            self._reporter.start()

        # Context manager ile kaynağı aç/kapat
        # 'with' bloğu hata durumunda bile kaynağı kapatmayı garantiler
        with self._source:
            self._setup_output_writer()   # Video yazıcıyı hazırla
            self._process_frames()         # Ana işleme döngüsü
            self._visualizer.release_writer()  # Yazıcıyı kapat

        # OpenCV pencerelerini kapat
        cv2.destroyAllWindows()

        # Pipeline sonunda rapor oluştur
        self._generate_report()

        logger.info("Pipeline tamamlandı.")

    def _process_frames(self) -> None:
        """
        Kare kare işleme döngüsü — pipeline'ın kalbi.

        Her kare için sıralı işlem adımları:
            1. Ön-işleme (resize, denoise, CLAHE, sharpen)
            2. Tespit (HOG + SVM)
            3. Koordinat ölçekleme (resize yapıldıysa geri dönüştür)
            4. Son-işleme (güven filtresi, boyut filtresi, NMS)
            5. Görselleştirme (bounding box, panel çizimi)
            6. Frame örnekleme (diske kaydetme, opsiyonel)
            7. İstatistik kaydetme (rapor için, opsiyonel)
            8. Ekrana gösterme ve çıkış kontrolü
        """
        frame_count = 0  # İşlenen toplam kare sayısı

        while True:
            # --- Kare Okuma ---
            frame = self._source.read_frame()
            if frame is None:
                # Video sonu veya kamera bağlantısı koptu
                logger.info("Video sonu (toplam %d kare işlendi).", frame_count)
                break

            # FPS sayacını güncelle (kayan ortalama hesabı için)
            self._fps_counter.tick()
            frame_count += 1

            # === Adım 1: Ön-İşleme ===
            # Resize + görüntü iyileştirme → dedektör için hazırlık
            processed = self._preprocessor.process(frame)

            # === Adım 2: Tespit ===
            # HOG + SVM ile yaya tespiti → ham tespitler
            detections = self._detector.detect(processed)

            # === Adım 3: Koordinat Ölçekleme ===
            # Eğer ön-işlemede resize yapıldıysa, tespit koordinatları
            # küçültülmüş görüntüye göre. Orijinal boyuta geri dönüştür.
            scale = self._preprocessor.scale_factor
            if scale != 1.0:
                detections = [d.scale(scale) for d in detections]

            # === Adım 4: Son-İşleme (NMS) ===
            # Güven filtresi → boyut/oran filtresi → NMS
            detections = self._postprocessor.process(detections)

            # === Adım 5: Görselleştirme ===
            # Bounding box + güven etiketi + bilgi paneli → orijinal frame üzerine
            output = self._visualizer.draw(
                frame, detections, self._fps_counter.fps
            )

            # === Adım 6: Frame Örnekleme (Opsiyonel) ===
            # Tespit içeren kareleri belirli aralıklarla diske kaydeder
            if self._sampler:
                self._sampler.process(
                    frame_number=frame_count,
                    raw_frame=frame,             # Orijinal (çizimsiz) kare
                    annotated_frame=output,      # Çizim yapılmış kare
                    detections=detections,
                )

            # === Adım 7: İstatistik Kaydı (Opsiyonel) ===
            # Güven skorları ve FPS değerini kare bazlı kaydet
            if self._reporter:
                confidences = [d.confidence for d in detections]
                self._reporter.record_frame(
                    frame_number=frame_count,
                    detection_count=len(detections),
                    confidences=confidences,
                    fps=self._fps_counter.fps,
                )

            # === Adım 8: Ekrana Gösterme ===
            cv2.imshow(self._config.display_window_name, output)

            # --- Çıkış Kontrolü ---
            # waitKey(1): 1ms bekle ve tuş basımını kontrol et
            # 0xFF mask: Yalnızca ASCII değeri al (platform uyumluluğu)
            key = cv2.waitKey(1) & 0xFF
            if key == ord(self._config.quit_key):
                logger.info("Kullanıcı tarafından durduruldu.")
                break

    def _generate_report(self) -> None:
        """
        Pipeline sonunda detaylı rapor oluşturur.

        Rapor içeriği:
            - Video meta verileri (kaynak, çözünürlük, FPS, süre)
            - İşleme istatistikleri (toplam kare, tespit sayıları)
            - Güven dağılımı (ortalama, min, max)
            - FPS performans bilgisi
            - Kullanılan konfigürasyon özeti
            - En yoğun 5 kare ({frame_no, tespit_sayisi, güvenler})
        """
        if self._reporter is None:
            return  # Raporlama devre dışı

        # Video çözünürlüğünü al
        w, h = self._source.frame_size

        # Konfigürasyon özetini dict'e dönüştür (JSON serileştirme için)
        config_summary = {
            "detection": asdict(self._config.detection),
            "preprocess": asdict(self._config.preprocess),
        }

        # Raporu oluştur ve JSON'a yaz
        self._reporter.generate(
            video_source=str(getattr(self._source, '_file_path', 'camera')),
            video_resolution=f"{w}x{h}",
            video_fps=self._source.fps,
            total_video_frames=getattr(self._source, 'total_frames', 0),
            config_summary=config_summary,
        )

        # Örnekleme istatistiklerini logla
        if self._sampler:
            logger.info(
                "Frame örnekleme: %d kare kaydedildi (%d tespitli kareden)",
                self._sampler.total_saved,
                self._sampler.frames_with_detections,
            )

    def _setup_output_writer(self) -> None:
        """
        Çıktı video yazıcıyı ayarlar (konfigürasyonda etkinse).

        Çıktı dizini yoksa otomatik oluşturulur.
        Yazıcı, Visualizer sınıfına aktarılır ve draw() çağrılarında
        otomatik olarak her kareyi dosyaya yazar.
        """
        # save_output kapalıysa yazıcı kurma
        if not self._config.visualization.save_output:
            return

        output_path = self._config.visualization.output_path
        output_dir = os.path.dirname(output_path)

        # Çıktı dizini yoksa oluştur
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Visualizer'a yazıcı kur (FPS ve boyut video kaynağından alınır)
        self._visualizer.setup_writer(
            output_path=output_path,
            fps=self._source.fps,
            frame_size=self._source.frame_size,
        )
