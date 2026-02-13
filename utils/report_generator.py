"""
Tespit Raporu Oluşturma Modülü
=================================
Video işleme sırasında kare başı istatistik toplar ve
pipeline sonunda detaylı JSON raporu üretir.

Rapor İçeriği:
    - Video bilgileri (kaynak, çözünürlük, FPS, toplam kare)
    - İşleme istatistikleri (tespitli/tespitsiz kare, toplam tespit)
    - Güven skoru dağılımı (ortalama, min, max)
    - FPS performans bilgisi (ortalama, min, max)
    - Toplam işleme süresi
    - En yoğun N kare detayları
    - Kullanılan konfigürasyon parametreleri

Çıktı Örneği:
    output/detection_report.json

Kullanım:
    reporter = ReportGenerator(output_dir="output")
    reporter.start()
    for frame in video:
        reporter.record_frame(frame_no, det_count, confidences, fps)
    reporter.generate(video_source="video.mp4", ...)
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict

from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# VERİ SINIFLARI
# =============================================================================
@dataclass
class FrameStats:
    """
    Tek bir karenin tespit istatistikleri.

    Her işlenen kare için bir FrameStats nesnesi oluşturulur
    ve ReportGenerator'ın listesine eklenir.

    Attributes:
        frame_number: Kare numarası (1'den başlar).
        detection_count: Bu karedeki tespit sayısı.
        confidences: Her tespite ait güven skorları listesi.
        fps: Bu kare işlenirken ölçülen anlık FPS.
    """

    frame_number: int
    detection_count: int
    confidences: List[float]
    fps: float


@dataclass
class DetectionReport:
    """
    Video işleme sonuç raporu — JSON'a serileştirilir.

    Bu dataclass, raporun tüm alanlarını tanımlar.
    asdict() ile dict'e dönüştürülür ve json.dump() ile dosyaya yazılır.
    """

    # --- Video Bilgileri ---
    video_source: str = ""              # Video dosya yolu veya "camera"
    video_resolution: str = ""          # "1280x720" formatında çözünürlük
    video_fps: float = 0.0              # Kaynak video FPS değeri
    total_video_frames: int = 0         # Kaynak videodaki toplam kare sayısı

    # --- İşleme İstatistikleri ---
    total_processed_frames: int = 0     # İşlenen toplam kare sayısı
    frames_with_detections: int = 0     # En az 1 tespit içeren kare sayısı
    frames_without_detections: int = 0  # Hiç tespit olmayan kare sayısı
    total_detections: int = 0           # Tüm karelerdeki toplam tespit sayısı

    # --- Güven Skoru Dağılımı ---
    avg_confidence: float = 0.0         # Ortalama güven skoru
    min_confidence: float = 0.0         # Minimum güven skoru
    max_confidence: float = 0.0         # Maksimum güven skoru

    # --- Performans Bilgisi ---
    avg_fps: float = 0.0                # Ortalama işlem FPS'i
    min_fps: float = 0.0                # Minimum FPS (en yavaş kare)
    max_fps: float = 0.0                # Maksimum FPS (en hızlı kare)
    total_processing_time_sec: float = 0.0  # Toplam işleme süresi (saniye)

    # --- Kare Detayları ---
    # En çok tespit içeren N kare (detaylı bilgi ile)
    top_detection_frames: List[Dict] = field(default_factory=list)

    # --- Konfigürasyon Özeti ---
    # Hangi parametrelerle çalıştırıldığının kaydı
    config_summary: Dict = field(default_factory=dict)


# =============================================================================
# RAPOR OLUŞTURUCU
# =============================================================================
class ReportGenerator:
    """
    Pipeline çalışması sırasında istatistik toplar ve
    sonunda detaylı JSON raporu üretir.

    İki fazlı çalışma:
        1. Toplama fazı: record_frame() ile kare başı veri topla
        2. Üretim fazı: generate() ile tüm veriyi analiz et ve raporla
    """

    def __init__(self, output_dir: str = "output") -> None:
        """
        ReportGenerator'u başlatır ve çıktı dizinini oluşturur.

        Args:
            output_dir: Rapor dosyasının kaydedileceği dizin.
        """
        self._output_dir = output_dir

        # --- Veri Toplama Listeleri ---
        self._frame_stats: List[FrameStats] = []     # Kare başı istatistikler
        self._all_confidences: List[float] = []       # Tüm güven skorları (düz liste)
        self._all_fps: List[float] = []               # Tüm FPS değerleri

        # İşleme başlangıç zamanı (start() ile set edilir)
        self._start_time: float = 0.0

        # Çıktı dizinini oluştur
        os.makedirs(output_dir, exist_ok=True)
        logger.info("ReportGenerator başlatıldı | Dizin: %s", output_dir)

    def start(self) -> None:
        """
        İşleme başlangıç zamanını kaydeder.
        Pipeline başlamadan hemen önce çağrılmalıdır —
        toplam işleme süresi bu zamandan itibaren hesaplanır.
        """
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

        Bu metot pipeline'ın her karesinde çağrılır.
        Veriler bellekte biriktirilir ve generate() çağrıldığında analiz edilir.

        Args:
            frame_number: Kare numarası.
            detection_count: Bu karedeki tespit sayısı.
            confidences: Her tespite ait güven skorları.
            fps: Anlık FPS değeri (FPSCounter'dan).
        """
        # FrameStats nesnesi oluştur ve listeye ekle
        stats = FrameStats(
            frame_number=frame_number,
            detection_count=detection_count,
            confidences=confidences,
            fps=fps,
        )
        self._frame_stats.append(stats)

        # Güven skorlarını düz listeye ekle (global istatistik için)
        self._all_confidences.extend(confidences)

        # FPS değerini kaydet (0 olanları hariç tut — başlangıç gecikmesi)
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
        Toplanan verileri analiz eder ve JSON rapor dosyası oluşturur.

        Adımlar:
            1. Toplanan istatistikleri hesapla (ortalama, min, max)
            2. En yoğun 5 kareyi bul
            3. DetectionReport dataclass'ını doldur
            4. JSON'a serileştir ve dosyaya yaz
            5. Konsola özet logla

        Args:
            video_source: Video dosya yolu veya "camera".
            video_resolution: "1280x720" formatında çözünürlük.
            video_fps: Kaynak video FPS değeri.
            total_video_frames: Kaynak videodaki toplam kare sayısı.
            config_summary: Kullanılan parametrelerin dict'i.

        Returns:
            Oluşturulan rapor dosyasının yolu.
        """
        # Toplam işleme süresini hesapla
        elapsed = time.time() - self._start_time if self._start_time else 0

        # Tespitli kare sayıları
        frames_with = sum(1 for s in self._frame_stats if s.detection_count > 0)
        total_det = sum(s.detection_count for s in self._frame_stats)

        # DetectionReport nesnesini oluştur
        report = DetectionReport(
            # Video bilgileri
            video_source=video_source,
            video_resolution=video_resolution,
            video_fps=video_fps,
            total_video_frames=total_video_frames,
            # İşleme istatistikleri
            total_processed_frames=len(self._frame_stats),
            frames_with_detections=frames_with,
            frames_without_detections=len(self._frame_stats) - frames_with,
            total_detections=total_det,
            # Güven dağılımı
            avg_confidence=self._safe_avg(self._all_confidences),
            min_confidence=min(self._all_confidences) if self._all_confidences else 0,
            max_confidence=max(self._all_confidences) if self._all_confidences else 0,
            # FPS performansı
            avg_fps=self._safe_avg(self._all_fps),
            min_fps=min(self._all_fps) if self._all_fps else 0,
            max_fps=max(self._all_fps) if self._all_fps else 0,
            total_processing_time_sec=round(elapsed, 2),
            # En yoğun 5 kare
            top_detection_frames=self._get_top_frames(5),
            # Konfigürasyon
            config_summary=config_summary or {},
        )

        # --- JSON Dosyasına Yaz ---
        output_path = os.path.join(self._output_dir, "detection_report.json")
        report_dict = asdict(report)  # dataclass → dict dönüşümü

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        logger.info("Rapor oluşturuldu: %s", output_path)

        # Konsola özet bilgi yazdır
        self._log_summary(report)

        return output_path

    def _get_top_frames(self, n: int) -> List[Dict]:
        """
        En çok tespit içeren N kareyi döndürür.

        Tespit sayısına göre azalan sıralama yapılır.
        Bu bilgi, kalabalık sahneleri veya sorunlu kareleri
        hızlıca bulmak için kullanılır.

        Args:
            n: Döndürülecek kare sayısı.

        Returns:
            FrameStats dict'lerinin listesi.
        """
        sorted_stats = sorted(
            self._frame_stats,
            key=lambda s: s.detection_count,
            reverse=True,  # Azalan sıralama (en yoğun ilk)
        )
        return [asdict(s) for s in sorted_stats[:n]]

    def _log_summary(self, report: DetectionReport) -> None:
        """
        Rapor özetini konsola yazdırır.
        Görsel ayraçlar ile okunabilirlik artırılır.
        """
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
        """
        Güvenli ortalama hesaplama.
        Boş liste durumunda ZeroDivisionError yerine 0.0 döndürür.

        Args:
            values: Sayısal değerler listesi.

        Returns:
            Ortalama değer (4 ondalık basamağa yuvarlanmış) veya 0.0.
        """
        if not values:
            return 0.0
        return round(sum(values) / len(values), 4)
