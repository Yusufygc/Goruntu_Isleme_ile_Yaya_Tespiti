"""
Yaya Tespit Sistemi - Ana Giriş Noktası
==========================================
Bu dosya uygulamanın başlangıç noktasıdır. Komut satırından alınan
argümanları ayrıştırır, konfigürasyonu oluşturur, video kaynağını
hazırlar ve tespit pipeline'ını çalıştırır.

Kullanım Örnekleri:
    # Video dosyası ile çalıştırma
    python main.py --source file --input input/video.mp4

    # Kamera ile canlı tespit
    python main.py --source camera --camera-index 0

    # Çıktıyı kaydetme
    python main.py --source file --input input/video.mp4 --save-output

Akış Şeması:
    1. Komut satırı argümanlarını ayrıştır (parse_arguments)
    2. Konfigürasyon nesnesi oluştur (build_config)
    3. Video kaynağını fabrika ile üret (SourceFactory)
    4. DetectionPipeline oluştur ve çalıştır
    5. Ctrl+C veya 'q' tuşu ile durdurulabilir
"""

import argparse
import sys

# Proje iç modülleri
from config.settings import PipelineConfig           # Merkezi ayarlar
from core.source.source_factory import SourceFactory, SourceType  # Video kaynağı üretici
from pipeline.detection_pipeline import DetectionPipeline          # Ana pipeline
from utils.logger import get_logger                                # Loglama yardımcısı

# Bu modülün logger'ı — loglar "main" etiketiyle yazılır
logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Komut satırı argümanlarını ayrıştırır.

    Desteklenen argümanlar:
        --source       : Video kaynağı ('file' veya 'camera')
        --input        : Video dosya yolu (file modu için zorunlu)
        --camera-index : Kamera cihaz indeksi (varsayılan: 0)
        --target-width : Ön-işleme hedef genişlik piksel (varsayılan: 640)
        --save-output  : Çıktı videosunu kaydetme bayrağı
        --output-path  : Çıktı video dosya yolu

    Returns:
        argparse.Namespace: Ayrıştırılmış argümanlar.
    """
    parser = argparse.ArgumentParser(
        description="Yaya Tespit Sistemi - HOG + SVM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python main.py --source file --input input/video.mp4
  python main.py --source camera --camera-index 0
  python main.py --source file --input input/video.mp4 --save-output
        """,
    )

    # --- Kaynak Ayarları ---
    parser.add_argument(
        "--source",
        type=str,
        choices=["file", "camera"],
        default="file",
        help="Video kaynağı tipi (varsayılan: file)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Video dosya yolu (source=file için zorunlu)",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Kamera indeksi (varsayılan: 0)",
    )

    # --- İşleme Ayarları ---
    parser.add_argument(
        "--target-width",
        type=int,
        default=640,
        help="Ön-işleme hedef genişlik (varsayılan: 640)",
    )

    # --- Çıktı Ayarları ---
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Çıktı videosunu kaydet",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="output/result.avi",
        help="Çıktı video dosya yolu (varsayılan: output/result.avi)",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """
    Komut satırı argümanlarından PipelineConfig nesnesi oluşturur.

    Varsayılan tespit/ön-işleme ayarları config/settings.py'den gelir.
    Burada yalnızca komut satırından geçirilen değerler override edilir.

    Args:
        args: Ayrıştırılmış komut satırı argümanları.

    Returns:
        PipelineConfig: Pipeline için hazır konfigürasyon nesnesi.
    """
    from config.settings import (
        DetectionConfig,       # HOG+SVM tespit parametreleri
        PreprocessConfig,      # Görüntü ön-işleme parametreleri
        VisualizationConfig,   # Görsel çıktı parametreleri
    )

    # Her alt konfigürasyon kendi varsayılanlarını taşır,
    # yalnızca komut satırından gelen değerler burada override edilir
    config = PipelineConfig(
        detection=DetectionConfig(),
        preprocess=PreprocessConfig(target_width=args.target_width),
        visualization=VisualizationConfig(
            save_output=args.save_output,
            output_path=args.output_path,
        ),
    )

    return config


def main() -> None:
    """
    Ana uygulama fonksiyonu.

    Adımlar:
        1. Argümanları ayrıştır
        2. Kaynak tipini belirle (FILE veya CAMERA)
        3. Video kaynağını oluştur
        4. Konfigürasyonu hazırla
        5. Pipeline'ı başlat ve çalıştır
        6. Hata yönetimini sağla (Ctrl+C, beklenmeyen hatalar)
    """
    args = parse_arguments()

    # Başlatma banner'ı — logları ayırt etmek için görsel çizgi
    logger.info("=" * 50)
    logger.info("Yaya Tespit Sistemi Başlatılıyor")
    logger.info("=" * 50)

    # --- Kaynak Tipi Belirleme ---
    # Kullanıcı "file" seçtiyse --input zorunlu; "camera" ise indeks yeterli
    if args.source == "file":
        source_type = SourceType.FILE
        if args.input is None:
            logger.error("--source file seçildi, ancak --input belirtilmedi.")
            sys.exit(1)  # Hatalı kullanım — çıkış kodu 1
    else:
        source_type = SourceType.CAMERA

    # --- Video Kaynağı Oluşturma ---
    # SourceFactory (Factory Pattern) uygun kaynak nesnesini üretir
    try:
        source = SourceFactory.create(
            source_type=source_type,
            path=args.input,
            camera_index=args.camera_index,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Kaynak oluşturma hatası: %s", e)
        sys.exit(1)

    # --- Konfigürasyon Oluşturma ---
    config = build_config(args)

    # --- Pipeline Oluşturma ve Çalıştırma ---
    # DetectionPipeline tüm bileşenleri orkestre eder
    pipeline = DetectionPipeline(source=source, config=config)

    try:
        pipeline.run()  # Ana döngü burada başlar
    except KeyboardInterrupt:
        # Ctrl+C basıldığında nazik kapanış
        logger.info("Ctrl+C ile durduruldu.")
    except Exception as e:
        # Beklenmeyen hata — tam stack trace ile logla
        logger.error("Beklenmeyen hata: %s", e, exc_info=True)
        sys.exit(1)


# Python doğrudan bu dosyayı çalıştırırsa main() çağrılır;
# import edildiğinde çağrılmaz (modül olarak kullanılabilir)
if __name__ == "__main__":
    main()
