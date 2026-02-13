"""
Yaya Tespit Sistemi - Ana Giriş Noktası

Kullanım:
    python main.py --source file --input input/video.mp4
    python main.py --source camera --camera-index 0
"""

import argparse
import sys

from config.settings import PipelineConfig
from core.source.source_factory import SourceFactory, SourceType
from pipeline.detection_pipeline import DetectionPipeline
from utils.logger import get_logger

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Komut satırı argümanlarını ayrıştırır."""
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
    parser.add_argument(
        "--target-width",
        type=int,
        default=640,
        help="Ön-işleme hedef genişlik (varsayılan: 640)",
    )
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
    """Argümanlardan PipelineConfig oluşturur."""
    from config.settings import (
        DetectionConfig,
        PreprocessConfig,
        VisualizationConfig,
    )

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
    """Ana uygulama fonksiyonu."""
    args = parse_arguments()

    logger.info("=" * 50)
    logger.info("Yaya Tespit Sistemi Başlatılıyor")
    logger.info("=" * 50)

    # Kaynak tipi belirleme
    if args.source == "file":
        source_type = SourceType.FILE
        if args.input is None:
            logger.error("--source file seçildi, ancak --input belirtilmedi.")
            sys.exit(1)
    else:
        source_type = SourceType.CAMERA

    # Kaynak oluştur
    try:
        source = SourceFactory.create(
            source_type=source_type,
            path=args.input,
            camera_index=args.camera_index,
        )
    except (FileNotFoundError, ValueError) as e:
        logger.error("Kaynak oluşturma hatası: %s", e)
        sys.exit(1)

    # Konfigürasyon
    config = build_config(args)

    # Pipeline oluştur ve çalıştır
    pipeline = DetectionPipeline(source=source, config=config)

    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Ctrl+C ile durduruldu.")
    except Exception as e:
        logger.error("Beklenmeyen hata: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
