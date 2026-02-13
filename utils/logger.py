"""
Merkezi Logging Yardımcısı
============================
Tüm modüller bu fonksiyon üzerinden logger alır.
Singleton benzeri bir yapı ile root logger bir kez yapılandırılır.

Log Formatı:
    HH:MM:SS | modül_adı                | SEVİYE  | Mesaj

Örnek Çıktı:
    20:26:06 | core.detection.hog_detector | INFO    | HOG Dedektör başlatıldı

Kullanım:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Bir mesaj")
    logger.error("Hata: %s", hata_detayi)

Neden Merkezi Logger?
    - Tüm modüller aynı formatta log üretir
    - Log seviyesi tek yerden kontrol edilir
    - Handler konfigürasyonu tekrarlanmaz
"""

import logging
import sys
from typing import Optional

# Log çıktı formatı
# %(asctime)s     → Saat (HH:MM:SS)
# %(name)-25s     → Logger adı (modül yolu, 25 karakter sola hizalı)
# %(levelname)-7s → Log seviyesi (INFO, WARNING vb., 7 karakter)
# %(message)s     → Log mesajı
_LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Flag: Root logger'ın sadece bir kez yapılandırılmasını sağlar
_configured = False


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Modül adına göre yapılandırılmış logger döndürür.

    İlk çağrıda root logger yapılandırılır (handler eklenir).
    Sonraki çağrılarda yalnızca modül logger'ı döndürülür.

    Args:
        name: Logger adı. Genellikle __name__ kullanılır,
              bu sayede log çıktısında modül yolu görünür.
              Örn: "core.detection.hog_detector"
        level: Opsiyonel log seviyesi override.
               Varsayılan: INFO (root logger'dan miras).
               Kullanım: get_logger(__name__, logging.DEBUG)

    Returns:
        Yapılandırılmış Logger nesnesi.
    """
    global _configured

    # Root logger'ı ilk kez yapılandır (sadece bir kez çalışır)
    if not _configured:
        _setup_root_logger()
        _configured = True

    # Modül adına göre logger al (hiyerarşik yapıda)
    logger = logging.getLogger(name)

    # Opsiyonel seviye override
    if level is not None:
        logger.setLevel(level)

    return logger


def _setup_root_logger() -> None:
    """
    Root logger'ı bir kez yapılandırır.

    - Seviye: INFO (DEBUG mesajları gösterilmez)
    - Handler: StreamHandler → stdout (konsola yazdırır)
    - Formatter: Saat | Modül | Seviye | Mesaj formatı

    NOT: Mevcut handler'lar temizlenir — çift log çıktısını önler.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Mevcut handler'ları temizle — aynı handler'ın birden fazla
    # eklenmesini önler (importlarda tekrarlayan çağrılar)
    root_logger.handlers.clear()

    # Konsol handler'ı — stdout'a yazdırır
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(handler)
