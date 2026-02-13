"""
Merkezi logging yardımcısı.
Tüm modüller bu fonksiyon üzerinden logger alır.
"""

import logging
import sys
from typing import Optional


_LOG_FORMAT = "%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_configured = False


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Modül adına göre yapılandırılmış logger döndürür.

    Args:
        name: Logger adı (genellikle __name__).
        level: Opsiyonel log seviyesi. Varsayılan INFO.

    Returns:
        Yapılandırılmış Logger nesnesi.
    """
    global _configured

    if not _configured:
        _setup_root_logger()
        _configured = True

    logger = logging.getLogger(name)

    if level is not None:
        logger.setLevel(level)

    return logger


def _setup_root_logger() -> None:
    """Root logger'ı bir kez yapılandırır."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Mevcut handler'ları temizle
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root_logger.addHandler(handler)
