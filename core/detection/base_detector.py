"""
Soyut Tespit Motoru Arayüzü
=============================
Template Method Pattern ile tespit algoritması iskeleti tanımlar.

Bu modül iki temel yapı sunar:
    1. Detection (dataclass): Tek bir tespitin bilgilerini taşır
       (konum, boyut, güven skoru).
    2. BaseDetector (ABC): Tüm dedektörlerin uygulaması gereken
       soyut arayüz. Yeni bir tespit algoritması eklendiğinde
       bu sınıftan türetilmelidir.

Tasarım Prensibi:
    Open/Closed Principle — Yeni dedektörler (YOLO, SSD vb.)
    bu arayüzü implement ederek sisteme eklenebilir;
    mevcut kod değiştirilmez.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


# =============================================================================
# TESPİT VERİ SINIFI
# =============================================================================
@dataclass
class Detection:
    """
    Tek bir tespit sonucunu temsil eder (bounding box + güven skoru).

    Attributes:
        x: Bounding box sol üst köşe X koordinatı (piksel).
        y: Bounding box sol üst köşe Y koordinatı (piksel).
        w: Bounding box genişliği (piksel).
        h: Bounding box yüksekliği (piksel).
        confidence: SVM karar fonksiyonundan gelen güven skoru.
                    Yüksek değer → dedektörün "yaya" kararına daha güvenli.
                    Tipik aralık: 0.5 - 3.0+ (HOG+SVM için).
    """

    x: int          # Sol üst köşe X
    y: int          # Sol üst köşe Y
    w: int          # Genişlik
    h: int          # Yükseklik
    confidence: float  # SVM güven skoru

    @property
    def area(self) -> int:
        """
        Bounding box alanını döndürür.
        NMS gibi algoritmalarda IoU hesaplaması için kullanılır.
        """
        return self.w * self.h

    @property
    def center(self) -> tuple[int, int]:
        """
        Bounding box merkez noktasını döndürür.
        Takip (tracking) algoritmalarında nesne konumunu belirlemek için kullanılır.
        """
        return (self.x + self.w // 2, self.y + self.h // 2)

    def scale(self, factor: float) -> "Detection":
        """
        Koordinatları verilen faktörle ölçekler (ters yönde).

        Ön-işlemede görüntü küçültüldüğünde, tespit koordinatları
        küçültülmüş görüntüye göre olur. Bu metot orijinal
        boyuta geri dönüştürme için kullanılır.

        Örnek:
            Resim %50 küçültüldüyse (factor=0.5), koordinatlar
            2 ile çarpılarak orijinal boyuta geri döner:
            x_orijinal = x_küçük / 0.5 = x_küçük * 2

        Args:
            factor: Ölçekleme faktörü (0-1 arası küçültme).

        Returns:
            Ölçeklenmiş koordinatlara sahip yeni Detection nesnesi.
        """
        return Detection(
            x=int(self.x / factor),
            y=int(self.y / factor),
            w=int(self.w / factor),
            h=int(self.h / factor),
            confidence=self.confidence,  # Güven skoru değişmez
        )


# =============================================================================
# SOYUT DEDEKTÖR SINIFI
# =============================================================================
class BaseDetector(ABC):
    """
    Tespit algoritması için soyut temel sınıf.

    Her dedektör bu iki metodu implement etmelidir:
        - initialize(): Model/dedektör yükleme (bir kez çağrılır)
        - detect(): Kare üzerinde tespit yapma (her frame için çağrılır)

    Kullanım:
        detector = HOGDetector(config)  # Alt sınıf oluştur
        detector.initialize()            # Modeli yükle
        detections = detector.detect(frame)  # Tespit yap
    """

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Frame üzerinde nesne tespiti yapar.

        Args:
            frame: Girdi frame (BGR veya gri tonlama numpy array).

        Returns:
            Tespit edilen nesnelerin listesi (boş liste olabilir).
        """

    @abstractmethod
    def initialize(self) -> None:
        """
        Dedektörü başlatır.
        Model yükleme, SVM ağırlıklarını atama vb. işlemler
        burada yapılır. Pipeline başlamadan önce bir kez çağrılır.
        """
