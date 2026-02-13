"""
Soyut tespit motoru arayüzü.
Template Method Pattern ile tespit algoritması iskeleti tanımlar.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Detection:
    """Tek bir tespit sonucunu temsil eder."""

    x: int
    y: int
    w: int
    h: int
    confidence: float

    @property
    def area(self) -> int:
        """Bounding box alanını döndürür."""
        return self.w * self.h

    @property
    def center(self) -> tuple[int, int]:
        """Bounding box merkez noktasını döndürür."""
        return (self.x + self.w // 2, self.y + self.h // 2)

    def scale(self, factor: float) -> "Detection":
        """Koordinatları verilen faktörle ölçekler."""
        return Detection(
            x=int(self.x / factor),
            y=int(self.y / factor),
            w=int(self.w / factor),
            h=int(self.h / factor),
            confidence=self.confidence,
        )


class BaseDetector(ABC):
    """Tespit algoritması için soyut temel sınıf (Template Method)."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Frame üzerinde nesne tespiti yapar.

        Args:
            frame: Girdi frame (BGR veya gri tonlama).

        Returns:
            Tespit edilen nesnelerin listesi.
        """

    @abstractmethod
    def initialize(self) -> None:
        """Dedektörü başlatır (model yükleme vb.)."""
