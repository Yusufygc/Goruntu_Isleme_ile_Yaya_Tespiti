"""
HOG + SVM Tabanlı Yaya Tespit Motoru
=======================================
OpenCV'nin yerleşik HOGDescriptor ve DefaultPeopleDetector kullanır.

Çalışma Prensibi:
    1. HOG (Histogram of Oriented Gradients) görüntüdeki kenar yönelimlerini
       hesaplar ve bunları bir özellik vektörüne dönüştürür.
    2. Önceden eğitilmiş SVM (Support Vector Machine) bu vektörü alarak
       "yaya mı / değil mi" kararı verir.
    3. Kayan pencere (sliding window) + görüntü piramidi ile farklı boyut
       ve konumlardaki yayalar tespit edilir.

Multi-Pass Tespit:
    İsteğe bağlı olarak farklı parametrelerle ikinci bir tarama yapılır.
    Bu, kalabalık ortamda birinci geçişin kaçırdığı yayaları yakalamaya
    çalışır. Performans maliyeti yüksektir (varsayılan kapalı).

Sınıf Hiyerarşisi:
    BaseDetector (abstract) → HOGDetector (concrete)
"""

from typing import List

import cv2
import numpy as np

from config.settings import DetectionConfig
from core.detection.base_detector import BaseDetector, Detection
from utils.logger import get_logger

logger = get_logger(__name__)


class HOGDetector(BaseDetector):
    """
    HOG (Histogram of Oriented Gradients) + SVM yaya dedektörü.

    OpenCV'nin cv2.HOGDescriptor sınıfını kullanır.
    DefaultPeopleDetector, INRIA veri seti üzerinde eğitilmiş
    SVM ağırlıklarını içerir.
    """

    def __init__(self, config: DetectionConfig) -> None:
        """
        Args:
            config: Tespit konfigürasyonu (eşikler, boyutlar, multi-pass ayarları).
        """
        self._config = config
        # HOG descriptor — initialize() çağrılana kadar None
        self._hog: cv2.HOGDescriptor = None

    def initialize(self) -> None:
        """
        HOG descriptor oluşturur ve SVM ağırlıklarını yükler.

        Bu metot pipeline başlamadan önce bir kez çağrılmalıdır.
        DefaultPeopleDetector, OpenCV'deki hazır yaya tanıma
        ağırlıklarını kullanır (INRIA dataset'i üzerinde eğitilmiş).
        """
        # HOGDescriptor oluştur — varsayılan pencere boyutu 64x128 piksel
        self._hog = cv2.HOGDescriptor()

        # INRIA Pedestrian Dataset üzerinde eğitilmiş SVM ağırlıklarını ata
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

        logger.info(
            "HOG Dedektör başlatıldı | winStride: %s | scale: %.2f | "
            "multi-pass: %s",
            self._config.win_stride,
            self._config.scale,
            self._config.enable_multi_pass,
        )

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        HOG + SVM ile yaya tespiti yapar.

        İşlem Akışı:
            1. Geçiş 1: Standart parametrelerle tarama
            2. Geçiş 2 (opsiyonel): Daha yoğun parametrelerle tekrar tarama
            3. İki geçişin sonuçları birleştirilir
            4. Post-processing (NMS) ayrı modülde yapılır

        Args:
            frame: BGR veya gri tonlama girdi frame (ön-işlenmiş).

        Returns:
            Tespit edilen yayaların listesi (filtrelenmemiş ham sonuçlar).

        Raises:
            RuntimeError: Dedektör henüz başlatılmadıysa.
        """
        # Güvenlik kontrolü — initialize() çağrılmadan detect() yapılamaz
        if self._hog is None:
            raise RuntimeError("Dedektör başlatılmadı. Önce initialize() çağırın.")

        # --- Geçiş 1: Standart parametreler ---
        # Ana tarama — hızlı ve genel amaçlı
        detections = self._single_pass(
            frame,
            win_stride=self._config.win_stride,
            padding=self._config.padding,
            scale=self._config.scale,
        )

        # --- Geçiş 2: Yoğun tarama (kalabalık ortam için) ---
        # Daha küçük adım ve daha ince piramit ile ek tarama
        # Bu geçiş YALNIZCA enable_multi_pass=True olduğunda çalışır
        if self._config.enable_multi_pass:
            second_pass = self._single_pass(
                frame,
                win_stride=self._config.second_pass_win_stride,
                padding=self._config.second_pass_padding,
                scale=self._config.second_pass_scale,
                hit_threshold=self._config.second_pass_hit_threshold,
            )
            # İki geçişin sonuçları birleştirilir
            # Çift tespitler post-processing'de NMS ile elenir
            detections.extend(second_pass)

        return detections

    def _single_pass(
        self,
        frame: np.ndarray,
        win_stride: tuple,
        padding: tuple,
        scale: float,
        hit_threshold: float = 0.0,
    ) -> List[Detection]:
        """
        Tek geçişlik HOG tespiti yapar.

        OpenCV'nin detectMultiScale fonksiyonunu çağırır:
            - Görüntü piramidi oluşturur (farklı ölçeklerde tarama)
            - Her ölçekte kayan pencere ile HOG özelliklerini hesaplar
            - SVM ile her pencereyi sınıflandırır

        Args:
            frame: Girdi frame.
            win_stride: Kayan pencere adımı (piksel).
            padding: Kenar dolgusu (piksel).
            scale: Piramit ölçek faktörü.
            hit_threshold: SVM karar eşiği (0.0 → tüm sonuçları al).

        Returns:
            Minimum boyut filtresini geçen tespitlerin listesi.
        """
        # detectMultiScale → (regions, weights)
        # regions: Tespit edilen kutuların [x, y, w, h] dizisi
        # weights: Her kutu için SVM güven skorları
        regions, weights = self._hog.detectMultiScale(
            frame,
            winStride=win_stride,
            padding=padding,
            scale=scale,
            hitThreshold=hit_threshold,
        )

        detections: List[Detection] = []

        # Her tespit bölgesi için Detection nesnesi oluştur
        for (x, y, w, h), weight in zip(regions, weights):
            confidence = float(weight)  # numpy → Python float dönüşümü

            # --- Minimum boyut filtresi ---
            # Çok küçük tespitler genellikle gürültüdür (uzaktaki objeler,
            # sensör artefaktları vb.). Minimum boyutu geçemeyenler elenir.
            if w < self._config.min_detection_size[0]:
                continue  # Genişlik çok küçük
            if h < self._config.min_detection_size[1]:
                continue  # Yükseklik çok küçük

            detections.append(Detection(
                x=int(x), y=int(y), w=int(w), h=int(h),
                confidence=confidence,
            ))

        return detections
