"""
Son-İşleme (Postprocessing) Modülü
=====================================
Ham tespit sonuçlarını filtreler ve düzenler.

İşlem Zinciri:
    1. Güven eşiği filtreleme → Düşük güvenli tespitleri ele
    2. Boyut ve en-boy oranı filtresi → Yaya olmayan şekilleri ele
    3. Non-Maximum Suppression (NMS) → Çakışan kutuları birleştir

Neden Post-Processing Gerekli?
    - HOG dedektörü aynı yaya için birden fazla kayan pencerede
      "yaya" kararı verebilir → çakışan kutular oluşur
    - Bazı yaya olmayan nesneler (tabela, direk) de yaya olarak
      tespit edilebilir → false positive'ler oluşur
    - Post-processing bu problemleri temizler
"""

from typing import List

import cv2
import numpy as np

from config.settings import DetectionConfig
from core.detection.base_detector import Detection
from utils.logger import get_logger

logger = get_logger(__name__)


class Postprocessor:
    """
    Tespit sonuçlarını filtreler ve düzenler.

    Üç aşamalı filtreleme pipeline'ı uygular:
        güven_eşiği → boyut/oran_filtresi → NMS
    """

    def __init__(self, config: DetectionConfig) -> None:
        """
        Args:
            config: Tespit konfigürasyonu (eşik değerleri ve boyut sınırları).
        """
        self._config = config
        logger.info(
            "Postprocessor başlatıldı | NMS eşiği: %.2f | Güven eşiği: %.2f",
            config.nms_threshold,
            config.confidence_threshold,
        )

    def process(self, detections: List[Detection]) -> List[Detection]:
        """
        Tespit sonuçlarına filtreleme ve NMS uygular.

        İşlem sırası önemlidir:
            1. Güven eşiği → En hızlı filtre (basit karşılaştırma)
            2. Boyut/oran → Yaya olmayan şekilleri ele
            3. NMS → En pahalı işlem (IoU hesaplaması), son adımda yapılır

        Args:
            detections: HOG dedektöründen gelen ham tespit listesi.

        Returns:
            Filtrelenmiş ve NMS uygulanmış nihai tespit listesi.
        """
        # Boş liste kontrolü — gereksiz işlem yapma
        if not detections:
            return []

        # --- Adım 1: Güven Eşiği Filtreleme ---
        # confidence_threshold altındaki tespitler elenir.
        # Bu en hızlı filtredir, çoğu sahte tespiti burada yakalar.
        filtered = [
            d for d in detections
            if d.confidence >= self._config.confidence_threshold
        ]

        # --- Adım 2: Boyut ve En-Boy Oranı Filtresi ---
        # Yaya olmayan şekilleri eler:
        #   - Çok büyük kutular (birden fazla kişiyi kaplayan)
        #   - Yatay kutular (araç, tabela gibi)
        filtered = [
            d for d in filtered
            if self._is_valid_detection(d)
        ]

        if not filtered:
            return []

        # --- Adım 3: Non-Maximum Suppression (NMS) ---
        # Çakışan kutuları birleştirir — en güvenlisini tutar
        return self._apply_nms(filtered)

    def _is_valid_detection(self, det: Detection) -> bool:
        """
        Tespitteki bounding box'ın yaya boyutuna ve oranına
        uygun olup olmadığını kontrol eder.

        İki kriter uygulanır:
            1. Maksimum boyut → Dev kutular tek yaya olamaz
            2. En-boy oranı → Yayalar dikey yapıdadır (h/w > 1.3)

        Args:
            det: Kontrol edilecek tespit.

        Returns:
            True ise tespit geçerli (yaya olabilir), False ise elenir.
        """
        # --- Maksimum boyut kontrolü ---
        # Çok büyük kutular genellikle birden fazla kişiyi veya
        # arka plan yapılarını (bina, vitrin) kapsar → sahte tespit
        max_w, max_h = self._config.max_detection_size
        if det.w > max_w or det.h > max_h:
            return False

        # --- En-boy oranı kontrolü ---
        # İnsan vücudu dikey yapıdadır: tipik oran h/w ≈ 2.0-2.5
        # min_aspect_ratio=1.3 → kısmen eğilmiş/oturan kişiler dahil
        # max_aspect_ratio=3.5 → aşırı ince/uzun nesneler (direkler) hariç
        if det.w == 0:
            return False  # Sıfıra bölme koruması
        ratio = det.h / det.w
        return self._config.min_aspect_ratio <= ratio <= self._config.max_aspect_ratio

    def _is_valid_aspect_ratio(self, det: Detection) -> bool:
        """
        Tespitteki bounding box'ın yaya oranına uygun olup olmadığını kontrol eder.
        Yayalar genellikle dikey yapıdadır (yükseklik/genişlik > 1.2).

        NOT: Bu metot _is_valid_detection içine entegre edilmiştir.
        Geriye dönük uyumluluk için korunmuştur.
        """
        if det.w == 0:
            return False
        ratio = det.h / det.w
        return self._config.min_aspect_ratio <= ratio <= self._config.max_aspect_ratio

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """
        Non-Maximum Suppression (NMS) uygular.

        NMS Algoritması:
            1. Tüm kutuları güven skoruna göre sırala (yüksekten düşüğe)
            2. En yüksek güvenli kutuyu al → sonuç listesine ekle
            3. Bu kutuyla yüksek IoU'ya sahip diğer kutuları ele
            4. Kalan kutular için adım 2'ye dön
            5. Hiç kutu kalmayana kadar devam et

        IoU (Intersection over Union):
            İki kutunun kesişim alanı / birleşim alanı
            IoU > nms_threshold → kutular "aynı nesne" kabul edilir

        NOT: OpenCV'nin cv2.dnn.NMSBoxes fonksiyonu kullanılır —
        verimli C++ implementasyonudur.

        Args:
            detections: Güven ve boyut filtrelerini geçmiş tespitler.

        Returns:
            NMS sonrası kalan tespitler (çakışmalar elenmiş).
        """
        # Detection listesini numpy array'lere dönüştür
        # (OpenCV NMS bu formatı bekler)
        boxes = np.array(
            [[d.x, d.y, d.w, d.h] for d in detections], dtype=np.int32
        )
        confidences = np.array(
            [d.confidence for d in detections], dtype=np.float32
        )

        # OpenCV NMS çağrısı
        # bboxes: [x, y, w, h] formatında kutu listesi
        # scores: Her kutunun güven skoru
        # score_threshold: Bu altındakiler doğrudan elenir
        # nms_threshold: IoU eşiği (bu üstünde çakışanlar elenir)
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=confidences.tolist(),
            score_threshold=self._config.confidence_threshold,
            nms_threshold=self._config.nms_threshold,
        )

        # NMS hiç kutu bırakmadıysa boş liste döndür
        if len(indices) == 0:
            return []

        # indices farklı OpenCV sürümlerinde farklı boyutta olabilir
        # (eski sürümler: [[0], [2], [5]], yeni sürümler: [0, 2, 5])
        # flatten() ile her durumda 1D array elde ederiz
        indices = np.array(indices).flatten()

        # Yalnızca NMS'ten geçen tespitleri döndür
        return [detections[i] for i in indices]
