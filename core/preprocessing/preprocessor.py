"""
Görüntü Ön-İşleme Modülü
==========================
Tespit öncesi frame hazırlığını yapar. HOG+SVM dedektörünün
doğru çalışabilmesi için görüntü kalitesini iyileştirir.

İşlem Sırası (pipeline olarak sıralı uygulanır):
    1. resize   → Performans için boyut küçültme
    2. denoise  → Sensör gürültüsünü azaltma (GaussianBlur)
    3. CLAHE    → Kontrast iyileştirme (gölgeli alanlar için)
    4. sharpen  → Kenar bilgisini güçlendirme (HOG kenar tabanlıdır)
    5. gray     → (Opsiyonel) Gri tonlama dönüşümü

Her adım konfigürasyondan bağımsız olarak etkinleştirilebilir/devre dışı bırakılabilir.

Neden Bu Adımlar Önemli?
    - HOG dedektörü kenar yönelimlerine (gradient) dayalıdır
    - Bulanık/gürültülü görüntüler → zayıf gradientler → kaçırılan tespitler
    - Düşük kontrast → aydınlatma farklılıkları → tutarsız sonuçlar
    - Ön-işleme bu sorunları dedektöre girmeden ÖNCE çözer
"""

import cv2
import numpy as np

from config.settings import PreprocessConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """
    Frame ön-işleme işlemlerini yönetir.

    Constructor'da ağır nesneler (CLAHE, kernel) bir kez oluşturulur
    ve her frame'de yeniden kullanılır (performans optimizasyonu).
    """

    def __init__(self, config: PreprocessConfig) -> None:
        """
        Preprocessor'u başlatır ve yeniden kullanılacak nesneleri oluşturur.

        Args:
            config: Ön-işleme konfigürasyonu.
        """
        self._config = config

        # Ölçekleme faktörü — resize sırasında hesaplanır,
        # sonradan koordinat dönüşümü için saklanır
        self._scale_factor: float = 1.0

        # --- CLAHE nesnesi ---
        # CLAHE'yi her frame'de yeniden oluşturmak pahalıdır,
        # bu yüzden bir kez oluşturup tekrar kullanırız
        self._clahe: cv2.CLAHE = None
        if config.enable_clahe:
            self._clahe = cv2.createCLAHE(
                clipLimit=config.clahe_clip_limit,   # Kontrast sınırı (aşırı parlamayı önler)
                tileGridSize=config.clahe_grid_size, # Bölge ızgarası (lokal işlem boyutu)
            )

        # --- Keskinleştirme çekirdeği ---
        # Konvolüsyon kerneli bir kez hesaplanır, her frame'de uygulanır
        self._sharpen_kernel: np.ndarray = None
        if config.enable_sharpening:
            self._sharpen_kernel = self._build_sharpen_kernel(
                config.sharpen_strength
            )

        # Başlatma bilgisi logla
        logger.info(
            "Preprocessor başlatıldı | Genişlik: %d | CLAHE: %s | "
            "Keskinleştirme: %s | Gürültü azaltma: %s",
            config.target_width,
            config.enable_clahe,
            config.enable_sharpening,
            config.enable_denoising,
        )

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Frame üzerinde ön-işleme zinciri uygular.

        İşlem sırası önemlidir:
            resize  → Boyut küçültme (sonraki adımları hızlandırır)
            denoise → Gürültü azaltma (CLAHE'den ÖNCE — çünkü CLAHE gürültüyü artırabilir)
            CLAHE   → Kontrast iyileştirme
            sharpen → Keskinleştirme (son adım — önceki adımların bulanıklığını telafi eder)

        Args:
            frame: BGR formatında girdi frame (orijinal boyut).

        Returns:
            İşlenmiş frame (boyutu küçültülmüş olabilir).
        """
        # 1. Boyut küçültme — hem performans hem de HOG pencere boyutu uyumu için
        processed = self._resize(frame)

        # 2. Gürültü azaltma — CLAHE'den önce uygulanmalı
        if self._config.enable_denoising:
            processed = self._denoise(processed)

        # 3. CLAHE — kontrast iyileştirme (gölgeli alanları aydınlatır)
        if self._config.enable_clahe:
            processed = self._apply_clahe(processed)

        # 4. Keskinleştirme — kenar bilgisini güçlendirir
        if self._config.enable_sharpening:
            processed = self._sharpen(processed)

        # 5. Gri tonlama dönüşümü (opsiyonel)
        if self._config.convert_to_gray:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

        return processed

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """
        En-boy oranını koruyarak yeniden boyutlandırır.

        Neden boyut küçültüyoruz?
            - 1920x1080 çözünürlükte HOG taraması çok yavaş (~5 FPS)
            - 640 genişliğe küçültme ile ~25 FPS elde edilir
            - HOG 64x128 pencere boyutu ile çalışır, küçük yayalar
              zaten görünmez olacağından bilgi kaybı minimeldir

        NOT: Ölçekleme faktörü saklanır — tespit koordinatlarını
        orijinal boyuta geri dönüştürmek için gereklidir.
        """
        height, width = frame.shape[:2]

        # Eğer frame zaten hedef genişlikten küçükse, resize yapma
        if width <= self._config.target_width:
            self._scale_factor = 1.0
            return frame

        # Ölçekleme faktörünü hesapla (0-1 arası)
        self._scale_factor = self._config.target_width / width
        new_height = int(height * self._scale_factor)

        # INTER_AREA: Küçültme için en iyi interpolasyon yöntemi
        # (piksel bilgisini korur, moiré efekti önler)
        return cv2.resize(
            frame,
            (self._config.target_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    def _denoise(self, frame: np.ndarray) -> np.ndarray:
        """
        Hafif Gaussian blur ile gürültü azaltma.

        Neden Gaussian Blur?
            - Sensör gürültüsü HOG gradientlerini bozar
            - Küçük kernel (3x3 veya 5x5) kenar bilgisini korurken gürültüyü azaltır
            - Daha güçlü gürültü azaltma (bilaterali filter vb.) çok yavaş olur

        Kernel boyutu kuralı:
            - OpenCV GaussianBlur tek sayı kernel gerektirir (3, 5, 7...)
            - Çift sayı girilirse +1 ile düzeltilir
        """
        k = self._config.denoise_strength
        # Kernel boyutu tek sayı olmalı (OpenCV gereksinimi)
        if k % 2 == 0:
            k += 1
        return cv2.GaussianBlur(frame, (k, k), 0)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast Limited Adaptive Histogram Equalization) uygular.

        Normal histogram eşitleme tüm görüntüye global uygulanır ve
        bazı bölgelerde aşırı parlama yaratır. CLAHE ise:
            - Görüntüyü küçük bölgelere (tile) böler
            - Her bölgeye ayrı ayrı histogram eşitleme uygular
            - clipLimit ile aşırı kontrastı sınırlar

        Neden LAB renk uzayı?
            - BGR'de doğrudan CLAHE uygulamak renkleri bozar
            - LAB uzayında L (aydınlık) kanalına CLAHE uygulanır
            - A (kırmızı-yeşil) ve B (mavi-sarı) kanalları korunur
            - Sonuç: Doğal renklerde iyileştirilmiş kontrast
        """
        # BGR → LAB dönüşümü
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Yalnızca L (aydınlık) kanalına CLAHE uygula
        l_channel = self._clahe.apply(l_channel)

        # Kanalları birleştir ve BGR'ye geri dönüştür
        lab = cv2.merge([l_channel, a_channel, b_channel])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _sharpen(self, frame: np.ndarray) -> np.ndarray:
        """
        Unsharp Mask tabanlı keskinleştirme uygular.

        Neden keskinleştirme?
            - HOG dedektörü kenar yönelimlerine (gradient) dayalıdır
            - Bulanık karelerde gradient zayıfladığından tespit düşer
            - Keskinleştirme kenar bilgisini güçlendirir → daha güçlü HOG özellikleri

        Çekirdek (kernel), 2D konvolüsyon ile her pikseli
        komşularının ağırlıklı ortalaması ile değiştirir.
        """
        return cv2.filter2D(frame, -1, self._sharpen_kernel)

    @staticmethod
    def _build_sharpen_kernel(strength: float) -> np.ndarray:
        """
        Ayarlanabilir keskinleştirme çekirdeği (kernel) oluşturur.

        İki çekirdek arasında interpolasyon yapılır:
            - identity (birim): Orijinal görüntü → strength=0
            - sharpen (keskin): Maksimum keskinleştirme → strength=1

        Matematiksel olarak:
            kernel = identity * (1 - strength) + sharpen * strength

        Args:
            strength: Keskinleştirme gücü (0.0 = etkisiz, 1.0 = maksimum).

        Returns:
            3x3 float32 numpy array (konvolüsyon çekirdeği).
        """
        # Keskinleştirme çekirdeği: Merkez piksel güçlendirilir,
        # komşu pikseller çıkarılır → kenarlar belirginleşir
        base = np.array(
            [[ 0, -1,  0],
             [-1,  5, -1],
             [ 0, -1,  0]], dtype=np.float32
        )
        # Birim çekirdeği: Görüntüyü değiştirmez
        identity = np.array(
            [[0, 0, 0],
             [0, 1, 0],
             [0, 0, 0]], dtype=np.float32
        )
        # Orijinal ile keskin arası yumuşak geçiş (interpolasyon)
        return identity * (1 - strength) + base * strength

    @property
    def scale_factor(self) -> float:
        """
        Son işlemdeki ölçekleme faktörünü döndürür.

        Bu değer, ön-işleme sonrası tespit koordinatlarını
        orijinal frame boyutuna geri dönüştürmek için kullanılır.

        Örnek: scale_factor=0.5 → koordinatlar 2 ile çarpılmalı
        """
        return self._scale_factor
