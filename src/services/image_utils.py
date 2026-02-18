# Dosya: src/services/image_utils.py
import cv2
import numpy as np

def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Röntgen görüntülerinde detayları belirginleştirmek için CLAHE uygular.
    Stage 3 (Hastalık Tespiti) öncesi kullanılması önerilir.
    """
    if image is None: return None
    
    # Görüntü renkli ise LAB dönüşümü yap
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # Gri ise direkt uygula
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(image)

def validate_image(nparr):
    """Gelen byteların geçerli bir resim olup olmadığını kontrol eder."""
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Görüntü verisi bozuk veya okunamadı.")
    return img