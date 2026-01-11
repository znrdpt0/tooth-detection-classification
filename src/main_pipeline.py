import cv2
import os
import sys
import time
from ultralytics import YOLO

# --- YAPILANDIRMA (CONFIG) ---
MODEL_PATHS = {
    "stage1": "../models/stage1_n_640/train/weights/best.pt",
    "stage2": "../models/stage2_m_640/weights/best.pt",
    "stage3": "../models/stage3_m_224_cls/weights/best.pt"
}


COLORS = {
    "Caries": (0, 165, 255),        # orange
    "Deep_Caries": (0, 0, 255),     # red
    "Impacted": (255, 0, 255),      # purple
    "Periapical_Lesion": (0, 255, 255) # yellow
}

def load_models():
    models = {}
    try:
        for key, path in MODEL_PATHS.items():
            if os.path.exists(path):
                models[key] = YOLO(path)
            else:
                print(f" Hata: Model dosyası eksik -> {path}")
                sys.exit(1)
    except Exception as e:
        print(f"Beklenmeyen Hata: {e}")
        sys.exit(1)
        
    return models

def check_containment(inner_box, outer_box):
    """Dişin merkezi, Quadrant kutusunun içinde mi?"""
    ix_center = (inner_box[0] + inner_box[2]) / 2
    iy_center = (inner_box[1] + inner_box[3]) / 2
    ox1, oy1, ox2, oy2 = outer_box
    return (ox1 < ix_center < ox2) and (oy1 < iy_center < oy2)

def analyze_image(image_path, models):
    """
    CORE FUNCTION: Resmi analiz eder ve saf veri döndürür.
    Çizim yapmaz, sadece hesaplar. API bu fonksiyonu kullanacak.
    """
    img = cv2.imread(image_path)
    if img is None: return None, []

    h_img, w_img = img.shape[:2]
    detected_pathologies = [] 

    # --- Stage 1: Quadrant  ---
    q_results = models["stage1"].predict(img, conf=0.5, verbose=False)
    q_boxes = q_results[0].boxes
    
    if len(q_boxes) == 0:
        print("⚠️ Uyarı: Quadrant bulunamadı.")
        return img, []

    quadrants = []
    for box in q_boxes:
        coords = box.xyxy[0].cpu().numpy()
        label = q_results[0].names[int(box.cls[0])].replace("quadrant_", "Q").replace("Quadrant ", "Q")
        quadrants.append((label, coords))

    # --- Stage 2: Diş Tespiti ---
    t_results = models["stage2"].predict(img, conf=0.05, verbose=False)
    teeth_boxes = t_results[0].boxes

    # --- Stage 3: Analiz Döngüsü ---
    for box in teeth_boxes:
        tx1, ty1, tx2, ty2 = map(int, box.xyxy[0].cpu().numpy())
        tooth_box = [tx1, ty1, tx2, ty2]
        
        # 3.1: Hiyerarşi Kontrolü
        assigned_q = None
        for q_lbl, q_coords in quadrants:
            if check_containment(tooth_box, q_coords):
                assigned_q = q_lbl
                break
        
        if assigned_q is None: continue 

        # 3.2: Diş Türü
        tooth_type = t_results[0].names[int(box.cls[0])]

        # 3.3: Hastalık Kontrolü
        tx1_c, ty1_c = max(0, tx1), max(0, ty1)
        tx2_c, ty2_c = min(w_img, tx2), min(h_img, ty2)
        crop = img[ty1_c:ty2_c, tx1_c:tx2_c]
        if crop.size == 0: continue

        d_results = models["stage3"].predict(crop, verbose=False)
        disease_id = d_results[0].probs.top1
        disease_name = d_results[0].names[disease_id]
        conf = d_results[0].probs.top1conf.item()

        # FİLTRE: Sağlıklı dişleri listeye ekleme
        if disease_name == "Healthy": continue

        # BULGUYU KAYDET (Saf Veri)
        detected_pathologies.append({
            "quadrant": assigned_q,
            "tooth_type": tooth_type,
            "disease": disease_name,
            "confidence": conf,
            "bbox": [tx1, ty1, tx2, ty2] # Koordinatları da sakla
        })

    return img, detected_pathologies

def visualize_results(img, pathologies):
    """
    HELPER FUNCTION: Analiz verisini alır ve resmin üzerine çizer.
    Sadece görselleştirme içindir.
    """
    final_img = img.copy()
    
    for item in pathologies:
        x1, y1, x2, y2 = item["bbox"]
        color = COLORS.get(item["disease"], (0, 0, 255))
        
        # Etiket: "Q1 Molar | Caries %98"
        label = f"{item['quadrant']} {item['tooth_type']} | {item['disease']} %{int(item['confidence']*100)}"
        
        # Kutu ve Yazı
        cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
        
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(final_img, (x1, y1 - 20), (x1 + w + 10, y1), color, -1)
        cv2.putText(final_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return final_img
