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

# Sadece hastalıkları çizmek için renk paleti
COLORS = {
    "Caries": (0, 165, 255),        # Turuncu
    "Deep_Caries": (0, 0, 255),     # Kırmızı
    "Impacted": (255, 0, 255),      # Mor
    "Periapical_Lesion": (0, 255, 255) # Sarı
}

def load_models():
    """Modelleri yükler ve bir sözlük olarak döndürür."""
    print("🤖 Sistem Başlatılıyor...")
    models = {}
    try:
        for key, path in MODEL_PATHS.items():
            if os.path.exists(path):
                models[key] = YOLO(path)
            else:
                print(f"❌ Kritik Hata: Model dosyası eksik -> {path}")
                sys.exit(1)
    except Exception as e:
        print(f"❌ Beklenmeyen Hata: {e}")
        sys.exit(1)
        
    print("✅ Yapay Zeka Motoru Hazır!")
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
    detected_pathologies = [] # Sonuçları burada toplayacağız

    # --- ADIM 1: Quadrant (Stage 1) ---
    q_results = models["stage1"].predict(img, conf=0.5, verbose=False)
    q_boxes = q_results[0].boxes
    
    if len(q_boxes) == 0:
        print("⚠️ Uyarı: Quadrant bulunamadı.")
        return img, [] # Boş liste dön

    quadrants = []
    for box in q_boxes:
        coords = box.xyxy[0].cpu().numpy()
        label = q_results[0].names[int(box.cls[0])].replace("quadrant_", "Q").replace("Quadrant ", "Q")
        quadrants.append((label, coords))

    # --- ADIM 2: Diş Tespiti (Stage 2) ---
    t_results = models["stage2"].predict(img, conf=0.25, verbose=False)
    teeth_boxes = t_results[0].boxes

    # --- ADIM 3: Analiz Döngüsü (Stage 3) ---
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

# --- MAIN BLOCK (Terminalden çalışınca burası çalışır) ---
if __name__ == "__main__":
    # Test Resmi (Bunu değiştirebilir veya argüman olarak alabilirsin)
    TEST_IMAGE = "../data/raw/val/validation_data/quadrant_enumeration_disease/xrays/val_1.png"
    OUTPUT_FILE = "final_output.jpg"

    # 1. Modelleri Yükle
    ai_models = load_models()
    
    if os.path.exists(TEST_IMAGE):
        print(f"🔍 Analiz Ediliyor: {TEST_IMAGE}")
        start = time.time()
        
        # 2. Analiz Et (Logic)
        original_img, results = analyze_image(TEST_IMAGE, ai_models)
        
        # 3. Sonuçları Bas (Report)
        print(f"\n📊 TESPİT EDİLEN PATOLOJİLER ({len(results)} adet):")
        print("-" * 40)
        for res in results:
            print(f"🦷 {res['quadrant']} {res['tooth_type']:<10} -> {res['disease']} (%{int(res['confidence']*100)})")
        print("-" * 40)
        
        # 4. Çiz ve Kaydet (Visualization)
        if original_img is not None:
            painted_img = visualize_results(original_img, results)
            cv2.imwrite(OUTPUT_FILE, painted_img)
            print(f"✅ Görsel Kaydedildi: {OUTPUT_FILE}")
            
        print(f"⏱️ Toplam Süre: {time.time() - start:.2f} sn")
            
    else:
        print(f"❌ Test resmi bulunamadı: {TEST_IMAGE}")