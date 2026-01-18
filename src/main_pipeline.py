import cv2
import os
import sys
import time
from ultralytics import YOLO

# --- YAPILANDIRMA (CONFIG) ---
# --- CONFIG ---
MODEL_PATHS = {
    "stage1": "../models/stage1_n_640/train/weights/best.pt",
    "stage2": "../models/stage2_m_640_upgrade/weights/best.pt",
    "stage3": "../models/stage3_final_v2/weights/best.pt"
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
                print(f" Error: Model missing -> {path}")
                sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}")
        sys.exit(1)
    return models

def get_fdi_number(quadrant_label, class_id):
    """
    Maps Class ID (0-7) + Quadrant (Q1-4) -> FDI Tooth Number (11-48)
    Class 0 = Central Incisor (#1)
    Class 7 = Third Molar (#8)
    """
    try:
        # Extract Quadrant Number (e.g., 'Q1' -> 1, 'Quadrant 3' -> 3)
        q_num = int(''.join(filter(str.isdigit, quadrant_label)))
        
        tooth_offset = class_id 
        fdi_num = f"{q_num}{tooth_offset}"
        
        # Name lookup for display (0=Null, 1=Central, ...)
        names = ["Null", "Central Incisor", "Lateral Incisor", "Canine", "1st Premolar", "2nd Premolar", "1st Molar", "2nd Molar", "3rd Molar"]
        name = names[class_id]
        
        return fdi_num, name
    except:
        return "Unknown", "Unknown"

def analyze_image(image_path, models):
    img = cv2.imread(image_path)
    if img is None: return None, []

    h_img, w_img = img.shape[:2]
    detected_pathologies = [] 

    # --- Stage 1: Quadrants ---
    # We lower conf to ensure we find quadrants. Missing a quadrant = missing 8 teeth.
    q_results = models["stage1"].predict(img, conf=0.25, verbose=False)
    q_boxes = q_results[0].boxes
    
    if len(q_boxes) == 0:
        print("⚠️ Warning: No quadrants found.")
        return img, []

    # Store Logic: List of (label, [x1,y1,x2,y2])
    quadrants = []
    for box in q_boxes:
        coords = list(map(int, box.xyxy[0].cpu().numpy()))
        label = q_results[0].names[int(box.cls[0])].replace("quadrant_", "Q").replace("Quadrant ", "Q")
        quadrants.append((label, coords))

    # --- Stage 2: Teeth Detection (Per Quadrant) ---
    all_teeth = []
    
    for q_label, q_box in quadrants:
        qx1, qy1, qx2, qy2 = q_box
        
        # ADD PADDING (Match training logic: 10%)
        q_w = qx2 - qx1
        pad = int(q_w * 0.10)
        
        crop_x1 = max(0, qx1 - pad)
        crop_y1 = max(0, qy1 - pad)
        crop_x2 = min(w_img, qx2 + pad)
        crop_y2 = min(h_img, qy2 + pad)
        
        q_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if q_crop.size == 0: continue
        
        # Predict on Crop
        t_results = models["stage2"].predict(q_crop, conf=0.25, verbose=False)
        
        for t_box in t_results[0].boxes:
            # Crop Coords
            cx1, cy1, cx2, cy2 = map(int, t_box.xyxy[0].cpu().numpy())
            cls_id = int(t_box.cls[0])
            
            # Map back to Global Image Coords
            gx1 = cx1 + crop_x1
            gy1 = cy1 + crop_y1
            gx2 = cx2 + crop_x1
            gy2 = cy2 + crop_y1
            
            # Get Precise Tooth Name
            fdi_num, tooth_name = get_fdi_number(q_label, cls_id)
            
            all_teeth.append({
                'bbox': [gx1, gy1, gx2, gy2],
                'fdi': fdi_num,
                'name': tooth_name,
                'quadrant': q_label
            })

    # --- Stage 3: Disease Classification ---
    for tooth in all_teeth:
        tx1, ty1, tx2, ty2 = tooth['bbox']
        
        # Crop Tooth from Original Image
        # Clamp coordinates
        tx1, ty1 = max(0, tx1), max(0, ty1)
        tx2, ty2 = min(w_img, tx2), min(h_img, ty2)
        
        t_crop = img[ty1:ty2, tx1:tx2]
        if t_crop.size == 0: continue

        d_results = models["stage3"].predict(t_crop, verbose=False)
        
        # Classification Result
        disease_id = d_results[0].probs.top1
        disease_name = d_results[0].names[disease_id]
        conf = d_results[0].probs.top1conf.item()

        # Filter Healthy
        if disease_name == "Healthy": continue

        detected_pathologies.append({
            "quadrant": tooth['quadrant'],
            "tooth_type": f"{tooth['fdi']} ({tooth['name']})", # Display Name: "11 (Central Incisor)"
            "disease": disease_name,
            "confidence": conf,
            "bbox": [tx1, ty1, tx2, ty2]
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
        
        # Label: "Q1 16 (1st Molar) | Caries %98"
        label = f"{item['tooth_type']} | {item['disease']} %{int(item['confidence']*100)}"
        
        cv2.rectangle(final_img, (x1, y1), (x2, y2), color, 2)
        
        # Dynamic Text placement
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_txt = y1 - 10 if y1 - 25 > 0 else y1 + 20
        
        cv2.rectangle(final_img, (x1, y_txt - 15), (x1 + w + 10, y_txt + 5), color, -1)
        cv2.putText(final_img, label, (x1 + 5, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
    return final_img
