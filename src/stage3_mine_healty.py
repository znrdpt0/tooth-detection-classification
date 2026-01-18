import json, os, cv2
from ultralytics import YOLO
from stage2_prepare import apply_clahe
from stage3_config import datasets, OUTPUT_DIR



MODEL_PATH_S1 = "../models/stage1_n_640/train/weights/best.pt"
MODEL_PATH_S2 = "../models/stage2_m_640_upgrade/weights/best.pt" 

def calculate_iou(box1, box2):
    """
    İki kutu arasındaki örtüşme oranını (IoU) hesaplar.
    box: [x1, y1, x2, y2] formatında olmalı.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0

def mine_healthy_teeth(target_img_dir, target_json_path, max_count=None):
    print(f"   ⛏️ Mining from: {target_img_dir}")
    if max_count:
        print(f"   🎯 Target Limit: {max_count}")

    model_s1 = YOLO(MODEL_PATH_S1)
    model_s2 = YOLO(MODEL_PATH_S2)
    
    collected_crops = []

    if not os.path.exists(target_json_path):
        print("   ❌ JSON Bulunamadı")
        return []

    with open(target_json_path, 'r') as f:
        data = json.load(f)
        
    # Ground Truth Kutularını Hazırla
    gt_boxes_map = {}
    images_map = {img['id']: img['file_name'] for img in data['images']}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        x, y, w, h = ann['bbox']
        box = [x, y, x + w, y + h] # [x1, y1, x2, y2]
        
        if img_id not in gt_boxes_map:
            gt_boxes_map[img_id] = []
        gt_boxes_map[img_id].append(box)

    valid_extensions = ('.PNG', '.png', '.jpg', '.jpeg')
    img_files = [os.path.join(target_img_dir, f) for f in os.listdir(target_img_dir) if f.lower().endswith(valid_extensions)]
    
    # Shuffle processing order to get random healthy samples if we hit limit early
    import random
    random.shuffle(img_files) 

    for img_path in img_files:
        if max_count and len(collected_crops) >= max_count:
            break

        file_name = os.path.basename(img_path)
        
        # Read Image
        img = cv2.imread(img_path)
        if img is None: continue
        h_img, w_img = img.shape[:2]
        
        current_img_id = None
        for k, v in images_map.items():
            if v == file_name:
                current_img_id = k
                break
        
        known_diseases = gt_boxes_map.get(current_img_id, [])
        
        # 1. Detect Quadrants (Stage 1)
        res1 = model_s1.predict(img, verbose=False, conf=0.25)
        if not res1: continue
        s1_res = res1[0]
        
        quadrants = []
        for box in s1_res.boxes:
            coords = list(map(int, box.xyxy[0].cpu().numpy()))
            quadrants.append(coords)

        # 2. Iterate Quadrants -> Stage 2 -> Map Back
        detected_teeth_global = []

        for q_box in quadrants:
            qx1, qy1, qx2, qy2 = q_box
            
            # Padding (10%)
            q_w = qx2 - qx1
            pad = int(q_w * 0.10)
            crop_x1 = max(0, qx1 - pad)
            crop_y1 = max(0, qy1 - pad)
            crop_x2 = min(w_img, qx2 + pad)
            crop_y2 = min(h_img, qy2 + pad)
            
            q_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if q_crop.size == 0: continue
            
            # Stage 2 on Crop
            res2 = model_s2.predict(q_crop, verbose=False, conf=0.4)
            if not res2: continue
            s2_res = res2[0]
            
            for t_box in s2_res.boxes:
                cx1, cy1, cx2, cy2 = map(int, t_box.xyxy[0].cpu().numpy())
                
                # Map to Global
                gx1 = cx1 + crop_x1
                gy1 = cy1 + crop_y1
                gx2 = cx2 + crop_x1
                gy2 = cy2 + crop_y1
                
                detected_teeth_global.append([gx1, gy1, gx2, gy2])

        # 3. Check for Disease Overlap (Global Coords)
        for i, pred_box in enumerate(detected_teeth_global):
            if max_count and len(collected_crops) >= max_count:
                break
                
            x1, y1, x2, y2 = pred_box
            
            # Çakışma Kontrolü (STRICT MODE)
            is_sick = False
            for disease_box in known_diseases:
                # IoU yerine Intersection > 0 kontrolü
                x_overlap = max(0, min(x2, disease_box[2]) - max(x1, disease_box[0]))
                y_overlap = max(0, min(y2, disease_box[3]) - max(y1, disease_box[1]))
                if x_overlap * y_overlap > 0:
                    is_sick = True
                    break
            
            # SADECE SAĞLAMSA Ekle
            if not is_sick:
                # Crop with 20% Bottom Padding preference (User Request)?
                # The user asked for "Genişletilmiş Crop (Padding): Tüm sınıflara o bahsettiğimiz %20 Alt Padding"
                # But here we are just mining raw. We should apply consistent padding logic.
                # Actually, better to apply the SAME logic as stage3_prepare here.
                # Let's keep it standard here (maybe 10% is hardcoded here). 
                # Let's update this to 20% padding to match main prep if possible, 
                # or just extract raw-ish and let prepare handle? 
                # Wait, this function does the cropping. So I should update padding here too.
                
                gw = x2 - x1
                gh = y2 - y1
                pad_w = int(gw * 0.15)
                # Extra padding at bottom for roots? User said "20% Alt Padding".
                # Standard padding 15% all around + extra bottom?
                # Let's do 15% all around for now as established in prepare, or 20% as requested.
                # User said: "%20 Alt Padding".
                pad_h_top = int(gh * 0.15)
                pad_h_bot = int(gh * 0.20)
                
                tx1 = max(0, x1 - pad_w)
                ty1 = max(0, y1 - pad_h_top)
                tx2 = min(w_img, x2 + pad_w)
                ty2 = min(h_img, y2 + pad_h_bot)

                tooth_crop = img[ty1:ty2, tx1:tx2]
                if tooth_crop.size == 0: continue
                
                tooth_crop = apply_clahe(tooth_crop)
                collected_crops.append(tooth_crop)

    print(f"   ✅ Collected {len(collected_crops)} healthy crops.")
    return collected_crops

if __name__ == "__main__":
    # Test block
    if len(datasets) > 0:
        mine_healthy_teeth(datasets[0]["img_dir"], datasets[0]["json_path"])