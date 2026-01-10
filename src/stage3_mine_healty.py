import json, os, cv2
from ultralytics import YOLO
from stage2_prepare import apply_clahe
from stage3_prepare import datasets, OUTPUT_DIR



MODEL_PATH = "../models/stage2_m_640/weights/best.pt" 

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

def mine_healthy_teeth():

    model = YOLO(MODEL_PATH)
    
    total_healthy = 0

    for info in datasets:
        split = info["split"]
        img_dir = info["img_dir"]
        json_path = info["json_path"]
        
        
        if not os.path.exists(json_path):
            print("JSON Bulunamadı")
            continue

        with open(json_path, 'r') as f:
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

        save_dir = os.path.join(OUTPUT_DIR, split, "Healthy")
        os.makedirs(save_dir, exist_ok=True)

        valid_extensions = ('.PNG')
        img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(valid_extensions)]
        

        for img_path in img_files:
            file_name = os.path.basename(img_path)
            
            try:
                results = model.predict(img_path, verbose=False, conf=0.5)
            except Exception as e:
                # Bozuk resim vs varsa atla
                continue

            if not results: continue
            
            # Görüntüyü oku
            img = cv2.imread(img_path)
            if img is None: continue
            
            current_img_id = None
            for k, v in images_map.items():
                if v == file_name:
                    current_img_id = k
                    break
            
            known_diseases = gt_boxes_map.get(current_img_id, [])

            for i, box in enumerate(results[0].boxes.xyxy.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box[:4])
                pred_box = [x1, y1, x2, y2]
                
                # Çakışma Kontrolü (IoU)
                is_sick = False
                for disease_box in known_diseases:
                    if calculate_iou(pred_box, disease_box) > 0.3:
                        is_sick = True
                        break
                
                # SADECE SAĞLAMSA KAYDET
                if not is_sick:
                    h_img, w_img = img.shape[:2]
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(w_img, x2); y2 = min(h_img, y2)

                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0: continue
                    
                    crop = apply_clahe(crop)
                    
                    name_without_ext = os.path.splitext(file_name)[0]
                    save_name = f"{name_without_ext}_h_{i}.png"
                    save_path = os.path.join(save_dir, save_name)
                    
                    cv2.imwrite(save_path, crop)
                    total_healthy += 1

    print(f"Bitti! Toplam {total_healthy} adet Healthy diş klasörlere eklendi.")

if __name__ == "__main__":
    mine_healthy_teeth()