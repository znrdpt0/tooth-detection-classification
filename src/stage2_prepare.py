import json, os, shutil, cv2, random
import numpy as np
from pathlib import Path

JSON_PATH = "../data/raw/train/training_data/quadrant_enumeration/train_quadrant_enumeration.json"
IMG_DIR = "../data/raw/train/training_data/quadrant_enumeration/xrays"

OUTPUT_DIR = "../data/processed/stage2_enumeration"

def setup_directories():
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok= True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok= True)

# CLAHE: Kontrast 
def apply_clahe(image): 
    if len(image.shape) == 3 :
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)     

def get_tooth_class(cat2_id):
    
    tooth_num = int(cat2_id) % 10 # 11->1, 26->6
    
    if tooth_num in [1, 2]: 
        return 0 
    elif tooth_num == 3:     
        return 1 
    elif tooth_num in [4, 5]: 
        return 2 
    elif tooth_num in [6, 7, 8]: 
        return 3 
    
    return None

def main():
    setup_directories()
    
    print(f"📖 JSON okunuyor: {JSON_PATH}")
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Annotation'ları resim ID'sine göre grupla
    img_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns: 
            img_anns[img_id] = []
        img_anns[img_id].append(ann)
    
    # Train / Val Split
    all_images = data['images']
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.9)
    
    datasets = [('train', all_images[:split_idx]), ('val', all_images[split_idx:])]
    
    
    for split_name, img_list in datasets:
        for img_info in img_list:
            file_name = img_info['file_name']
            src_path = os.path.join(IMG_DIR, file_name)
            
            if not os.path.exists(src_path): 
                continue
                
            img = cv2.imread(src_path)
            if img is None: continue
            
            anns = img_anns.get(img_info['id'], [])
            
            # ADIM 1: Dişleri Quadrant ID'sine göre grupla
            quad_groups = {1:[], 2:[], 3:[], 4:[]}
            for ann in anns:
                qid = ann.get('category_id_1') # Quadrant ID
                if qid in quad_groups:
                    quad_groups[qid].append(ann)
            
            # ADIM 2: Her Quadrant'ı Kes ve İşle
            for qid, q_anns in quad_groups.items():
                if not q_anns: continue
                
                # Quadrant sınırlarını dişlerden hesapla
                bboxes = [a['bbox'] for a in q_anns]
                min_x = min([b[0] for b in bboxes])
                min_y = min([b[1] for b in bboxes])
                max_x = max([b[0]+b[2] for b in bboxes])
                max_y = max([b[1]+b[3] for b in bboxes])
                
                #Dinamik padding
                q_width = max_x - min_x
                pad = int(q_width * 0.10) 
                
                h_img, w_img = img.shape[:2]
                qx = max(0, int(min_x - pad))
                qy = max(0, int(min_y - pad))
                end_x = min(w_img, int(max_x + pad))
                end_y = min(h_img, int(max_y + pad))
                
                qw = end_x - qx
                qh = end_y - qy
                
                if qw <= 0 or qh <= 0: continue

                # Crop (Kesme)
                crop = img[qy:end_y, qx:end_x]
                if crop.size == 0: continue
                
                # İyileştirme
                enhanced = apply_clahe(crop)

                fname = f"{os.path.splitext(file_name)[0]}_q{qid}.png"
                save_path = f"{OUTPUT_DIR}/images/{split_name}/{fname}"
                cv2.imwrite(save_path, enhanced)

                # ADIM 3: Label Oluşturma
                valid_lines = []
                for obj in q_anns:
                    cls_id = get_tooth_class(obj['category_id_2'])
                    
                    if cls_id is None: continue 

                    ox, oy, ow, oh = obj['bbox']

                    #(YOLO formatı)
                    nx = ((ox - qx) + ow / 2) / qw
                    ny = ((oy - qy) + oh / 2) / qh
                    nw = ow / qw
                    nh = oh / qh
                    
                    #
                    nx = max(0, min(1, nx))
                    ny = max(0, min(1, ny))
                    nw = max(0, min(1, nw))
                    nh = max(0, min(1, nh))
                    
                    valid_lines.append(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

                if valid_lines:
                    lbl_path = save_path.replace('/images/', '/labels/').replace('.png', '.txt')
                    with open(lbl_path, 'w') as f:
                        f.write("\n".join(valid_lines))

if __name__ == "__main__":
    main()