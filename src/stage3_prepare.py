import json, os, shutil, cv2
import numpy as np
from pathlib import Path
from stage2_prepare import apply_clahe

datasets = [
    {
        "split" : "train",
        "img_dir" : "../data/raw/train/training_data/quadrant-enumeration-disease/xrays",
        "json_path" : "../data/raw/train/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
    },

    {
        "split" : "val",
        "img_dir" : "../data/raw/val/validation_data/quadrant_enumeration_disease/xrays",
        "json_path" : "../data/raw/validation_triple.json"
    }
] 

OUTPUT_DIR = "../data/processed/stage3_classifier"

DISEASE_MAP = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical_Lesion",
    3: "Deep_Caries"
}

def setup_directories ():
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/Healthy", exist_ok=True)
        for name in DISEASE_MAP.values():
            os.makedirs(f"{OUTPUT_DIR}/{split}/{name}", exist_ok=True)

def process_dataset(info):
    split = info["split"]
    img_dir = info["img_dir"]
    json_path = info["json_path"]

    with open(json_path, 'r') as f :
        data = json.load(f)
        
    images_map = {img['id']: img['file_name'] for img in data['images']}
    
    stats = {k: 0 for k in DISEASE_MAP.values()}
    stats['Healthy'] = 0

    for ann in data['annotations']:
        img_id = ann['image_id']
        file_name = images_map.get(img_id)
        if not file_name : continue

        src_path =f"{img_dir}/{file_name}"
        if not os.path.exists(src_path): 
            continue
        
        disease_id = ann.get('category_id_3')

        if disease_id in DISEASE_MAP:
            label = DISEASE_MAP[disease_id]
        else:
            label = "Healthy"

        img = cv2.imread(src_path)
        if img is None: continue
        
        x, y, w, h = map(int, ann['bbox'])
        h_img, w_img = img.shape[:2]
        x = max(0, x); y = max(0, y)
        w = min(w, w_img - x); h = min(h, h_img - y)
        
        crop = img[y:y+h, x:x+w]
        if crop.size == 0: continue

        crop = apply_clahe(crop)

        save_name = f"{Path(file_name).stem}_{ann['id']}.png"
        save_path = f"{OUTPUT_DIR}/{split}/{label}/{save_name}"
        
        cv2.imwrite(save_path, crop)
        stats[label] += 1

    print(f"📊 {split.upper()} Raporu:")
    for k, v in stats.items():
        print(f"   - {k}: {v}")

def main():
    setup_directories()
    for ds in datasets:
        process_dataset(ds)
    print(f"\n✅ İşlem Tamam. Çıktı: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()