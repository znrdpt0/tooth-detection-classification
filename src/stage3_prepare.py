import json, os, shutil, cv2
import numpy as np
import random
from pathlib import Path
from stage2_prepare import apply_clahe
from stage3_mine_healty import mine_healthy_teeth
import sys

# Append path to import augment_stage3 if needed locally, though it's in same dir
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from augment_stage3 import get_augmentation
# Will import stage3_balance at the end

from stage3_config import datasets, OUTPUT_DIR, DISEASE_MAP

def setup_directories():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR) # Clean start
    
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/Healthy", exist_ok=True)
        for name in DISEASE_MAP.values():
            os.makedirs(f"{OUTPUT_DIR}/{split}/{name}", exist_ok=True)


def process_dataset(info, buffer):
    split = info["split"]
    img_dir = info["img_dir"]
    json_path = info["json_path"]

    with open(json_path, 'r') as f :
        data = json.load(f)
        
    images_map = {img['id']: img['file_name'] for img in data['images']}
    
    # Process Annotated Diseases
    for ann in data['annotations']:
        img_id = ann['image_id']
        file_name = images_map.get(img_id)
        if not file_name : continue

        src_path =f"{img_dir}/{file_name}"
        if not os.path.exists(src_path): continue
        
        disease_id = ann.get('category_id_3')
        
        # Determine Label
        if disease_id in DISEASE_MAP:
            label = DISEASE_MAP[disease_id]
        else:
            # Skip manual healthy labels to avoid noise, rely on mining
            # or keep them if explicit?
            # User wants controlled healthy count.
            # Let's Skip explicit healthy here and rely on mine_healthy for Consistency?
            # OR keep them and count towards limit.
            # "Sınıf Birleştirme: Canine_Caries -> Caries" is handled by DISEASE_MAP implicitly 
            # (assuming JSON uses same ID for all Caries, or we map them).
            # If JSON has specific IDs for Canine_Caries vs Molar_Caries, we map them here.
            # Current DISEASE_MAP: 1: Caries, 3: Deep_Caries... seems global.
            label = "Healthy" 

        img = cv2.imread(src_path)
        if img is None: continue
        
        # Padding applied: 15% side/top, 20% bottom (Root visibility)
        x, y, w, h = map(int, ann['bbox'])
        h_img, w_img = img.shape[:2]
        
        pad_w = int(w * 0.15)
        pad_h_top = int(h * 0.15)
        pad_h_bot = int(h * 0.20)
        
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h_top)
        x2 = min(w_img, x + w + pad_w)
        y2 = min(h_img, y + h + pad_h_bot)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue

        crop = apply_clahe(crop) 

        # Store in buffer
        if split not in buffer: buffer[split] = {}
        if label not in buffer[split]: buffer[split][label] = []
        
        buffer[split][label].append(crop)

    # Process Mine Healthy
    # User Request: "healthy kodumuz sadece train için ... aynı şekilde val içinde hazılarsın"
    # User Request: "healthy train verisi 2000i geçmesin"
    
    target_cap = 2000 if split == 'train' else 500
    
    print(f"⛏️ Mining Healthy teeth for {split} (Limit: {target_cap})...")
    
    # Check how many we already have from annotations (likely 0 if we skipped, or some)
    existing_healthy = len(buffer[split].get('Healthy', []))
    remaining_quota = target_cap - existing_healthy
    
    if remaining_quota > 0:
        healthy_crops = mine_healthy_teeth(img_dir, json_path, max_count=remaining_quota)
        print(f"   Found {len(healthy_crops)} mined healthy candidates.")
        
        if 'Healthy' not in buffer[split]: buffer[split]['Healthy'] = []
        buffer[split]['Healthy'].extend(healthy_crops)
    else:
        print("   ⚠️ Healthy quota full from annotations. Skipping mining.")


def save_buffered_data(buffer):
    print("\n💾 Saving Data...")
    
    for split, labels in buffer.items():
        print(f"   Processing {split}...")
        
        for label, crops in labels.items():
            final_crops = crops
            
            # Enforce Cap strictly
            if label == "Healthy":
                cap = 2000 if split == 'train' else 500
                if len(final_crops) > cap:
                    print(f"   🔻 Capping Healthy {split} from {len(final_crops)} to {cap}")
                    final_crops = random.sample(final_crops, cap)
            
            # Note: We do NOT augment Deep_Caries/Lesion here.
            # That is the job of stage3_balance.py (Offline Augmentation).
            
            # Save
            save_dir = f"{OUTPUT_DIR}/{split}/{label}"
            os.makedirs(save_dir, exist_ok=True) # Ensure dir exists
            
            for i, img in enumerate(final_crops):
                cv2.imwrite(f"{save_dir}/{i}.png", img)

def main():
    setup_directories()
    
    data_buffer = {} 
    
    for ds in datasets:
        process_dataset(ds, data_buffer)
        
    save_buffered_data(data_buffer)
    
    print("\n⚖️ Running Stage 3 Balance (Offline Augmentation)...")
    import stage3_balance
    stage3_balance.balance_classes()
    
    print(f"\n✅ All Stages Complete. Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()