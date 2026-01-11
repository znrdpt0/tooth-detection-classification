import os
import glob
import cv2
import random
import numpy as np
from tqdm import tqdm
from augment_stage3 import get_augmentation

DATA_DIR = "../data/processed/stage3_classifier/train"

TARGET_COUNT = 1000 

def clean_old_augmentations():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Klasör bulunamadı: {DATA_DIR}")
        return

    total_removed = 0
    for class_name in os.listdir(DATA_DIR):
        class_path = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_path): continue
        
        # Remove old augmentations
        for f in glob.glob(os.path.join(class_path, "aug_*.png")):
            try:
                os.remove(f)
                total_removed += 1
            except OSError as e:
                print(f"Error removing {f}: {e}")

def balance_classes():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Klasör bulunamadı: {DATA_DIR}")
        return

    # cleaning
    clean_old_augmentations()
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    print(f"⚖️ Classes Balancing (Target: {TARGET_COUNT}, Albumentations)...")
    
    for cls in classes:
        class_dir = os.path.join(DATA_DIR, cls)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg')) and not f.startswith('aug_')]
        count = len(images)
        
        
        if count >= TARGET_COUNT:
            continue
        
        needed = TARGET_COUNT - count
        print(f"   ➕ {needed} adet yeni veri üretilecek...")
        
        if count == 0:
            continue

        generated = 0
        pbar = tqdm(total=needed, desc=f"   Üretiliyor ({cls})")
        
        while generated < needed:
            # Rastgele bir orijinal resim seç
            src_img_name = random.choice(images)
            src_img_path = os.path.join(class_dir, src_img_name)
            img = cv2.imread(src_img_path)
            
            if img is None: continue
            
            # --- Advanced Augmentation (Albumentations) ---
            try:
                aug_img = get_augmentation(img)
            except Exception as e:
                print(f"Augmentation Hatası: {e}")
                continue
            
            # Kaydet: aug_{sayi}_{orijinal_isim}
            new_name = f"aug_{generated}_{os.path.splitext(src_img_name)[0]}.png"
            cv2.imwrite(os.path.join(class_dir, new_name), aug_img)
            
            generated += 1
            pbar.update(1)
            
        pbar.close()


if __name__ == "__main__":
    balance_classes()