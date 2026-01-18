import os
import glob
import shutil
import cv2


DATA_DIR = "../data/processed/stage3_classifier/train" # Config style path
TARGET_COUNT = 2000 

def balance_classes():
    """
    Stage 3 Balance (Offline Augmentation).
    Only augments MINORITY classes to reach TARGET_COUNT.
    Skips Majority classes.
    """
    if not os.path.exists(DATA_DIR):
        print(f"❌ Directory not found: {DATA_DIR}")
        return

    print("⚖️ Balancing Stage: Focused Offline Augmentation...")
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    for cls in classes:
        class_dir = os.path.join(DATA_DIR, cls)
        
        # 1. Clean old augmentations (aug_ prefix)
        removed_count = 0
        for f in glob.glob(os.path.join(class_dir, "aug_*.png")):
            try:
                os.remove(f)
                removed_count += 1
            except: pass
            
        if removed_count > 0:
            print(f"   🧹 Removed {removed_count} legacy synthetic images from {cls}")

        # 2. Check Count
        # Only original files remaining now
        images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        
        # 3. Strategy: Only augment minority
        if count >= TARGET_COUNT:
            print(f"   ✅ {cls}: {count} images (Sufficient). Skipping.")
            continue
        else:
            needed = TARGET_COUNT - count
            print(f"   ⚠️ {cls}: {count} images (Low). Generating {needed} synthetic...")
            
            # Simple Augmentation Loop
            # We cycle through original images and apply aug until filled
            import random
            from augment_stage3 import get_augmentation
            
            generated = 0
            while generated < needed:
                src_name = random.choice(images)
                src_path = os.path.join(class_dir, src_name)
                
                img = cv2.imread(src_path)
                if img is None: continue
                
                # Apply Augmentation
                aug_img = get_augmentation(img)
                
                # Save
                new_name = f"aug_{generated}_{src_name}"
                cv2.imwrite(os.path.join(class_dir, new_name), aug_img)
                generated += 1
            
            print(f"      ✨ Generated {generated} images. Total: {count + generated}")

    print("\n✅ Balancing Complete.")

if __name__ == "__main__":
    balance_classes()