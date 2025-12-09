import os
import json
import shutil
import random
from pathlib import Path

#Configuration
BASE_DIR = Path("../data")
QUADRANT_DIR = BASE_DIR / "raw/train/training_data/quadrant"
IMAGE_DIR = QUADRANT_DIR / "xrays"
JSON_PATH = QUADRANT_DIR / "train_quadrant.json"
OUTPUT_DIR = BASE_DIR / "processed/stage1_quadrant"

#Create directories
for split in ['train', 'val']:
    os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok=True)

#Load data
with open(JSON_PATH, 'r') as f:
    data = json.load(f)

#Group annotations
annotations_map = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_map:
        annotations_map[img_id] = []
    annotations_map[img_id].append(ann)

all_images = data['images']
random.seed(42)
random.shuffle(all_images)
split_index = int(len(all_images) * 0.9)

train_images = all_images[:split_index]
val_images = all_images[split_index:]

# Process images
def procces(images, split_name):
    count = 0
    for img_info in images:
        img_id = img_info['id']
        file_name = img_info['file_name']
        
        # Paths
        src_path = f"{IMAGE_DIR}/{file_name}"
        dst_img_path = f"{OUTPUT_DIR}/images/{split_name}/{file_name}"
        dst_label_path = f"{OUTPUT_DIR}/labels/{split_name}/{file_name.replace('.png', '.txt').replace('.jpg', '.txt')}"
        
        #Copy Image
        if not os.path.exists(src_path):
            continue
        shutil.copy(src_path, dst_img_path)

        #Convert to YOLO Format
        yolo_lines = []
        if img_id in annotations_map:
            for ann in annotations_map[img_id]:
                
                x, y, w, h = ann['bbox']
                
                #YOLO Normalized
                x_center = (x + w / 2) / img_info['width']
                y_center = (y + h / 2) / img_info['height']
                w_norm = w / img_info['width']
                h_norm = h / img_info['height']
                
                # Categories: 1,2,3,4 -> YOLO: 0,1,2,3
                class_id = ann['category_id']
                
                yolo_lines.append(f"{class_id} {x_center:.8f} {y_center:.8f} {w_norm:.8f} {h_norm:.8f}")
    
        # 3. Write Label File
        with open(dst_label_path, 'w') as f:
            f.write("\n".join(yolo_lines))
    
        count += 1
    return count

procces(train_images, 'train')
print("train set is ready.")
procces(val_images, 'val')
print("val set is ready.")