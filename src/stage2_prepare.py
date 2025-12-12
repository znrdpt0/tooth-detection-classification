import json, os, shutil, cv2, random
import numpy as np
from pathlib import Path

JSON_PATH = "../data/raw/train/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
IMG_DIR = "../data/raw/train/training_data/quadrant-enumeration-disease/xrays"

OUTPUT_DIR = "../data/processed/stage2_teeth"

def setup_directories ():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/images/{split}", exist_ok= True)
        os.makedirs(f"{OUTPUT_DIR}/labels/{split}", exist_ok= True)

#apply clahe 
#BGR -> GRAY(Clahe only works with single-channel ) -> BGR(YOLO works with triple-channel)
def apply_clahe(image): 
    #convert gray
    if len(image.shape) == 3 :
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)     

#classifaciton
def get_classification(cat2_id, cat3_id):
    classes = []
    
    #teeth
    if cat2_id + 1 in [1,2] : classes.append(0) #incisor
    elif cat2_id + 1 == 3     : classes.append(1) #canine
    elif cat2_id + 1 in [4,5] : classes.append(2) #premolar
    elif cat2_id + 1 in [6,7,8] : classes.append(3) #molar


    #diseases
    if cat3_id == 0 : classes.append(4) #impacted
    elif cat3_id == 1 : classes.append(5) #caries
    elif cat3_id == 2 : classes.append(6) #periapical lesion
    elif cat3_id == 3 : classes.append(7) #deep caries

    return classes

def main() :
    
    setup_directories()
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    #grouping annotation with image id
    img_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns: 
            img_anns[img_id] = []
        img_anns[img_id].append(ann)
    
    #train / val
    all_images = data['images']
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.9)
    
    datasets = [('train', all_images[:split_idx]), ('val', all_images[split_idx:])]
    for split_name, img_list in datasets :
        for img_info in img_list:
            file_name = img_info['file_name']
            src_path = os.path.join(IMG_DIR, file_name)
            if not os.path.exists(src_path): continue
            img =cv2.imread(src_path)
            anns = img_anns.get(img_info['id'], [])
            
            #step 1 : grouping teeth with quadrant
            quad_groups ={0:[], 1:[], 2:[], 3:[]}
            for ann in anns:
                qid = ann.get('category_id_1')
                quad_groups[qid].append(ann)
            #step 2 : Cut and process each quarter.
            for qid, q_anns in quad_groups.items():
                if not q_anns: continue
                
                bboxes = [a['bbox'] for a in q_anns]
                min_x = min([b[0] for b in bboxes])
                min_y = min([b[1] for b in bboxes])
                max_x = max([b[0]+b[2] for b in bboxes])
                max_y = max([b[1]+b[3] for b in bboxes])
                
                # Padding
                pad = 50 
                h_img, w_img = img.shape[:2]
                qx = max(0, int(min_x - pad))
                qy = max(0, int(min_y - pad))
                end_x = min(w_img, int(max_x + pad))
                end_y = min(h_img, int(max_y + pad))
                
                qw = end_x - qx
                qh = end_y - qy
                if qw <= 0 or qh <= 0: continue

                crop = img[qy:end_y, qx:end_x]
                if crop.size == 0: continue
                
                enhanced = apply_clahe(crop)

                fname = f"{os.path.splitext(file_name)[0]}_q{qid+1}.jpg"
                save_path = f"{OUTPUT_DIR}/images/{split_name}/{fname}"
                cv2.imwrite(save_path, enhanced)

                # step 3 : create labels
                valid_lines = []
                for obj in q_anns:
                    classes = get_classification(obj['category_id_2'], obj['category_id_3'])
                    ox, oy, ow, oh = obj['bbox']

                    for cls_id in classes:
                        nx = ((ox - qx) + ow / 2) / qw
                        ny = ((oy - qy) + oh / 2) / qh
                        nw = ow / qw
                        nh = oh / qh
                        
                        nx = max(0, min(1, nx))
                        ny = max(0, min(1, ny))
                        nw = max(0, min(1, nw))
                        nh = max(0, min(1, nh))
                        
                        valid_lines.append(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

                if valid_lines:
                    # save folders
                    lbl_path = save_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
                    with open(lbl_path, 'w') as f:
                        f.write("\n".join(valid_lines))

main()



