import cv2 , os
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from stage2_prepare import apply_clahe

stage1_model_path = "../models/stage1/train/weights/best.pt"
stage2_model_path = "../models/stage2/train/weights/best.pt"
model_s1 = YOLO(stage1_model_path)
model_s2 = YOLO(stage2_model_path)

CLASS_NAMES = {
    0: 'Incisor', 1: 'Canine', 2: 'Premolar', 3: 'Molar',
    4: 'Impacted', 5: 'Caries', 6: 'Periapical Lesion', 7: 'Deep Caries'
}
#NMS (Non-Maximum Suppression):
#Aynı nesneyi temsil eden, üst üste binen kutulardan sadece en güvenilir olanı tutma işlemidir.

def gloabal_NMS (detections, iou_thresh=0.5):
    if len(detections) == 0 :
        return []
    final_detections = []

    class_grouped = {}
    #tespitleri sınıflara göre grupla
    for det in detections:
        cls_id = det['cls']
        if cls_id not in class_grouped:    
            class_grouped[cls_id] = []
        class_grouped[cls_id].append(det)
    # Her sınıf için ayrı ayrı NMS
    for cls_id, class_dets in class_grouped.items():
        boxes = np.array([d['bbox'] for d in class_dets])
        scores = np.array([d['conf'] for d in class_dets])

        # OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=0.0,
            nms_threshold=iou_thresh
        )

        if len(indices) > 0:
            indices = indices.flatten()
            for i in indices:
                final_detections.append(class_dets[i])

    return final_detections

def get_pipeline_predictions(img_path):
    original_img = cv2.imread(img_path)
    if original_img is None:
        raise ValueError(f"Resim Okunamadı : {img_path}")
    
    h_org, w_org = original_img.shape[:2]

    final_detections = []
    #stage1
    results_s1 = model_s1(original_img, verbose = False)[0]

    for box in results_s1.boxes :
        q_x1, q_y1, q_x2, q_y2 = map(int, box.xyxy[0].cpu())
        # Dinamik Padding
        pad = int(0.1 * (q_x2 - q_x1))
        crop_x1 = max(0, q_x1 - pad)
        crop_y1 = max(0, q_y1 - pad)
        crop_x2 = min(w_org, q_x2 + pad)
        crop_y2 = min(h_org, q_y2 + pad)
        
        crop_img = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop_img.size == 0: continue
        
        enhanced_crop = apply_clahe(crop_img)

    #stage2
    results_s2 = model_s2(enhanced_crop, verbose = False)[0]

    for s2_box in results_s2.boxes :
        lx1, ly1, lx2, ly2 = s2_box.xyxy[0].cpu().numpy()
        cls_id = int(s2_box.cls)
        conf = float(s2_box.conf)
            
        gx1 = int(lx1 + crop_x1)
        gy1 = int(ly1 + crop_y1)
        gx2 = int(lx2 + crop_x1)
        gy2 = int(ly2 + crop_y1)
            
        final_detections.append({
            'cls': cls_id,
            'conf': conf,
            'bbox': [gx1, gy1, gx2, gy2]
        })

    # --- CLASS-SPECIFIC NMS ---
    final_detections = gloabal_NMS(final_detections, iou_thresh=0.5)

    return final_detections, original_img