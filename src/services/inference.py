import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from src.services.image_utils import apply_clahe

# --- AYARLAR ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATHS = {
    "stage1": BASE_DIR / "models/weights/stage1_quadrant.pt",
    "stage2": BASE_DIR / "models/weights/stage2_teeth.pt",
    "stage3": BASE_DIR / "models/weights/stage3_classifier.pt"
}

# Dinamik Thresholds
THRESHOLDS = {
    "Periapical_Lesion": 0.20,
    "Deep_Caries": 0.30,
    "Caries": 0.45,
    "Impacted": 0.60
}

class DentalPredictor:
    def __init__(self):
        self.models = {}
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚙️ Çalışma Ortamı: {self.device}")

    def load_models(self):
        print("📥 Models uploading...")
        try:
            for key, path in MODEL_PATHS.items():
                if path.exists():
                    self.models[key] = YOLO(path)
                    print(f"✅ {key} uploaded.")
                else:
                    raise FileNotFoundError(f"❌ Model not found: {path}")
        except Exception as e:
            print(f"KRİTİK HATA: {e}")
            raise e

    def predict(self, original_img: np.ndarray):
        if not self.models:
            raise RuntimeError("models not uploaded!")

        results_data = []
        h_img, w_img = original_img.shape[:2]

        # --- ADIM 1: Quadrant ---
        s1_results = self.models['stage1'].predict(original_img, conf=0.6, verbose=False)[0]

        for q_box in s1_results.boxes:
            qx1, qy1, qx2, qy2 = map(int, q_box.xyxy[0].cpu().numpy())
            q_label = s1_results.names[int(q_box.cls[0])]

            # Padding
            q_w = qx2 - qx1
            q_h = qy2 - qy1
            pad_x = int(q_w * 0.10)
            pad_y = int(q_h * 0.10)
            crop_x1 = max(0, qx1 - pad_x)
            crop_y1 = max(0, qy1 - pad_y)
            crop_x2 = min(w_img, qx2 + pad_x)
            crop_y2 = min(h_img, qy2 + pad_y)

            q_crop = original_img[crop_y1:crop_y2, crop_x1:crop_x2]
            if q_crop.size == 0: continue

            # --- ADIM 2: Diş Tespiti ---
            s2_results = self.models['stage2'].predict(q_crop, conf=0.4, verbose=False)[0]

            for t_box in s2_results.boxes:
                cx1, cy1, cx2, cy2 = map(int, t_box.xyxy[0].cpu().numpy())
                t_cls_id = int(t_box.cls[0].item())
                t_name = s2_results.names[t_cls_id]

                # Global Koordinat
                gx1 = cx1 + crop_x1
                gy1 = cy1 + crop_y1
                gx2 = cx2 + crop_x1
                gy2 = cy2 + crop_y1
                
                # --- ADIM 3: Hastalık Tespiti ---
                t_crop = original_img[gy1:gy2, gx1:gx2]
                if t_crop.size == 0: continue

                # Stage 3 Tahmin
                s3_res = self.models['stage3'].predict(t_crop, verbose=False)[0]
                
                probs = s3_res.probs
                top1_id = probs.top1
                top1_conf = probs.top1conf.item()
                disease_name = s3_res.names[top1_id]

                # Healhy ignore
                if disease_name == "Healthy":
                    continue

                # Threshold Kontrolü
                required_conf = THRESHOLDS.get(disease_name, 0.5)

                if top1_conf >= required_conf:
                    tooth_data = {
                        "quadrant": q_label,
                        "tooth_type": t_name,
                        "bbox": [gx1, gy1, gx2, gy2],
                        "diagnosis": disease_name,
                        "confidence": round(top1_conf, 2)
                    }
                    results_data.append(tooth_data)

        return results_data