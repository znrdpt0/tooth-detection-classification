import os
import json
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer

# --- COLAB ve T4 ---
data = '/content/dataset/stage3_classifier'
weight = '/content/dataset/stage3_classifier/class_weights.json'

class_weights = []

def load_weights():
    global class_weights
    if os.path.exists(weight):
        with open(weight, 'r') as f:
            data = json.load(f)
            
            sequential_classes = sorted(data.keys())
            class_weights = [data[k] for k in sequential_classes]
    else:
        print("Ağırlık dosyası bulunamadı.")

class trainer_with_weight(ClassificationTrainer):
    def get_criterion(self, split):
        criterion = super().get_criterion(split)
        if split == 'train' and len(class_weights) > 0:
            device = self.device
            weights_tensor = torch.tensor(class_weights, device=device).float()
            criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        return criterion

def main():
    load_weights()
    model = YOLO('yolov8s-cls.pt')
    
    print(f"Eğitim Başlıyor: {model}")

    train_settings = dict(
        model = model,
        data = data,
        epochs = 100,
        imgsz = 224,
        batch = 64,
        device = 0,
        workers = 8,
        project = '/content/drive/MyDrive/ToothDetectionModels',
        name = 'stage3_optimum',
        exist_ok = True,
        patience = 15,
        dropout = 0.2,
        augment = True,
        degrees = 15.0,
        translate = 0.1,
        scale = 0.2,
        fliplr = 0.5,
        erasing = 0.1,
        optimizer = 'AdamW',
        lr0 = 0.001
    )

    try:
        if len(class_weights) > 0:
            trainer = trainer_with_weight(overrides=train_settings)
            trainer.train()
        else:
            model.train(**train_settings)
            
        print("✅ Eğitim Başarıyla Tamamlandı.")
        
    except Exception as e:
        print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()
