from ultralytics import YOLO
import torch
import os

def main():
    # PATHS
    # Ensure this matches where you upload the data in Colab
    yaml_file = '/content/tooth-detection-classification/configs/stage2_upgrade.yaml' 
    project_dir = "/content/drive/MyDrive/ToothDetection/models"
    
    # Check for GPU
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training on: {device}")

    # Load Model (Transfer Learning from YOLOv8m)
    model = YOLO('yolov8m.pt')

    # Train
    model.train(
        data = yaml_file,
        epochs = 100,            # Sufficient for convergence
        imgsz = 640,
        batch = 16,             # Colab T4 usually handles 16
        device = device,
        project = project_dir,
        name = 'stage2_m_upgrade_8cls',
        exist_ok = True,

        # Optimization
        patience = 20,          # Stop if no improvement
        save = True,            # Save best checkpoints
        optimizer = 'AdamW',    # Robust optimizer
        lr0 = 0.001,            # Standard YOLO learning rate
        cos_lr = True,
        dropout = 0.1,          # Prevent overfitting

        # Augmentation (Standard YOLO)
        augment = True,
        close_mosaic = 10,      # Disable mosaic in last 10 epochs for precision
        
        verbose = True
    )

if __name__ == "__main__":
    main()
