from ultralytics import YOLO
import torch
import os

def main():
    
    data_yaml = '../configs/stage1_quadrant.yaml'
    model = YOLO("yolov8n.pt")
    project_dir = '../models'

    results = model.train(
        data = data_yaml,
        epochs = 50,
        imgsz = 640,
        batch = 8,
        device = 'mps',
        project = project_dir,
        exist_ok = True,
        patience = 10,
        verbose = True
    )

main()    