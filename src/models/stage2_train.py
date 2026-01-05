from ultralytics import YOLO
import torch

def main ():
    yaml_file = 'stage2_teeth.yaml'

    if not torch.cuda.is_available():
        print("GPU is not available")
    else:
        print("GPU active")

    model = YOLO('yolov8x.pt')

    model.train(
        data = yaml_file,
        epochs = 100,
        imgsz =1280,
        batch = 8,
        device = 0,
        project = 'models',
        name = 'stage2_x_1280',
        exist_ok = True,

        patience = 20,
        save = True,
        cos_lr = True,
        optimizer = 'AdamW',
        lr0 = 0.001,
        dropout = 0.1,
        verbose = True
    )
main()            