from ultralytics import YOLO
import torch

def main ():
    yaml_file = '../../configs/stage2_enumeration.yaml'
    project_dir = "../../models"

    model = YOLO('yolov8m.pt')

    model.train(
        data = yaml_file,
        epochs = 100,
        imgsz =640,
        batch = 8,
        device = 'mps',
        project = project_dir,
        name = 'stage2_m_640',
        exist_ok = True,

        patience = 15,
        save = True,
        cos_lr = True,
        optimizer = 'auto',
        lr0 = 0.001,
        dropout = 0.1,

        augment = True,
        close_mosaic = 10,
        verbose = True
    )
main()            