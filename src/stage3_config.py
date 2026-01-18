
# Common configuration for Stage 3 scripts to avoid circular imports

datasets = [
    {
        "split" : "train",
        "img_dir" : "../data/raw/train/training_data/quadrant-enumeration-disease/xrays",
        "json_path" : "../data/raw/train/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
    },
    {
        "split" : "val",
        "img_dir" : "../data/raw/val/validation_data/quadrant_enumeration_disease/xrays",
        "json_path" : "../data/raw/validation_triple.json"
    }
] 

OUTPUT_DIR = "../data/processed/stage3_classifier"

DISEASE_MAP = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical_Lesion",
    3: "Deep_Caries"
}
