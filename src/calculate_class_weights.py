import os
import json
import numpy as np

# Paths
DATA_DIR = '../data/processed/stage3_classifier/train'
OUTPUT_FILE = '../data/processed/stage3_classifier/class_weights.json'

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory not found -> {DATA_DIR}")
        return

    # Get class names
    classes = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    class_counts = {} 
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(DATA_DIR, class_name)
        
        # Count images
        files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))] 
        count = len(files)
        
        class_counts[class_name] = count
        total_images += count
        print(f" - {class_name}: {count} images")

    # Weight Calculation Formula:
    # Weight = Total Images / (Number of Classes * Images in Class)
    
    weights = {}
    num_classes = len(classes)
    
    for class_name, count in class_counts.items():
        if count == 0:
            weight = 0
        else:
            weight = total_images / (num_classes * count)
            
        weights[class_name] = weight

    print("\nCalculated Weights:")
    print(json.dumps(weights, indent=4))
    
    # Save as JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(weights, f, indent=4)
        
    print(f"\nFile saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
