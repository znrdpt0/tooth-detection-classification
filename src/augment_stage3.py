import cv2
import albumentations as A
import numpy as np

def get_augmentation(image):
    """
    Refined augmentation for dental classification.
    Removes severe distortions (color noise).
    Keeps: Geometric, CLAHE, Texture/Contrast.
    """
    transform = A.Compose([
        # 1. Geometric (Flip, Rotate, ShiftScaleRotate) - Moderate
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        
        # 2. Geometric Distortion (Grid/Elastic) - Mild
        A.OneOf([
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=30, p=0.5),
        ], p=0.3),

        # 3. Contrast & Texture (CLAHE - crucial for X-ray)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0), # Always apply or high prob
        
        # 4. Brightness/Contrast - Mild
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        
        # Note: MixUp/Mosaic are typically applied during batch loading in training loop 
        # or require mixing two images. For single image function, we focus on shape/texture.
    ])

    return transform(image=image)['image']
