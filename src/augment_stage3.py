import cv2
import albumentations as A

def get_augmentation(image):    

    transform = A.Compose([
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=1, sigma=50, alpha_affine=50), # Elastik bükme
            A.GridDistortion(p=0.5, num_steps=5, distort_limit=0.1),       # Izgara bükme
        ], p=0.7), 
        
        A.Rotate(limit=15, p=0.7),    
        A.HorizontalFlip(p=0.5),
        
        # 2. Renk ve Gürültü
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.5), # Karıncalanma
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.5), 
        ], p=0.5),

        A.OneOf([
            A.RandomBrightnessContrast(p=0.5), 
        ], p=0.5),
        
        # 3. Veri Kaybı Simülasyonu
        A.CoarseDropout(max_holes=1, max_height=16, max_width=16, p=0.2),
    ])

    # Dönüştürme işlemini yap
    result = transform(image=image)
    return result['image']
