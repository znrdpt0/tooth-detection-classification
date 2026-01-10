import os, cv2, random
import numpy as np
from pathlib import Path
from tqdm import tqdm


DATA_DIR = "../data/processed/stage3_classifier/train"

target_count = 2000

def augment_image(image):
    rows, cols, _ = image.shape
    choice = random.randint(0, 3)

    #flip(aynalama)
    if choice == 0:
        return cv2.flip(image, 1)
    
    #rotation
    elif choice == 1:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)
    
    #Brightness
    elif choice == 2:
        value = random.uniform(0.7, 1.3)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = hsv[:, :, 2] * value
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    #Noise
    elif choice == 3:
        gauss = np.random.normal(0, 0.1**0.5, image.size)
        gauss = gauss.reshape(image.shape[0], image.shape[1], image.shape[2]).astype('uint8')
        # Gürültüyü çok hafif ekle
        noise_img = cv2.addWeighted(image, 0.95, gauss, 0.05, 0)
        return noise_img
    
def balance_classes():
    if not os.path.exists(DATA_DIR):
        print(f"❌ Klasör bulunamadı: {DATA_DIR}")
        return

    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    print(f"⚖️ Sınıflar dengeleniyor (Hedef: {target_count})...")
    
    for cls in classes:
        class_dir = os.path.join(DATA_DIR, cls)
        images = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        
        print(f"\n📂 Sınıf: {cls} | Mevcut: {count}")
        
        if count >= target_count:
            print(f"   ✅ Yeterli sayıya sahip. (Atlanıyor)")
            continue
            
        # Eksik miktar
        needed = target_count - count
        print(f"   ➕ {needed} yeni resim üretilecek...")
        
        # Döngüyle üret
        generated = 0

        pbar = tqdm(total=needed, desc=f"   Üretiliyor ({cls})")        
        
        while generated < needed:
            # Rastgele bir orijinal resim seç
            src_img_name = random.choice(images)
            src_img_path = os.path.join(class_dir, src_img_name)
            img = cv2.imread(src_img_path)
            
            if img is None: continue
            
            # Augmentasyon yap
            aug_img = augment_image(img)
            
            # Kaydet: aug_{sayi}_{orijinal_isim}
            new_name = f"aug_{generated}_{src_img_name}"
            cv2.imwrite(os.path.join(class_dir, new_name), aug_img)
            
            generated += 1
            pbar.update(1)
            
        pbar.close()

    print("\n✅ Dengeleme Tamamlandı!")

if __name__ == "__main__":
    balance_classes()    