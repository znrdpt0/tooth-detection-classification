from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn
import numpy as np
import cv2
import time
from src.services.inference import DentalPredictor
from src.services.image_utils import validate_image

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Sistem Başlatılıyor...")
    
    predictor = DentalPredictor() 
    predictor.load_models()
    
    models["predictor"] = predictor
    yield
    print("🛑 Sistem Kapanıyor...")
    models.clear()

app = FastAPI(
    title="Dental AI Diagnosis API",
    version="2.0",
    description="Quadrant -> Tooth -> Disease Pipeline",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "active", "service": "Dental AI v2.0"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    # 1. Dosya Tipi Kontrolü
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Sadece JPEG veya PNG formatı desteklenir.")

    start_time = time.time()

    try:
        # 2. Resmi Okuma
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        
        # Resmi doğrula
        img = validate_image(nparr)

        # 3. Tahmin (Pipeline)
        predictor = models["predictor"]
        results = predictor.predict(img)

        process_time = round(time.time() - start_time, 3)

        # 4. Yanıt Oluşturma
        return {
            "success": True,
            "process_time": process_time,
            "detected_teeth_count": len(results), # Sadece hasta dişlerin sayısı
            "results": results 
        }

    except Exception as e:
        print(f"HATA: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=True)
    #docker caching deploy testing