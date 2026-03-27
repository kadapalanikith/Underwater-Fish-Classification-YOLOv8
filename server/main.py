import os
import io
import base64
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("HF-Backend")

app = FastAPI()

# IMPORTANT: Allow your Vercel URL here
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace with your-app.vercel.app
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model - ensures it looks in the models folder
MODEL_PATH = os.getenv("MODEL_PATH", "../models/best.pt")
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = YOLO(MODEL_PATH)
        logger.info("✅ YOLOv8 Model Loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

@app.get("/")
def home():
    return {"status": "running", "model": "YOLOv8-Fish-Classifier"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    results = model.predict(image, conf=0.25)
    res = results[0]
    
    # Annotated Image (already in BGR from res.plot)
    im_array = res.plot()
    _, buffer = cv2.imencode('.jpg', im_array)
    img_str = base64.b64encode(buffer).decode()
    
    detections = []
    for box in res.boxes:
        detections.append({
            "species": res.names[int(box.cls[0])],
            "confidence": float(box.conf[0])
        })
        
    return {
        "success": True,
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{img_str}"
    }

if __name__ == "__main__":
    import uvicorn
    # Default to 8000 for local/Render, but respect environmental variable
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
