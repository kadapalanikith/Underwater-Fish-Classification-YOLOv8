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

# Load model - ensures it looks in the current directory
MODEL_PATH = "best.pt" 
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
    
    # Annotated Image
    im_array = res.plot()
    im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(im_rgb, cv2.COLOR_RGB2BGR))
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
    # Hugging Face Spaces use port 7860
    uvicorn.run(app, host="0.0.0.0", port=7860)
