import os
import io
import base64
import logging
import cv2
import numpy as np
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DeepSeaAI")

# Load environment variables
load_dotenv()

app = FastAPI(title="DeepSea AI Production", version="2.1.0")

# Security: CORS Policy
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
MODEL_PATH = os.getenv("MODEL_PATH", "../models/best.pt")
STATIC_DIR = os.getenv("STATIC_DIR", "../client/dist")

# Load model globally
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        logger.info(f"YOLOv8 model loaded from {MODEL_PATH}")
    else:
        logger.error(f"Model file not found at {MODEL_PATH}")
        model = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

@app.get("/health")
def health_check():
    return {"status": "healthy" if model else "unhealthy"}

@app.post("/api/predict")
def predict(file: UploadFile = File(...)):
    """API endpoint for YOLOv8 inference."""
    import time
    start_time = time.time()
    
    if not model:
        logger.error("Predict called but model not loaded.")
        raise HTTPException(status_code=503, detail="AI Service is currently unavailable.")

    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Invalid file type.")

    try:
        # Using sync read for sync function
        contents = file.file.read()
        logger.info(f"File received: {file.filename}, Size: {len(contents)} bytes")
        
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Inference
        predict_start = time.time()
        results = model.predict(image, conf=0.25, verbose=False)
        predict_end = time.time()
        logger.info(f"Inference completed in {predict_end - predict_start:.3f}s")
        
        detections = []
        img_str = ""

        if results:
            res = results[0]
            # Process annotated image
            render_start = time.time()
            im_array = res.plot()
            im_rgb = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(im_rgb)
            
            buffered = io.BytesIO()
            im_pil.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            render_end = time.time()
            logger.info(f"Image rendering/encoding completed in {render_end - render_start:.3f}s")

            # Process metadata
            for box in res.boxes:
                detections.append({
                    "species": res.names[int(box.cls[0])],
                    "confidence": float(box.conf[0])
                })

        total_time = time.time() - start_time
        logger.info(f"Total prediction request time: {total_time:.3f}s")

        return {
            "success": True,
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_str}" if img_str else None
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

# Serve Frontend static files
if os.path.exists(STATIC_DIR):
    logger.info(f"Serving frontend from {STATIC_DIR}")
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    logger.warning(f"Static directory {STATIC_DIR} not found. Frontend will not be served.")

# Catch-all for SPA routing (React)
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Frontend not found."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
