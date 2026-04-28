import os
import io
import asyncio
import base64
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# Load .env file so PORT, MODEL_PATH, etc. are available
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Fish-Backend")

MODEL_PATH = os.getenv("MODEL_PATH", "../models/best.pt")
model = None


# ── Lifespan (replaces deprecated @app.on_event) ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    try:
        model = YOLO(MODEL_PATH)
        logger.info("✅ YOLOv8 Model Loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
    yield  # app runs here
    model = None


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

    # Annotated image (BGR from res.plot → encode to JPEG → base64)
    im_array = res.plot()
    _, buffer = cv2.imencode(".jpg", im_array)
    img_str = base64.b64encode(buffer).decode()

    detections = []
    for box in res.boxes:
        detections.append({
            "species": res.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
        })

    return {
        "success": True,
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{img_str}",
    }


if __name__ == "__main__":
    import uvicorn

    # ── Fix: Windows + Python 3.10 asyncio event loop crash ───────────
    # Without this, uvicorn starts and immediately shuts down on Windows.
    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
