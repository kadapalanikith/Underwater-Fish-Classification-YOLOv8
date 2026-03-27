# 🌊 Underwater Fish Classification Using YOLOv8
### Minor Project Review — Presentation Document
**Project:** DeepSea AI | **Model:** YOLOv8 (best.pt) | **Team:** Nikith Kadapalaneni

---

## 1. Problem Statement

Underwater environments present some of the most challenging conditions for automated visual recognition:

- **Poor Visibility** — Low ambient light, murky water, and particles scatter camera sensors
- **Motion Blur** — Fish move fast; underwater cameras capture ghosting and trail artifacts
- **Occlusion** — Fish partially hide behind corals, rocks, or each other
- **Color Distortion** — Water absorbs red wavelengths, making images appear greenish-blue
- **Scale Variation** — A clownfish and a surgeonfish can appear in the same frame at very different sizes

> Traditional monitoring methods rely on trained marine biologists physically surveying underwater environments — a process that is expensive, slow, geographically limited, and not scalable.

### Real-World Relevance

| Domain | Application |
|---|---|
| 🔬 Marine Research | Automated species census for biodiversity studies |
| 🐟 Fisheries Management | Population tracking, illegal fishing detection |
| 🌿 Conservation | Health monitoring of coral reef ecosystems |
| 🤿 Aquaculture | Automated fish counting in fish farms |

---

## 2. Objective

**To build an end-to-end, real-time fish species detection and classification system that:**

- Accurately detects **13 tropical fish species** from a single uploaded underwater image
- Draws bounding boxes around each fish with **species label and confidence score**
- Achieves a **mAP@50 ≥ 0.90** on the validation set
- Is deployable as a **web application** accessible from any browser
- Runs efficiently on **CPU hardware** without requiring a GPU server

---

## 3. Proposed Solution

### Why YOLOv8?

**YOLO** stands for **You Only Look Once** — it is a single-pass object detection algorithm. Unlike traditional CNNs that classify a whole image, YOLOv8 simultaneously:

1. Divides the image into a grid
2. Predicts bounding boxes and class labels for every cell in one forward pass
3. Returns results in **real-time** (under 45ms per image)

| Approach | How It Works | Speed | Accuracy |
|---|---|---|---|
| Traditional CNN | Classifies the whole image, no localization | Fast | Low (no boxes) |
| R-CNN / Faster R-CNN | Two-stage: propose regions, then classify | Slow | High |
| **YOLOv8 (Ours)** | Single-pass detection + classification | **Very Fast** | **High** |

> We chose YOLOv8 because it is the state-of-the-art single-stage detector with the best balance of speed and accuracy, a strong Python API (`ultralytics`), and can run on CPU — which matters for accessibility.

### How the Pipeline Works

```
User Uploads Image
        ↓
  [Preprocess] → Resize to 640×640, normalize pixel values
        ↓
  [YOLOv8 Model] → Forward pass through neural network (best.pt)
        ↓
  [Detect] → Non-max suppression, confidence filtering (≥25%)
        ↓
  [Output] → Annotated image + species list + confidence scores
```

---

## 4. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      USER BROWSER                       │
│   React 19 + Vite Frontend (http://localhost:5173)      │
│   Pages: Home | Demo | Research | Contact               │
└───────────────────────┬─────────────────────────────────┘
                        │  HTTP POST /api/predict
                        │  (sends image as multipart form)
┌───────────────────────▼─────────────────────────────────┐
│              FastAPI Backend (http://localhost:8000)     │
│   - Receives image file                                  │
│   - Loads YOLOv8 model once at startup                  │
│   - Runs inference                                       │
│   - Returns JSON: { annotated_image, detections[] }     │
└───────────────────────┬─────────────────────────────────┘
                        │
┌───────────────────────▼───────────┐
│       YOLOv8 Model (best.pt)      │
│   Trained on 2,000+ labeled       │
│   underwater fish images          │
│   13 species | mAP@50: 0.91      │
└───────────────────────────────────┘
```

### Step-by-Step Flow (Easy to Say Out Loud)

1. **User visits the web app** and goes to the Demo page
2. **Uploads an underwater image** (JPEG/PNG, up to 10 MB)
3. The **frontend sends the image** to the FastAPI backend via a REST call
4. The **backend loads the YOLOv8 model** (loaded once at startup, reused for every request)
5. **YOLOv8 runs inference** — predicts bounding boxes, class labels, and confidence scores
6. The backend **annotates the image** using OpenCV and encodes it to base64
7. The **frontend displays** the annotated image alongside a list of detected species

---

## 5. Dataset & Training

### Data Collection

- **Source:** Curated labeled images from **Kaggle** and **Roboflow** underwater fish datasets
- **Volume:** 2,000+ annotated instances across 13 fish classes
- **Format:** YOLO annotation format (`.txt` files with normalized bounding box coordinates)

### Fish Classes (13 Species)

`AngelFish` · `BlueTang` · `ButterflyFish` · `ClownFish` · `GoldFish` · `Gourami` · `MorishIdol` · `PlatyFish` · `RibbonedSweetlips` · `ThreeStripedDamselfish` · `YellowCichlid` · `YellowTang` · `ZebraFish`

### Challenges Addressed During Training

| Challenge | Solution Applied |
|---|---|
| Low-light images | Brightness/contrast augmentation |
| Motion blur | Blur augmentation during training |
| Class imbalance | Weighted sampling and mosaic augmentation |
| Small fish size | Multi-scale training at 640px |

### Training Approach

- **Base Model:** YOLOv8n (nano variant — lightest, CPU-friendly)
- **Transfer Learning:** Fine-tuned from COCO-pretrained weights
- **Epochs:** 10 (5 frozen backbone + 5 full fine-tune)
- **Output:** `best.pt` — the checkpoint with the highest validation mAP

---

## 6. Implementation Details

### Technology Stack

| Component | Technology |
|---|---|
| AI Model | YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| Backend API | FastAPI + Python |
| Image Processing | OpenCV, Pillow |
| Frontend | React 19 + Vite |
| Animation | Framer Motion |
| Containerization | Docker |

### How Inference Works

```python
# 1. Load model once at startup
model = YOLO("models/best.pt")

# 2. Run prediction on incoming image
results = model.predict(image, conf=0.25)

# 3. Draw bounding boxes and extract metadata
annotated = results[0].plot()   # OpenCV image with boxes
detections = [{ species, confidence } for each box]
```

### How Users Interact

1. Open browser → go to `/demo`
2. Upload any JPG/PNG (drag-and-drop supported)
3. Click **"Process with YOLOv8"**
4. View: annotated image + species + confidence %

---

## 7. Key Features & Innovation

- **🔴 Real-Time Detection** — Full inference pipeline completes in under 45ms per image
- **🌊 Underwater-Optimized** — Trained specifically on underwater conditions (blur, low light, occlusion)
- **⚡ Lightweight** — YOLOv8 Nano runs on CPU — no GPU required
- **🌐 Full-Stack Web App** — React frontend + FastAPI backend, accessible from any browser
- **📊 Animated Pipeline UI** — Visual pipeline display (Input → Preprocess → YOLOv8 → Detect → Output) helps users understand the AI process
- **🔌 Open REST API** — `/api/predict` endpoint can be integrated into any client or IoT device
- **🐳 Docker Support** — One-command deployment using Docker

---

## 8. Results & Output

### What the System Produces Per Image

- ✅ **Annotated Image** — Original image with colored bounding boxes drawn over each detected fish
- ✅ **Species Label** — Class name for each detection (e.g., "ClownFish")
- ✅ **Confidence Score** — Probability value (e.g., 92.4%)
- ✅ **Multiple Detections** — Handles multiple fish of different species in the same frame

### Performance Metrics (Validation Set)

| Metric | Value | What It Means |
|---|---|---|
| mAP@50 | **0.91** | Model correctly locates & classifies 91% of fish |
| Precision | **0.86** | 86% of detections are correct |
| Recall | **0.87** | Model finds 87% of all fish in images |
| mAP@50–95 | **0.66** | Strict accuracy across multiple IoU thresholds |
| Inference Time | **≤ 45ms** | Per image on CPU |

---

## 9. Limitations

We're being honest here — every system has constraints:

- **Dataset Size** — 2,000+ images is good for a minor project, but production systems use millions
- **Extreme Visibility** — Very murky water (visibility < 1m) can fool the model
- **Species Scope** — Only 13 fish species; won't recognize anything outside the trained classes
- **Static Input Only** — Current version works on images; real-time video streaming is not yet implemented
- **No Tracking** — The same fish seen in multiple frames is not linked across time

---

## 10. Future Enhancements

- **📹 Real-Time Video** — Process live video streams (RTSP / webcam feed)
- **📱 Mobile App** — React Native or Flutter app for field researchers
- **☁️ Cloud Deployment** — Host on Hugging Face Spaces / Render / AWS
- **🔢 Multi-Species Tracking** — DeepSORT integration for tracking fish across video frames
- **🧬 Species Expansion** — Scale to 50+ species with larger curated datasets
- **📊 Marine Analytics Dashboard** — Historical population trends, per-location heatmaps
- **🌡️ Integration with Sensors** — Combine AI with water quality / temperature sensors

---

## 11. Conclusion

> Underwater fish classification is a genuinely hard computer vision problem.
> By applying **YOLOv8** — the leading single-stage detector — trained on real labeled underwater images, we have built a system that is **fast, accurate, lightweight, and accessible via a modern web interface**.

This project demonstrates the practical application of deep learning in **marine science**, **biodiversity monitoring**, and **aquaculture management**. More importantly, it is fully deployable, open-sourced, and extensible — a real foundation for marine AI research.

**mAP@50 of 0.91 on 13 species. Single image. Under 45ms. Running on CPU.**

That's the power of YOLOv8, applied to the ocean.

---
*Prepared by: Nikith Kadapalaneni | Minor Project Review 2026*
