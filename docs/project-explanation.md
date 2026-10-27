# 🌊 Underwater Fish Classification Using YOLOv8
### Minor Project Review — Presentation Document
**Project:** DeepSea AI | **Model:** YOLOv8 (best.pt) | **Team:** Nikith Kadapalaneni

> **Validation note (2026-10-28):** This presentation contains historical project descriptions. Historical notebook validation metrics: precision 0.865, recall 0.874, mAP@50 0.908 and mAP@50–95 0.666; only 12/13 classes have validation instances. A new local re-evaluation of the saved checkpoint recorded mAP@50 0.841 / mAP@50–95 0.636 on a 399-image leakage-filtered test subset (0.840 / 0.622 on 697 full-test images); the test split is not fully independent. A labeled enhancement comparison found no supported improvement; only the mAP@50–95 degradations for gray-world and fusion survived Holm correction. A zero-shot OBSEA check recorded class-agnostic mAP@50 0.036, not 13-class species accuracy. Real-video evaluation was run but failed to produce tracks or counts. The included research PDFs are not evidence of publication or peer review.

---

## Current evidence update (2026-10-28)

> This section supersedes the older validation note above where they conflict.

- Current test suite: **57 passed in 11.30 s** in `.venv` (Python 3.11.9). Tests validate code paths and schemas, not fish accuracy.
- Frontend: `npm run build` and `npm run check:routes` pass. The route check is an SSR smoke test.
- Held-out `test_clean`: 399 images / 639 instances; mAP@50 **0.841** (95% CI 0.823-0.880), mAP@50-95 **0.636** (95% CI 0.615-0.694). Leakage-filtered, not source-independent.
- `test_full`: 697 / 1,129; mAP@50 **0.840** (95% CI 0.826-0.874), mAP@50-95 **0.622** (95% CI 0.604-0.670). Not fully independent.
- `valid_full`: 698 / 1,193; mAP@50 **0.910**, mAP@50-95 **0.658**; AngelFish has no validation instances.
- Enhancement: CLAHE was approximately neutral. Only the mAP@50-95 degradations for gray-world and fusion were supported after Holm correction. No enhancement improvement was supported.
- OBSEA external check (CC BY 4.0, DOI 10.5281/zenodo.14888440): 618 images / 5,369 fish instances, class-agnostic zero-shot, P 0.124, R 0.094, mAP@50 **0.036**, mAP@50-95 **0.017**. This is not 13-class species accuracy and is not universal generalization.
- RT-DETR: the saved checkpoint is COCO-pretrained, 80 classes, and not fish fine-tuned. The latency result is timing-only. A fine-tuning plan and opt-in script exist, but no RT-DETR training or fish-accuracy evaluation was run.
- Tracking: no real fish video is present. The smoke test and API test use synthetic frames; persistent IDs and counts are unvalidated.

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

- Runs YOLOv8 detection for the configured fish classes on an uploaded image; classwise performance is not established for every class
- Draws bounding boxes around each fish with **species label and confidence score**
- Records both the historical validation log and a new local held-out re-evaluation; the test split is not fully independent
- Is deployable as a **web application** accessible from any browser
- Runs efficiently on **CPU hardware** without requiring a GPU server

---

## 3. Proposed Solution

### Why YOLOv8?

**YOLO** stands for **You Only Look Once** — it is a single-pass object detection algorithm. Unlike traditional CNNs that classify a whole image, YOLOv8 simultaneously:

1. Divides the image into a grid
2. Predicts bounding boxes and class labels for every cell in one forward pass
3. Returns detections in a single forward pass; measured CPU inference was about 45–61 ms per image in this workspace, depending on protocol

| Approach | How It Works | Speed | Accuracy |
|---|---|---|---|
| Traditional CNN | Classifies the whole image, no localization | Fast | Low (no boxes) |
| R-CNN / Faster R-CNN | Two-stage: propose regions, then classify | Slow | High |
| **YOLOv8 (Ours)** | Single-pass detection + classification | **Very Fast** | **High** |

> YOLOv8 was selected for the existing implementation because it provides a single-stage detector and a usable Ultralytics API. This project has not established that it is state-of-the-art or the best speed/accuracy trade-off for this dataset.

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
│   8,229 images / 13 classes      │
│   underwater fish images          │
│   12/13 validation classes       │
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
- **Volume:** 8,229 images (6,834 train / 698 valid / 697 test); 14,144 / 1,193 / 1,129 instances after polygon-to-box conversion
- **Format:** YOLO-format `.txt` labels; most rows are boxes, and some are polygons (converted to enclosing boxes for evaluation)

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
- **Epochs:** 10 (from the notebook training log); a staged freeze schedule is not verified in this audit
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

- **🔴 CPU Detection** — A 30-image CPU run measured about 61 ms mean YOLOv8 inference per image; timings vary with hardware and settings
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
| mAP@50 | **0.908** | Original validation log; 12 classes represented |
| Precision | **0.865** | Original validation log; not test performance |
| Recall | **0.874** | Original validation log; not test performance |
| mAP@50–95 | **0.666** | Original validation log; not test performance |
| Held-out test mAP@50 (leakage-filtered) | **0.841** | 399 images / 639 instances; not fully independent |
| Held-out test mAP@50–95 (leakage-filtered) | **0.636** | Same subset; several low-support classes |
| Held-out test mAP@50 (full test) | **0.840** | 697 images / 1,129 instances; not fully independent |
| Held-out test mAP@50–95 (full test) | **0.622** | Same full test |
| Inference Time | **60.7 ms mean** | 30 test images, CPU, imgsz 640, conf 0.25; timing only |

---

## 9. Limitations

We're being honest here — every system has constraints:

- **Dataset evaluation** — 8,229 images were audited; the test split has exact duplicates and source-stem overlaps with training data, and test_clean is a leakage-filtered subset, not fully independent
- **Extreme Visibility** — Very murky water (visibility < 1m) can fool the model
- **Species Scope** — Only 13 fish species; won't recognize anything outside the trained classes
- **Video validation** — tracking code exists, but only a synthetic rectangle smoke test is recorded; fish ID persistence and counting are not validated on real footage

---

## 10. Future Enhancements

- **📹 Real-Time Video** — Process live video streams (RTSP / webcam feed)
- **📱 Mobile App** — React Native or Flutter app for field researchers
- **☁️ Cloud Deployment** — Host on Hugging Face Spaces / Render / AWS
- **🔢 Tracking follow-up** — Evaluate real-video tracking and counting against annotated footage before presenting accuracy claims
- **🧬 Species Expansion** — Scale to 50+ species with larger curated datasets
- **📊 Marine Analytics Dashboard** — Historical population trends, per-location heatmaps
- **🌡️ Integration with Sensors** — Combine AI with water quality / temperature sensors

---

## 11. Conclusion

> Underwater fish classification is a genuinely hard computer vision problem.
> The repository contains a YOLOv8 fish detection demonstration with historical validation logs, a new local held-out re-evaluation, and one preliminary class-agnostic external-domain check. The local test split is not fully independent; OBSEA zero-shot detection was very low, and real-video counting accuracy remains unvalidated. On a labeled subset, CLAHE was approximately neutral while gray-world and fusion reduced accuracy.

This project demonstrates the practical application of deep learning in **marine science**, **biodiversity monitoring**, and **aquaculture management**. More importantly, it is fully deployable, open-sourced, and extensible — a real foundation for marine AI research.

**Historical validation mAP@50: 0.908 (12 classes represented). Local held-out re-evaluation: mAP@50 0.841 on 399 leakage-filtered test images (0.840 on 697 full-test images). CPU inference: 60.7 ms mean on 30 test images. These are not fully independent results.**

That's the power of YOLOv8, applied to the ocean.

---
*Prepared by: Nikith Kadapalaneni | Minor Project Review 2026*
