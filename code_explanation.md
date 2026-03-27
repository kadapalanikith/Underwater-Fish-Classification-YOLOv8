# 🌊 DeepSea AI — Code Explanation
### Minor Project Review — Implementation Walkthrough
**Project:** Underwater Fish Classification Using YOLOv8 | **Model File:** `best.pt`

---

## 1. High-Level Code Overview

The project is a **full-stack web application** split into two distinct parts:

```
fish_offline_app/
├── server/
│   ├── main.py              ← The entire FastAPI backend (AI lives here)
│   └── requirements.txt     ← Python dependencies
├── client/
│   └── src/
│       ├── App.jsx           ← Router — wires all pages together
│       ├── Navbar.jsx        ← Top navigation bar
│       ├── Footer.jsx        ← Footer component
│       ├── index.css         ← Full design system (colors, layouts, pipeline UI)
│       └── pages/
│           ├── HomePage.jsx  ← Landing page, stats, how-to guide
│           ├── DemoPage.jsx  ← The actual AI inference UI (most important)
│           ├── ResearchPage.jsx ← Research paper viewer + metrics
│           └── ContactPage.jsx  ← Team info + contact form
└── models/
    └── best.pt               ← Trained YOLOv8 weights (this IS the AI brain)
```

> There are only **2 main layers**: a Python AI server and a React web frontend. They talk to each other via one single API endpoint: `POST /api/predict`.

---

## 2. Execution Flow

Here is exactly what happens from the moment a user opens the app to the moment they see results:

```
[1] User opens http://localhost:5173 in browser
         ↓
[2] React app loads (App.jsx → routes to HomePage)
         ↓
[3] User clicks "Demo" → DemoPage.jsx renders
         ↓
[4] User uploads an underwater fish image
    → handleImageChange() fires
    → image stored in state, preview URL created
    → Pipeline Step 0 "Input" lights up (blue)
         ↓
[5] User clicks "Process with YOLOv8"
    → handlePredict() fires
    → Two things start AT THE SAME TIME (Promise.allSettled):
         a) Animated pipeline UI: Input → Preprocess → YOLOv8 → Detect
         b) HTTP POST to FastAPI: sends image as multipart/form-data
         ↓
[6] FastAPI (main.py) receives the image
    → file.read() gets raw bytes
    → PIL opens the image
    → model.predict() runs YOLOv8 inference
    → results contain bounding boxes, labels, confidence scores
    → OpenCV draws boxes → image encoded as base64 JPEG
    → JSON response sent back: { detections, annotated_image }
         ↓
[7] Frontend receives response
    → Pipeline moves to Step 4 "Output" → all steps turn green
    → Annotated image displayed on right side
    → Species names + confidence % listed below
```

---

## 3. Model Loading — `server/main.py`

```python
MODEL_PATH = os.getenv("MODEL_PATH", "../models/best.pt")
model = None

@app.on_event("startup")
def load_model():
    global model
    model = YOLO(MODEL_PATH)
```

**Why load at startup, not per request?**

- YOLOv8 model weights (`best.pt`) are ~6MB and take ~1-2 seconds to load into memory
- If we loaded the model on every request, each user would wait 1-2 seconds just for setup
- Loading **once at startup** means every subsequent request uses the already-loaded model
- The `global model` variable is shared across all requests — safe because inference is read-only

> Think of it like opening a book once and leaving it on your desk, rather than going to the library every time you need to check a fact.

---

## 4. Input Handling

### Frontend (DemoPage.jsx)

```javascript
const handleImageChange = (e) => {
  const file = e.target.files[0];
  if (file.size > 10 * 1024 * 1024) { setError('Image exceeds 10 MB'); return; }
  setSelectedImage(file);
  setPreviewUrl(URL.createObjectURL(file)); // Show instant preview
  setPipelineStep(0); // Light up "Input" step
};
```

- Accepts any `image/*` file type (JPEG, PNG, WebP, etc.)
- Validates file size (max 10 MB) immediately in-browser, before sending to server
- Creates a local object URL for instant preview — no server round-trip needed to show the image

### Backend (main.py)

```python
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
```

- FastAPI's `UploadFile` automatically handles multipart form parsing
- `io.BytesIO` converts raw bytes into a file-like object that PIL can open
- `.convert("RGB")` strips any alpha channel (PNG transparency) before passing to YOLOv8

---

## 5. Inference Logic

```python
results = model.predict(image, conf=0.25)
res = results[0]
```

**What happens inside `model.predict()`?**

1. **Resize** — Image is automatically resized to 640×640 (YOLOv8's input size)
2. **Normalize** — Pixel values scaled from 0–255 → 0.0–1.0
3. **Forward Pass** — Image passes through the YOLOv8 neural network layers
4. **Decode Predictions** — Raw grid-cell predictions converted to bounding boxes
5. **NMS (Non-Max Suppression)** — Removes overlapping duplicate boxes, keeps the best one
6. **Confidence Filter** — Only boxes with confidence ≥ 25% (`conf=0.25`) are kept

**What does a detection look like?**

```python
for box in res.boxes:
    species = res.names[int(box.cls[0])]    # e.g., "ClownFish"
    confidence = float(box.conf[0])          # e.g., 0.924 (92.4%)
    # box.xyxy contains the pixel coordinates of the bounding box
```

> In simple terms: YOLOv8 looks at the image, draws imaginary boxes around every fish it sees, and says "I'm 92% sure this is a ClownFish".

---

## 6. Post-Processing

```python
# Draw bounding boxes on the image
im_array = res.plot()  # Returns BGR numpy array with boxes already drawn

# Encode to base64 JPEG for easy JSON transport
_, buffer = cv2.imencode('.jpg', im_array)
img_str = base64.b64encode(buffer).decode()
```

- `res.plot()` uses Ultralytics' built-in rendering — draws colored boxes with labels automatically
- The result is a **NumPy ndarray** (a pixel grid), not a file
- `cv2.imencode('.jpg', ...)` compresses it to JPEG bytes
- `base64.b64encode(...)` converts bytes → text string (so it can travel in JSON)

**Confidence Threshold = 0.25** was chosen as a balance:
- Too high (0.8+) → misses partially visible or small fish
- Too low (0.1) → detects false positives (rocks, coral mistaken for fish)

---

## 7. Output Generation

### What the backend returns:

```json
{
  "success": true,
  "annotated_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "detections": [
    { "species": "ClownFish",  "confidence": 0.924 },
    { "species": "BlueTang",   "confidence": 0.871 }
  ]
}
```

### How the frontend renders it:

```jsx
<img src={result.annotated_image} alt="Annotated" />

{result.detections.map((d, i) => (
  <div key={i}>
    <Fish size={16} /> {d.species}
    <span>{(d.confidence * 100).toFixed(1)}%</span>
  </div>
))}
```

- The `data:image/jpeg;base64,...` string is a **self-contained image** — no file needed, just paste it into an `<img src>` tag
- Detections are rendered as a styled list with a fish icon, species name, and confidence %
- If `detections.length === 0`, a "No fish detected at 25% confidence" message is shown

---

## 8. Key Functions Breakdown

### `handlePredict()` — DemoPage.jsx
- **Purpose:** Orchestrates the entire predict flow from the user's click to result display
- **Inputs:** Uses `selectedImage` from React state
- **What it does:**
  1. Starts the animated pipeline UI (runs in parallel with the API call)
  2. Posts image to `/api/predict`
  3. On success: sets result state, marks all pipeline steps green
  4. On error: shows error message, resets pipeline to Input step

### `predict()` — main.py
- **Purpose:** The core inference endpoint
- **Input:** `file: UploadFile` (multipart image)
- **Output:** JSON with `detections` array and `annotated_image` base64 string

### `load_model()` — main.py
- **Purpose:** Startup hook to load model weights exactly once
- **Input:** Path from environment variable or default `../models/best.pt`
- **Output:** Sets `global model` variable

### `getStepState()` — DemoPage.jsx
- **Purpose:** Drives the pipeline animation visual state machine
- **Input:** `stepIdx, activeIdx, done`
- **Output:** CSS class string: `'idle' | 'active' | 'done'`

---

## 9. Design Decisions

### Why FastAPI instead of Flask?

| Feature | Flask | FastAPI |
|---|---|---|
| Speed | Synchronous | Async (faster I/O) |
| Auto Docs | No | Yes (`/docs`) |
| Validation | Manual | Automatic |
| Type Safety | No | Yes (Pydantic) |

> FastAPI is modern, faster for I/O-heavy work, and auto-generates API docs at `/docs`.

### Why a Decoupled Frontend + Backend?

- Frontend (React) can be hosted on **Vercel** (free, global CDN)
- Backend (Python) can be hosted on **Hugging Face Spaces** or **Render** (free GPU-tier possible)
- They communicate via a single REST endpoint — easy to swap or scale either side independently

### Why YOLOv8 Nano (`yolov8n`)?

- Smallest model variant → runs on **CPU without GPU**
- 91% mAP@50 is still excellent for a 13-class domain-specific task
- Model size is ~6MB — practical for deployment on limited hardware

### Why Base64 Image Encoding?

- Avoids storing temporary files on the server
- Result is a single JSON object — easy to consume by any frontend or mobile app
- Trade-off: base64 is ~33% larger than binary, but acceptable for web use

---

## 10. Error Handling / Edge Cases

| Scenario | What Happens |
|---|---|
| No fish detected | Returns empty `detections: []`, frontend shows "No fish detected at 25% confidence" |
| Invalid file (not image) | PIL raises exception → `HTTPException(400)` returned |
| Model not loaded | Guard `if not model: raise HTTPException(503)` prevents crash |
| Image > 10 MB | Caught in frontend before upload — no server load |
| Request timeout (90s) | Axios timeout triggers → "Request timed out" error shown |
| Network down | `ECONNABORTED` code detected → "Connection error" message shown |

---

## 11. Optimization Choices

- **Model loaded once** → Saves 1-2 seconds per request
- **`conf=0.25` threshold** → Balanced precision/recall for underwater images
- **JPEG encoding at output** → Smaller than PNG, acceptable quality for display
- **`Promise.allSettled()`** → Animation and API call run in parallel for faster perceived response
- **`convert("RGB")`** → Handles PNG alpha channels without crashes
- **Framer Motion animations** → Pure CSS transitions can't handle the complex state machine; FM handles it cleanly

---

## 12. Scalability

The codebase is designed to extend easily:

| Extension | How to Do It |
|---|---|
| Video Streaming | Add `/api/predict_video` endpoint, process frames in a loop |
| More Species | Add labeled data, retrain YOLOv8 with new classes, drop in new `best.pt` |
| GPU Inference | Set `device='cuda'` in `model.predict()` — zero other code changes |
| Multiple Models | Use environment variable `MODEL_PATH` to swap models without code changes |
| Mobile App | React frontend API calls are already REST — any mobile app can consume `/api/predict` |
| Cloud (HF/Render) | `Dockerfile`, `Dockerfile.huggingface`, `Dockerfile.render` are all ready |
| Authentication | Add FastAPI OAuth2 middleware — one file change |

---

*Prepared by: Nikith Kadapalaneni | Minor Project Review 2026*
