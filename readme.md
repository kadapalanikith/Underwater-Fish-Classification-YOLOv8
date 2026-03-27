# 🌊 DeepSea AI — Underwater Fish Classification
### YOLOv8 · FastAPI · React 19 · Docker

[![Model](https://img.shields.io/badge/YOLO-v8-blue.svg)](https://ultralytics.com)
[![Frontend](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev)
[![Backend](https://img.shields.io/badge/FastAPI-0.110-05998b.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Academic-orange.svg)](#license)

**DeepSea AI** is an end-to-end computer vision web application that detects and classifies **13 underwater fish species** in real time using a custom-trained YOLOv8 model (`best.pt`).

---

## 🗂️ Project Structure

```
fish_offline_app/
├── client/                  # React 19 + Vite Frontend
│   ├── src/
│   │   ├── App.jsx           # Router — Home | Demo | Research | Contact
│   │   ├── pages/
│   │   │   ├── DemoPage.jsx  ← AI inference UI (main feature)
│   │   │   ├── HomePage.jsx  ← Landing page
│   │   │   ├── ResearchPage.jsx ← Paper + metrics
│   │   │   └── ContactPage.jsx  ← Team info
│   │   └── index.css        # Design system
│   ├── public/
│   │   └── research/
│   │       └── RE_Paper.pdf  # Research paper (served statically)
│   └── .env                 # Frontend environment variables
├── server/
│   ├── main.py              # FastAPI inference server
│   └── requirements.txt     # Python dependencies
├── models/
│   └── best.pt              # YOLOv8 trained weights (mAP@50: 0.91)
├── Dockerfile               # Unified build (for self-hosted)
├── Dockerfile.render        # Backend-only — Render.com
├── Dockerfile.huggingface   # Backend-only — Hugging Face Spaces
├── explanation.md           # Project explanation for review committee
└── code_explanation.md      # Code walkthrough for review committee
```

---

## 🐟 Fish Species (13 Classes)

`AngelFish` · `BlueTang` · `ButterflyFish` · `ClownFish` · `GoldFish` · `Gourami` · `MorishIdol` · `PlatyFish` · `RibbonedSweetlips` · `ThreeStripedDamselfish` · `YellowCichlid` · `YellowTang` · `ZebraFish`

---

## 💻 How to Run Locally (Step-by-Step)

### Prerequisites

Make sure you have these installed before starting:

| Tool | Version | Check with |
|---|---|---|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |
| Git | Any | `git --version` |

---

### Step 1 — Clone the Repository

```bash
git clone https://github.com/kadapalanikith/Underwater-Fish-Classification-YOLOv8.git
cd Underwater-Fish-Classification-YOLOv8
```

---

### Step 2 — Start the Backend (Python / FastAPI)

Open a terminal and run:

```bash
cd server
pip install -r requirements.txt
python main.py
```

> ✅ You should see:
> ```
> INFO:     Uvicorn running on http://0.0.0.0:8000
> INFO:HF-Backend:✅ YOLOv8 Model Loaded
> ```

**The backend is now running at:** `http://localhost:8000`

- Visit `http://localhost:8000` → should return `{"status":"running","model":"YOLOv8-Fish-Classifier"}`
- Visit `http://localhost:8000/docs` → interactive API documentation

> **Note:** First run may take a few minutes to download PyTorch and Ultralytics.

---

### Step 3 — Configure Frontend Environment

The frontend needs to know where the backend is. Open `client/.env`:

```env
VITE_API_URL=http://localhost:8000
```

This is already set correctly for local development. ✅ No changes needed.

---

### Step 4 — Start the Frontend (React / Vite)

Open a **new terminal** (keep the backend running):

```bash
cd client
npm install
npm run dev
```

> ✅ You should see:
> ```
>   VITE v5.x ready in 300ms
>   ➜  Local:   http://localhost:5173/
> ```

**The frontend is now running at:** `http://localhost:5173`

---

### Step 5 — Use the App

1. Open `http://localhost:5173` in your browser
2. Click **"Try the Model"** or navigate to **Demo**
3. Upload any underwater fish photo (JPEG/PNG, max 10 MB)
4. Click **"Process with YOLOv8"**
5. View annotated image with bounding boxes, species names, and confidence scores

---

## 🐳 Docker (All-in-One)

If you have Docker Desktop installed:

```bash
# Build the unified image
docker build -t deepsea-ai .

# Run it
docker run -p 8000:8000 deepsea-ai
```

Open `http://localhost:8000` — the React build is served by FastAPI in this mode.

---

## 🧪 Test the API Directly

You can test the backend without the frontend using `curl`:

```bash
curl -X POST http://localhost:8000/api/predict \
  -F "file=@/path/to/your/fish.jpg"
```

Or use the interactive docs at `http://localhost:8000/docs`.

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| mAP@50 | **0.91** |
| Precision | **0.86** |
| Recall | **0.87** |
| mAP@50–95 | **0.66** |
| Inference Time | **≤ 45ms** (CPU) |
| Input Size | 640 × 640 px |
| Confidence Threshold | 0.25 (default) |

---

## 🚀 Production Deployment

### Frontend → Vercel

1. Push this repo to GitHub
2. Connect repo to [vercel.com](https://vercel.com)
3. Set **Root Directory** to `client`
4. Add Environment Variable: `VITE_API_URL` → your backend URL
5. Deploy ✅

### Backend → Hugging Face Spaces

1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose **Docker** as the SDK
3. Upload `Dockerfile.huggingface` (rename to `Dockerfile`) and `server/` contents
4. Upload `models/best.pt` to the Space files
5. Space will build and run automatically ✅

### Backend → Render.com

1. Create a new **Web Service** at [render.com](https://render.com)
2. Connect GitHub repo, set **Root Directory** to `.` (repo root)
3. Set **Dockerfile Path** to `Dockerfile.render`
4. Deploy ✅

---

## 🔧 Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: ultralytics` | Run `pip install -r server/requirements.txt` |
| Backend shows `❌ Failed to load model` | Check that `models/best.pt` exists |
| Frontend shows "Connection error" | Make sure backend is running on port 8000 |
| `CORS error` in browser | Backend allows `*` origins by default — check it's running |
| `npm install` fails | Ensure Node.js 18+ is installed |
| Port 8000 already in use | Kill the process: `npx kill-port 8000` |

---

## 📄 Documents

| File | Purpose |
|---|---|
| [`explanation.md`](./explanation.md) | Project explanation for review committee (problem → solution → results) |
| [`code_explanation.md`](./code_explanation.md) | Code walkthrough for review committee (architecture → functions → design decisions) |

---

## 👨‍🔬 Team

**Nikith Kadapalaneni** — Computer Vision & Full-Stack Development
GitHub: [kadapalanikith](https://github.com/kadapalanikith)

---

## 📜 License

Academic/Educational use only. No unauthorized commercial redistribution.
