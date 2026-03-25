# 🌊 DeepSea AI | Advanced YOLOv8 Full-Stack v3.0

[![Production Ready](https://img.shields.io/badge/Production-Ready-brightgreen.svg)](https://github.com/kadapalanikith/Underwater-Fish-Classification-YOLOv8)
[![Model](https://img.shields.io/badge/YOLO-v8-blue.svg)](https://ultralytics.com)
[![Frontend](https://img.shields.io/badge/React-19-61dafb.svg)](https://react.dev)
[![Backend](https://img.shields.io/badge/FastAPI-v0.110-05998b.svg)](https://fastapi.tiangolo.com)

**DeepSea AI** is a professional, production-grade computer vision platform designed for real-time underwater fish species classification. It features a highly optimized YOLOv8 neural network integrated into a modern decoupled architecture.

---

## 🏗️ Enterprise Architecture

This project is engineered for high availability and decoupled scalability:

-   **Frontend (Vercel Optimized)**: A sleek, glassmorphism UI built with **React 19**, **Vite**, and **Framer Motion**. Optimized for global CDN delivery.
-   **Backend (Hugging Face / Render / Docker)**: A high-performance **FastAPI** inference service serving the YOLOv8 model.
-   **Neural Engine**: Custom-trained **YOLOv8** weights (`best.pt`) supporting 13 unique marine species with ultra-low latency.
-   **Inference Pipeline**: Real-time visual feedback with base64 encoded annotated frames and confidence metadata.

---

## 📂 Project Structure

```text
fish_offline_app/
├── client/              # React + Vite Frontend (Optimized for Vercel)
│   ├── src/             # Core UI components & Pages
│   └── public/          # Static assets
├── server/              # FastAPI Backend (Optimized for HF/Render)
│   ├── main.py          # Professional REST API with CORS & Logging
│   └── requirements.txt # Enterprise-grade Python dependencies
├── models/              # Neural Network weights
│   └── best.pt          # YOLOv8 Weights (mAP@50: 0.91)
├── Dockerfile           # Unified production build (Vite + FastAPI)
├── Dockerfile.render    # Backend-only build optimized for Render.com
├── Dockerfile.huggingface # Specific config for Hugging Face Spaces
└── README.md            # System Documentation
```

---

## 🚀 Deployment Guide (Production)

### 1. Backend (The Engine)
We recommend hosting the API on **Hugging Face Spaces** or **Render**:
-   **Hugging Face**: Use `Dockerfile.huggingface` (rename to `Dockerfile` on HF).
-   **Render**: Use `Dockerfile.render`. 
-   **Port**: The API matches the environment default (7860 for HF, 8000 for Render/Local).

### 2. Frontend (The UI)
Host on **Vercel** for optimal global speed:
1.  Connect this GitHub repo to Vercel.
2.  Add Environment Variable: `VITE_API_URL` = `YOUR_BACKEND_URL`.
3.  Deploy.

---

## 💻 Local Developer Execution

### Option A: The "Direct" Way (Recommended for Dev)
1.  **Start Backend**: 
    ```bash
    cd server && pip install -r requirements.txt && python main.py
    ```
2.  **Start Frontend**: 
    ```bash
    cd client && npm install && npm run dev
    ```

### Option B: The Docker Way (Recommended for Testing)
Ensure Docker Desktop is running, then:
```bash
docker build -t deepsea-ai .
docker run -p 8000:8000 deepsea-ai
```
Access at `http://localhost:8000`.

---

## 📊 Model Intelligence
-   **Architecture**: YOLOv8 (Ultralytics)
-   **Confidence Threshold**: 0.25 (User Configurable)
-   **Classes**: AngelFish, BlueTang, ButterflyFish, ClownFish, GoldFish, Gourami, MorishIdol, PlatyFish, RibbonedSweetlips, ThreeStripedDamselfish, YellowCichlid, YellowTang, ZebraFish.

---

## 👨‍🔬 Research & Development
DeepSea AI is maintained by **Nikith** with a focus on marine biodiversity monitoring through automated visual recognition.

---

## 📜 License
Standard academic/educational license applies. No unauthorized commercial redistribution.
