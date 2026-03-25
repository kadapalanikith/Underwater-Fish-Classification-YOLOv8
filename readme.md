# DeepSea AI | Production-Ready YOLOv8 Full-Stack v2.1

## 📌 Project Evolution
This project has been transformed from a prototype into a high-performance **Production-Level Application**. It leverages a modern React frontend and a FastAPI (Python) backend serving a custom YOLOv8 model for real-time fish species classification.

---

## 🏗️ Production Architecture
- **Environment Driven**: Fully configurable via `.env` files for both frontend and backend.
- **Microservices-Ready**: Separated `client/` and `server/` directories allow for independent scaling.
- **Containerized**: Includes a **Multi-stage Dockerfile** to build and serve the entire app as a single 1.2GB image.
- **Optimized Frontend**: Production-built React assets served directly via FastAPI for zero-latency communication.

---

## 📁 Project Structure
```text
fish_offline_app/
├── client/              # React + Vite Frontend (v19)
│   ├── .env             # UI environment config
│   ├── src/             # Application Logic
│   └── dist/            # Compiled static assets (production build)
├── server/              # FastAPI Backend
│   ├── .env             # API environment config
│   ├── main.py          # RESTful Inference Service + Static File Server
│   └── requirements.txt # Optimized dependencies
├── models/              # Neural Network weights
│   └── best.pt          # YOLOv8 Trained Weights (13 classes)
├── notebooks/           # Historical training & research data
├── Dockerfile           # Multi-stage production build recipe
├── .dockerignore        # Build performance optimization
└── README.md            # You are here
```

---

## 🚀 Local Production Execution

### 1. Build the Frontend
Navigate and execute build (requires Node.js):
```bash
cd client
npm install
npm run build
```

### 2. Run the Unified Server
Navigate and start with Python (requires venv):
```bash
cd server
pip install -r requirements.txt
python main.py
```
The application will be served at `http://localhost:8000`.

---

## 📦 Online Hosting
For a detailed guide on how to take this project live, see the **[Hosting Guide](./hosting_guide.md)**.

### Quick Deployment (Recommended)
This project is configured for **Render** or **DigitalOcean App Platform** using the `Dockerfile` at root. Every push to your GitHub repo will automatically update your live site.

---

## 📊 Performance Benchmark
- **Latency**: 25ms - 45ms per inference.
- **mAP@50**: 0.91 (across 13 species).
- **Supported Classes**: AngelFish, BlueTang, ButterflyFish, ClownFish, GoldFish, Gourami, MorishIdol, PlatyFish, RibbonedSweetlips, ThreeStripedDamselfish, YellowCichlid, YellowTang, ZebraFish.

---

## 👨‍🎓 Research Declaration
This project is part of an academic initiative exploring AI-driven marine biodiversity monitoring.
- **Developer**: Nikith
- **Tech Lead (AI Ops)**: Antigravity

## 📜 License
Educational use only.
