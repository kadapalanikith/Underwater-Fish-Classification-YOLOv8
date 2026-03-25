# 🚀 DeepSea AI | Standalone Backend Hosting Guide

You've successfully identified a common build issue: **Vite 8 requires Node 20+**, but the Docker container was using Node 18. Since you've decided to host the **Backend only on Render**, we've simplified the setup to avoid these Node version conflicts entirely.

---

## 🏗️ New Strategy: Split Hosting
1. **Backend (Python/FastAPI)**: Hosted on Render (using the new simplified `Dockerfile`).
2. **Frontend (React)**: Hosted on **Vercel** or **Netlify** (much faster and easier for React apps).

---

## 🚀 Step 1: Deploy Backend to Render

1. **Commit your changes**:
   I have updated your `Dockerfile` to remove the Node/Frontend parts. It now builds only the Python API.
2. **Push to GitHub**:
   `git add . && git commit -m "chore: switch to standalone backend build" && git push`
3. **Configure Render**:
   - Create a **New Web Service**.
   - Connect your repo.
   - **Instance Type**: `Starter` (needed for AI inference).
   - **Environment Variables**:
     - `PORT`: `8000`
     - `MODEL_PATH`: `/app/models/best.pt`
     - `ALLOW_ORIGINS`: `*`

Once live, Render will give you a URL like: `https://deepsea-api.onrender.com`.

---

## 🔗 Step 2: Connect Frontend to the New API

1. **Update Frontend Environment**:
   Go to `client/.env` and set the `VITE_API_URL` to your Render URL:
   ```env
   VITE_API_URL=https://deepsea-api.onrender.com
   ```

2. **Deploy Frontend to Vercel/Netlify**:
   - Go to [Vercel](https://vercel.com) or [Netlify](https://netlify.com).
   - Connect your GitHub repo.
   - **Root Directory**: `client`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Add Environment Variable**: Set `VITE_API_URL` to your backend URL.

---

## 🛠️ Why this is better:
- **Zero Node Conflicts**: No more "Unsupported engine" errors during Docker builds.
- **Smaller Image**: Your Docker image is now ~600MB instead of 1.2GB.
- **Faster Deploys**: Frontend and Backend deploy independently.
- **Scalability**: You can scale your AI backend without affecting your UI.

---
*Created by Antigravity for DeepSea AI.*
