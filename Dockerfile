# Stage 1: Build the React Frontend
FROM node:18-alpine AS frontend-builder
WORKDIR /app/client
COPY client/package*.json ./
RUN npm install
COPY client/ ./
RUN npm run build

# Stage 2: Build the FastAPI Backend
FROM python:3.10-slim
WORKDIR /app

# Install system dependencies for OpenCV and YOLO
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY server/requirements.txt ./server/
RUN pip install --no-cache-dir -r server/requirements.txt

# Copy the trained model
COPY models/ ./models/

# Copy the built frontend from Stage 1
COPY --from=frontend-builder /app/client/dist ./client/dist

# Copy the server code
COPY server/ ./server/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV STATIC_DIR=/app/client/dist
ENV MODEL_PATH=/app/models/best.pt

EXPOSE 8000

# Start command
CMD ["python", "server/main.py"]
