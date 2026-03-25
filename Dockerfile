# Build only the FastAPI Backend (Standalone AI API)
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

# Copy the server code
COPY server/ ./server/

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV MODEL_PATH=/app/models/best.pt
# We leave STATIC_DIR empty as we are hosting the frontend separately
ENV STATIC_DIR=/tmp/null 

EXPOSE 8000

# Start command
CMD ["python", "server/main.py"]
