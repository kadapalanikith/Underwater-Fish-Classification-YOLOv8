"""
API integration tests for the FastAPI extension endpoints.

These tests exercise the real application lifespan (YOLO/RT-DETR loading) and
do not download data, train models, or overwrite saved experiment artifacts.
Tracking is checked with a synthetic video only; it is not a real-footage
tracking evaluation.
"""
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from server.main import app

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def _image_bytes() -> bytes:
    sample = sorted((ROOT / "test_sample").glob("*.jpg"))
    if sample:
        return sample[0].read_bytes()
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _synthetic_video_bytes(frames: int = 6) -> bytes:
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240))
    if not writer.isOpened():
        os.remove(path)
        pytest.skip("OpenCV MP4 writer is unavailable in this environment")
    try:
        for i in range(frames):
            frame = np.full((240, 320, 3), (50, 100, 80), dtype=np.uint8)
            cv2.rectangle(frame, (20 + i * 10, 80), (80 + i * 10, 140), (0, 165, 255), -1)
            writer.write(frame)
    finally:
        writer.release()
    try:
        return Path(path).read_bytes()
    finally:
        os.remove(path)


def test_health_reports_loaded_models(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert "yolov8n" in payload["models_loaded"]
    assert payload["device"] in ("cpu", "cuda:0", "mps")


def test_models_endpoint_distinguishes_pretrained_rtdetr(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    models = {item["id"]: item for item in response.json()["models"]}

    assert models["yolov8n"]["fish_fine_tuned"] is True
    if "rtdetr" in models:
        assert models["rtdetr"]["fish_fine_tuned"] is False
        assert "not fish fine-tuned" in models["rtdetr"]["notes"]


def test_metrics_endpoint_preserves_provenance(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200
    payload = response.json()

    assert payload["baseline"]["validation_metrics"]["mAP50"] == 0.908
    assert "enhancement_audit" in payload
    assert "enhancement_benchmark" in payload
    assert payload["bootstrap_uncertainty"]["results"]["test_clean"]["images"] == 399
    assert payload["obsea_evaluation"]["test_audit"]["images"] == 618
    assert payload["obsea_evaluation"]["dataset"]["license"] == "CC BY 4.0"
    assert "not accuracy measurements" in payload["provenance"]["notes"]
    assert "class-agnostic" in payload["provenance"]["notes"]


def test_predict_returns_expected_schema(client):
    response = client.post(
        "/api/predict",
        files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")},
        data={"enhancement": "raw", "model_name": "yolov8n", "conf": "0.25"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["success"] is True
    assert payload["model_used"] == "yolov8n"
    assert payload["model_fallback"] is False
    assert isinstance(payload["detections"], list)
    assert payload["annotated_image"].startswith("data:image/jpeg;base64,")


def test_predict_unknown_model_falls_back_and_reports_it(client):
    response = client.post(
        "/api/predict",
        files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")},
        data={"enhancement": "raw", "model_name": "not-a-model", "conf": "0.25"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["requested_model"] == "not-a-model"
    assert payload["model_used"] == "yolov8n"
    assert payload["model_fallback"] is True


def test_predict_rejects_invalid_image(client):
    response = client.post(
        "/api/predict",
        files={"file": ("not-an-image.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_predict_rejects_out_of_range_confidence(client):
    response = client.post(
        "/api/predict",
        files={"file": ("sample.jpg", _image_bytes(), "image/jpeg")},
        data={"conf": "1.5"},
    )
    assert response.status_code == 422


def test_track_synthetic_video_pipeline(client):
    response = client.post(
        "/api/track",
        files={"file": ("synthetic.mp4", _synthetic_video_bytes(frames=6), "video/mp4")},
        data={"enhancement": "raw", "conf": "0.25", "max_frames": "6"},
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["success"] is True
    assert payload["frames_processed"] == 6
    assert isinstance(payload["crossing_events"], list)
    assert payload["video_base64"].startswith("data:video/mp4;base64,")
