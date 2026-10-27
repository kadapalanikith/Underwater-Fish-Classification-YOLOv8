import os
import sys
import json
import time
import numpy as np
from ultralytics import YOLO, RTDETR
from src.utils.device import get_optimal_device

device = get_optimal_device()
print(f"Device for benchmark: {device}")

# Model 1: YOLOv8n (Custom trained fish baseline)
yolo_path = "models/best.pt"
print(f"Loading YOLOv8n from {yolo_path}...")
yolo_model = YOLO(yolo_path)

# Model 2: RT-DETR (Real-Time DEtection TRansformer)
rtdetr_path = "models/rtdetr-l.pt"
print("Loading RT-DETR (rtdetr-l.pt)...")
try:
    rtdetr_model = RTDETR("rtdetr-l.pt")
    # Save checkpoint to models/
    if not os.path.exists("models/rtdetr-l.pt"):
        import shutil
        if os.path.exists("rtdetr-l.pt"):
            shutil.move("rtdetr-l.pt", "models/rtdetr-l.pt")
    rtdetr_available = True
except Exception as e:
    print(f"Could not load RT-DETR: {e}")
    rtdetr_available = False

dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)

def benchmark_model(model, name, arch_type, weights_path):
    # Warmup
    for _ in range(3):
        _ = model(dummy_input, verbose=False, device=device)

    latencies = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = model(dummy_input, verbose=False, device=device)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_lat = float(np.mean(latencies))
    fps = round(1000.0 / avg_lat, 2) if avg_lat > 0 else 0.0

    try:
        params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        params = 0

    size_mb = round(os.path.getsize(weights_path) / (1024 * 1024), 2) if os.path.exists(weights_path) else 0.0

    return {
        "model_name": name,
        "architecture_type": arch_type,
        "parameters": params,
        "file_size_mb": size_mb,
        "device": device,
        "avg_latency_ms": round(avg_lat, 2),
        "fps": fps
    }

benchmarks = []
benchmarks.append(benchmark_model(yolo_model, "YOLOv8n-Fish", "CNN (CSPDarknet + PAN-FPN)", yolo_path))

if rtdetr_available:
    benchmarks.append(benchmark_model(rtdetr_model, "RT-DETR-L", "Transformer (Hybrid Encoder + Deformable DETR Decoder)", "models/rtdetr-l.pt" if os.path.exists("models/rtdetr-l.pt") else "rtdetr-l.pt"))

print("\n--- Model Architecture Benchmark Comparison ---")
print(json.dumps(benchmarks, indent=2))

with open("outputs/model_benchmark_comparison.json", "w") as f:
    json.dump({"device": device, "models": benchmarks}, f, indent=2)
print("Saved to outputs/model_benchmark_comparison.json")
