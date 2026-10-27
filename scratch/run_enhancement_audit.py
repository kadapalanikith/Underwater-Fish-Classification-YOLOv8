"""
Definitive Enhancement Benchmark
Runs all 4 enhancement methods on a fixed, sorted set of qualitative_samples images.
Records per-image detection counts, per-image confidence list, processing time.
Explicitly flags the absence of ground-truth labels and marks this as a
qualitative inference benchmark, NOT an accuracy comparison.
"""
import sys, os, json, time, glob, hashlib
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')

from src.preprocessing.enhancement import UnderwaterEnhancer
from src.utils.device import get_optimal_device
from ultralytics import YOLO
import numpy as np

# Fixed, reproducible image order
images = sorted(glob.glob('qualitative_samples/*.jpg'))
assert len(images) == 40, f"Expected 40 images, got {len(images)}"

device = get_optimal_device()
model = YOLO('models/best.pt')
enhancer = UnderwaterEnhancer()

# Verify no ground-truth labels exist for these images
has_ground_truth = any(
    os.path.exists(p.replace('qualitative_samples/', 'qualitative_samples/labels/').replace('.jpg', '.txt'))
    for p in images
)

results = {
    "_metadata": {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S'),
        "image_set": "qualitative_samples (40 fixed images, sorted alphabetically)",
        "image_set_source": "Roboflow .rf. naming — same Kaggle/Roboflow distribution as training data",
        "model": "models/best.pt (YOLOv8n, trained on this dataset)",
        "ground_truth_labels_available": has_ground_truth,
        "experiment_type": "Qualitative inference benchmark — no accuracy claims possible without ground truth",
        "device": device,
        "warning": "Detection count differences between methods do NOT confirm accuracy improvement. "
                   "A method finding more boxes may be producing false positives. "
                   "Ground-truth annotations are required to compute precision/recall/mAP."
    }
}

CONF_THRESHOLD = 0.25
methods = ['raw', 'clahe', 'gray_world', 'fusion']

for method in methods:
    per_image = []
    all_confidences = []
    preproc_times = []
    infer_times = []

    for img_path in images:
        t0 = time.perf_counter()
        enhanced, emeta = enhancer.enhance(img_path, method=method)
        preproc_ms = (time.perf_counter() - t0) * 1000.0
        preproc_times.append(preproc_ms)

        t1 = time.perf_counter()
        res = model.predict(enhanced, conf=CONF_THRESHOLD, verbose=False, device=device)[0]
        infer_ms = (time.perf_counter() - t1) * 1000.0
        infer_times.append(infer_ms)

        boxes = res.boxes
        n = len(boxes) if boxes is not None else 0
        confs = [round(float(b.conf[0]), 4) for b in (boxes or [])]
        all_confidences.extend(confs)

        per_image.append({
            "image": os.path.basename(img_path),
            "detections": n,
            "confidences": confs,
            "preproc_ms": round(preproc_ms, 2),
            "infer_ms": round(infer_ms, 2)
        })

    results[method] = {
        "method": method,
        "images_evaluated": len(images),
        "total_detections": sum(r["detections"] for r in per_image),
        "images_with_at_least_1_detection": sum(1 for r in per_image if r["detections"] > 0),
        "avg_confidence_all_boxes": round(float(np.mean(all_confidences)), 4) if all_confidences else None,
        "median_confidence": round(float(np.median(all_confidences)), 4) if all_confidences else None,
        "avg_preproc_ms": round(float(np.mean(preproc_times)), 2),
        "avg_infer_ms": round(float(np.mean(infer_times)), 2),
        "avg_total_ms": round(float(np.mean([p + i for p, i in zip(preproc_times, infer_times)])), 2),
        "per_image": per_image
    }
    print(f"[{method.upper()}] done: {results[method]['total_detections']} detections, "
          f"hit_rate={results[method]['images_with_at_least_1_detection']}/40, "
          f"avg_conf={results[method]['avg_confidence_all_boxes']}, "
          f"avg_total_ms={results[method]['avg_total_ms']}")

os.makedirs('outputs', exist_ok=True)
with open('outputs/enhancement_audit.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print("\nSaved to outputs/enhancement_audit.json")
print("\n[AUDIT NOTE] No ground-truth labels found for qualitative_samples images.")
print("All results reflect inference counts only — not validated accuracy.")
