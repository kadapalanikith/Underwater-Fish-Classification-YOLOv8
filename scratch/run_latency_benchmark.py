"""Timing-only comparison of the fish YOLOv8 checkpoint and pretrained RT-DETR-L.

This is NOT an accuracy comparison. The RT-DETR checkpoint is pretrained on COCO
and has not been fine-tuned on the fish dataset, so its detections are not
fish-species predictions. Only per-image latency is reported.
"""

import statistics
import sys
import time

from eval_common import (
    DERIVED,
    OUTPUTS,
    ROOT,
    RTDETR_CHECKPOINT,
    CHECKPOINT,
    environment_info,
    sha256_file,
    timestamp,
    write_json,
)

sys.path.insert(0, str(ROOT))

IMAGE_COUNT = 30
WARMUP_COUNT = 3
IMGSZ = 640
CONF = 0.25
DEVICE = "cpu"


def summarize(values):
    ordered = sorted(values)
    return {
        "samples": len(values),
        "mean_ms": round(statistics.fmean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def benchmark(model, image_paths, label):
    for image_path in image_paths[:WARMUP_COUNT]:
        model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            verbose=False,
            save=False,
        )

    records = []
    for image_path in image_paths:
        start = time.perf_counter()
        results = model.predict(
            source=str(image_path),
            imgsz=IMGSZ,
            conf=CONF,
            device=DEVICE,
            verbose=False,
            save=False,
        )
        wall_ms = (time.perf_counter() - start) * 1000.0
        speed = results[0].speed
        records.append(
            {
                "image": image_path.name,
                "wall_ms": wall_ms,
                "preprocess_ms": float(speed.get("preprocess", 0.0)),
                "inference_ms": float(speed.get("inference", 0.0)),
                "postprocess_ms": float(speed.get("postprocess", 0.0)),
                "detections": int(len(results[0].boxes)),
            }
        )

    return {
        "label": label,
        "wall": summarize([item["wall_ms"] for item in records]),
        "preprocess": summarize([item["preprocess_ms"] for item in records]),
        "inference": summarize([item["inference_ms"] for item in records]),
        "postprocess": summarize([item["postprocess_ms"] for item in records]),
        "mean_detections": round(statistics.fmean([item["detections"] for item in records]), 3),
        "records": records,
    }


def main():
    from ultralytics import RTDETR, YOLO

    image_paths = sorted((DERIVED / "test_clean" / "images").glob("*"))[:IMAGE_COUNT]
    if not image_paths:
        raise SystemExit("Missing outputs/derived/test_clean/images; run held-out evaluation first.")

    yolo = YOLO(str(CHECKPOINT))
    rtdetr = RTDETR(str(RTDETR_CHECKPOINT))

    print(f"Benchmarking {len(image_paths)} images with warmup={WARMUP_COUNT}, imgsz={IMGSZ}, device={DEVICE}")
    yolo_result = benchmark(yolo, image_paths, "yolov8_fish")
    rtdetr_result = benchmark(rtdetr, image_paths, "rtdetr_l_coco_pretrained")

    payload = {
        "timestamp": timestamp(),
        "environment": environment_info(),
        "protocol": {
            "images": len(image_paths),
            "warmup_images": WARMUP_COUNT,
            "imgsz": IMGSZ,
            "conf": CONF,
            "device": DEVICE,
            "batch_size": 1,
            "api": "ultralytics predict",
            "note": (
                "Timing only. RT-DETR-L is a pretrained COCO checkpoint, not a "
                "fish-fine-tuned model. Do not interpret detection counts or classes "
                "as fish-species accuracy."
            ),
        },
        "checkpoints": {
            "yolov8_fish": {
                "path": str(CHECKPOINT),
                "sha256": sha256_file(CHECKPOINT),
                "classes": len(yolo.names),
            },
            "rtdetr_l_coco_pretrained": {
                "path": str(RTDETR_CHECKPOINT),
                "sha256": sha256_file(RTDETR_CHECKPOINT),
                "classes": len(rtdetr.names),
            },
        },
        "results": {
            "yolov8_fish": yolo_result,
            "rtdetr_l_coco_pretrained": rtdetr_result,
        },
    }
    output_path = OUTPUTS / "latency_yolov8_vs_rtdetr.json"
    write_json(output_path, payload)

    print(f"\nWrote {output_path}")
    for key, result in payload["results"].items():
        print(
            f"{key:26s} wall={result['wall']['mean_ms']:8.2f} ms  "
            f"inference={result['inference']['mean_ms']:8.2f} ms  "
            f"detections={result['mean_detections']:.1f}"
        )


if __name__ == "__main__":
    main()
