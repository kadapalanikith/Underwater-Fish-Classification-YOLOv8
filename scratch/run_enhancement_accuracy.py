"""Evaluate preprocessing methods on the leakage-filtered labeled test subset.

The YOLOv8 checkpoint is fixed. Only the input images change. The comparison uses
the same 399 images and converted box labels as the clean held-out baseline.

Methods:
  raw_original : hardlinked original JPEGs (control, matches held-out baseline)
  raw_jpeg95   : raw pixels re-encoded at JPEG quality 95 (encoding control)
  clahe        : LAB-space CLAHE
  gray_world   : gray-world white balance
  fusion       : gray-world followed by CLAHE
"""

import json
import os
import shutil
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from eval_common import (
    CLASS_NAMES,
    DERIVED,
    OUTPUTS,
    ROOT,
    environment_info,
    evaluate_yolo,
    sha256_file,
    timestamp,
    write_json,
)

sys.path.insert(0, str(ROOT))
from src.preprocessing.enhancement import UnderwaterEnhancer  # noqa: E402

TEST_CLEAN = DERIVED / "test_clean"
BASELINE_JSON = OUTPUTS / "heldout_yolov8_metrics.json"
OUTPUT_JSON = OUTPUTS / "enhancement_accuracy_test_clean.json"

METHODS = ["raw_original", "raw_jpeg95", "clahe", "gray_world", "fusion"]
JPEG_QUALITY = 95
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(directory):
    return sorted(p for p in directory.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def write_data_yaml(dataset_dir):
    data_yaml = dataset_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": dataset_dir.resolve().as_posix(),
                "train": "images",
                "val": "images",
                "test": "images",
                "nc": len(CLASS_NAMES),
                "names": CLASS_NAMES,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return data_yaml


def build_dataset(method, image_paths, label_dir, enhancer):
    dataset_dir = DERIVED / f"enhancement_{method}"
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    out_images = dataset_dir / "images"
    out_labels = dataset_dir / "labels"
    out_images.mkdir(parents=True)
    out_labels.mkdir(parents=True)

    latencies = []
    mean_channels = []
    for index, image_path in enumerate(image_paths):
        source_label = label_dir / f"{image_path.stem}.txt"
        target_label = out_labels / source_label.name
        if target_label.exists():
            target_label.unlink()
        os.link(source_label, target_label)

        target_image = out_images / image_path.name
        if method == "raw_original":
            if target_image.exists():
                target_image.unlink()
            os.link(image_path, target_image)
            continue

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not read {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if method == "raw_jpeg95":
            enhanced = rgb
        else:
            if index < 3:
                enhancer.enhance(rgb, method)
            start = time.perf_counter()
            enhanced, _ = enhancer.enhance(rgb, method)
            latencies.append((time.perf_counter() - start) * 1000.0)

        mean_channels.append(np.mean(enhanced, axis=(0, 1)))
        out_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
        ok = cv2.imwrite(
            str(target_image),
            out_bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError(f"Could not write {target_image}")

    data_yaml = write_data_yaml(dataset_dir)
    channel_means = (
        np.mean(np.asarray(mean_channels), axis=0).round(3).tolist()
        if mean_channels
        else None
    )
    latency_stats = None
    if latencies:
        ordered = sorted(latencies)
        latency_stats = {
            "samples": len(latencies),
            "mean_ms": round(statistics.fmean(latencies), 3),
            "median_ms": round(statistics.median(latencies), 3),
            "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3),
            "min_ms": round(min(latencies), 3),
            "max_ms": round(max(latencies), 3),
        }
    return dataset_dir, data_yaml, latency_stats, channel_means


def main():
    if not TEST_CLEAN.exists():
        raise SystemExit(f"Missing {TEST_CLEAN}; run scratch/run_heldout_evaluation.py first.")
    if not BASELINE_JSON.exists():
        raise SystemExit(f"Missing {BASELINE_JSON}; run scratch/run_heldout_evaluation.py first.")

    image_paths = list_images(TEST_CLEAN / "images")
    label_dir = TEST_CLEAN / "labels"
    if not image_paths:
        raise SystemExit(f"No images in {TEST_CLEAN / 'images'}")

    baseline_payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
    baseline = baseline_payload["results"]["test_clean"]["overall"]
    enhancer = UnderwaterEnhancer()

    results = {}
    for method in METHODS:
        print(f"\n=== {method} ===", flush=True)
        dataset_dir, data_yaml, latency_stats, channel_means = build_dataset(
            method, image_paths, label_dir, enhancer
        )
        evaluation = evaluate_yolo(
            data_yaml=data_yaml,
            split="val",
            run_name=f"enhancement_{method}",
            image_count=len(image_paths),
        )
        evaluation["preprocessing"] = {
            "method": method,
            "jpeg_quality": None if method == "raw_original" else JPEG_QUALITY,
            "enhancement_latency": latency_stats,
            "mean_rgb_channels": channel_means,
            "dataset_dir": str(dataset_dir),
        }
        evaluation["delta_vs_raw_original"] = None
        results[method] = evaluation
        print(
            f"{method}: mAP50={evaluation['overall']['mAP50']:.4f} "
            f"mAP50-95={evaluation['overall']['mAP50_95']:.4f} "
            f"P={evaluation['overall']['precision']:.4f} "
            f"R={evaluation['overall']['recall']:.4f}",
            flush=True,
        )

    raw_overall = results["raw_original"]["overall"]
    for method, evaluation in results.items():
        evaluation["delta_vs_raw_original"] = {
            key: round(evaluation["overall"][key] - raw_overall[key], 6)
            for key in ("precision", "recall", "mAP50", "mAP50_95")
        }

    payload = {
        "timestamp": timestamp(),
        "environment": environment_info(),
        "checkpoint": {
            "path": str(ROOT / "models" / "best.pt"),
            "sha256": sha256_file(ROOT / "models" / "best.pt"),
        },
        "dataset": {
            "source": "outputs/derived/test_clean",
            "images": len(image_paths),
            "instances": baseline_payload["results"]["test_clean"]["instances_evaluated"],
            "leakage_filter": baseline_payload["_metadata"]["leakage_filter"],
            "labels": "converted box labels (polygons converted to enclosing boxes)",
            "heldout_baseline_overall": baseline,
        },
        "protocol": {
            "imgsz": 640,
            "conf": 0.001,
            "iou": 0.6,
            "device": "cpu",
            "workers": 0,
            "plots": False,
            "jpeg_quality_for_enhanced_images": JPEG_QUALITY,
            "note": (
                "raw_original is hardlinked from the clean test set and should match "
                "the held-out baseline. raw_jpeg95 controls for re-encoding. "
                "Enhancement methods are compared against raw_jpeg95, not raw_original."
            ),
        },
        "results": results,
    }
    write_json(OUTPUT_JSON, payload)

    print("\n=== Summary ===")
    print(f"{'method':14s} {'mAP50':>8s} {'mAP50-95':>9s} {'P':>8s} {'R':>8s} {'d50':>8s} {'d50-95':>8s}")
    for method in METHODS:
        overall = results[method]["overall"]
        delta = results[method]["delta_vs_raw_original"]
        print(
            f"{method:14s} {overall['mAP50']:8.4f} {overall['mAP50_95']:9.4f} "
            f"{overall['precision']:8.4f} {overall['recall']:8.4f} "
            f"{delta['mAP50']:+8.4f} {delta['mAP50_95']:+8.4f}"
        )
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
