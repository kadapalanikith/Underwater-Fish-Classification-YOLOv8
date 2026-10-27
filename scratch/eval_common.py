import hashlib
import json
import os
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "Dataset"
OUTPUTS = ROOT / "outputs"
DERIVED = OUTPUTS / "derived"
CHECKPOINT = ROOT / "models" / "best.pt"
RTDETR_CHECKPOINT = ROOT / "models" / "rtdetr-l.pt"

CLASS_NAMES = [
    "AngelFish",
    "BlueTang",
    "ButterflyFish",
    "ClownFish",
    "GoldFish",
    "Gourami",
    "MorishIdol",
    "PlatyFish",
    "RibbonedSweetlips",
    "ThreeStripedDamselfish",
    "YellowCichlid",
    "YellowTang",
    "ZebraFish",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_stem(filename):
    stem = filename.split(".rf.", 1)[0]
    lowered = stem.lower()
    for suffix in ("_jpeg", "_jpg", "_png", "_bmp", "_webp"):
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return Path(stem).stem


def list_images(split):
    image_dir = DATASET / split / "images"
    return sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for_image(image_path):
    return DATASET / image_path.parent.parent.name / "labels" / f"{image_path.stem}.txt"


def parse_label_line(line):
    parts = line.split()
    if not parts:
        return {"error": "blank"}
    try:
        class_id = int(float(parts[0]))
    except ValueError:
        return {"error": "non_numeric_class"}
    if class_id < 0 or class_id >= len(CLASS_NAMES):
        return {"error": "class_out_of_range", "class_id": class_id}
    if len(parts) == 5:
        try:
            x, y, w, h = (float(value) for value in parts[1:])
        except ValueError:
            return {"error": "non_numeric_box", "class_id": class_id}
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            return {"error": "box_out_of_bounds", "class_id": class_id}
        return {
            "class_id": class_id,
            "format": "box",
            "box": (x, y, w, h),
            "out_of_range": False,
        }
    if len(parts) > 5:
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            return {"error": "non_numeric_polygon", "class_id": class_id}
        if len(values) % 2 != 0:
            return {"error": "odd_polygon_coordinates", "class_id": class_id}
        points = np.asarray(values, dtype=np.float64).reshape(-1, 2)
        if len(points) < 3:
            return {"error": "polygon_too_short", "class_id": class_id}
        xmin, ymin = points.min(axis=0)
        xmax, ymax = points.max(axis=0)
        out_of_range = bool(
            xmin < 0.0 or ymin < 0.0 or xmax > 1.0 or ymax > 1.0
        )
        xmin = float(np.clip(xmin, 0.0, 1.0))
        ymin = float(np.clip(ymin, 0.0, 1.0))
        xmax = float(np.clip(xmax, 0.0, 1.0))
        ymax = float(np.clip(ymax, 0.0, 1.0))
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0.0 or h <= 0.0:
            return {"error": "degenerate_polygon", "class_id": class_id}
        return {
            "class_id": class_id,
            "format": "polygon",
            "box": ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, w, h),
            "out_of_range": out_of_range,
        }
    return {"error": "wrong_field_count", "class_id": class_id}


def format_box(class_id, box):
    x, y, w, h = box
    return f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def read_label_file(label_path):
    if not label_path.exists():
        return []
    return [
        line for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()
    ]


def prepare_dataset(split, name, include_names=None, convert_polygons=True):
    source_images = list_images(split)
    if include_names is not None:
        include_names = set(include_names)
        source_images = [path for path in source_images if path.name in include_names]
    dataset_dir = DERIVED / name
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    stats = {
        "source_split": split,
        "source_images": len(list_images(split)),
        "included_images": len(source_images),
        "excluded_images": len(list_images(split)) - len(source_images),
        "images_linked": 0,
        "images_copied": 0,
        "box_rows": 0,
        "polygon_rows": 0,
        "converted_polygon_boxes": 0,
        "invalid_rows": 0,
        "invalid_row_examples": [],
        "out_of_range_polygon_rows": 0,
        "images_with_polygon_rows": 0,
        "images_with_mixed_rows": 0,
        "empty_label_files": 0,
        "duplicate_box_rows": 0,
        "instances_per_class": {class_name: 0 for class_name in CLASS_NAMES},
    }

    for image_path in source_images:
        label_path = DATASET / split / "labels" / f"{image_path.stem}.txt"
        rows = read_label_file(label_path)
        converted = []
        formats = set()
        seen_boxes = set()
        for line in rows:
            parsed = parse_label_line(line)
            if "error" in parsed:
                stats["invalid_rows"] += 1
                if len(stats["invalid_row_examples"]) < 20:
                    stats["invalid_row_examples"].append(
                        {"file": label_path.name, "line": line[:300], "error": parsed["error"]}
                    )
                continue
            if parsed["format"] == "polygon" and not convert_polygons:
                continue
            formats.add(parsed["format"])
            stats[f"{parsed['format']}_rows"] += 1
            if parsed["format"] == "polygon":
                stats["converted_polygon_boxes"] += 1
                if parsed["out_of_range"]:
                    stats["out_of_range_polygon_rows"] += 1
            text = format_box(parsed["class_id"], parsed["box"])
            if text in seen_boxes:
                stats["duplicate_box_rows"] += 1
            seen_boxes.add(text)
            class_name = CLASS_NAMES[parsed["class_id"]]
            stats["instances_per_class"][class_name] += 1
            converted.append(text)

        if "polygon" in formats:
            stats["images_with_polygon_rows"] += 1
        if "box" in formats and "polygon" in formats:
            stats["images_with_mixed_rows"] += 1
        if not converted:
            stats["empty_label_files"] += 1

        destination = image_dir / image_path.name
        try:
            os.link(image_path, destination)
            stats["images_linked"] += 1
        except OSError:
            shutil.copy2(image_path, destination)
            stats["images_copied"] += 1
        (label_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(converted) + ("\n" if converted else ""),
            encoding="utf-8",
        )

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
    stats["instances_total"] = int(sum(stats["instances_per_class"].values()))
    stats["dataset_dir"] = str(dataset_dir)
    stats["data_yaml"] = str(data_yaml)
    return dataset_dir, stats


def evaluate_yolo(
    data_yaml,
    split,
    run_name,
    image_count,
    checkpoint=CHECKPOINT,
    imgsz=640,
    conf=0.001,
    iou=0.6,
    device="cpu",
):
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    run_root = Path(os.environ.get("TEMP", str(Path.home() / "AppData" / "Local" / "Temp")))
    run_root = run_root / "opencode" / "ultralytics_runs"
    run_root.mkdir(parents=True, exist_ok=True)
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        conf=conf,
        iou=iou,
        workers=0,
        plots=False,
        verbose=False,
        project=str(run_root),
        name=run_name,
        exist_ok=True,
    )
    names = {int(key): value for key, value in results.names.items()}
    nt_per_class = getattr(results, "nt_per_class", None)
    ap_class_index = [int(value) for value in getattr(results.box, "ap_class_index", [])]
    per_class = {}
    for index, class_id in enumerate(ap_class_index):
        instances = None
        if nt_per_class is not None:
            instances = int(nt_per_class[class_id])
        per_class[names[class_id]] = {
            "class_id": class_id,
            "instances": instances,
            "precision": round(float(results.box.p[index]), 6),
            "recall": round(float(results.box.r[index]), 6),
            "ap50": round(float(results.box.ap50[index]), 6),
            "ap50_95": round(float(results.box.ap[index]), 6),
        }
    classes_without_gt = []
    for class_id, class_name in names.items():
        if class_name not in per_class:
            instances = 0
            if nt_per_class is not None:
                instances = int(nt_per_class[class_id])
            classes_without_gt.append(
                {"class_id": class_id, "name": class_name, "instances": instances}
            )
    low_support = sorted(
        [
            {"name": item["name"], "instances": item["instances"]}
            for item in classes_without_gt
            if item["instances"] > 0
        ]
        + [
            {"name": name, "instances": item["instances"]}
            for name, item in per_class.items()
            if item["instances"] is not None and item["instances"] < 10
        ],
        key=lambda item: item["instances"],
    )
    return {
        "split": split,
        "images_evaluated": int(image_count),
        "instances_evaluated": int(nt_per_class.sum()) if nt_per_class is not None else None,
        "overall": {
            "precision": round(float(results.results_dict.get("metrics/precision(B)", 0.0)), 6),
            "recall": round(float(results.results_dict.get("metrics/recall(B)", 0.0)), 6),
            "mAP50": round(float(results.results_dict.get("metrics/mAP50(B)", 0.0)), 6),
            "mAP50_95": round(float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 6),
            "mAP75": round(float(getattr(results.box, "map75", 0.0)), 6),
        },
        "per_class": per_class,
        "classes_without_ground_truth": classes_without_gt,
        "low_support_classes": low_support,
        "speed_ms_per_image": {
            key: round(float(value), 3) for key, value in results.speed.items()
        },
        "run_dir": str(results.save_dir),
        "settings": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_size_mb": round(checkpoint.stat().st_size / (1024 * 1024), 2),
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "device": device,
            "workers": 0,
            "plots": False,
        },
    }


def environment_info():
    import cv2
    import torch
    import ultralytics

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_used": "cpu",
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def timestamp():
    return time.strftime("%Y-%m-%dT%H:%M:%S")
