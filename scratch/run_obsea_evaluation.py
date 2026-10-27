"""Independent-domain class-agnostic fish evaluation on the OBSEA test split.

OBSEA is an external Mediterranean observatory dataset with 23 classes. The
local checkpoint predicts 13 aquarium species, so species-level transfer is
undefined. This script collapses both annotations and predictions to a single
"fish" class, excludes the non-fish Diver and Octopus annotations, keeps every
test image (including background-only images), and evaluates at the same
imgsz/conf/IoU/max_det settings used for the local held-out evaluation.

No training or tuning is performed on OBSEA. The derived test dataset contains
hard-linked images and converted labels under outputs/derived/obsea_fish_test.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT / "scratch"))

from eval_common import (  # noqa: E402
    CHECKPOINT,
    DERIVED,
    OUTPUTS,
    environment_info,
    sha256_file,
    timestamp,
    write_json,
)

OBSEA_ROOT = (
    SCRIPT_ROOT
    / "external_data"
    / "obsea"
    / "extracted"
    / "23sp_4120img_34945annots_2688res"
)
OBSEA_ZIP = SCRIPT_ROOT / "external_data" / "obsea" / "obsea_split_YOLO.zip"
OBSEA_DOI = "10.5281/zenodo.14888440"
OBSEA_LICENSE = "CC BY 4.0"
NON_FISH_NAMES = {"Diver", "Octopus vulgaris"}
DERIVED_NAME = "obsea_fish_test"
OUTPUT_JSON = OUTPUTS / "obsea_evaluation.json"
SETTINGS = {
    "imgsz": 640,
    "conf": 0.001,
    "iou": 0.6,
    "max_det": 300,
    "device": "cpu",
    "workers": 0,
    "single_cls": True,
    "agnostic_nms": True,
}


def read_rows(path: Path):
    return [
        line
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def audit_class_names():
    data = yaml.safe_load((OBSEA_ROOT / "data.yaml").read_text(encoding="utf-8"))
    names = list(data["names"])
    fish_ids = [index for index, name in enumerate(names) if name not in NON_FISH_NAMES]
    return names, fish_ids


def audit_split(split: str, names: list[str], fish_ids: list[int]):
    image_dir = OBSEA_ROOT / split / "images"
    label_dir = OBSEA_ROOT / split / "labels"
    images = sorted(path for path in image_dir.iterdir() if path.is_file())
    class_counts = {name: 0 for name in names}
    rows = 0
    invalid_rows = 0
    images_with_fish = 0
    images_without_fish = 0
    fish_instances = 0
    non_fish_instances = 0
    width_height = {}
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        has_fish = False
        for line in read_rows(label_path):
            parts = line.split()
            if len(parts) != 5:
                invalid_rows += 1
                continue
            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(names):
                invalid_rows += 1
                continue
            rows += 1
            class_counts[names[class_id]] += 1
            if class_id in fish_ids:
                has_fish = True
                fish_instances += 1
            else:
                non_fish_instances += 1
        if has_fish:
            images_with_fish += 1
        else:
            images_without_fish += 1
        try:
            from PIL import Image

            with Image.open(image_path) as image:
                key = f"{image.width}x{image.height}"
                width_height[key] = width_height.get(key, 0) + 1
        except Exception:
            pass
    return {
        "split": split,
        "images": len(images),
        "labels": len(list(label_dir.glob("*.txt"))),
        "annotation_rows": rows,
        "invalid_rows": invalid_rows,
        "fish_instances": fish_instances,
        "non_fish_instances": non_fish_instances,
        "images_with_fish": images_with_fish,
        "images_without_fish": images_without_fish,
        "class_counts": {name: count for name, count in class_counts.items() if count},
        "image_size_counts": dict(sorted(width_height.items())),
    }


def prepare_test(names: list[str], fish_ids: list[int], force: bool):
    dataset_dir = DERIVED / DERIVED_NAME
    if dataset_dir.exists():
        if not force:
            raise SystemExit(
                f"{dataset_dir} already exists; pass --force to rebuild this derived test set"
            )
        shutil.rmtree(dataset_dir)
    image_dir = dataset_dir / "images"
    label_dir = dataset_dir / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    source_images = sorted(
        path for path in (OBSEA_ROOT / "test" / "images").iterdir() if path.is_file()
    )
    fish_set = set(fish_ids)
    stats = {
        "source_images": len(source_images),
        "converted_instances": 0,
        "excluded_non_fish_instances": 0,
        "invalid_rows": 0,
        "images_linked": 0,
        "images_copied": 0,
        "empty_label_files": 0,
        "duplicate_rows_removed": 0,
    }
    seen_names = set()
    for image_path in source_images:
        if image_path.name in seen_names:
            raise RuntimeError(f"Duplicate OBSEA image name: {image_path.name}")
        seen_names.add(image_path.name)
        rows = []
        seen_rows = set()
        for line in read_rows(OBSEA_ROOT / "test" / "labels" / f"{image_path.stem}.txt"):
            parts = line.split()
            if len(parts) != 5:
                stats["invalid_rows"] += 1
                continue
            class_id = int(float(parts[0]))
            if class_id < 0 or class_id >= len(names):
                stats["invalid_rows"] += 1
                continue
            if class_id not in fish_set:
                stats["excluded_non_fish_instances"] += 1
                continue
            row = "0 " + " ".join(parts[1:])
            if row in seen_rows:
                stats["duplicate_rows_removed"] += 1
                continue
            seen_rows.add(row)
            rows.append(row)
            stats["converted_instances"] += 1
        if not rows:
            stats["empty_label_files"] += 1
        destination = image_dir / image_path.name
        try:
            destination.hardlink_to(image_path)
            stats["images_linked"] += 1
        except OSError:
            shutil.copy2(image_path, destination)
            stats["images_copied"] += 1
        (label_dir / f"{image_path.stem}.txt").write_text(
            "\n".join(rows) + ("\n" if rows else ""),
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
                "nc": 1,
                "names": ["fish"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    stats["dataset_dir"] = str(dataset_dir)
    stats["data_yaml"] = str(data_yaml)
    return dataset_dir, stats


def run_evaluation(data_yaml: Path):
    from ultralytics import YOLO

    model = YOLO(str(CHECKPOINT))
    run_root = Path.home() / "AppData" / "Local" / "Temp" / "opencode" / "ultralytics_runs"
    results = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=SETTINGS["imgsz"],
        conf=SETTINGS["conf"],
        iou=SETTINGS["iou"],
        max_det=SETTINGS["max_det"],
        device=SETTINGS["device"],
        workers=SETTINGS["workers"],
        single_cls=SETTINGS["single_cls"],
        agnostic_nms=SETTINGS["agnostic_nms"],
        plots=False,
        verbose=True,
        project=str(run_root),
        name="obsea_fish_test",
        exist_ok=True,
    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild the derived test set")
    args = parser.parse_args()

    if not (OBSEA_ROOT / "data.yaml").exists():
        raise SystemExit(f"OBSEA data.yaml not found at {OBSEA_ROOT}")
    names, fish_ids = audit_class_names()
    train_audit = audit_split("train", names, fish_ids)
    valid_audit = audit_split("valid", names, fish_ids)
    test_audit = audit_split("test", names, fish_ids)

    dataset_dir, derived_stats = prepare_test(names, fish_ids, args.force)
    results = run_evaluation(dataset_dir / "data.yaml")
    instances_evaluated = int(results.nt_per_class.sum())
    overall = {
        "precision": round(float(results.results_dict.get("metrics/precision(B)", 0.0)), 6),
        "recall": round(float(results.results_dict.get("metrics/recall(B)", 0.0)), 6),
        "mAP50": round(float(results.results_dict.get("metrics/mAP50(B)", 0.0)), 6),
        "mAP50_95": round(float(results.results_dict.get("metrics/mAP50-95(B)", 0.0)), 6),
        "mAP75": round(float(getattr(results.box, "map75", 0.0)), 6),
    }
    payload = {
        "_metadata": {
            "timestamp": timestamp(),
            "script": "scratch/run_obsea_evaluation.py",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "checkpoint_size_mb": round(CHECKPOINT.stat().st_size / (1024 * 1024), 2),
            "environment": environment_info(),
            "settings": {**SETTINGS, "split": "test"},
            "external_dataset": {
                "name": "OBSEA fish detection dataset (YOLO)",
                "doi": OBSEA_DOI,
                "url": "https://zenodo.org/records/14888440",
                "license": OBSEA_LICENSE,
                "zip": str(OBSEA_ZIP.relative_to(SCRIPT_ROOT)),
                "zip_md5": "80daf33935f424e66fb35e31ef3b1780",
                "root": str(OBSEA_ROOT.relative_to(SCRIPT_ROOT)),
                "test_images": test_audit["images"],
                "test_annotation_rows": test_audit["annotation_rows"],
                "class_names": names,
                "fish_class_ids": fish_ids,
                "non_fish_class_names": sorted(NON_FISH_NAMES),
            },
            "class_agnostic_protocol": {
                "ground_truth": "all fish-taxon annotations mapped to class 0; Diver and Octopus annotations excluded",
                "predictions": "all 13 local species predictions mapped to class 0; single_cls and agnostic NMS",
                "background_images": "all test images retained; images without fish labels are backgrounds",
                "deduplication": "identical class-agnostic boxes within an image are deduplicated",
                "species_metrics": "not computed; local and OBSEA taxonomies do not match",
                "tuning": "none; zero-shot external evaluation only",
            },
            "limitations": [
                "This measures class-agnostic fish detection under domain shift, not 13-class species accuracy.",
                "The local checkpoint was trained on the local Roboflow Dataset, not on OBSEA.",
                "OBSEA annotations include 21 fish taxa; Diver and Octopus annotations are excluded from fish ground truth.",
                "Predictions on excluded non-fish objects remain false positives because every prediction is treated as fish.",
                "The OBSEA test split is used once for reporting; no threshold or model selection was performed on it.",
            ],
        },
        "audit": {
            "train": train_audit,
            "valid": valid_audit,
            "test": test_audit,
            "fish_class_ids": fish_ids,
            "non_fish_class_names": sorted(NON_FISH_NAMES),
        },
        "derived_test": derived_stats,
        "instances_evaluated": instances_evaluated,
        "overall": overall,
        "per_class": {
            "fish": {
                "instances": instances_evaluated,
                "precision": overall["precision"],
                "recall": overall["recall"],
                "ap50": overall["mAP50"],
                "ap50_95": overall["mAP50_95"],
            }
        },
        "speed_ms_per_image": {
            key: round(float(value), 3) for key, value in results.speed.items()
        },
        "run_dir": str(results.save_dir),
    }
    write_json(OUTPUT_JSON, payload)
    print(json.dumps({"overall": overall, "test": test_audit, "derived": derived_stats}, indent=2))
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
