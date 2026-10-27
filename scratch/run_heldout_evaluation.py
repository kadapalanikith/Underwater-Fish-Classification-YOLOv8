import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import (
    CHECKPOINT,
    CLASS_NAMES,
    DATASET,
    OUTPUTS,
    environment_info,
    evaluate_yolo,
    list_images,
    prepare_dataset,
    sha256_file,
    source_stem,
    timestamp,
    write_json,
)


def leakage_filter():
    train_images = list_images("train")
    valid_images = list_images("valid")
    train_hashes = {sha256_file(path) for path in train_images}
    valid_hashes = {sha256_file(path) for path in valid_images}
    train_prefixes = {source_stem(path.name) for path in train_images}
    valid_prefixes = {source_stem(path.name) for path in valid_images}
    clean_names = []
    excluded = []
    for image_path in list_images("test"):
        digest = sha256_file(image_path)
        prefix = source_stem(image_path.name)
        reasons = []
        if digest in train_hashes:
            reasons.append("exact_duplicate_train")
        if digest in valid_hashes:
            reasons.append("exact_duplicate_valid")
        if prefix in train_prefixes:
            reasons.append("source_prefix_train")
        if prefix in valid_prefixes:
            reasons.append("source_prefix_valid")
        if reasons:
            excluded.append({"file": image_path.name, "reasons": reasons})
        else:
            clean_names.append(image_path.name)
    return clean_names, {
        "test_images": len(list_images("test")),
        "clean_images": len(clean_names),
        "excluded_images": len(excluded),
        "excluded_by_reason": {
            reason: sum(1 for item in excluded if reason in item["reasons"])
            for reason in (
                "exact_duplicate_train",
                "exact_duplicate_valid",
                "source_prefix_train",
                "source_prefix_valid",
            )
        },
        "excluded_examples": excluded[:30],
    }


def load_audit_summary():
    path = OUTPUTS / "dataset_audit.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        audit = json.load(handle)
    return {
        "path": str(path),
        "splits": {
            split: {
                "images": audit["splits"][split]["images"],
                "box_rows": audit["splits"][split]["box_rows"],
                "polygon_rows": audit["splits"][split]["polygon_rows"],
                "converted_instances": audit["splits"][split]["converted_instances_total"],
                "classes_without_instances": audit["splits"][split]["classes_without_instances"],
            }
            for split in ("train", "valid", "test")
        },
        "data_yaml_paths_valid": {
            key: value["exists"] for key, value in audit["data_yaml"]["resolved_split_paths"].items()
        },
        "exact_duplicate_counts": audit["leakage"]["exact_duplicate_counts"],
        "source_prefix_overlap_counts": {
            key: value["prefix_count"]
            for key, value in audit["leakage"]["source_prefix_overlaps"].items()
        },
        "test_sample": audit["test_sample"],
        "videos": audit["videos"],
    }


def main():
    clean_names, clean_stats = leakage_filter()
    prepared = {}
    for split, name, include in (
        ("test", "test_full", None),
        ("test", "test_clean", clean_names),
        ("valid", "valid_full", None),
    ):
        dataset_dir, stats = prepare_dataset(split, name, include_names=include)
        prepared[name] = {"dataset_dir": str(dataset_dir), "stats": stats}

    results = {}
    for name, split, run_name in (
        ("test_full", "test", "yolov8_test_full"),
        ("test_clean", "test", "yolov8_test_clean"),
        ("valid_full", "val", "yolov8_valid_full"),
    ):
        stats = prepared[name]["stats"]
        result = evaluate_yolo(
            prepared[name]["stats"]["data_yaml"],
            split=split,
            run_name=run_name,
            image_count=stats["included_images"],
            checkpoint=CHECKPOINT,
        )
        result["dataset_stats"] = stats
        results[name] = result

    warnings = [
        "The local Dataset/data.yaml split paths are invalid in the current layout; derived YAML files under outputs/derived were used instead.",
        "Local labels mix 5-field detection rows and >5-field polygon rows. Evaluation labels convert polygons to bounding boxes; 29 test polygons and 84 validation polygons were converted.",
        "The test split has 34 images that exact-match a train image (33 distinct hashes), 212 images sharing a train source stem (202 distinct prefixes), and 70 sharing a valid source stem (69 distinct prefixes); test_full is not a fully independent held-out set.",
        "test_clean excludes 298 test images whose exact hash or Roboflow source stem appears in train/valid; it is a conservative leakage-filtered subset, not proof of source independence.",
        "The validation split was used during training/model selection; valid_full is not an independent test result. AngelFish has no validation instances.",
        "Metrics were produced by Ultralytics validation on CPU with conf=0.001, iou=0.6, imgsz=640. They are not comparable to the original notebook's validation logs without rerunning the original training pipeline.",
    ]

    report = {
        "_metadata": {
            "timestamp": timestamp(),
            "script": "scratch/run_heldout_evaluation.py",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "checkpoint_size_mb": round(CHECKPOINT.stat().st_size / (1024 * 1024), 2),
            "dataset_root": str(DATASET),
            "class_names": CLASS_NAMES,
            "evaluation_protocol": {
                "framework": "Ultralytics YOLO.val",
                "split_requested": {"test_full": "test", "test_clean": "test", "valid_full": "val"},
                "imgsz": 640,
                "conf": 0.001,
                "iou": 0.6,
                "device": "cpu",
                "workers": 0,
                "plots": False,
            },
            "leakage_filter": clean_stats,
            "environment": environment_info(),
            "warnings": warnings,
            "audit_summary": load_audit_summary(),
        },
        "results": results,
    }
    output_path = OUTPUTS / "heldout_yolov8_metrics.json"
    write_json(output_path, report)

    print(json.dumps({
        "output": str(output_path),
        "leakage_filter": clean_stats,
        "results": {
            name: {
                "images": result["images_evaluated"],
                "instances": result["instances_evaluated"],
                "overall": result["overall"],
                "classes_without_ground_truth": result["classes_without_ground_truth"],
                "low_support_classes": result["low_support_classes"],
                "speed_ms_per_image": result["speed_ms_per_image"],
            }
            for name, result in results.items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
