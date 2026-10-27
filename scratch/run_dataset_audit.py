import collections
import json
import sys
from pathlib import Path

import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_common import (
    CLASS_NAMES,
    DATASET,
    IMAGE_EXTENSIONS,
    OUTPUTS,
    ROOT,
    VIDEO_EXTENSIONS,
    format_box,
    list_images,
    parse_label_line,
    read_label_file,
    sha256_file,
    source_stem,
    timestamp,
    write_json,
)

SPLITS = ["train", "valid", "test"]


def audit_split(split):
    images = list_images(split)
    label_dir = DATASET / split / "labels"
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}
    missing_labels = sorted(image_stems - label_stems)
    unpaired_labels = sorted(label_stems - image_stems)

    formats = collections.Counter()
    box_class_counts = collections.Counter()
    polygon_class_counts = collections.Counter()
    converted_class_counts = collections.Counter()
    invalid_examples = []
    duplicate_rows = 0
    files_with_polygon = 0
    files_with_mixed = 0
    files_with_boxes = 0
    empty_label_files = 0
    polygon_out_of_range = 0
    total_rows = 0
    converted_instances = 0

    for label_path in labels:
        rows = read_label_file(label_path)
        row_formats = set()
        seen = set()
        if not rows:
            empty_label_files += 1
        for line in rows:
            total_rows += 1
            parsed = parse_label_line(line)
            if "error" in parsed:
                formats["invalid"] += 1
                if len(invalid_examples) < 25:
                    invalid_examples.append(
                        {"file": label_path.name, "line": line[:300], "error": parsed["error"]}
                    )
                continue
            row_formats.add(parsed["format"])
            formats[parsed["format"]] += 1
            class_name = CLASS_NAMES[parsed["class_id"]]
            converted_class_counts[class_name] += 1
            converted_instances += 1
            if parsed["format"] == "box":
                box_class_counts[class_name] += 1
            else:
                polygon_class_counts[class_name] += 1
                if parsed["out_of_range"]:
                    polygon_out_of_range += 1
            text = format_box(parsed["class_id"], parsed["box"])
            if text in seen:
                duplicate_rows += 1
            seen.add(text)
        if "polygon" in row_formats:
            files_with_polygon += 1
        if "box" in row_formats:
            files_with_boxes += 1
        if "box" in row_formats and "polygon" in row_formats:
            files_with_mixed += 1

    sizes = collections.Counter()
    modes = collections.Counter()
    unreadable = []
    for image_path in images:
        try:
            with Image.open(image_path) as image:
                sizes[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                image.verify()
        except Exception as error:
            unreadable.append({"file": image_path.name, "error": str(error)})

    return {
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(missing_labels),
        "missing_label_examples": missing_labels[:20],
        "unpaired_labels": len(unpaired_labels),
        "unpaired_label_examples": unpaired_labels[:20],
        "total_label_rows": total_rows,
        "box_rows": formats["box"],
        "polygon_rows": formats["polygon"],
        "invalid_rows": formats["invalid"],
        "invalid_row_examples": invalid_examples,
        "converted_instances_total": converted_instances,
        "box_class_counts": dict(box_class_counts),
        "polygon_class_counts": dict(polygon_class_counts),
        "converted_class_counts": {
            class_name: converted_class_counts.get(class_name, 0) for class_name in CLASS_NAMES
        },
        "classes_without_instances": [
            class_name for class_name in CLASS_NAMES if converted_class_counts.get(class_name, 0) == 0
        ],
        "files_with_polygon_rows": files_with_polygon,
        "files_with_box_rows": files_with_boxes,
        "files_with_mixed_rows": files_with_mixed,
        "empty_label_files": empty_label_files,
        "duplicate_box_rows": duplicate_rows,
        "polygon_rows_out_of_range": polygon_out_of_range,
        "image_sizes": dict(sizes),
        "image_modes": dict(modes),
        "unreadable_images": unreadable,
    }


def hash_and_source_analysis(split_audits):
    hashes = {}
    hash_lists = {}
    prefixes = {}
    for split in SPLITS:
        split_hashes = {}
        split_hash_list = []
        split_prefixes = collections.defaultdict(list)
        for image_path in list_images(split):
            digest = sha256_file(image_path)
            split_hashes[digest] = image_path
            split_hash_list.append(digest)
            split_prefixes[source_stem(image_path.name)].append(image_path.name)
        hashes[split] = split_hashes
        hash_lists[split] = split_hash_list
        prefixes[split] = dict(split_prefixes)

    exact_pairs = []
    exact_counts = {}
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        common = set(hashes[first]) & set(hashes[second])
        exact_counts[f"{first}_vs_{second}"] = len(common)
        for digest in sorted(common):
            exact_pairs.append(
                {
                    "hash": digest,
                    "split_a": first,
                    "file_a": hashes[first][digest].name,
                    "split_b": second,
                    "file_b": hashes[second][digest].name,
                }
            )

    within_split_duplicates = {}
    for split in SPLITS:
        counts = collections.Counter(hash_lists[split])
        duplicates = sum(1 for count in counts.values() if count > 1)
        within_split_duplicates[split] = duplicates

    prefix_overlaps = {}
    for first, second in (("train", "valid"), ("train", "test"), ("valid", "test")):
        common = sorted(set(prefixes[first]) & set(prefixes[second]))
        prefix_overlaps[f"{first}_vs_{second}"] = {
            "prefix_count": len(common),
            "prefixes": common,
            "examples": [
                {
                    "prefix": prefix,
                    "split_a_images": prefixes[first][prefix],
                    "split_b_images": prefixes[second][prefix],
                }
                for prefix in common[:20]
            ],
        }

    return {
        "exact_duplicate_counts": exact_counts,
        "exact_duplicate_pairs": exact_pairs,
        "within_split_duplicate_hashes": within_split_duplicates,
        "source_prefix_overlaps": prefix_overlaps,
        "hashes": {split: dict(hashes[split]) for split in SPLITS},
        "prefixes": prefixes,
    }


def clean_test_analysis(leakage):
    test_images = list_images("test")
    train_hashes = set(leakage["hashes"]["train"])
    valid_hashes = set(leakage["hashes"]["valid"])
    train_prefixes = set(leakage["prefixes"]["train"])
    valid_prefixes = set(leakage["prefixes"]["valid"])
    excluded = []
    clean = []
    for image_path in test_images:
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
            excluded.append({"file": image_path.name, "prefix": prefix, "reasons": reasons})
        else:
            clean.append(image_path)

    class_counts = collections.Counter()
    polygon_boxes = 0
    for image_path in clean:
        label_path = DATASET / "test" / "labels" / f"{image_path.stem}.txt"
        for line in read_label_file(label_path):
            parsed = parse_label_line(line)
            if "error" in parsed:
                continue
            class_counts[CLASS_NAMES[parsed["class_id"]]] += 1
            if parsed["format"] == "polygon":
                polygon_boxes += 1

    return {
        "test_images": len(test_images),
        "excluded_images": len(excluded),
        "clean_images": len(clean),
        "excluded_by_reason": dict(collections.Counter(reason for item in excluded for reason in item["reasons"])),
        "excluded_examples": excluded[:30],
        "clean_instances_total": int(sum(class_counts.values())),
        "clean_polygon_derived_boxes": polygon_boxes,
        "clean_class_counts": {class_name: class_counts.get(class_name, 0) for class_name in CLASS_NAMES},
        "clean_classes_without_instances": [
            class_name for class_name in CLASS_NAMES if class_counts.get(class_name, 0) == 0
        ],
    }


def data_yaml_analysis():
    yaml_path = DATASET / "data.yaml"
    raw = yaml_path.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    resolved = {}
    for key in ("train", "val", "test"):
        value = parsed.get(key)
        if value is None:
            resolved[key] = None
            continue
        target = (DATASET / value).resolve()
        resolved[key] = {"value": value, "resolved": str(target), "exists": target.exists()}
    return {
        "path": str(yaml_path),
        "raw": raw,
        "parsed": parsed,
        "resolved_split_paths": resolved,
        "roboflow_readme_expected_images": 8242,
        "actual_images_total": sum(len(list_images(split)) for split in SPLITS),
    }


def checkpoint_analysis(data_yaml_info):
    from ultralytics import YOLO

    checkpoint = ROOT / "models" / "best.pt"
    model = YOLO(str(checkpoint))
    names = [model.names[index] for index in sorted(model.names)]
    yaml_names = data_yaml_info["parsed"].get("names", [])
    train_args = model.ckpt.get("train_args", {}) if hasattr(model, "ckpt") else {}
    return {
        "path": str(checkpoint),
        "size_mb": round(checkpoint.stat().st_size / (1024 * 1024), 2),
        "sha256": sha256_file(checkpoint),
        "task": model.task,
        "num_classes": len(names),
        "names": names,
        "names_match_data_yaml": names == yaml_names,
        "data_yaml_num_classes": data_yaml_info["parsed"].get("nc"),
        "train_args": {
            key: train_args.get(key)
            for key in ("data", "imgsz", "epochs", "batch", "model", "device", "pretrained")
        },
    }


def qualitative_samples_analysis(leakage):
    sample_dir = ROOT / "qualitative_samples"
    images = sorted(
        path for path in sample_dir.glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    labels_dir = sample_dir / "labels"
    labels = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
    dataset_hashes = set()
    dataset_prefixes = set()
    for split in SPLITS:
        dataset_hashes.update(leakage["hashes"][split])
        dataset_prefixes.update(leakage["prefixes"][split])
    exact_overlap = []
    prefix_overlap = []
    for image_path in images:
        digest = sha256_file(image_path)
        if digest in dataset_hashes:
            exact_overlap.append(image_path.name)
        prefix = source_stem(image_path.name)
        if prefix in dataset_prefixes:
            prefix_overlap.append({"file": image_path.name, "prefix": prefix})
    return {
        "path": str(sample_dir),
        "images": len(images),
        "labels": len(labels),
        "labels_directory_exists": labels_dir.exists(),
        "exact_duplicate_with_dataset": len(exact_overlap),
        "exact_duplicate_examples": exact_overlap[:20],
        "source_prefix_overlap_with_dataset": len(prefix_overlap),
        "source_prefix_overlap_examples": prefix_overlap[:20],
        "is_independent_annotated_dataset": False,
    }


def video_analysis():
    search_roots = [
        ROOT / "client" / "public",
        ROOT / "docs",
        ROOT / "scratch",
        ROOT / "server",
        ROOT / "qualitative_samples",
        ROOT / "outputs",
    ]
    videos = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*"):
            if "node_modules" in path.parts or ".venv" in path.parts:
                continue
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(str(path.relative_to(ROOT)))
    return {"video_files_found": videos, "real_fish_video_available": False}


def main():
    split_audits = {split: audit_split(split) for split in SPLITS}
    leakage = hash_and_source_analysis(split_audits)
    data_yaml_info = data_yaml_analysis()
    checkpoint_info = checkpoint_analysis(data_yaml_info)
    clean_test = clean_test_analysis(leakage)
    sample_info = qualitative_samples_analysis(leakage)
    videos = video_analysis()

    report = {
        "_metadata": {
            "timestamp": timestamp(),
            "script": "scratch/run_dataset_audit.py",
            "dataset_root": str(DATASET),
            "notes": [
                "This audit inspects the local dataset only; it does not download or modify dataset files.",
                "Label rows with 5 fields are YOLO detection boxes; rows with more than 5 fields are polygon annotations and are counted separately.",
                "Exact duplicate detection uses SHA-256 file hashes; source leakage uses Roboflow filename stems before '.rf.'.",
                "The clean test subset is conservative and is not a guarantee of source independence.",
            ],
        },
        "data_yaml": data_yaml_info,
        "checkpoint": checkpoint_info,
        "splits": split_audits,
        "leakage": {
            key: value
            for key, value in leakage.items()
            if key not in ("hashes", "prefixes")
        },
        "clean_test_candidate": clean_test,
        "qualitative_samples": sample_info,
        "videos": videos,
    }
    output_path = OUTPUTS / "dataset_audit.json"
    write_json(output_path, report)

    print(json.dumps({
        "output": str(output_path),
        "splits": {
            split: {
                "images": split_audits[split]["images"],
                "box_rows": split_audits[split]["box_rows"],
                "polygon_rows": split_audits[split]["polygon_rows"],
                "invalid_rows": split_audits[split]["invalid_rows"],
                "converted_instances": split_audits[split]["converted_instances_total"],
                "classes_without_instances": split_audits[split]["classes_without_instances"],
            }
            for split in SPLITS
        },
        "data_yaml_paths_valid": {
            key: value["exists"] for key, value in data_yaml_info["resolved_split_paths"].items()
        },
        "checkpoint_names_match": checkpoint_info["names_match_data_yaml"],
        "exact_duplicate_counts": leakage["exact_duplicate_counts"],
        "source_prefix_overlap_counts": {
            key: value["prefix_count"] for key, value in leakage["source_prefix_overlaps"].items()
        },
        "clean_test": {
            "images": clean_test["clean_images"],
            "excluded": clean_test["excluded_images"],
            "instances": clean_test["clean_instances_total"],
            "classes_without_instances": clean_test["clean_classes_without_instances"],
        },
        "qualitative_samples": {
            "images": sample_info["images"],
            "labels": sample_info["labels"],
            "exact_duplicate_with_dataset": sample_info["exact_duplicate_with_dataset"],
            "source_prefix_overlap_with_dataset": sample_info["source_prefix_overlap_with_dataset"],
        },
        "videos": videos,
    }, indent=2))


if __name__ == "__main__":
    main()
