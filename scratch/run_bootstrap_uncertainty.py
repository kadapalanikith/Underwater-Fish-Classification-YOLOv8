"""Bootstrap confidence intervals for held-out and enhancement evaluations.

The script runs the recorded Ultralytics validation protocol and captures each
image's true-positive matrix together with prediction confidences/classes and
ground-truth classes. It caches those stats under outputs/uncertainty_stats/,
checks that the reconstructed point estimates match the saved artifacts, then
resamples images and Roboflow source stems with replacement.

This is CPU inference over existing local images only. It does not train,
download, or modify checkpoints/datasets. The summary is written to
outputs/bootstrap_uncertainty.json.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT / "scratch"))

from eval_common import (  # noqa: E402
    CHECKPOINT,
    CLASS_NAMES,
    DERIVED,
    OUTPUTS,
    environment_info,
    sha256_file,
    source_stem,
    timestamp,
    write_json,
)
from src.evaluation.uncertainty import (  # noqa: E402
    cluster_bootstrap_indices,
    holm_adjust,
    image_bootstrap_indices,
    metrics_from_rows,
    paired_delta_summary,
    source_clusters,
    summarize_bootstrap,
)

STATS_DIR = OUTPUTS / "uncertainty_stats"
OUTPUT_JSON = OUTPUTS / "bootstrap_uncertainty.json"
HELDOUT_JSON = OUTPUTS / "heldout_yolov8_metrics.json"
ENHANCEMENT_JSON = OUTPUTS / "enhancement_accuracy_test_clean.json"

DATASETS = {
    "test_full": {"dir": DERIVED / "test_full"},
    "test_clean": {"dir": DERIVED / "test_clean"},
    "valid_full": {"dir": DERIVED / "valid_full"},
    "raw_original": {"dir": DERIVED / "enhancement_raw_original", "reuse": "test_clean"},
    "raw_jpeg95": {"dir": DERIVED / "enhancement_raw_jpeg95"},
    "clahe": {"dir": DERIVED / "enhancement_clahe"},
    "gray_world": {"dir": DERIVED / "enhancement_gray_world"},
    "fusion": {"dir": DERIVED / "enhancement_fusion"},
}

ENHANCEMENT_METHODS = ["raw_original", "raw_jpeg95", "clahe", "gray_world", "fusion"]
PRIMARY_METHODS = ["clahe", "gray_world", "fusion"]
CONTROL = "raw_jpeg95"
METRICS = ["mAP50", "mAP50_95", "precision", "recall"]
EVALUATION_SETTINGS = {
    "imgsz": 640,
    "conf": 0.001,
    "iou": 0.6,
    "max_det": 300,
    "device": "cpu",
    "batch": 8,
    "multi_label": True,
}
CACHE_SCHEMA = 2


def list_images_in(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )


def names_hash(paths: list[Path]) -> str:
    return hashlib.sha256("\n".join(path.name for path in paths).encode("utf-8")).hexdigest()


def labels_hash(paths: list[Path]) -> str:
    hasher = hashlib.sha256()
    for path in paths:
        hasher.update(path.name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def read_json_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_gz(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        json.dump(payload, handle, indent=None, separators=(",", ":"))


def make_validator(holder: dict):
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class CaptureValidator(DetectionValidator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.image_records = []
            holder["validator"] = self

        def update_metrics(self, preds, batch):
            for si, pred in enumerate(preds):
                self.seen += 1
                pbatch = self._prepare_batch(si, batch)
                cls = pbatch["cls"].cpu().numpy()
                predn = self._prepare_pred(pred)
                tp = self._process_batch(predn, pbatch)["tp"]
                conf = predn["conf"].cpu().numpy()
                pred_cls = predn["cls"].cpu().numpy()
                self.metrics.update_stats(
                    {
                        "tp": tp,
                        "target_cls": cls,
                        "target_img": np.unique(cls),
                        "conf": conf,
                        "pred_cls": pred_cls,
                        "im_name": Path(pbatch["im_file"]).name,
                    }
                )
                self.image_records.append(
                    {
                        "name": Path(pbatch["im_file"]).name,
                        "tp": tp.astype(np.uint8).tolist(),
                        "conf": conf.tolist(),
                        "pred_cls": pred_cls.tolist(),
                        "target_cls": cls.tolist(),
                    }
                )

    return CaptureValidator


def run_validation(dataset_dir: Path, cache_key: str):
    from ultralytics import YOLO

    holder = {}
    validator_class = make_validator(holder)
    model = YOLO(str(CHECKPOINT))
    started = time.perf_counter()
    print(
        f"Evaluating {dataset_dir.name} with Ultralytics val "
        f"(conf={EVALUATION_SETTINGS['conf']}, iou={EVALUATION_SETTINGS['iou']}, "
        f"multi_label={EVALUATION_SETTINGS['multi_label']})",
        flush=True,
    )
    model.val(
        data=str(dataset_dir / "data.yaml"),
        split="val",
        validator=validator_class,
        imgsz=EVALUATION_SETTINGS["imgsz"],
        conf=EVALUATION_SETTINGS["conf"],
        iou=EVALUATION_SETTINGS["iou"],
        max_det=EVALUATION_SETTINGS["max_det"],
        device=EVALUATION_SETTINGS["device"],
        batch=EVALUATION_SETTINGS["batch"],
        workers=0,
        plots=False,
        verbose=False,
        project=str(OUTPUTS / "diagnostics"),
        name=f"bootstrap_{cache_key}",
        exist_ok=True,
    )
    elapsed = time.perf_counter() - started
    validator = holder["validator"]
    records = validator.image_records
    print(
        f"  captured {len(records)} image rows in {elapsed:.1f}s "
        f"({1000 * elapsed / max(len(records), 1):.1f} ms/image)",
        flush=True,
    )
    return records


def stats_metadata(dataset_dir: Path, image_paths: list[Path]) -> dict:
    import ultralytics

    return {
        "schema": CACHE_SCHEMA,
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "settings": EVALUATION_SETTINGS,
        "image_count": len(image_paths),
        "image_names_sha256": names_hash(image_paths),
        "labels_sha256": labels_hash(
            [dataset_dir / "labels" / f"{path.stem}.txt" for path in image_paths]
        ),
        "ultralytics": ultralytics.__version__,
    }


def parse_stats(payload: dict) -> list[dict]:
    records = []
    for item in payload["records"]:
        records.append(
            {
                "name": item["name"],
                "tp": np.asarray(item["tp"], dtype=bool).reshape(-1, 10),
                "conf": np.asarray(item["conf"], dtype=np.float64).reshape(-1),
                "pred_cls": np.asarray(item["pred_cls"], dtype=np.int64).reshape(-1),
                "target_cls": np.asarray(item["target_cls"], dtype=np.int64).reshape(-1),
            }
        )
    return records


def build_stats(dataset_dir: Path, cache_key: str, force: bool = False) -> list[dict]:
    image_paths = list_images_in(dataset_dir / "images")
    if not image_paths:
        raise RuntimeError(f"No images in {dataset_dir / 'images'}")
    metadata = stats_metadata(dataset_dir, image_paths)
    cache_path = STATS_DIR / f"{cache_key}.json.gz"
    if cache_path.exists() and not force:
        cached = read_json_gz(cache_path)
        old = cached.get("metadata", {})
        if all(old.get(key) == metadata[key] for key in metadata):
            print(f"Using cached validation stats: {cache_path}", flush=True)
            return parse_stats(cached)

    if force and cache_path.exists():
        print(f"Rebuilding validation stats (--force): {cache_path}", flush=True)
    records = run_validation(dataset_dir, cache_key)
    if len(records) != len(image_paths):
        raise RuntimeError(
            f"Expected {len(image_paths)} validation rows, captured {len(records)}"
        )
    records.sort(key=lambda row: row["name"])
    payload = {
        "metadata": {
            "cache_key": cache_key,
            "created": timestamp(),
            "script": "scratch/run_bootstrap_uncertainty.py",
            **metadata,
            "environment": environment_info(),
        },
        "records": records,
    }
    write_json_gz(cache_path, payload)
    print(f"Wrote {cache_path}", flush=True)
    return parse_stats(payload)


def point_metrics(records):
    return metrics_from_rows(records, range(len(records)), names=CLASS_NAMES)


def bootstrap_distribution(records, resample_indices, fixed_classes, per_class=False):
    overall_values = {metric: [] for metric in METRICS}
    per_class_values = {
        class_name: {"ap50": [], "ap50_95": []} for class_name in CLASS_NAMES
    }
    for count, indices in enumerate(resample_indices, start=1):
        try:
            metrics = metrics_from_rows(records, indices, names=CLASS_NAMES)
        except ValueError:
            for metric in METRICS:
                overall_values[metric].append(np.nan)
            if per_class:
                for class_name in CLASS_NAMES:
                    per_class_values[class_name]["ap50"].append(np.nan)
                    per_class_values[class_name]["ap50_95"].append(np.nan)
        else:
            all_present = all(name in metrics["per_class"] for name in fixed_classes)
            for metric in METRICS:
                overall_values[metric].append(
                    metrics["overall"][metric] if all_present else np.nan
                )
            if per_class:
                present = metrics["per_class"]
                for class_name in CLASS_NAMES:
                    item = present.get(class_name)
                    per_class_values[class_name]["ap50"].append(
                        np.nan if item is None else item["ap50"]
                    )
                    per_class_values[class_name]["ap50_95"].append(
                        np.nan if item is None else item["ap50_95"]
                    )
        if count % 500 == 0:
            print(f"  bootstrap {count}/{len(resample_indices)}", flush=True)
    return overall_values, per_class_values


def summarize_distribution(values: dict, point: dict, per_class_values=None, point_per_class=None):
    summary = {
        metric: summarize_bootstrap(values[metric], point[metric])
        for metric in METRICS
    }
    if per_class_values is not None:
        classes = {}
        for class_name, class_values in per_class_values.items():
            full = (point_per_class or {}).get(class_name)
            if full is None:
                continue
            classes[class_name] = {
                "instances": full["instances"],
                "ap50": summarize_bootstrap(class_values["ap50"], full["ap50"]),
                "ap50_95": summarize_bootstrap(class_values["ap50_95"], full["ap50_95"]),
                "fraction_of_resamples_present": float(
                    np.mean(np.isfinite(class_values["ap50"]))
                ),
            }
        summary["per_class"] = classes
    return summary


def load_saved():
    heldout = json.loads(HELDOUT_JSON.read_text(encoding="utf-8"))
    enhancement = json.loads(ENHANCEMENT_JSON.read_text(encoding="utf-8"))
    return heldout, enhancement


def compare_saved(name, computed, saved, tolerance=1e-6):
    differences = {}
    for metric in ("precision", "recall", "mAP50", "mAP50_95"):
        if metric in saved and metric in computed:
            differences[metric] = float(computed[metric] - saved[metric])
    max_abs = max(abs(value) for value in differences.values()) if differences else None
    return {
        "dataset": name,
        "computed": computed,
        "saved": {
            key: saved[key]
            for key in ("precision", "recall", "mAP50", "mAP50_95")
            if key in saved
        },
        "delta": differences,
        "max_abs_delta": max_abs,
        "within_tolerance": None if max_abs is None else bool(max_abs <= tolerance),
    }


def paired_comparisons(distributions, point_estimates, indices_length):
    results = {}
    comparisons = {
        "raw_jpeg95_vs_raw_original": ("raw_jpeg95", "raw_original"),
        **{
            f"{method}_vs_{CONTROL}": (method, CONTROL)
            for method in PRIMARY_METHODS
        },
        **{
            f"{method}_vs_raw_original": (method, "raw_original")
            for method in PRIMARY_METHODS
        },
    }
    for label, (left_name, right_name) in comparisons.items():
        left = distributions[left_name]
        right = distributions[right_name]
        point_left = point_estimates[left_name]["overall"]
        point_right = point_estimates[right_name]["overall"]
        metrics = {}
        for metric in METRICS:
            left_values = np.asarray(left[metric], dtype=np.float64)
            right_values = np.asarray(right[metric], dtype=np.float64)
            if len(left_values) != len(right_values):
                raise RuntimeError("Paired distributions have different lengths")
            point_delta = float(point_left[metric] - point_right[metric])
            metrics[metric] = paired_delta_summary(left_values - right_values, point_delta)
        results[label] = {
            "left": left_name,
            "right": right_name,
            "delta_definition": "left minus right",
            "paired_resamples": int(indices_length),
            "metrics": metrics,
        }
    return results


def add_holm(comparisons):
    family = []
    for method in PRIMARY_METHODS:
        label = f"{method}_vs_{CONTROL}"
        for metric in ("mAP50", "mAP50_95"):
            family.append((label, metric, comparisons[label]["metrics"][metric]["p_two_sided"]))
    adjusted = holm_adjust(
        [item[2] if item[2] is not None else 1.0 for item in family]
    )
    for (label, metric, raw), adj in zip(family, adjusted):
        comparisons[label]["metrics"][metric]["p_two_sided_holm"] = float(adj)
        comparisons[label]["metrics"][metric]["holm_significant_0_05"] = bool(adj < 0.05)
    return comparisons


def supported_difference(summary: dict, alpha=0.05) -> bool:
    low = summary.get("ci_low")
    high = summary.get("ci_high")
    adj = summary.get("p_two_sided_holm", summary.get("p_two_sided"))
    if low is None or high is None or adj is None:
        return False
    return bool((low > 0.0 or high < 0.0) and adj < alpha)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--force-eval", action="store_true")
    parser.add_argument(
        "--no-cluster-bootstrap",
        action="store_true",
        help="Skip the source-stem cluster sensitivity analysis.",
    )
    args = parser.parse_args()
    if args.resamples < 100:
        raise SystemExit("Use at least 100 bootstrap resamples")

    if not HELDOUT_JSON.exists() or not ENHANCEMENT_JSON.exists():
        raise SystemExit("Run the held-out and enhancement evaluations first")
    heldout, enhancement = load_saved()

    stats = {}
    for key, spec in DATASETS.items():
        if spec.get("reuse"):
            continue
        stats[key] = build_stats(spec["dir"], key, force=args.force_eval)

    records = {}
    names = {}
    point = {}
    for key, spec in DATASETS.items():
        source_key = spec.get("reuse", key)
        records[key] = stats[source_key]
        names[key] = [row["name"] for row in records[key]]
        point[key] = point_metrics(records[key])
        print(
            f"{key}: {len(records[key])} images, "
            f"{sum(len(row['conf']) for row in records[key])} predictions, "
            f"mAP50={point[key]['overall']['mAP50']:.6f}, "
            f"mAP50-95={point[key]['overall']['mAP50_95']:.6f}",
            flush=True,
        )

    validation = []
    for key in ("test_full", "test_clean", "valid_full"):
        saved = heldout["results"][key]["overall"]
        validation.append(compare_saved(key, point[key]["overall"], saved))
    for key in ENHANCEMENT_METHODS:
        saved = enhancement["results"][key]["overall"]
        validation.append(compare_saved(f"enhancement_{key}", point[key]["overall"], saved))

    print("\nPoint-estimate validation against saved artifacts:", flush=True)
    for item in validation:
        print(
            f"  {item['dataset']}: max|delta|={item['max_abs_delta']:.10f} "
            f"within_tolerance={item['within_tolerance']}",
            flush=True,
        )

    if any(item["within_tolerance"] is False for item in validation):
        raise SystemExit(
            "Captured validation stats do not reproduce the saved metrics; "
            "inspect the evaluator before publishing bootstrap intervals."
        )

    baseline_bootstrap = {}
    for key, n_images in (
        ("test_full", len(records["test_full"])),
        ("test_clean", len(records["test_clean"])),
        ("valid_full", len(records["valid_full"])),
    ):
        print(f"\nBootstrapping {key} ({n_images} images, {args.resamples} resamples)", flush=True)
        indices = image_bootstrap_indices(n_images, args.resamples, args.seed)
        fixed_classes = list(point[key]["per_class"])
        overall_values, per_class_values = bootstrap_distribution(
            records[key], indices, fixed_classes, per_class=True
        )
        entry = summarize_distribution(
            overall_values,
            point[key]["overall"],
            per_class_values=per_class_values,
            point_per_class=point[key]["per_class"],
        )
        entry["n_images"] = n_images
        entry["n_instances"] = int(
            sum(len(row["target_cls"]) for row in records[key])
        )
        entry["fixed_classes"] = fixed_classes
        entry["overall_conditioning"] = (
            "overall metrics require every fixed class to have at least one "
            "ground-truth instance in the resample"
        )

        test_names = names[key]
        clusters = source_clusters(test_names, source_stem)
        entry["source_stem_clusters"] = len(clusters)
        if args.no_cluster_bootstrap:
            entry["source_cluster_bootstrap"] = None
        else:
            print(
                f"  source-stem cluster bootstrap ({len(clusters)} clusters)",
                flush=True,
            )
            cluster_indices = cluster_bootstrap_indices(
                clusters, args.resamples, args.seed + 1
            )
            cluster_values, cluster_per_class = bootstrap_distribution(
                records[key], cluster_indices, fixed_classes, per_class=True
            )
            entry["source_cluster_bootstrap"] = summarize_distribution(
                cluster_values,
                point[key]["overall"],
                per_class_values=cluster_per_class,
                point_per_class=point[key]["per_class"],
            )
            entry["source_cluster_bootstrap"]["n_clusters"] = len(clusters)
        baseline_bootstrap[key] = entry

    print("\nBootstrapping enhancement methods on the identical 399 images", flush=True)
    indices = image_bootstrap_indices(
        len(records["test_clean"]), args.resamples, args.seed
    )
    enhancement_fixed_classes = list(point["test_clean"]["per_class"])
    distributions = {}
    for method in ENHANCEMENT_METHODS:
        print(f"  {method}", flush=True)
        values, _ = bootstrap_distribution(
            records[method], indices, enhancement_fixed_classes
        )
        distributions[method] = values

    enhancement_point = {
        method: point[method]["overall"] for method in ENHANCEMENT_METHODS
    }
    method_summaries = {
        method: summarize_distribution(
            distributions[method], enhancement_point[method]
        )
        for method in ENHANCEMENT_METHODS
    }
    image_paired = add_holm(
        paired_comparisons(distributions, point, len(indices))
    )

    cluster_entry = None
    if not args.no_cluster_bootstrap:
        print("\nPaired source-stem cluster bootstrap for enhancement", flush=True)
        enhancement_clusters = source_clusters(names["test_clean"], source_stem)
        cluster_indices = cluster_bootstrap_indices(
            enhancement_clusters, args.resamples, args.seed + 2
        )
        cluster_distributions = {}
        for method in ENHANCEMENT_METHODS:
            print(f"  {method}", flush=True)
            values, _ = bootstrap_distribution(
                records[method], cluster_indices, enhancement_fixed_classes
            )
            cluster_distributions[method] = values
        cluster_paired = add_holm(
            paired_comparisons(cluster_distributions, point, len(cluster_indices))
        )
        cluster_method_summaries = {
            method: summarize_distribution(
                cluster_distributions[method], enhancement_point[method]
            )
            for method in ENHANCEMENT_METHODS
        }
        cluster_entry = {
            "n_clusters": len(enhancement_clusters),
            "method_summaries": cluster_method_summaries,
            "paired": cluster_paired,
        }

    primary_summary = []
    for method in PRIMARY_METHODS:
        label = f"{method}_vs_{CONTROL}"
        comparison = image_paired[label]
        primary_summary.append(
            {
                "method": method,
                "control": CONTROL,
                "delta_mAP50": comparison["metrics"]["mAP50"]["point"],
                "delta_mAP50_ci": [
                    comparison["metrics"]["mAP50"]["ci_low"],
                    comparison["metrics"]["mAP50"]["ci_high"],
                ],
                "delta_mAP50_holm_p": comparison["metrics"]["mAP50"].get(
                    "p_two_sided_holm"
                ),
                "delta_mAP50_supported": supported_difference(
                    comparison["metrics"]["mAP50"]
                ),
                "delta_mAP50_95": comparison["metrics"]["mAP50_95"]["point"],
                "delta_mAP50_95_ci": [
                    comparison["metrics"]["mAP50_95"]["ci_low"],
                    comparison["metrics"]["mAP50_95"]["ci_high"],
                ],
                "delta_mAP50_95_holm_p": comparison["metrics"]["mAP50_95"].get(
                    "p_two_sided_holm"
                ),
                "delta_mAP50_95_supported": supported_difference(
                    comparison["metrics"]["mAP50_95"]
                ),
            }
        )

    payload = {
        "_metadata": {
            "timestamp": timestamp(),
            "script": "scratch/run_bootstrap_uncertainty.py",
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256_file(CHECKPOINT),
            "environment": environment_info(),
            "evaluation_protocol": {
                **EVALUATION_SETTINGS,
                "implementation": "Ultralytics DetectionValidator with captured per-image TP rows",
                "matching": "Ultralytics greedy per-image same-class matching; multi_label=True NMS",
                "iou_thresholds": "0.50:0.05:0.95",
                "ap": "Ultralytics ap_per_class / 101-point interpolated AP",
                "bootstrap": "nonparametric percentile bootstrap",
                "resamples": args.resamples,
                "seed": args.seed,
                "unit": "image (primary); Roboflow source stem (sensitivity)",
                "paired_enhancement": True,
                "confidence_interval": "95% percentile interval",
                "p_values": "two-sided paired bootstrap; Holm-Bonferroni for three enhancement methods x two metrics",
            },
            "stats_cache": str(STATS_DIR),
            "limitations": [
                "Bootstrap intervals resample existing images; they do not create new domains or annotations.",
                "test_clean is leakage-filtered but still shares source-level provenance with training data; image-level intervals may be too narrow.",
                "The source-stem cluster bootstrap is a sensitivity analysis, not proof of source independence.",
                "Overall mAP intervals require every evaluated class to have at least one ground-truth instance in the resample; finite_resamples reports how many of the requested resamples satisfy this. Rare-class per-class intervals and fraction_of_resamples_present are reported separately.",
                "Enhancement comparisons use one checkpoint, one 399-image subset, and CPU inference.",
                "Bootstrap p-values are approximate and are not a substitute for external validation.",
            ],
        },
        "point_estimates": {key: point[key] for key in point},
        "point_estimate_validation": validation,
        "baseline_bootstrap": baseline_bootstrap,
        "enhancement": {
            "control": CONTROL,
            "methods": ENHANCEMENT_METHODS,
            "point_estimates": enhancement_point,
            "fixed_classes": enhancement_fixed_classes,
            "overall_conditioning": (
                "overall metrics require every fixed class to have at least "
                "one ground-truth instance in the resample"
            ),
            "image_method_summaries": method_summaries,
            "image_paired_bootstrap": image_paired,
            "source_cluster_paired_bootstrap": cluster_entry,
            "primary_summary": primary_summary,
        },
    }
    write_json(OUTPUT_JSON, payload)

    print("\n=== Enhancement paired comparisons (image bootstrap) ===")
    print(
        f"{'method':12s} {'dMAP50':>9s} {'95% CI':>20s} {'Holm p':>9s} "
        f"{'dMAP50-95':>11s} {'95% CI':>20s} {'Holm p':>9s}"
    )
    for row in primary_summary:
        ci50 = row["delta_mAP50_ci"]
        ci95 = row["delta_mAP50_95_ci"]
        p50 = row["delta_mAP50_holm_p"]
        p95 = row["delta_mAP50_95_holm_p"]
        p50_text = "NA" if p50 is None else f"{p50:.4f}"
        p95_text = "NA" if p95 is None else f"{p95:.4f}"
        print(
            f"{row['method']:12s} {row['delta_mAP50']:+9.4f} "
            f"[{ci50[0]:+.4f},{ci50[1]:+.4f}] {p50_text:>9s} "
            f"{row['delta_mAP50_95']:+11.4f} "
            f"[{ci95[0]:+.4f},{ci95[1]:+.4f}] {p95_text:>9s}"
        )
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
