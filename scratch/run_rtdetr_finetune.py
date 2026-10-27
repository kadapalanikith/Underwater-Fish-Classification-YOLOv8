"""Opt-in RT-DETR-L fine-tuning and evaluation for the local fish dataset.

The default invocation changes nothing. Training and evaluation require the
explicit --execute flag. Data preparation writes only derived hard-linked
copies under outputs/derived/. This script does not install packages, download
weights, or approve a training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

SCRIPT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SCRIPT_ROOT / "scratch"))

from eval_common import (  # noqa: E402
    CLASS_NAMES,
    DERIVED,
    OUTPUTS,
    RTDETR_CHECKPOINT,
    environment_info,
    prepare_dataset,
    timestamp,
    write_json,
)

TRAIN_DIR = DERIVED / "rtdetr_fish_train"
VALID_DIR = DERIVED / "rtdetr_fish_valid"
DATA_YAML = DERIVED / "rtdetr_fish" / "data.yaml"
PILOT_JSON = OUTPUTS / "rtdetr_pilot.json"
METRICS_JSON = OUTPUTS / "rtdetr_finetune_metrics.json"
RUNS_DIR = OUTPUTS / "rtdetr_runs"
EVAL_SPLITS = ("test_clean", "test_full")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--check-gpu", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", metavar="CHECKPOINT")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Required for --pilot, --train, and --evaluate.",
    )
    return parser.parse_args()


def require_execute(args):
    if not args.execute:
        print("No action taken. Re-run with --execute after reviewing the plan.")
        print("Plan: docs/rtdetr-finetune-plan.md")
        return False
    return True


def check_gpu():
    import torch

    print(json.dumps(environment_info(), indent=2))
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA build: {torch.version.cuda}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("GPU gate failed: CUDA is not available.")
        return 2
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"capability: {torch.cuda.get_device_capability(0)}")
    try:
        x = torch.ones((512, 512), device="cuda")
        value = (x @ x).sum().item()
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"GPU gate failed: {type(exc).__name__}: {exc}")
        return 2
    print(f"GPU tensor operation passed: {value}")
    return 0


def prepare_data(force=False):
    if force or not (TRAIN_DIR / "images").exists():
        print("Preparing RT-DETR training images...")
        prepare_dataset("train", TRAIN_DIR.name)
    else:
        print(f"Reusing {TRAIN_DIR}")
    if force or not (VALID_DIR / "images").exists():
        print("Preparing RT-DETR validation images...")
        prepare_dataset("valid", VALID_DIR.name)
    else:
        print(f"Reusing {VALID_DIR}")

    data = {
        "path": DATA_YAML.parent.as_posix(),
        "train": f"../{TRAIN_DIR.name}/images",
        "val": f"../{VALID_DIR.name}/images",
        "nc": len(CLASS_NAMES),
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    DATA_YAML.write_text(
        yaml.safe_dump(data, sort_keys=False),
        encoding="utf-8",
    )
    print(f"Wrote {DATA_YAML}")
    return data


def require_data():
    if not DATA_YAML.exists():
        raise SystemExit(
            "Training data is not prepared. Run --prepare-data first."
        )
    return DATA_YAML


def load_model(checkpoint):
    from ultralytics import RTDETR

    return RTDETR(str(checkpoint))


def run_training(args, pilot=False):
    import torch

    require_data()
    epochs = 1 if pilot else args.epochs
    name = "pilot" if pilot else f"rtdetr_l_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / name
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    device = str(args.device)
    if device != "cpu" and not torch.cuda.is_available():
        raise SystemExit("Requested GPU training but CUDA is unavailable.")
    if device != "cpu":
        try:
            x = torch.ones((256, 256), device="cuda")
            (x @ x).sum().item()
            torch.cuda.synchronize()
        except Exception as exc:
            raise SystemExit(f"GPU tensor gate failed: {type(exc).__name__}: {exc}")

    model = load_model(RTDETR_CHECKPOINT)
    started = time.perf_counter()
    if torch.cuda.is_available() and device != "cpu":
        torch.cuda.reset_peak_memory_stats()
    result = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        amp=True,
        cache=False,
        pretrained=True,
        project=str(RUNS_DIR),
        name=name,
        exist_ok=False,
        plots=not pilot,
        val=True,
        verbose=True,
    )
    elapsed = time.perf_counter() - started
    peak = None
    if torch.cuda.is_available() and device != "cpu":
        peak = torch.cuda.max_memory_allocated() / (1024**3)

    best = run_dir / "weights" / "best.pt"
    payload = {
        "_metadata": {
            "timestamp": timestamp(),
            "script": "scratch/run_rtdetr_finetune.py",
            "mode": "pilot" if pilot else "train",
            "checkpoint_start": str(RTDETR_CHECKPOINT),
            "data_yaml": str(DATA_YAML),
            "epochs": epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "device": device,
            "workers": args.workers,
            "seed": args.seed,
            "amp": True,
            "elapsed_seconds": elapsed,
            "peak_cuda_allocated_gb": peak,
            "run_dir": str(run_dir),
            "best_checkpoint": str(best) if best.exists() else None,
            "ultralytics_result": str(type(result).__name__),
            "limitations": [
                "Pilot metrics are one epoch and are not a final comparison.",
                "No YOLO baseline was retrained in this run.",
                "COCO-pretrained initialization does not make this a fish accuracy result.",
            ],
        }
    }
    output = PILOT_JSON if pilot else METRICS_JSON
    write_json(output, payload)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output}")
    return 0


def metrics_for_split(model, data_yaml, args, split):
    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=args.imgsz,
        conf=0.001,
        iou=0.6,
        max_det=300,
        device=args.device,
        workers=0,
        plots=False,
        verbose=False,
    )
    box = metrics.box
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    per_class = {}
    for index, class_id in enumerate(box.ap_class_index.tolist()):
        name = names.get(class_id, str(class_id))
        per_class[name] = {
            "class_id": int(class_id),
            "ap50": float(box.ap50[index]),
            "ap50_95": float(box.ap[index]),
            "precision": float(box.p[index]),
            "recall": float(box.r[index]),
        }
    return {
        "split": split,
        "data_yaml": str(data_yaml),
        "images": len(getattr(metrics, "images", []) or []),
        "overall": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "mAP50": float(box.map50),
            "mAP50_95": float(box.map),
            "mAP75": float(box.map75),
        },
        "per_class": per_class,
        "speed_ms_per_image": {
            key: float(value) for key, value in (getattr(metrics, "speed", {}) or {}).items()
        },
    }


def evaluate(args):
    require_data()
    checkpoint = Path(args.evaluate)
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    model = load_model(checkpoint)
    results = {}
    for split in EVAL_SPLITS:
        data_yaml = DERIVED / split / "data.yaml"
        if not data_yaml.exists():
            results[split] = {"error": f"Missing data.yaml: {data_yaml}"}
            continue
        print(f"Evaluating {checkpoint} on {split}...")
        results[split] = metrics_for_split(model, data_yaml, args, split)
    payload = {
        "_metadata": {
            "timestamp": timestamp(),
            "script": "scratch/run_rtdetr_finetune.py",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "environment": environment_info(),
            "settings": {
                "imgsz": args.imgsz,
                "conf": 0.001,
                "iou": 0.6,
                "max_det": 300,
                "device": args.device,
                "multi_label": True,
            },
            "protocol": "Same derived test sets and validation settings as the YOLOv8 held-out report.",
            "limitations": [
                "Metrics are only valid for the specified checkpoint and test sets.",
                "test_full is not fully independent; test_clean is leakage-filtered but not source-independent.",
                "No species mapping is applied to OBSEA.",
            ],
        },
        "results": results,
    }
    write_json(METRICS_JSON, payload)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {METRICS_JSON}")
    return 0


def main():
    args = parse_args()
    actions = [args.prepare_data, args.check_gpu, args.pilot, args.train, bool(args.evaluate)]
    if not any(actions):
        print("RT-DETR fine-tuning is opt-in; no action taken.")
        print("Plan: docs/rtdetr-finetune-plan.md")
        print("Data: --prepare-data")
        print("GPU gate: --check-gpu")
        print("Pilot: --pilot --execute")
        print("Full run: --train --execute --epochs 10")
        print("Evaluate: --evaluate PATH --execute")
        return 0

    if args.prepare_data:
        prepare_data(force=args.force_data)
        if not any([args.check_gpu, args.pilot, args.train, args.evaluate]):
            return 0
    if args.check_gpu:
        code = check_gpu()
        if code or not any([args.pilot, args.train, args.evaluate]):
            return code
    if args.pilot:
        if not require_execute(args):
            return 0
        return run_training(args, pilot=True)
    if args.train:
        if not require_execute(args):
            return 0
        return run_training(args, pilot=False)
    if args.evaluate:
        if not require_execute(args):
            return 0
        return evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
