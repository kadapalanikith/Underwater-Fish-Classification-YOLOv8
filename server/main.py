import os
import io
import sys
import time
import json
import base64
import logging
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO, RTDETR
from PIL import Image
import cv2
import numpy as np

# Ensure root directory is on Python path so src modules are accessible
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing.enhancement import UnderwaterEnhancer
from src.tracking.fish_tracker import FishVideoTracker
from src.utils.device import get_optimal_device

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepSea-Major-Backend")

# Model configuration paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "best.pt"))
RTDETR_PATH = os.getenv("RTDETR_PATH", str(BASE_DIR / "models" / "rtdetr-l.pt"))

models_dict = {}
enhancer = UnderwaterEnhancer()
device = "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models_dict, device
    device = get_optimal_device()
    logger.info(f"🚀 Initializing DeepSea AI backend on device: {device}")

    # Load YOLOv8 baseline model
    if os.path.exists(MODEL_PATH):
        try:
            models_dict["yolov8n"] = YOLO(MODEL_PATH)
            logger.info(f"✅ Loaded YOLOv8 fish model from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLOv8 model: {e}")
    else:
        logger.warning(f"⚠️ YOLOv8 checkpoint not found at {MODEL_PATH}")

    # Load RT-DETR model if available
    if os.path.exists(RTDETR_PATH):
        try:
            models_dict["rtdetr"] = RTDETR(RTDETR_PATH)
            logger.info(f"✅ Loaded RT-DETR model from {RTDETR_PATH}")
        except Exception as e:
            logger.warning(f"Could not load custom RT-DETR: {e}")

    yield
    models_dict.clear()


app = FastAPI(
    title="DeepSea AI - Underwater Fish Classification & Tracking API",
    version="2.0.0",
    description="Multi-model underwater fish detection, image enhancement, and video tracking pipeline.",
    lifespan=lifespan
)

def _cors_origins() -> List[str]:
    raw = os.getenv("ALLOW_ORIGINS", "*")
    if raw.strip() == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


CORS_ORIGINS = _cors_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ORIGINS != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "status": "running",
        "system": "DeepSea AI Major Project",
        "available_models": list(models_dict.keys()),
        "enhancement_methods": enhancer.available_methods,
        "device": device
    }


@app.get("/api/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": list(models_dict.keys()),
        "device": device
    }


@app.get("/api/models")
def get_models():
    """Lists available models, parameter counts, and architecture types."""
    info = []
    for name, m in models_dict.items():
        try:
            param_count = sum(p.numel() for p in m.model.parameters())
        except Exception:
            param_count = "N/A"

        is_fish_baseline = name == "yolov8n"
        info.append({
            "id": name,
            "name": "YOLOv8 Nano (Fish Baseline)" if is_fish_baseline else "RT-DETR-L (Pretrained Benchmark)",
            "architecture": "CNN (CSPDarknet)" if is_fish_baseline else "Hybrid Transformer",
            "parameters": param_count,
            "fish_fine_tuned": is_fish_baseline,
            "notes": (
                "Custom fish checkpoint."
                if is_fish_baseline
                else "Pretrained generic checkpoint; not fish fine-tuned and not an accuracy comparator."
            ),
            "active": True
        })
    return {"models": info}


@app.get("/api/metrics")
def get_metrics():
    """Returns the historical baseline, uncertainty, external, and enhancement artifacts."""
    outputs_dir = BASE_DIR / "outputs"

    def load_json(filename):
        path = outputs_dir / filename
        if not path.exists():
            return {}, path
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle), path
        except (OSError, json.JSONDecodeError):
            return {}, path

    audit_data, _ = load_json("enhancement_audit.json")
    benchmark_data, _ = load_json("enhancement_benchmark.json")
    heldout_data, heldout_path = load_json("heldout_yolov8_metrics.json")
    enhancement_accuracy_data, enhancement_accuracy_path = load_json(
        "enhancement_accuracy_test_clean.json"
    )
    latency_data, latency_path = load_json("latency_yolov8_vs_rtdetr.json")
    bootstrap_data, bootstrap_path = load_json("bootstrap_uncertainty.json")
    obsea_data, obsea_path = load_json("obsea_evaluation.json")

    baseline_audit = {
        "model": "YOLOv8n",
        "training_epochs": 10,
        "classes_evaluated": 12,
        "classes_total": 13,
        "validation_metrics": {
            "precision": 0.865,
            "recall": 0.874,
            "mAP50": 0.908,
            "mAP50_95": 0.666
        },
        "audit_notes": "12 of 13 classes had validation instances in minor project. AngelFish had 0 validation instances."
    }

    heldout_summary = {}
    if heldout_data:
        metadata = heldout_data.get("_metadata", {})
        heldout_summary = {
            "source": "outputs/heldout_yolov8_metrics.json",
            "timestamp": metadata.get("timestamp"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "evaluation_protocol": metadata.get("evaluation_protocol"),
            "leakage_filter": metadata.get("leakage_filter"),
            "warnings": metadata.get("warnings", []),
            "results": {
                split: {
                    "images": result.get("images_evaluated"),
                    "instances": result.get("instances_evaluated"),
                    "overall": result.get("overall"),
                    "classes_without_ground_truth": result.get("classes_without_ground_truth"),
                    "low_support_classes": result.get("low_support_classes"),
                }
                for split, result in heldout_data.get("results", {}).items()
            },
        }

    enhancement_accuracy_summary = {}
    if enhancement_accuracy_data:
        enhancement_accuracy_summary = {
            "source": "outputs/enhancement_accuracy_test_clean.json",
            "timestamp": enhancement_accuracy_data.get("timestamp"),
            "dataset": enhancement_accuracy_data.get("dataset"),
            "protocol": enhancement_accuracy_data.get("protocol"),
            "results": {
                method: {
                    "overall": result.get("overall"),
                    "delta_vs_raw_original": result.get("delta_vs_raw_original"),
                    "preprocessing": result.get("preprocessing"),
                }
                for method, result in enhancement_accuracy_data.get("results", {}).items()
            },
        }

    latency_summary = {}
    if latency_data:
        latency_summary = {
            "source": "outputs/latency_yolov8_vs_rtdetr.json",
            "timestamp": latency_data.get("timestamp"),
            "protocol": latency_data.get("protocol"),
            "checkpoints": latency_data.get("checkpoints"),
            "results": {
                key: {
                    "label": result.get("label"),
                    "wall": result.get("wall"),
                    "inference": result.get("inference"),
                    "mean_detections": result.get("mean_detections"),
                }
                for key, result in latency_data.get("results", {}).items()
            },
        }

    bootstrap_summary = {}
    if bootstrap_data:
        metadata = bootstrap_data.get("_metadata", {})
        points = bootstrap_data.get("point_estimates", {})
        baseline = bootstrap_data.get("baseline_bootstrap", {})
        enhancement = bootstrap_data.get("enhancement", {})
        bootstrap_summary = {
            "source": "outputs/bootstrap_uncertainty.json",
            "timestamp": metadata.get("timestamp"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "protocol": metadata.get("evaluation_protocol"),
            "results": {
                split: {
                    "images": summary.get("n_images"),
                    "instances": summary.get("n_instances"),
                    "overall": points.get(split, {}).get("overall"),
                    "intervals": {
                        metric: {
                            "ci_low": summary.get(metric, {}).get("ci_low"),
                            "ci_high": summary.get(metric, {}).get("ci_high"),
                            "finite_resamples": summary.get(metric, {}).get("finite_resamples"),
                            "resamples": summary.get(metric, {}).get("resamples"),
                        }
                        for metric in ("mAP50", "mAP50_95")
                    },
                    "source_stem_clusters": summary.get("source_stem_clusters"),
                    "source_cluster_intervals": {
                        metric: {
                            "ci_low": summary.get("source_cluster_bootstrap", {}).get(metric, {}).get("ci_low"),
                            "ci_high": summary.get("source_cluster_bootstrap", {}).get(metric, {}).get("ci_high"),
                        }
                        for metric in ("mAP50", "mAP50_95")
                    },
                }
                for split, summary in baseline.items()
            },
            "enhancement": {
                "control": enhancement.get("control"),
                "methods": enhancement.get("methods", []),
                "results": {
                    method: {
                        "overall": summary,
                        "intervals": {
                            metric: {
                                "ci_low": summary.get(metric, {}).get("ci_low"),
                                "ci_high": summary.get(metric, {}).get("ci_high"),
                            }
                            for metric in ("mAP50", "mAP50_95")
                        },
                    }
                    for method, summary in enhancement.get("image_method_summaries", {}).items()
                },
                "paired_deltas": {
                    comparison: {
                        metric: {
                            "point": value.get("point"),
                            "ci_low": value.get("ci_low"),
                            "ci_high": value.get("ci_high"),
                            "p_two_sided_holm": value.get("p_two_sided_holm"),
                            "holm_significant_0_05": value.get("holm_significant_0_05"),
                        }
                        for metric, value in result.get("metrics", {}).items()
                        if metric in ("mAP50", "mAP50_95")
                    }
                    for comparison, result in enhancement.get("image_paired_bootstrap", {}).items()
                },
            },
            "limitations": metadata.get("limitations", []),
        }

    obsea_summary = {}
    if obsea_data:
        metadata = obsea_data.get("_metadata", {})
        dataset = metadata.get("external_dataset", {})
        test_audit = obsea_data.get("audit", {}).get("test", {})
        obsea_summary = {
            "source": "outputs/obsea_evaluation.json",
            "timestamp": metadata.get("timestamp"),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"),
            "dataset": {
                "name": dataset.get("name"),
                "doi": dataset.get("doi"),
                "url": dataset.get("url"),
                "license": dataset.get("license"),
                "zip_md5": dataset.get("zip_md5"),
                "test_images": dataset.get("test_images"),
                "fish_classes": len(dataset.get("fish_class_ids", [])),
                "non_fish_classes": dataset.get("non_fish_class_names", []),
            },
            "protocol": metadata.get("class_agnostic_protocol"),
            "test_audit": {
                "images": test_audit.get("images"),
                "images_with_fish": test_audit.get("images_with_fish"),
                "images_without_fish": test_audit.get("images_without_fish"),
                "fish_instances": test_audit.get("fish_instances"),
                "non_fish_instances": test_audit.get("non_fish_instances"),
            },
            "derived_test": obsea_data.get("derived_test"),
            "instances_evaluated": obsea_data.get("instances_evaluated"),
            "overall": obsea_data.get("overall"),
            "speed_ms_per_image": obsea_data.get("speed_ms_per_image"),
            "limitations": metadata.get("limitations", []),
        }

    return {
        "baseline": baseline_audit,
        "heldout_evaluation": heldout_summary,
        "bootstrap_uncertainty": bootstrap_summary,
        "obsea_evaluation": obsea_summary,
        "enhancement_accuracy": enhancement_accuracy_summary,
        "latency_benchmark": latency_summary,
        "enhancement_audit": audit_data,
        "enhancement_benchmark": benchmark_data,
        "provenance": {
            "enhancement_audit_source": "outputs/enhancement_audit.json" if audit_data else None,
            "enhancement_benchmark_source": "outputs/enhancement_benchmark.json" if benchmark_data else None,
            "heldout_evaluation_source": str(heldout_path.relative_to(BASE_DIR)).replace("\\", "/") if heldout_data else None,
            "bootstrap_uncertainty_source": str(bootstrap_path.relative_to(BASE_DIR)).replace("\\", "/") if bootstrap_data else None,
            "obsea_evaluation_source": str(obsea_path.relative_to(BASE_DIR)).replace("\\", "/") if obsea_data else None,
            "enhancement_accuracy_source": str(enhancement_accuracy_path.relative_to(BASE_DIR)).replace("\\", "/") if enhancement_accuracy_data else None,
            "latency_benchmark_source": str(latency_path.relative_to(BASE_DIR)).replace("\\", "/") if latency_data else None,
            "notes": "Saved detection counts, confidence values and timings from the 40-image enhancement audit are not accuracy measurements. Held-out, bootstrap, and enhancement-accuracy metrics use local labels; test_full is not fully independent and test_clean is a leakage-filtered subset. Bootstrap intervals are conditional on all fixed classes being present and do not fix leakage. OBSEA is one external dataset and the result is class-agnostic detection, not 13-class species accuracy. RT-DETR latency is timing only; its checkpoint is pretrained COCO, not fish fine-tuned, and no RT-DETR fine-tuning run has been executed. Video tracking has only synthetic smoke-test coverage."
        }
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    enhancement: str = Form("raw"),
    model_name: str = Form("yolov8n"),
    conf: float = Form(0.25, ge=0.0, le=1.0)
):
    """
    Image inference endpoint. Supports modular underwater enhancement
    and model selection (YOLOv8n vs RT-DETR).
    """
    requested_model = (model_name or "yolov8n").lower()
    selected_model = models_dict.get(requested_model)
    model_fallback = selected_model is None
    if model_fallback:
        selected_model = models_dict.get("yolov8n")
        used_model = "yolov8n"
    else:
        used_model = requested_model

    if not selected_model:
        raise HTTPException(status_code=503, detail="No detection model loaded on server.")

    # Validate file size (< 25MB)
    contents = await file.read()
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 25 MB limit.")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file format.")

    t_start = time.perf_counter()

    # Step 1: Optional Underwater Image Enhancement
    try:
        enhanced_rgb, enh_meta = enhancer.enhance(image, method=enhancement)
    except Exception as e:
        logger.warning(f"Enhancement failed ({e}), falling back to raw image.")
        enhanced_rgb = np.array(image)
        enh_meta = {"method": "raw", "latency_ms": 0.0}

    # Step 2: Object Detection
    t_det_start = time.perf_counter()
    results = selected_model.predict(
        enhanced_rgb,
        conf=conf,
        verbose=False,
        device=device
    )
    res = results[0]
    det_time_ms = round((time.perf_counter() - t_det_start) * 1000.0, 2)
    total_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

    # Step 3: Draw Annotations
    im_array = res.plot()
    _, buffer = cv2.imencode(".jpg", im_array)
    img_str = base64.b64encode(buffer).decode()

    # Step 4: Extract Detection Bounding Boxes & Confidence
    detections = []
    species_summary = {}
    if res.boxes is not None:
        for box in res.boxes:
            cls_id = int(box.cls[0])
            sp = res.names.get(cls_id, f"Fish_{cls_id}")
            cf = float(box.conf[0])
            detections.append({
                "species": sp,
                "confidence": round(cf, 4),
                "bbox": [round(float(c), 2) for c in box.xyxy[0].tolist()]
            })
            species_summary[sp] = species_summary.get(sp, 0) + 1

    return {
        "success": True,
        "model_used": used_model,
        "requested_model": requested_model,
        "model_fallback": model_fallback,
        "enhancement_applied": enh_meta["method"],
        "enhancement_latency_ms": enh_meta["latency_ms"],
        "detection_latency_ms": det_time_ms,
        "total_latency_ms": total_time_ms,
        "total_detections": len(detections),
        "species_summary": species_summary,
        "detections": detections,
        "annotated_image": f"data:image/jpeg;base64,{img_str}"
    }


@app.post("/api/track")
async def track_video(
    file: UploadFile = File(...),
    enhancement: str = Form("raw"),
    model_name: str = Form("yolov8n"),
    conf: float = Form(0.25, ge=0.0, le=1.0),
    line_y: float = Form(0.50, ge=0.0, le=1.0),
    max_frames: int = Form(100, ge=0, le=1000)
):
    """
    Video tracking and automated fish counting endpoint.
    Applies ByteTrack to persist IDs and count fish crossing virtual gate.
    """
    requested_model = (model_name or "yolov8n").lower()
    selected_model = models_dict.get(requested_model)
    model_fallback = selected_model is None
    if model_fallback:
        selected_model = models_dict.get("yolov8n")
        used_model = "yolov8n"
    else:
        used_model = requested_model

    if not selected_model:
        raise HTTPException(status_code=503, detail="No detection model loaded on server.")

    contents = await file.read()
    if len(contents) > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Video size exceeds 100 MB limit.")

    # Save uploaded video to temporary file
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in_path = tmp_in.name
        tmp_in.write(contents)

    tmp_out_path = tmp_in_path.replace(suffix, f"_annotated{suffix}")

    try:
        tracker = FishVideoTracker(
            model=selected_model,
            enhancer=enhancer,
            enhancement_method=enhancement,
            conf_threshold=conf
        )

        tracking_results = tracker.process_video(
            input_video_path=tmp_in_path,
            output_video_path=tmp_out_path,
            line_start=(0.05, line_y),
            line_end=(0.95, line_y),
            max_frames=max_frames
        )

        # Base64 encode the resulting annotated video
        with open(tmp_out_path, "rb") as vf:
            video_bytes = vf.read()
            video_b64 = base64.b64encode(video_bytes).decode("utf-8")

        return {
            "success": True,
            "model_used": used_model,
            "requested_model": requested_model,
            "model_fallback": model_fallback,
            "total_fish_counted": tracking_results["total_fish_counted"],
            "species_counts": tracking_results["species_counts"],
            "frames_processed": tracking_results["total_frames_processed"],
            "crossing_events": tracking_results["crossing_events"],
            "video_base64": f"data:video/mp4;base64,{video_b64}"
        }

    except Exception as e:
        logger.error(f"Tracking error: {e}")
        raise HTTPException(status_code=500, detail=f"Tracking pipeline failed: {str(e)}")

    finally:
        # Cleanup temp files
        for p in [tmp_in_path, tmp_out_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


if __name__ == "__main__":
    import uvicorn

    if os.name == "nt":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    port = int(os.getenv("PORT", 8080))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting DeepSea AI Major Server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
