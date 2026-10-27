"""Run the fish tracker on the acquired DIGI4ECO OBSEA clip and compare with CVAT.

This is a bounded CPU evaluation of one 150-frame sequence. It reports
descriptive matching and counting results; it does not fine-tune the detector.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from collections import defaultdict
from statistics import mean
import time
import xml.etree.ElementTree as ET
import yaml

import cv2
import torch
import ultralytics
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracking.fish_tracker import FishVideoTracker


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "external_data/digi4eco/extracted/OBSEA/video.mp4"
DEFAULT_ANNOTATIONS = ROOT / "external_data/digi4eco/extracted/OBSEA/annotations.xml"
DEFAULT_CHECKPOINT = ROOT / "models/best.pt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reference_boxes(annotation_path: Path) -> tuple[dict[int, list[dict]], dict]:
    """Read visible, non-occluded CVAT rectangles grouped by frame."""
    root = ET.parse(annotation_path).getroot()
    width = int(root.findtext("./meta/original_size/width", "0"))
    height = int(root.findtext("./meta/original_size/height", "0"))
    job = root.find("./meta/job")
    if width <= 0 or height <= 0 or job is None:
        raise ValueError("CVAT XML must contain original dimensions and job metadata")

    by_frame: dict[int, list[dict]] = {}
    visible_box_count = 0
    for track in root.findall("./track"):
        track_id = str(track.get("id"))
        for box in track.findall("./box"):
            if box.get("outside") == "1" or box.get("occluded") == "1":
                continue
            frame = int(box.get("frame"))
            observation = {
                "track_id": track_id,
                "xyxy": [
                    float(box.get("xtl")),
                    float(box.get("ytl")),
                    float(box.get("xbr")),
                    float(box.get("ybr")),
                ],
            }
            by_frame.setdefault(frame, []).append(observation)
            visible_box_count += 1

    metadata = {
        "width": width,
        "height": height,
        "start_frame": int(job.findtext("start_frame", "0")),
        "stop_frame": int(job.findtext("stop_frame", "0")),
        "track_count": len(root.findall("./track")),
        "all_box_count": sum(len(track.findall("./box")) for track in root.findall("./track")),
        "visible_unoccluded_box_count": visible_box_count,
    }
    return by_frame, metadata


def inspect_video(video_path: Path) -> dict:
    """Read container metadata without decoding or altering the source video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video file: {video_path}")
    metadata = {
        "declared_frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": capture.get(cv2.CAP_PROP_FPS),
    }
    capture.release()
    return metadata


def box_iou(a: list[float], b: list[float]) -> float:
    """Calculate intersection-over-union for two xyxy boxes."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def match_frame(reference: list[dict], predicted: list[dict], threshold: float) -> list[dict]:
    """Greedily match highest-IoU pairs once per frame, without species labels."""
    candidates = []
    for gt_index, gt in enumerate(reference):
        for pred_index, pred in enumerate(predicted):
            overlap = box_iou(gt["xyxy"], pred["xyxy"])
            if overlap >= threshold:
                candidates.append((overlap, gt_index, pred_index))

    matches = []
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    for overlap, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append(
            {
                "ground_truth_id": reference[gt_index]["track_id"],
                "predicted_id": int(predicted[pred_index]["track_id"]),
                "iou": round(overlap, 4),
            }
        )
    return matches


def summarize_id_matching(
    reference_by_frame: dict[int, list[dict]],
    predicted_by_frame: dict[int, list[dict]],
    *,
    iou_threshold: float,
) -> dict:
    """Report frame matching, track-ID fragmentation, and adjacent-frame switches."""
    matched_boxes = 0
    gt_boxes = sum(len(boxes) for boxes in reference_by_frame.values())
    predicted_boxes = sum(len(boxes) for boxes in predicted_by_frame.values())
    id_sets: dict[str, set[int]] = {}
    last_match: dict[str, tuple[int, int]] = {}
    switches = []

    for frame in sorted(set(reference_by_frame) | set(predicted_by_frame)):
        matches = match_frame(
            reference_by_frame.get(frame, []),
            predicted_by_frame.get(frame, []),
            iou_threshold,
        )
        matched_boxes += len(matches)
        for match in matches:
            gt_id = match["ground_truth_id"]
            pred_id = match["predicted_id"]
            id_sets.setdefault(gt_id, set()).add(pred_id)
            previous = last_match.get(gt_id)
            if previous and previous[0] + 1 == frame and previous[1] != pred_id:
                switches.append(
                    {
                        "ground_truth_id": gt_id,
                        "frame": frame,
                        "previous_predicted_id": previous[1],
                        "current_predicted_id": pred_id,
                    }
                )
            last_match[gt_id] = (frame, pred_id)

    gt_track_count = len({item["track_id"] for boxes in reference_by_frame.values() for item in boxes})
    fragmented = {track_id: sorted(ids) for track_id, ids in id_sets.items() if len(ids) > 1}
    return {
        "iou_match_threshold": iou_threshold,
        "reference_visible_unoccluded_boxes": gt_boxes,
        "predicted_tracked_boxes": predicted_boxes,
        "matched_boxes": matched_boxes,
        "box_precision_at_iou_threshold": round(matched_boxes / predicted_boxes, 4) if predicted_boxes else 0.0,
        "box_recall_at_iou_threshold": round(matched_boxes / gt_boxes, 4) if gt_boxes else 0.0,
        "reference_tracks_with_any_match": len(id_sets),
        "reference_track_count": gt_track_count,
        "reference_tracks_with_multiple_predicted_ids": fragmented,
        "identity_switches_on_consecutive_matched_frames": switches,
        "identity_switch_count": len(switches),
        "identity_scope_note": "ID checks use class-agnostic greedy IoU matches; gaps and unmatched frames are not counted as adjacent-frame switches.",
    }


def match_crossing_events(reference: list[dict], predicted: list[dict], tolerance: int) -> dict:
    """Greedily match count events within a fixed frame tolerance."""
    pairs = []
    for gt_index, gt in enumerate(reference):
        for pred_index, pred in enumerate(predicted):
            frame_delta = abs(int(gt["frame_after"]) - int(pred["frame"]))
            if frame_delta <= tolerance:
                pairs.append((frame_delta, gt_index, pred_index))
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for delta, gt_index, pred_index in sorted(pairs):
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matches.append(
            {
                "reference_track_id": reference[gt_index]["track_id"],
                "predicted_track_id": int(predicted[pred_index]["track_id"]),
                "reference_frame": int(reference[gt_index]["frame_after"]),
                "predicted_frame": int(predicted[pred_index]["frame"]),
                "absolute_frame_delta": delta,
            }
        )
    precision = len(matches) / len(predicted) if predicted else 0.0
    recall = len(matches) / len(reference) if reference else 0.0
    return {
        "frame_tolerance": tolerance,
        "reference_event_count": len(reference),
        "predicted_event_count": len(predicted),
        "matched_event_count": len(matches),
        "event_precision": round(precision, 4),
        "event_recall": round(recall, 4),
        "event_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "matched_events": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/digi4eco_obsea_tracking_validation.json")
    parser.add_argument("--annotated-video", type=Path, help="Optional annotated video output; never overwrites")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--event-tolerance", type=int, default=2)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()

    for required in (args.video, args.annotations, args.checkpoint):
        if not required.is_file():
            parser.error(f"required input does not exist: {required}")
    for destination in (args.output, args.annotated_video):
        if destination is None:
            continue
        if destination.exists():
            parser.error(f"refusing to overwrite existing output: {destination}")

    video_metadata = inspect_video(args.video)
    reference_by_frame, annotation_metadata = load_reference_boxes(args.annotations)
    video_frame_count = video_metadata["declared_frame_count"]
    annotation_boxes_beyond_video = sum(
        len(boxes) for frame_index, boxes in reference_by_frame.items() if frame_index >= video_frame_count
    )
    # Do not penalize the detector for annotated frames that are absent from the video file.
    reference_by_frame = {
        frame_index: boxes
        for frame_index, boxes in reference_by_frame.items()
        if frame_index < video_frame_count
    }
    model = YOLO(str(args.checkpoint))
    tracker_config_path = Path(ultralytics.__file__).parent / "cfg/trackers/bytetrack.yaml"
    tracker_config = yaml.safe_load(tracker_config_path.read_text(encoding="utf-8"))
    tracker = FishVideoTracker(
        model=model,
        conf_threshold=args.confidence,
        iou_threshold=args.iou,
        tracker_type="bytetrack.yaml",
        device=args.device,
    )

    started = time.perf_counter()
    tracking_result = tracker.process_video(
        input_video_path=str(args.video),
        output_video_path=str(args.annotated_video) if args.annotated_video else None,
        line_start=(0.05, 0.5),
        line_end=(0.95, 0.5),
        max_frames=annotation_metadata["stop_frame"] - annotation_metadata["start_frame"] + 1,
        include_track_observations=True,
    )
    runtime_sec = time.perf_counter() - started

    observations = tracking_result.pop("track_observations")
    predicted_by_frame = {item["frame"]: item["tracks"] for item in observations}
    predictions_by_id: dict[int, list[dict]] = defaultdict(list)
    predicted_observation_count = 0
    confidence_values = []
    raw_detection_count = 0
    raw_detection_confidences = []
    raw_detections_without_track_id = 0
    for frame in observations:
        for detection in frame["detections"]:
            raw_detection_count += 1
            raw_detection_confidences.append(detection["confidence"])
            if detection["track_id"] is None:
                raw_detections_without_track_id += 1
        for track in frame["tracks"]:
            predicted_observation_count += 1
            confidence_values.append(track["confidence"])
            predictions_by_id[track["track_id"]].append(
                {"frame": frame["frame"], "class_id": track["class_id"], "confidence": track["confidence"]}
            )
    predicted_track_summary = []
    for track_id, samples in sorted(predictions_by_id.items()):
        predicted_track_summary.append(
            {
                "track_id": track_id,
                "first_frame": samples[0]["frame"],
                "last_frame": samples[-1]["frame"],
                "observed_frames": len(samples),
                "mean_confidence": round(mean(sample["confidence"] for sample in samples), 4),
                "class_ids_seen": sorted({sample["class_id"] for sample in samples}),
            }
        )
    predicted_events = tracking_result["crossing_events"]
    if tracking_result["total_fish_counted"] != len(predicted_events):
        raise RuntimeError("Tracker count total does not match its crossing event log")
    reference_result = analyze_reference_for_gate(args.annotations)
    reference_events = reference_result["crossing_events"]
    expected_count = reference_result["expected_total_crossings"]
    predicted_count = tracking_result["total_fish_counted"]
    signed_error = predicted_count - expected_count
    annotated_video_result = None
    if args.annotated_video:
        annotated_capture = cv2.VideoCapture(str(args.annotated_video))
        if not annotated_capture.isOpened():
            annotated_capture.release()
            raise RuntimeError("Annotated output video could not be reopened for verification")
        annotated_video_metadata = {
            "frame_count": int(annotated_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(annotated_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(annotated_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": annotated_capture.get(cv2.CAP_PROP_FPS),
        }
        annotated_capture.release()
        annotated_video_result = {
            "path": str(args.annotated_video),
            "sha256": sha256_file(args.annotated_video),
            **annotated_video_metadata,
        }

    result = {
        "status": "completed_descriptive_real_video_evaluation",
        "dataset": {
            "name": "DIGI4ECO Underwater Fish Tracking Dataset, OBSEA sequence",
            "doi": "10.5281/zenodo.20701225",
            "license": "CC BY 4.0",
            "annotation_metadata": annotation_metadata,
            "video_metadata": video_metadata,
            "annotation_boxes_beyond_decodable_video": annotation_boxes_beyond_video,
        },
        "inputs": {
            "video": str(args.video),
            "video_sha256": sha256_file(args.video),
            "annotations": str(args.annotations),
            "annotations_sha256": sha256_file(args.annotations),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": __import__("torchvision").__version__,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_used": "cpu",
        },
        "settings": {
            "confidence": args.confidence,
            "iou": args.iou,
            "tracker": "bytetrack.yaml",
            "tracker_thresholds": {
                key: tracker_config[key]
                for key in ("track_high_thresh", "track_low_thresh", "new_track_thresh")
            },
            "imgsz": 640,
            "counting_gate": {"start": [0.05, 0.5], "end": [0.95, 0.5]},
            "crossing_match_tolerance_frames": args.event_tolerance,
            "box_match_iou_threshold": args.match_iou,
            "class_agnostic_reference_matching": True,
        },
        "runtime": {
            "total_seconds": round(runtime_sec, 3),
            "milliseconds_per_processed_frame": round(runtime_sec * 1000 / max(1, tracking_result["total_frames_processed"]), 3),
        },
        "tracking_result": tracking_result,
        "predicted_track_summary": predicted_track_summary,
        "prediction_summary": {
            "raw_detections_after_confidence_filter": raw_detection_count,
            "raw_detections_without_tracker_id": raw_detections_without_track_id,
            "mean_raw_detection_confidence": round(mean(raw_detection_confidences), 4) if raw_detection_confidences else None,
            "minimum_raw_detection_confidence": round(min(raw_detection_confidences), 4) if raw_detection_confidences else None,
            "maximum_raw_detection_confidence": round(max(raw_detection_confidences), 4) if raw_detection_confidences else None,
            "raw_detections_below_tracker_low_threshold": sum(
                confidence < tracker_config["track_low_thresh"] for confidence in raw_detection_confidences
            ),
            "unique_predicted_track_ids": len(predictions_by_id),
            "tracked_box_observations": predicted_observation_count,
            "mean_detection_confidence": round(mean(confidence_values), 4) if confidence_values else None,
        },
        "id_matching": summarize_id_matching(
            reference_by_frame,
            predicted_by_frame,
            iou_threshold=args.match_iou,
        ),
        "count_comparison": {
            "reference_crossings": expected_count,
            "predicted_crossings": predicted_count,
            "signed_error_predicted_minus_reference": signed_error,
            "absolute_error": abs(signed_error),
            "absolute_relative_error": round(abs(signed_error) / expected_count, 4) if expected_count else None,
            "crossing_event_matching": match_crossing_events(reference_events, predicted_events, args.event_tolerance),
        },
        "annotated_video": annotated_video_result,
        "limitations": [
            "This is one short OBSEA video, not evidence of general tracking performance.",
            "The model predicts 13 local species classes, while this sequence is annotated as one fish class; box/ID/count matching is class-agnostic.",
            "Reference boxes marked outside or occluded are excluded from frame-level box matching and gate-crossing extraction.",
            "The video container has 148 frames while the annotation XML covers frames 0–149; frame annotations beyond the video are excluded from box matching.",
            "Frame-level ID association uses greedy IoU matching, not TrackEval; MOTA/IDF1/HOTA are not reported.",
            "No species-level accuracy is computed for this video.",
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.annotated_video:
        args.annotated_video.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Result saved: {args.output}")
    if args.annotated_video:
        print(f"Annotated video saved: {args.annotated_video}")


def analyze_reference_for_gate(annotation_path: Path) -> dict:
    """Import the project parser by module path so this runner has one gate rule."""
    from scratch.analyze_cvat_gate_reference import analyze_reference

    return analyze_reference(
        annotation_path,
        line_y_fraction=0.5,
        gate_start_x_fraction=0.05,
        gate_end_x_fraction=0.95,
    )


if __name__ == "__main__":
    main()
