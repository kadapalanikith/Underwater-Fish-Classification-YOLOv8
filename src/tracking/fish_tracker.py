"""
Video Tracking and Automated Fish Counting Module
Uses ByteTrack / BoT-SORT multi-object tracking to assign persistent IDs to fish,
track movement trajectories, and count crossings across virtual gates/lines without duplicates.
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
from src.utils.device import get_optimal_device


class FishVideoTracker:
    def __init__(
        self,
        model,
        enhancer=None,
        enhancement_method: str = "raw",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        tracker_type: str = "bytetrack.yaml",
        device: Optional[str] = None
    ):
        """
        Args:
            model: Ultralytics YOLO or RT-DETR model instance.
            enhancer: UnderwaterEnhancer instance (optional).
            enhancement_method: Enhancement method ('raw', 'clahe', 'gray_world', 'fusion').
            conf_threshold: Minimum detection confidence.
            iou_threshold: NMS IoU threshold.
            tracker_type: Tracking configuration ('bytetrack.yaml' or 'botsort.yaml').
            device: Device string ('cpu' or 'cuda:0'). Defaults to get_optimal_device().
        """
        self.model = model
        self.enhancer = enhancer
        self.enhancement_method = enhancement_method
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_type = tracker_type
        self.device = device or get_optimal_device()

    @staticmethod
    def _ccw(A, B, C):
        """Tests if 3 points are in counter-clockwise order."""
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    @classmethod
    def _intersect(cls, p1, p2, p3, p4):
        """Returns True if line segment p1p2 intersects line segment p3p4."""
        return (cls._ccw(p1, p3, p4) != cls._ccw(p2, p3, p4)) and (cls._ccw(p1, p2, p3) != cls._ccw(p1, p2, p4))

    def process_video(
        self,
        input_video_path: str,
        output_video_path: Optional[str] = None,
        line_start: Optional[Tuple[float, float]] = None,
        line_end: Optional[Tuple[float, float]] = None,
        max_frames: Optional[int] = None,
        include_track_observations: bool = False
    ) -> Dict[str, Any]:
        """
        Tracks fish in video and counts line crossings.

        Args:
            input_video_path: Path to input MP4/AVI video.
            output_video_path: Path to write annotated output video (if requested).
            line_start: Normalized (x, y) coordinates of virtual gate start [0.0 - 1.0].
            line_end: Normalized (x, y) coordinates of virtual gate end [0.0 - 1.0].
            max_frames: Limit frames for quick evaluation.
            include_track_observations: Include per-frame boxes and IDs for evaluation. Leave
                disabled for long videos to avoid retaining all detections in memory.

        Returns:
            Dictionary with tracking metrics, species counts, and event log.
        """
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be a positive integer when provided")

        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video file: {input_video_path}")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError(f"Video has invalid frame dimensions: {width}x{height}")
        if not np.isfinite(fps) or fps <= 0:
            fps = 25.0

        # Default virtual counting line: horizontal line at 50% height
        gate_p1 = (
            int((line_start[0] if line_start else 0.05) * width),
            int((line_start[1] if line_start else 0.50) * height)
        )
        gate_p2 = (
            int((line_end[0] if line_end else 0.95) * width),
            int((line_end[1] if line_end else 0.50) * height)
        )

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                writer.release()
                cap.release()
                raise ValueError(f"Could not create output video file: {output_video_path}")

        track_history = defaultdict(list)
        # A track contributes at most one crossing for the whole video, regardless of direction.
        # This is a unique-fish counter, not an in/out counter; a fish crossing back is not counted again.
        counted_track_ids = set()
        species_counts = defaultdict(int)
        crossing_events = []
        track_observations = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if max_frames is not None and frame_idx >= max_frames:
                break

            # Frame enhancement if requested
            if self.enhancer and self.enhancement_method != "raw":
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                enhanced_rgb, _ = self.enhancer.enhance(frame_rgb, method=self.enhancement_method)
                infer_frame = cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
            else:
                infer_frame = frame

            # Run detection + tracking
            try:
                results = self.model.track(
                    infer_frame,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    persist=True,
                    tracker=self.tracker_type,
                    device=self.device,
                    verbose=False
                )
            except Exception:
                # Release file handles before propagating model or tracker errors.
                cap.release()
                if writer:
                    writer.release()
                raise

            annotated_frame = frame.copy()
            # Draw counting line
            cv2.line(annotated_frame, gate_p1, gate_p2, (0, 255, 255), 2)
            cv2.putText(
                annotated_frame,
                "Virtual Counting Gate",
                (gate_p1[0], max(20, gate_p1[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            frame_tracks = []
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                classes = results[0].boxes.cls.int().cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                names = results[0].names

                for box, track_id, cls_id, conf in zip(boxes, track_ids, classes, confs):
                    frame_tracks.append({
                        "track_id": int(track_id),
                        "class_id": int(cls_id),
                        "confidence": float(conf),
                        "xyxy": [float(value) for value in box],
                    })
                    x1, y1, x2, y2 = map(int, box)
                    centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
                    species_name = names.get(cls_id, f"Fish_{cls_id}")

                    # Update track history
                    history = track_history[track_id]
                    history.append(centroid)
                    if len(history) > 30:
                        history.pop(0)

                    # Compare consecutive centroids so a stationary detection on the gate is not counted.
                    # The event is recorded only once per tracker ID by the policy above.
                    if len(history) >= 2 and track_id not in counted_track_ids:
                        prev_point = history[-2]
                        curr_point = history[-1]

                        if self._intersect(prev_point, curr_point, gate_p1, gate_p2):
                            counted_track_ids.add(track_id)
                            species_counts[species_name] += 1
                            crossing_events.append({
                                "frame": frame_idx,
                                "timestamp_sec": round(frame_idx / fps, 2),
                                "track_id": int(track_id),
                                "species": species_name,
                                "confidence": round(float(conf), 3),
                                "crossing_point": curr_point
                            })

                    # Draw bounding box and track ID
                    color = (0, 200, 0) if track_id in counted_track_ids else (255, 100, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = f"#{track_id} {species_name} {conf:.2f}"
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                    # Draw movement trail
                    if len(history) > 1:
                        points = np.array(history, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], False, color, 1)

            if include_track_observations:
                frame_detections = []
                if results and results[0].boxes:
                    detection_boxes = results[0].boxes.xyxy.cpu().numpy()
                    detection_classes = results[0].boxes.cls.int().cpu().numpy()
                    detection_confidences = results[0].boxes.conf.cpu().numpy()
                    detection_ids = (
                        results[0].boxes.id.int().cpu().numpy()
                        if results[0].boxes.id is not None
                        else [None] * len(detection_boxes)
                    )
                    for box, class_id, confidence, track_id in zip(
                        detection_boxes, detection_classes, detection_confidences, detection_ids
                    ):
                        frame_detections.append({
                            "track_id": int(track_id) if track_id is not None else None,
                            "class_id": int(class_id),
                            "confidence": float(confidence),
                            "xyxy": [float(value) for value in box],
                        })
                track_observations.append(
                    {"frame": frame_idx, "tracks": frame_tracks, "detections": frame_detections}
                )

            # Display real-time total count on frame
            total_counted = len(counted_track_ids)
            overlay_text = f"Counted: {total_counted} fish"
            cv2.rectangle(annotated_frame, (10, 10), (280, 50), (0, 0, 0), -1)
            cv2.putText(
                annotated_frame,
                overlay_text,
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            if writer:
                writer.write(annotated_frame)

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()

        output = {
            "total_frames_processed": frame_idx,
            "total_fish_counted": len(counted_track_ids),
            "species_counts": dict(species_counts),
            "crossing_events": crossing_events,
            "active_tracks_tracked": len(track_history),
            "output_video_path": output_video_path
        }
        if include_track_observations:
            output["track_observations"] = track_observations
        return output
